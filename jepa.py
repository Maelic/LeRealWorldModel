"""JEPA Implementation"""

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

def detach_clone(v):
    return v.detach().clone() if torch.is_tensor(v) else v

class JEPA(nn.Module):

    def __init__(
        self,
        encoder,
        predictor,
        action_encoder,
        projector=None,
        pred_proj=None,
        cam_fuser=None,
    ):
        super().__init__()

        self.encoder = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.projector = projector or nn.Identity()
        self.pred_proj = pred_proj or nn.Identity()
        self.cam_fuser = cam_fuser  # None = single-camera mode

    def _encode_pixels(self, pixels: torch.Tensor) -> torch.Tensor:
        """Encode a (B, T, C, H, W) pixel tensor → (B, T, embed_dim)."""
        b = pixels.size(0)
        flat = rearrange(pixels, "b t ... -> (b t) ...")
        cls = self.encoder(flat, interpolate_pos_encoding=True).last_hidden_state[:, 0]
        emb = self.projector(cls)
        return rearrange(emb, "(b t) d -> b t d", b=b)

    def encode(self, info):
        """Encode observations and actions into embeddings.
        info: dict with pixels (and optionally pixels2) and action keys.
        """
        emb = self._encode_pixels(info['pixels'].float())

        if 'pixels2' in info and self.cam_fuser is not None:
            emb2 = self._encode_pixels(info['pixels2'].float())
            b, t, d = emb.shape
            fused = torch.cat([emb, emb2], dim=-1)          # (B, T, 2*embed_dim)
            emb = self.cam_fuser(rearrange(fused, "b t d -> (b t) d"))
            emb = rearrange(emb, "(b t) d -> b t d", b=b)

        info["emb"] = emb

        if "action" in info:
            info["act_emb"] = self.action_encoder(info["action"])

        return info

    def predict(self, emb, act_emb):
        """Predict next state embedding
        emb: (B, T, D)
        act_emb: (B, T, A_emb)
        """
        preds = self.predictor(emb, act_emb)
        preds = self.pred_proj(rearrange(preds, "b t d -> (b t) d"))
        preds = rearrange(preds, "(b t) d -> b t d", b=emb.size(0))
        return preds

    ####################
    ## Inference only ##
    ####################

    def rollout(self, info, action_sequence, history_size: int = 3):
        """Rollout the model given an initial info dict and action sequence.
        pixels: (B, S, T, C, H, W)
        action_sequence: (B, S, T, action_dim)
         - S is the number of action plan samples
         - T is the time horizon
        """

        assert "pixels" in info, "pixels not in info_dict"
        H = info["pixels"].size(2)
        B, S, T = action_sequence.shape[:3]
        act_0, act_future = torch.split(action_sequence, [H, T - H], dim=2)
        info["action"] = act_0
        n_steps = T - H

        # Encode initial history with delta actions
        _init = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
        act_0_init = _init["action"]  # (B, H, A)
        # Use prev_action as anchor for Δa[0] if provided (mid-episode anchor fix)
        if "prev_action" in _init:
            prev_anchor = _init.pop("prev_action").unsqueeze(1)  # (B, 1, A)
        else:
            prev_anchor = torch.zeros_like(act_0_init[:, :1])
        prev_init = torch.cat([prev_anchor, act_0_init[:, :-1]], dim=1)
        _init["action"] = act_0_init - prev_init
        _init = self.encode(_init)
        emb = info["emb"] = _init["emb"].unsqueeze(1).expand(B, S, -1, -1)
        _init = {k: detach_clone(v) for k, v in _init.items()}

        # flatten batch and sample dimensions for rollout
        emb = rearrange(emb, "b s ... -> (b s) ...").clone()
        act = rearrange(act_0, "b s ... -> (b s) ...")
        act_future = rearrange(act_future, "b s ... -> (b s) ...")

        # compute delta actions for history; track prev_act for future steps
        if "prev_action" in info and torch.is_tensor(info["prev_action"]):
            prev_action_flat = rearrange(info["prev_action"], "b s a -> (b s) a")  # (BS, A)
            prev_anchor_flat = prev_action_flat.unsqueeze(1)  # (BS, 1, A)
        else:
            prev_anchor_flat = torch.zeros_like(act[:, :1])
        prev = torch.cat([prev_anchor_flat, act[:, :-1]], dim=1)
        act_delta = act - prev  # (BS, H, A)
        prev_act = act[:, -1:]  # last history action

        # rollout predictor autoregressively for n_steps
        HS = history_size
        for t in range(n_steps):
            act_emb = self.action_encoder(act_delta)
            emb_trunc = emb[:, -HS:]  # (BS, HS, D)
            act_trunc = act_emb[:, -HS:]  # (BS, HS, A_emb)
            pred_emb = self.predict(emb_trunc, act_trunc)[:, -1:]  # (BS, 1, D)
            emb = torch.cat([emb, pred_emb], dim=1)

            next_act = act_future[:, t : t + 1, :]  # (BS, 1, A)
            act_delta = torch.cat([act_delta, next_act - prev_act], dim=1)
            prev_act = next_act
            act = torch.cat([act, next_act], dim=1)

        # predict the last state
        act_emb = self.action_encoder(act_delta)
        emb_trunc = emb[:, -HS:]  # (BS, HS, D)
        act_trunc = act_emb[:, -HS:]  # (BS, HS, A_emb)
        pred_emb = self.predict(emb_trunc, act_trunc)[:, -1:]  # (BS, 1, D)
        emb = torch.cat([emb, pred_emb], dim=1)

        # unflatten batch and sample dimensions
        pred_rollout = rearrange(emb, "(b s) ... -> b s ...", b=B, s=S)
        info["predicted_emb"] = pred_rollout

        return info

    def criterion(self, info_dict: dict):
        """Compute the cost between predicted embeddings and goal embeddings."""
        pred_emb = info_dict["predicted_emb"]  # (B,S, T-1, dim)
        goal_emb = info_dict["goal_emb"]  # (B, S, T, dim)

        goal_emb = goal_emb[..., -1:, :].expand_as(pred_emb)

        # return last-step cost per action candidate
        cost = F.mse_loss(
            pred_emb[..., -1:, :],
            goal_emb[..., -1:, :].detach(),
            reduction="none",
        ).sum(dim=tuple(range(2, pred_emb.ndim)))  # (B, S)

        return cost

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor):
        """ Compute the cost of action candidates given an info dict with goal and initial state."""

        assert "goal" in info_dict, "goal not in info_dict"

        device = next(self.parameters()).device
        for k in list(info_dict.keys()):
            if torch.is_tensor(info_dict[k]):
                info_dict[k] = info_dict[k].to(device)

        goal = {k: v[:, 0] for k, v in info_dict.items() if torch.is_tensor(v)}
        goal["pixels"] = goal["goal"]

        for k in info_dict:
            if k.startswith("goal_"):
                goal[k[len("goal_") :]] = goal.pop(k)

        goal.pop("action")
        goal = self.encode(goal)

        info_dict["goal_emb"] = goal["emb"]
        info_dict = self.rollout(info_dict, action_candidates)

        cost = self.criterion(info_dict)
        
        return cost
