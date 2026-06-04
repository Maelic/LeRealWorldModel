"""Run a trained lewm world model on a real SO-100 arm via random-shooting MPC.

Usage example::

    python -m lewm.deploy_so100 \
        --ckpt ~/.stable_worldmodel/<run_id>/lewm_epoch_100_object.ckpt \
        --normalizers ~/.stable_worldmodel/<run_id>/lewm_normalizers.pt \
        --goal-image ./goal.png \
        --port /dev/ttyACM0 \
        --camera-key observation.images.front \
        --camera-index 0 --camera-width 640 --camera-height 480 \
        --history-size 3 --horizon 8 --num-samples 256 \
        --frameskip 5 --max-steps 600 --fps 30

Dry-run (no robot — pulls observations from a LeRobotDataset and only logs
chosen actions)::

    python -m lewm.deploy_so100 \
        --ckpt ... --goal-image ... \
        --dry-run-replay-from lerobot/svla_so100_pickplace \
        --max-steps 60

The deploy script does NOT touch the robot until ``--max-steps`` is set and the
``--dry-run-replay-from`` flag is absent. The default ``max_relative_target``
is left to whatever the robot config carries — set ``--max-relative-target``
small (e.g. 5 degrees) for the first hardware run.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Make the repo root importable so `import jepa` works (utils, module, etc.).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Re-export JEPA early so `torch.load` can resolve the pickled class.
import lewm_robot  # noqa: F401, E402

from utils import get_img_preprocessor  # noqa: E402

from lewm_robot.planning import RandomShootingPlanner  # noqa: E402

logger = logging.getLogger("lewm.deploy_so100")


# ───────────────────────────── helpers ───────────────────────────────


def _load_image(path: Path, img_size: int) -> torch.Tensor:
    """Load a PNG/JPG into a (3, H, W) uint8 CHW tensor."""
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img)  # (H, W, 3) uint8
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def _preprocess_pixels(pixels_uint8: torch.Tensor, img_size: int) -> torch.Tensor:
    """Apply the same image preprocessing used at training time.

    ``pixels_uint8`` is ``(T, 3, H, W)`` uint8 CHW. Returns a float ImageNet-
    normalized tensor of shape ``(T, 3, img_size, img_size)``.
    """
    transform = get_img_preprocessor(source="pixels", target="pixels", img_size=img_size)
    sample = {"pixels": pixels_uint8}
    return transform(sample)["pixels"]


def _normalize_action(action: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (action - mean) / std


def _denormalize_action(action: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return action * std + mean


def _build_robot(args, action_motor_names: list[str]):
    """Construct an SOFollower robot with one OpenCV camera."""
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    from lerobot.robots.so_follower.config_so_follower import SO100FollowerConfig
    from lerobot.robots.so_follower.so_follower import SOFollower

    camera_name = args.camera_key.split(".")[-1]  # e.g. "front"
    cameras = {
        camera_name: OpenCVCameraConfig(
            index_or_path=args.camera_index,
            width=args.camera_width,
            height=args.camera_height,
            fps=args.fps,
        )
    }

    cfg_kwargs = dict(port=args.port, cameras=cameras)
    if args.max_relative_target is not None:
        cfg_kwargs["max_relative_target"] = args.max_relative_target

    config = SO100FollowerConfig(**cfg_kwargs)
    robot = SOFollower(config)
    robot.connect()
    logger.info("Robot connected on %s with camera %s", args.port, camera_name)
    return robot, camera_name


def _read_obs_pixels(obs: dict, camera_name: str) -> torch.Tensor:
    """Pull the camera frame from a robot observation as ``(3, H, W)`` uint8 CHW."""
    cam = obs[camera_name]
    if isinstance(cam, np.ndarray):
        if cam.ndim == 3 and cam.shape[-1] in (1, 3):  # HWC
            cam = np.transpose(cam, (2, 0, 1))
        cam = torch.from_numpy(cam.copy())
    elif isinstance(cam, torch.Tensor):
        if cam.ndim == 3 and cam.shape[-1] in (1, 3):
            cam = cam.permute(2, 0, 1).contiguous()
    else:
        raise TypeError(f"Unexpected camera frame type: {type(cam)}")
    return cam


def _read_obs_state(obs: dict, motor_names: list[str]) -> torch.Tensor:
    """Pull joint positions from a robot observation as ``(action_dim,)`` float."""
    vals = []
    for name in motor_names:
        key = f"{name}.pos" if not name.endswith(".pos") else name
        vals.append(float(obs[key]))
    return torch.tensor(vals, dtype=torch.float32)


# ─────────────────────────── replay backend ──────────────────────────


class _DatasetReplay:
    """Stand-in for SOFollower that yields observations from a LeRobotDataset."""

    def __init__(self, repo_id: str, image_key: str, motor_names: list[str], max_steps: int):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        self._ds = LeRobotDataset(repo_id=repo_id, return_uint8=True)
        self._image_key = image_key
        self._motor_names = motor_names
        self._idx = 0
        self._max_steps = min(max_steps, len(self._ds))
        self._sent: list[dict] = []
        # Cache absolute frame index for the camera/state lookup.
        self._camera_name = image_key.split(".")[-1]

    def get_observation(self) -> dict:
        item = self._ds[self._idx]
        obs = {}
        # Camera frame (already CHW uint8 from return_uint8=True).
        cam = item[self._image_key]
        obs[self._camera_name] = cam
        # Per-motor positions from the state vector.
        state = item.get("observation.state")
        if state is not None:
            for i, name in enumerate(self._motor_names):
                obs[f"{name}.pos"] = float(state[i].item() if torch.is_tensor(state[i]) else state[i])
        self._idx = min(self._idx + 1, self._max_steps - 1)
        return obs

    def send_action(self, action: dict) -> dict:
        self._sent.append(dict(action))
        return action

    def disconnect(self):
        logger.info("Replay backend: %d actions logged", len(self._sent))


# ───────────────────────────── main loop ─────────────────────────────


@torch.no_grad()
def encode_goal(model, goal_image_chw: torch.Tensor, img_size: int, device) -> torch.Tensor:
    """Pre-encode the goal image to its latent embedding."""
    goal = _preprocess_pixels(goal_image_chw.unsqueeze(0), img_size).to(device)
    info = {"pixels": goal.unsqueeze(0)}  # (B=1, T=1, C, H, W)
    out = model.encode(info)
    return out["emb"]  # (1, 1, D)


def deploy(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    logger.info("Loading world model from %s on %s", args.ckpt, device)
    model = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.eval().to(device)

    # Load the saved action / proprio normalizers.
    norm_path = Path(args.normalizers) if args.normalizers else None
    action_stats = None
    if norm_path is not None and norm_path.exists():
        stats = torch.load(norm_path, map_location="cpu")
        action_stats = stats.get("action")
        logger.info("Loaded normalizers from %s; keys=%s", norm_path, list(stats.keys()))
    if action_stats is None:
        logger.warning(
            "No action normalizer found — assuming unit-mean / unit-std. "
            "This will likely produce garbage predictions on a model trained "
            "with normalization enabled."
        )
        action_mean = torch.zeros(args.action_dim)
        action_std = torch.ones(args.action_dim)
    else:
        action_mean = action_stats["mean"]
        action_std = action_stats["std"]

    # Motor names — order must match the dataset's action feature.
    motor_names = args.motor_names.split(",")
    if len(motor_names) != args.action_dim:
        raise ValueError(
            f"--motor-names has {len(motor_names)} entries but --action-dim={args.action_dim}"
        )

    # Goal image → latent goal.
    goal_chw = _load_image(Path(args.goal_image), args.img_size)
    goal_emb = encode_goal(model, goal_chw, args.img_size, device)
    goal_pixels_pre = _preprocess_pixels(goal_chw.unsqueeze(0), args.img_size).to(device)
    logger.info("Goal embedding shape: %s", tuple(goal_emb.shape))

    # Build planner.
    planner = RandomShootingPlanner(
        history_size=args.history_size,
        horizon=args.horizon,
        frameskip=args.frameskip,
        action_dim=args.action_dim,
        num_samples=args.num_samples,
        sample_std=args.sample_std,
        device=device,
    )

    # Choose robot backend.
    if args.dry_run_replay_from:
        logger.info("Dry-run mode: replaying from %s", args.dry_run_replay_from)
        robot = _DatasetReplay(
            args.dry_run_replay_from, args.camera_key, motor_names, args.max_steps
        )
        camera_name = robot._camera_name
    else:
        robot, camera_name = _build_robot(args, motor_names)

    try:
        history_size = args.history_size
        history_pixels: list[torch.Tensor] = []  # uint8 CHW each
        history_actions: list[torch.Tensor] = []  # normalized chunks each (frameskip*action_dim)
        last_action_chunk = torch.zeros(
            args.frameskip * args.action_dim, dtype=torch.float32
        )

        chunk_queue: list[torch.Tensor] = []  # 6-D actions (un-normalized) waiting to be sent

        control_dt = 1.0 / args.fps
        for step in range(args.max_steps):
            t0 = time.perf_counter()
            obs = robot.get_observation()
            cam_uint8 = _read_obs_pixels(obs, camera_name)

            history_pixels.append(cam_uint8)
            history_pixels = history_pixels[-history_size:]
            # Pad history with the first observation until we have history_size frames.
            while len(history_pixels) < history_size:
                history_pixels.insert(0, cam_uint8)

            # Re-plan only when the action chunk queue is empty.
            if not chunk_queue:
                # Pad action history if needed.
                while len(history_actions) < history_size:
                    history_actions.insert(0, last_action_chunk)
                action_hist = torch.stack(history_actions[-history_size:])  # (H, frameskip*ad)

                # Build info_dict for model.get_cost. The expected layout is
                #   pixels: (B=1, S=1, H, C, h, w)        ← S broadcasts to num_samples in rollout
                #   action: (B=1, S=1, H, frameskip*ad)
                #   goal:   (B=1, S=1, T_g=1, C, h, w)
                pixels_pre = _preprocess_pixels(
                    torch.stack(history_pixels), args.img_size
                ).to(device)  # (H, 3, h, w)
                info_dict = {
                    "pixels": pixels_pre.unsqueeze(0).unsqueeze(0),  # (1, 1, H, 3, h, w)
                    "action": action_hist.unsqueeze(0).unsqueeze(0).to(device),  # (1, 1, H, F*ad)
                    "goal": goal_pixels_pre.unsqueeze(0).unsqueeze(0),  # (1, 1, 1, 3, h, w)
                }
                planned = planner.plan(model, info_dict, last_action_chunk)
                # planned: (frameskip, action_dim) in normalized space.
                # Update the action history with the new normalized chunk.
                last_action_chunk = planned.reshape(-1)
                history_actions.append(last_action_chunk)
                history_actions = history_actions[-history_size:]
                # Denormalize and queue the per-tick actions.
                planned_real = _denormalize_action(planned, action_mean, action_std)
                chunk_queue = [planned_real[i] for i in range(args.frameskip)]
                logger.info(
                    "step=%d replanned chunk; first action=%s",
                    step,
                    chunk_queue[0].tolist(),
                )

            # Send the next action.
            next_action = chunk_queue.pop(0)
            action_dict = {f"{m}.pos": float(v) for m, v in zip(motor_names, next_action)}
            sent = robot.send_action(action_dict)
            logger.debug("step=%d sent=%s", step, sent)

            dt = time.perf_counter() - t0
            sleep = control_dt - dt
            if sleep > 0:
                time.sleep(sleep)
            elif step % 10 == 0:
                logger.warning(
                    "step=%d slow loop: dt=%.3fs (target %.3fs)", step, dt, control_dt
                )
    finally:
        robot.disconnect()


# ───────────────────────────── CLI glue ──────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--ckpt", required=True, help="Path to lewm_epoch_*_object.ckpt")
    p.add_argument("--normalizers", default=None, help="Path to lewm_normalizers.pt")
    p.add_argument("--goal-image", required=True, help="Goal image (PNG/JPG)")

    # Robot wiring
    p.add_argument("--port", default="/dev/ttyACM0")
    p.add_argument("--camera-key", default="observation.images.front")
    p.add_argument("--camera-index", type=int, default=0)
    p.add_argument("--camera-width", type=int, default=640)
    p.add_argument("--camera-height", type=int, default=480)
    p.add_argument("--max-relative-target", type=float, default=None,
                   help="Per-motor relative-position cap (degrees). Leave unset to use the robot config default.")
    p.add_argument(
        "--motor-names",
        default="shoulder_pan,shoulder_lift,elbow_flex,wrist_flex,wrist_roll,gripper",
        help="Comma-separated motor names matching the dataset's action feature order.",
    )

    # Planner
    p.add_argument("--history-size", type=int, default=3)
    p.add_argument("--horizon", type=int, default=8)
    p.add_argument("--frameskip", type=int, default=5)
    p.add_argument("--action-dim", type=int, default=6)
    p.add_argument("--num-samples", type=int, default=256)
    p.add_argument("--sample-std", type=float, default=0.5)

    # Loop
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--img-size", type=int, default=224)

    # Dry run
    p.add_argument(
        "--dry-run-replay-from",
        default=None,
        help="LeRobotDataset repo_id; if set, observations are pulled from this dataset instead of the robot.",
    )
    return p


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
    args = _build_parser().parse_args()
    deploy(args)


if __name__ == "__main__":
    main()
