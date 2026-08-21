"""Adapter that exposes a LeRobotDataset through the stable_worldmodel.Dataset API.

This lets ``train.py`` and ``train_bidir_enc.py`` consume a LeRobot-format
dataset (parquet + mp4 videos + meta json) without converting to HDF5.

The class subclasses ``stable_worldmodel.data.Dataset``, so ``__len__``,
``__getitem__``, and ``clip_indices`` come from the base class; we only
implement ``_load_slice`` and the column-introspection helpers needed by
``utils.get_column_normalizer`` and ``train.py``.

Action stacking semantics match ``HDF5Dataset``: actions are returned at the
native frame rate (``end - start`` rows) and the base class then reshapes
them into ``(num_steps, frameskip * action_dim)``. Other modalities are
downsampled by ``frameskip`` to give ``(num_steps, ...)``.

Slice loading bypasses ``LeRobotDataset.delta_timestamps`` and queries the
underlying ``hf_dataset`` and video reader directly. This keeps episode
boundaries clean (the base class' ``clip_indices`` filter already drops any
clip that would extend past episode end), avoids the per-frame pad masks,
and amortizes one video-decoder call per ``__getitem__`` instead of one per
frame.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from stable_worldmodel.data.dataset import Dataset

# LeRobotDataset is a heavy import (drags in datasets, huggingface_hub, ffmpeg
# bindings). Imported lazily inside __init__ to keep `import lewm.data` cheap.


class LeRobotWMDataset(Dataset):
    """LeRobotDataset adapter for the lewm world-model training loop.

    Args:
        repo_id: Hub repo id (e.g. ``"lerobot/svla_so100_pickplace"``).
        root: Optional local directory. If unset, falls back to the HF
            cache. Pass an absolute path when working offline.
        image_key: Primary camera feature key.
        image_key2: Optional second camera feature key. When set, a ``pixels2``
            key is added to each batch item with the second view.
        proprio_key: State feature key (joint positions). Defaults to
            ``"observation.state"``.
        action_key: Action feature key. Defaults to ``"action"``.
        state_key: Optional second state feature. If unset, the ``state``
            key is dropped (most LeRobot datasets only have ``observation.state``).
        frameskip: Number of native frames between sampled steps.
        num_steps: Number of sampled steps per ``__getitem__`` call. Should
            equal ``wm.history_size + wm.num_preds``.
        keys_to_load: Which output keys to populate. Subset of
            ``["pixels", "pixels2", "action", "proprio", "state"]``. Defaults
            to all available.
        return_uint8: If True, video frames come back as uint8 CHW tensors
            (the lewm image preprocessor will convert to float and ImageNet-
            normalize). Strongly recommended — avoids a redundant
            float-conversion in the LeRobotDataset reader.
        transform: Optional transform applied to the per-step dict. Same
            contract as :class:`stable_worldmodel.data.HDF5Dataset`.
        episodes: Optional list of episode indices to subset.
    """

    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        image_key: str = "observation.images.front",
        image_key2: str | None = None,
        proprio_key: str = "observation.state",
        action_key: str = "action",
        state_key: str | None = None,
        frameskip: int = 1,
        num_steps: int = 1,
        keys_to_load: list[str] | None = None,
        return_uint8: bool = True,
        transform: Callable[[dict], dict] | None = None,
        episodes: list[int] | None = None,
    ) -> None:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        self.repo_id = repo_id
        self.image_key = image_key
        self.image_key2 = image_key2
        self.proprio_key = proprio_key
        self.action_key = action_key
        self.state_key = state_key
        self._return_uint8 = return_uint8

        self._lerobot = LeRobotDataset(
            repo_id=repo_id,
            root=root,
            episodes=episodes,
            # tolerance_s must exceed max rounding error at the dataset fps (1/30 ≈ 0.0333s
            # can accumulate ~0.0003s of floating-point error, which exceeds the default 0.0001).
            tolerance_s=0.04,
            # Force the pyav backend: the dataset videos are AV1-encoded and the
            # Alvis system FFmpeg that torchcodec links against lacks an AV1 decoder
            # ("Could not push packet to decoder: Function not implemented"). pyav
            # bundles its own FFmpeg with libdav1d, so it decodes AV1 fine.
            video_backend="pyav",
            # No delta_timestamps and no image_transforms — we drive both
            # ourselves in _load_slice so we can pack into (T, C, H, W) batches
            # and apply the same preprocessor used at training time.
        )

        meta = self._lerobot.meta
        for key, label in [(image_key, "image_key"), (image_key2, "image_key2")]:
            if key is None:
                continue
            if key not in meta.video_keys and key not in meta.camera_keys:
                raise KeyError(
                    f"{label}={key!r} not in dataset video keys "
                    f"{list(meta.video_keys)} or camera keys {list(meta.camera_keys)}"
                )

        # Build per-episode lengths/offsets. ``meta.episodes`` always describes the
        # *full* dataset with global frame indices, but when ``episodes`` subsets the
        # data LeRobotDataset compacts ``hf_dataset`` to just those episodes,
        # re-indexed from 0. We therefore compute offsets that index into the
        # (possibly compacted) ``hf_dataset`` for the parquet slice, and separately
        # keep each clip-slot's *global* episode id — the video reader needs it, as
        # it indexes the full ``meta.episodes`` and resolves the per-episode chunk
        # file/offset. (Without ``episodes`` the two coincide.)
        episodes_meta = meta.episodes
        ep_from = np.asarray(episodes_meta["dataset_from_index"], dtype=np.int64)
        ep_to = np.asarray(episodes_meta["dataset_to_index"], dtype=np.int64)
        ep_ids = np.asarray(episodes_meta["episode_index"], dtype=np.int64)
        all_lengths = ep_to - ep_from

        if episodes is None:
            lengths = all_lengths
            offsets = ep_from
            self._global_ep_ids = ep_ids
        else:
            sel = sorted({int(e) for e in episodes})
            id_to_row = {int(e): i for i, e in enumerate(ep_ids)}
            try:
                rows = [id_to_row[e] for e in sel]
            except KeyError as exc:
                raise ValueError(
                    f"episode {exc.args[0]} not found in dataset {repo_id!r} "
                    f"({len(ep_ids)} episodes available)"
                ) from None
            lengths = all_lengths[rows]
            # Compacted offsets: cumulative within the subset (hf_dataset is
            # re-indexed from 0, in ascending episode order).
            offsets = np.concatenate(([0], np.cumsum(lengths)[:-1])).astype(np.int64)
            self._global_ep_ids = np.asarray(sel, dtype=np.int64)
            # Guard the ascending-compaction assumption against the loaded rows.
            n_rows = len(self._lerobot.hf_dataset)
            if int(lengths.sum()) != n_rows:
                raise RuntimeError(
                    f"episode subset length mismatch: sum(lengths)={int(lengths.sum())} "
                    f"!= len(hf_dataset)={n_rows}; episode ordering assumption broken"
                )

        # Filter to requested keys.
        available = ["pixels", "action", "proprio"]
        if image_key2 is not None:
            available.append("pixels2")
        if state_key is not None:
            available.append("state")
        if keys_to_load is None:
            keys_to_load = list(available)
        else:
            for k in keys_to_load:
                if k not in available:
                    raise ValueError(
                        f"keys_to_load={keys_to_load} contains unknown key {k!r}; "
                        f"valid keys are {available}"
                    )
        self._keys = list(keys_to_load)

        # Map our generic keys to the LeRobot feature names.
        self._key_map = {
            "pixels": image_key,
            "action": action_key,
            "proprio": proprio_key,
        }
        if image_key2 is not None:
            self._key_map["pixels2"] = image_key2
        if state_key is not None:
            self._key_map["state"] = state_key

        # Lazy cache for column data used by the normalizer at train startup.
        self._col_cache: dict[str, np.ndarray] = {}

        super().__init__(lengths, offsets, frameskip, num_steps, transform)

    @property
    def column_names(self) -> list[str]:
        return list(self._keys)

    @property
    def fps(self) -> int:
        return self._lerobot.meta.fps

    def _load_slice(self, ep_idx: int, start: int, end: int) -> dict[str, Any]:
        """Load one (history + predictions) clip from a single episode.

        ``start`` and ``end`` are local-to-episode frame indices; the parent
        class ensures ``end - start == num_steps * frameskip``.
        """
        ep_offset = int(self.offsets[ep_idx])
        g_start = ep_offset + start
        g_end = ep_offset + end

        steps: dict[str, Any] = {}

        # Non-pixel keys: read a contiguous slice of the parquet-backed
        # hf_dataset and stack into (n_rows, feat_dim). Then downsample by
        # frameskip for everything except the action stream (the base class'
        # __getitem__ reshapes the un-downsampled action stream into chunks).
        pixel_keys = [k for k in self._keys if k.startswith("pixels")]
        non_pixel = [k for k in self._keys if not k.startswith("pixels")]
        if non_pixel:
            hf_keys = [self._key_map[k] for k in non_pixel]
            hf = self._lerobot.hf_dataset
            slice_rows = hf.select(range(g_start, g_end))
            for our_key, hf_key in zip(non_pixel, hf_keys):
                col = slice_rows[hf_key]
                # `col` is a datasets.Column (Sequence-like); each element is a Tensor.
                arr = torch.stack([torch.as_tensor(c) for c in col])
                if our_key != "action":
                    arr = arr[:: self.frameskip]
                steps[our_key] = arr

        # Pixel keys: query the video reader once per camera for num_steps frames.
        if pixel_keys:
            local_indices = [start + i * self.frameskip for i in range(self.num_steps)]
            timestamps = [idx / self.fps for idx in local_indices]
            video_keys_query = {self._key_map[pk]: timestamps for pk in pixel_keys}
            # The reader indexes the *full* meta.episodes, so map the compacted
            # clip-slot index back to its global episode id.
            global_ep_idx = int(self._global_ep_ids[ep_idx])
            frames_dict = self._lerobot.reader._query_videos(video_keys_query, global_ep_idx)
            for pk in pixel_keys:
                frames = frames_dict[self._key_map[pk]]  # (T, C, H, W) float32 [0,1]
                if frames.ndim == 3:
                    frames = frames.unsqueeze(0)
                if self._return_uint8 and frames.is_floating_point():
                    frames = (frames.clamp(0.0, 1.0) * 255.0).to(torch.uint8)
                steps[pk] = frames

        if self.transform is not None:
            steps = self.transform(steps)
        return steps

    def get_col_data(self, col: str) -> np.ndarray:
        """Materialize a non-pixel column as a flat ``(N_frames, feat_dim)`` array.

        Used once at training startup by ``utils.get_column_normalizer`` to
        compute mean/std. Cached after the first call.
        """
        if col.startswith("pixels"):
            raise KeyError(
                f"get_col_data({col!r}) is not supported — videos are decoded "
                "lazily per slice."
            )
        if col not in self._key_map:
            raise KeyError(col)
        if col in self._col_cache:
            return self._col_cache[col]

        hf_key = self._key_map[col]
        hf = self._lerobot.hf_dataset
        col_data = hf[hf_key]
        stacked = torch.stack([torch.as_tensor(c) for c in col_data])
        arr = stacked.cpu().numpy()
        self._col_cache[col] = arr
        return arr

    def get_dim(self, col: str) -> int:
        """Return the per-frame feature dimension for ``col``."""
        if col.startswith("pixels"):
            lerobot_key = self._key_map.get(col)
            if lerobot_key is None:
                raise KeyError(col)
            shape = self._lerobot.meta.features[lerobot_key]["shape"]
            return int(np.prod(shape))
        if col not in self._key_map:
            raise KeyError(col)
        hf_key = self._key_map[col]
        shape = self._lerobot.meta.features[hf_key]["shape"]
        if len(shape) <= 1:
            return int(shape[0]) if shape else 1
        return int(np.prod(shape))

    def get_row_data(self, row_idx: int | list[int]) -> dict[str, Any]:
        hf = self._lerobot.hf_dataset
        if isinstance(row_idx, int):
            row = hf[row_idx]
            return {our: row[hf_key] for our, hf_key in self._key_map.items() if our != "pixels"}
        rows = hf.select(row_idx)
        return {
            our: rows[hf_key]
            for our, hf_key in self._key_map.items()
            if our != "pixels"
        }
