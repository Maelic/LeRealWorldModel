import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset
from stable_pretraining import data as dt
from lightning.pytorch.callbacks import Callback


class MirroredDataset(Dataset):
    """Wraps an existing dataset and exposes twice as many samples.

    The first half (indices 0..N-1) return the original sample unchanged.
    The second half (indices N..2N-1) return a time-reversed copy: both
    ``pixels`` and ``action`` are flipped along the time axis (dim 1).

    Normalisation statistics are computed on the original dataset only and are
    not affected by the mirroring because flipping does not change per-channel
    statistics.  Actions at sequence boundaries may contain NaNs; downstream
    code must handle them with ``torch.nan_to_num`` as usual.
    """

    def __init__(self, dataset: Dataset) -> None:
        super().__init__()
        self.dataset = dataset

    def __len__(self) -> int:
        return 2 * len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        base_idx = idx % len(self.dataset)
        sample = self.dataset[base_idx]
        if idx < len(self.dataset):
            return sample
        # Return a time-reversed copy without modifying the original dict
        mirrored = {k: v for k, v in sample.items()}
        if "pixels" in mirrored:
            mirrored["pixels"] = mirrored["pixels"].flip(0)
        if "action" in mirrored:
            mirrored["action"] = mirrored["action"].flip(0)
        return mirrored

def get_img_preprocessor(source: str, target: str, img_size: int = 224):
    imagenet_stats = dt.dataset_stats.ImageNet
    to_image = dt.transforms.ToImage(**imagenet_stats, source=source, target=target)
    resize = dt.transforms.Resize(img_size, source=source, target=target)
    return dt.transforms.Compose(to_image, resize)


def get_column_normalizer(dataset, source: str, target: str):
    """Get normalizer for a specific column in the dataset.

    The returned transform also carries ``.mean`` and ``.std`` attributes so
    callers can persist the per-column statistics (used by deploy code).
    """
    col_data = dataset.get_col_data(source)
    data = torch.from_numpy(np.array(col_data))
    if data.dim() == 1:
        data = data.unsqueeze(1)
    data = data[~torch.isnan(data).any(dim=1)]
    mean = data.mean(0).clone()   # (feat_dim,) — broadcasts over (T, feat_dim) or (T,)
    std = data.std(0).clone()

    def norm_fn(x):
        return ((x - mean) / std).float()

    normalizer = dt.transforms.WrapTorchTransform(norm_fn, source=source, target=target)
    normalizer.mean = mean
    normalizer.std = std
    return normalizer

class ModelObjectCallBack(Callback):
    """Callback to pickle model object after each epoch."""

    def __init__(self, dirpath, filename="model_object", epoch_interval: int = 1):
        super().__init__()
        self.dirpath = Path(dirpath)
        self.filename = filename
        self.epoch_interval = epoch_interval

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)

        epoch = trainer.current_epoch + 1
        should_save = trainer.is_global_zero and (
            epoch % self.epoch_interval == 0 or epoch == trainer.max_epochs
        )

        # Dual-encoder case
        model_bwd = getattr(pl_module, "model_bwd", None)
        if model_bwd is not None:
            bwd_path = self.dirpath / f"{self.filename}_epoch_{epoch}_bwd_object.ckpt"
            if should_save:
                self._dump_model(model_bwd, bwd_path)

        # Forward model / single-model case
        model_fwd = getattr(pl_module, "model_fwd", None) or getattr(pl_module, "model", None)
        if model_fwd is not None:
            suffix = "fwd_object" if model_bwd is not None else "object"
            fwd_path = self.dirpath / f"{self.filename}_epoch_{epoch}_{suffix}.ckpt"
            if should_save:
                self._dump_model(model_fwd, fwd_path)

    def _dump_model(self, model, path):
        try:
            torch.save(model, path)
        except Exception as e:
            print(f"Error saving model object: {e}")