from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


LEFT_TEETH = (34, 35, 36, 37, 38)
RIGHT_TEETH = (44, 45, 46, 47, 48)
LEFT_IAN = 3
RIGHT_IAN = 4
LEFT_UPPER_POSTERIOR_TEETH = (24, 25, 26, 27, 28)
RIGHT_UPPER_POSTERIOR_TEETH = (14, 15, 16, 17, 18)
LEFT_MAXILLARY_SINUS = 5
RIGHT_MAXILLARY_SINUS = 6


def image_path_from_label(label_path: Path, images_dir: Path) -> Path:
    name = label_path.name
    if not name.endswith(".nii.gz"):
        raise ValueError(f"unexpected label suffix: {name}")
    return images_dir / f"{name[:-7]}_0000.nii.gz"


class NPZRelationDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        crop_size: int | None = None,
        random_crop: bool = False,
        max_jitter_voxels: int = 0,
        fixed_offset: tuple[int, int, int] = (0, 0, 0),
    ):
        self.root = Path(root)
        self.files = sorted(self.root.glob("*.npz"))
        if not self.files:
            raise FileNotFoundError(f"no npz files in {self.root}")
        self.crop_size = crop_size
        self.random_crop = random_crop
        self.max_jitter_voxels = max_jitter_voxels
        self.fixed_offset = fixed_offset

    def _crop_arrays(self, arrays: list[np.ndarray]) -> list[np.ndarray]:
        if self.crop_size is None:
            return arrays
        spatial_shape = arrays[0].shape[-3:]
        if any(size < self.crop_size for size in spatial_shape):
            raise ValueError(f"crop size {self.crop_size} exceeds cached shape {spatial_shape}")
        margin = tuple((size - self.crop_size) // 2 for size in spatial_shape)
        if self.random_crop:
            offsets = tuple(
                int(torch.randint(-min(m, self.max_jitter_voxels), min(m, self.max_jitter_voxels) + 1, ()).item())
                for m in margin
            )
        else:
            offsets = self.fixed_offset
        starts = tuple(max(0, min(size - self.crop_size, m + offset)) for size, m, offset in zip(spatial_shape, margin, offsets))
        slices = tuple(slice(start, start + self.crop_size) for start in starts)
        return [array[(...,) + slices] for array in arrays]

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        path = self.files[index]
        with np.load(path, allow_pickle=False) as z:
            image, sdf, mask, critical = self._crop_arrays(
                [z["image"], z["sdf"], z["mask"], z["critical"]]
            )
            item: dict[str, torch.Tensor | str] = {
                "image": torch.from_numpy(image.copy()).float().unsqueeze(0),
                "sdf": torch.from_numpy(sdf.copy()).float(),
                "mask": torch.from_numpy(mask.copy()).float(),
                "critical": torch.from_numpy(critical.copy()).float(),
                "distance_mm": torch.tensor(float(z["distance_mm"]), dtype=torch.float32),
                "sample_id": str(z["sample_id"]),
            }
        return item


def save_json(path: str | Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    tmp.replace(path)
