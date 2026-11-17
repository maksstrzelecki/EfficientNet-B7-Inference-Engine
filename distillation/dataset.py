import json
import urllib.request
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder, Imagenette
from torchvision.transforms import functional as TF


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

CLASS_INDEX_URL = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"


def get_wnid_to_imagenet_idx(cache_dir: Path) -> dict[str, int]:
    """Pobiera i zwraca słownik mapujący kod folderu (WNID) na oficjalny indeks 0..999."""
    cache_file = cache_dir / "imagenet_class_index.json"
    cache_dir.mkdir(parents=True, exist_ok=True)

    if not cache_file.exists():
        urllib.request.urlretrieve(CLASS_INDEX_URL, cache_file)

    with open(cache_file, "r", encoding="utf-8") as f:
        raw_index = json.load(f)

    
    return {val[0]: int(idx_str) for idx_str, val in raw_index.items()}


class DualResolutionImagenetteDataset(Dataset):
    """
    Zbiór Imagenette (train lub val) zwracający parę tensorów (600x600, 224x224)
    oraz prawdziwy globalny indeks klasy 0..999 zgodny z ImageNet-1K i FastAPI.
    """

    def __init__(
        self, root: str, split: str, wnid_to_idx: dict[str, int], is_train: bool
    ) -> None:
        root_path = Path(root)
        download = not (root_path / "imagenette2").exists()
        self.base_ds = Imagenette(
            root=root, split=split, size="full", download=download
        )
        self.is_train = is_train

        
        self.local_to_global_idx = {
            local_idx: wnid_to_idx[wnid]
            for local_idx, wnid in enumerate(self.base_ds.wnids)
        }
        self.allowed_global_indices = set(self.local_to_global_idx.values())

        self.normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        self.to_tensor = transforms.ToTensor()

        
        self.val_transform_600 = transforms.Compose(
            [
                transforms.Resize(
                    600, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.CenterCrop(600),
                transforms.ToTensor(),
                self.normalize,
            ]
        )
        self.val_transform_224 = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                self.normalize,
            ]
        )

    def __len__(self) -> int:
        return len(self.base_ds)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        img, local_label = self.base_ds[idx]
        img = img.convert("RGB")
        global_label = self.local_to_global_idx[local_label]

        if not self.is_train:
            return (
                self.val_transform_600(img),
                self.val_transform_224(img),
                global_label,
            )

        # Wspólna augmentacja geometryczna dla Nauczyciela i Ucznia podczas treningu
        i, j, h, w = transforms.RandomResizedCrop.get_params(
            img, scale=(0.5, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)
        )
        img = TF.crop(img, i, j, h, w)
        if torch.rand(1).item() < 0.5:
            img = TF.hflip(img)

        img_600 = TF.resize(
            img, [600, 600], interpolation=transforms.InterpolationMode.BICUBIC
        )
        img_224 = TF.resize(
            img, [224, 224], interpolation=transforms.InterpolationMode.BILINEAR
        )

        tensor_600 = self.normalize(self.to_tensor(img_600))
        tensor_224 = self.normalize(self.to_tensor(img_224))

        return tensor_600, tensor_224, global_label


def get_dataloaders(
    data_dir: str = "data", batch_size: int = 32, num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """Główny interfejs wywoływany przez train.py oraz evaluate.py."""
    data_path = Path(data_dir)
    wnid_to_idx = get_wnid_to_imagenet_idx(data_path)

    train_dataset = DualResolutionImagenetteDataset(
        root=data_dir, split="train", wnid_to_idx=wnid_to_idx, is_train=True
    )
    val_dataset = DualResolutionImagenetteDataset(
        root=data_dir, split="val", wnid_to_idx=wnid_to_idx, is_train=False
    )

    print(
        f"Załadowano dane: {len(train_dataset)} zdjęć treningowych "
        f"oraz {len(val_dataset)} zdjęć walidacyjnych (10 klas z indeksami ImageNet-1K)."
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


if __name__ == "__main__":
    train_loader, val_loader = get_dataloaders(
        data_dir="data", batch_size=4, num_workers=0
    )
    imgs_600, imgs_224, labels = next(iter(train_loader))
    print("Kształt batcha Nauczyciela (600x600):", imgs_600.shape)
    print("Kształt batcha Ucznia (224x224):", imgs_224.shape)
    print("Indeksy klas ImageNet-1K w batchu:", labels.tolist())