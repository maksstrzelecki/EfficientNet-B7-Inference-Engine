import argparse
from collections import defaultdict
import json
import random
import sys
import time
from pathlib import Path
from typing import Tuple

from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import (
    EfficientNet_B7_Weights,
    efficientnet_b0,
    efficientnet_b7,
)
from torchvision.transforms import functional as TF

# Umożliwia import dataset.py niezależnie od tego, skąd uruchamiany jest skrypt
sys.path.append(str(Path(__file__).resolve().parent))
from dataset import IMAGENET_MEAN, IMAGENET_STD, get_dataloaders

# --- DOMYŚLNA KONFIGURACJA HIPERPARAMETRÓW ---
BATCH_SIZE = 32          # Liczba zdjęć przetwarzanych jednocześnie na RTX 5080
EPOCHS = 25              # Liczba epok treningu
LEARNING_RATE = 1e-3     # Szybkość uczenia się Ucznia
TEMPERATURE = 4.0        # Temperatura spłaszczająca pewność siebie Nauczyciela
ALPHA = 0.7              # 0.7 = destylacja z B7 | 0.0 = zwykły trening B0 (baseline)
SAVE_PATH = "student_b0.pth"


def set_seed(seed: int) -> None:
    """Ustawia ziarno losowości dla powtarzalności eksperymentów."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class DualResolutionImageNetV2Dataset(Dataset):
    """
    Zbiór ImageNetV2 zwracający parę tensorów (600x600 dla Nauczyciela, 224x224 dla Ucznia)
    oraz globalny indeks klasy 0..999 zgodny z ImageNet-1K.
    """

    def __init__(
        self, samples: list[tuple[Path, int]], is_train: bool, need_600: bool = True
    ) -> None:
        self.samples = samples
        self.is_train = is_train
        self.need_600 = need_600
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
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        img_path, class_id = self.samples[idx]
        with Image.open(img_path) as img:
            img = img.convert("RGB")

        if not self.is_train:
            tensor_600 = self.val_transform_600(img) if self.need_600 else torch.empty(0)
            return (
                tensor_600,
                self.val_transform_224(img),
                class_id,
            )

        # Wspólna augmentacja geometryczna dla Nauczyciela i Ucznia
        i, j, h, w = transforms.RandomResizedCrop.get_params(
            img, scale=(0.5, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)
        )
        img = TF.crop(img, i, j, h, w)
        if torch.rand(1).item() < 0.5:
            img = TF.hflip(img)

        img_224 = TF.resize(
            img, [224, 224], interpolation=transforms.InterpolationMode.BILINEAR
        )
        tensor_224 = self.normalize(self.to_tensor(img_224))

        if self.need_600:
            img_600 = TF.resize(
                img, [600, 600], interpolation=transforms.InterpolationMode.BICUBIC
            )
            tensor_600 = self.normalize(self.to_tensor(img_600))
        else:
            tensor_600 = torch.empty(0)

        return tensor_600, tensor_224, class_id


def get_imagenetv2_dataloaders(
    data_dir: str = "data",
    batch_size: int = 32,
    num_workers: int = 4,
    val_split: float = 0.2,
    seed: int = 42,
    need_600_train: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """Ładuje pobrany ImageNetV2 i dzieli go stratyfikowanie (równo w każdej z 1000 klas) na train/val."""
    root = Path(data_dir)
    if (root / "manifest.json").exists():
        dataset_root = root
    else:
        dataset_root = root / "imagenetv2"

    manifest_path = dataset_root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Nie znaleziono {manifest_path}. Pobierz zbiór poleceniem: .\\.venv\\Scripts\\python.exe imagenetv2.py"
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_class: dict[int, list[Path]] = defaultdict(list)
    for item in manifest["samples"]:
        by_class[int(item["class_id"])].append(dataset_root / item["file"])

    rng = random.Random(seed)
    train_samples: list[tuple[Path, int]] = []
    val_samples: list[tuple[Path, int]] = []

    for class_id in sorted(by_class.keys()):
        paths = sorted(by_class[class_id])
        rng.shuffle(paths)
        val_count = max(1, int(round(len(paths) * val_split)))
        for p in paths[:val_count]:
            val_samples.append((p, class_id))
        for p in paths[val_count:]:
            train_samples.append((p, class_id))

    train_dataset = DualResolutionImageNetV2Dataset(
        train_samples, is_train=True, need_600=need_600_train
    )
    # Podczas walidacji sprawdzamy tylko Ucznia (224x224), więc nie skalujemy do 600x600
    val_dataset = DualResolutionImageNetV2Dataset(
        val_samples, is_train=False, need_600=False
    )

    print(
        f"Załadowano ImageNetV2 ({len(by_class)} klas): "
        f"{len(train_dataset)} zdjęć treningowych oraz {len(val_dataset)} walidacyjnych."
    )

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    return train_loader, val_loader


def distillation_loss(student_logits, teacher_logits, labels, T: float, alpha: float):
    """
    Łączna funkcja straty dla Knowledge Distillation:
    1. Soft Loss (KL Divergence): porównuje rozkład Ucznia i Nauczyciela w temperaturze T.
    2. Hard Loss (Cross-Entropy): porównuje przewidywania Ucznia (T=1) z prawdziwą etykietą.
    """
    hard_loss = F.cross_entropy(student_logits, labels)
    if alpha == 0.0:
        return hard_loss

    soft_teacher = F.softmax(teacher_logits / T, dim=-1)
    soft_student = F.log_softmax(student_logits / T, dim=-1)
    soft_loss = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (T ** 2)

    return alpha * soft_loss + (1.0 - alpha) * hard_loss


@torch.no_grad()
def evaluate_val(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> tuple[float, float]:
    """Sprawdza celność Top-1 i Top-5 na zbiorze walidacyjnym po każdej epoce."""
    model.eval()
    correct_top1 = 0
    correct_top5 = 0
    total = 0

    for _, imgs_224, labels in val_loader:
        imgs_224 = imgs_224.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=amp_enabled):
            logits = model(imgs_224)

        _, top5_preds = logits.topk(5, dim=-1, largest=True, sorted=True)
        matches = top5_preds.eq(labels.view(-1, 1).expand_as(top5_preds))
        correct_top1 += matches[:, 0].sum().item()
        correct_top5 += matches.any(dim=-1).sum().item()
        total += labels.size(0)

    return 100.0 * correct_top1 / total, 100.0 * correct_top5 / total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trening Knowledge Distillation: EfficientNet-B7 -> EfficientNet-B0"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["imagenetv2", "imagenette"],
        default="imagenetv2",
        help="Zbiór danych do treningu (domyślnie: imagenetv2).",
    )
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument(
        "--alpha",
        type=float,
        default=ALPHA,
        help="Waga destylacji (0.0 = baseline bez Nauczyciela, 0.7 = destylacja z B7).",
    )
    parser.add_argument("--save-path", type=str, default=SAVE_PATH)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def train(args: argparse.Namespace) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        if torch.cuda.is_bf16_supported():
            amp_enabled, amp_dtype, use_scaler = True, torch.bfloat16, False
            precision_label = "CUDA bfloat16"
        else:
            amp_enabled, amp_dtype, use_scaler = True, torch.float16, True
            precision_label = "CUDA float16 + GradScaler"
    else:
        amp_enabled, amp_dtype, use_scaler = False, torch.float32, False
        precision_label = "CPU float32"

    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    print(f"Urządzenie: {device} | Precyzja: {precision_label} | Zbiór: {args.dataset}")

    # 1. Ładowanie danych
    if args.dataset == "imagenetv2":
        train_loader, val_loader = get_imagenetv2_dataloaders(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            val_split=args.val_split,
            seed=args.seed,
            need_600_train=(args.alpha > 0.0),
        )
    else:
        train_loader, val_loader = get_dataloaders(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

    # 2. Ładowanie i zamrożenie Nauczyciela (EfficientNet-B7)
    teacher = None
    if args.alpha > 0.0:
        print("Ładowanie Nauczyciela (EfficientNet-B7)...")
        teacher = efficientnet_b7(weights=EfficientNet_B7_Weights.DEFAULT).to(device)
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad = False

    # 3. Inicjalizacja Ucznia (EfficientNet-B0 od zera, 1000 klas)
    print("Inicjalizacja Ucznia (EfficientNet-B0)...")
    student = efficientnet_b0(weights=None, num_classes=1000).to(device)

    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0

    # 4. Główna pętla treningowa
    for epoch in range(1, args.epochs + 1):
        student.train()
        running_loss = 0.0
        start_time = time.time()

        for imgs_600, imgs_224, labels in train_loader:
            imgs_224 = imgs_224.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=amp_enabled):
                teacher_logits = None
                if teacher is not None:
                    imgs_600 = imgs_600.to(device, non_blocking=True)
                    with torch.no_grad():
                        teacher_logits = teacher(imgs_600)

                student_logits = student(imgs_224)
                loss = distillation_loss(
                    student_logits,
                    teacher_logits,
                    labels,
                    T=args.temperature,
                    alpha=args.alpha,
                )

            if use_scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            running_loss += loss.item()

        scheduler.step()
        val_top1, val_top5 = evaluate_val(
            student, val_loader, device, amp_enabled=amp_enabled, amp_dtype=amp_dtype
        )
        elapsed = time.time() - start_time
        avg_loss = running_loss / len(train_loader)
        max_vram_gb = (
            torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            if device.type == "cuda"
            else 0.0
        )

        print(
            f"Epoka [{epoch:02d}/{args.epochs:02d}] | "
            f"Loss: {avg_loss:.4f} | "
            f"Val Top-1: {val_top1:.2f}% | "
            f"Val Top-5: {val_top5:.2f}% | "
            f"Czas: {elapsed:.1f}s | "
            f"VRAM: {max_vram_gb:.2f} GB"
        )

        # 5. Zapisujemy tylko wagi z najlepszym wynikiem walidacji
        if val_top1 > best_val_acc:
            best_val_acc = val_top1
            torch.save(student.state_dict(), args.save_path)
            print(
                f"  -> Nowy rekord! Zapisano model do {args.save_path} ({best_val_acc:.2f}%)"
            )

    print(f"\nZakończono! Najlepszy wynik Val Top-1: {best_val_acc:.2f}%")


if __name__ == "__main__":
    train(parse_args())
