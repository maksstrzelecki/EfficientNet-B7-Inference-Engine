import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import (
    EfficientNet_B7_Weights,
    efficientnet_b0,
    efficientnet_b7,
)

# Umożliwia import dataset.py niezależnie od tego, skąd uruchamiany jest skrypt
sys.path.append(str(Path(__file__).resolve().parent))
from dataset import get_dataloaders

# --- KONFIGURACJA HIPERPARAMETRÓW ---
BATCH_SIZE = 32          # Liczba zdjęć przetwarzanych jednocześnie na RTX 5080
EPOCHS = 25              # 25 epok wystarczy na solidny wynik na Imagenette
LEARNING_RATE = 1e-3     # Szybkość uczenia się Ucznia
TEMPERATURE = 4.0        # Temperatura spłaszczająca pewność siebie Nauczyciela
ALPHA = 0.7              # 0.7 = destylacja z B7 | 0.0 = zwykły trening B0 (baseline)
SAVE_PATH = "student_b0.pth"


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
def evaluate_val(model: nn.Module, val_loader: DataLoader, device: torch.device) -> float:
    """Sprawdza celność Top-1 na zbiorze walidacyjnym po każdej epoce."""
    model.eval()
    correct = 0
    total = 0

    for _, imgs_224, labels in val_loader:
        imgs_224 = imgs_224.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(imgs_224)

        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return 100.0 * correct / total


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Uruchamianie na urządzeniu: {device}")

    # 1. Pobranie prawdziwych danych z pliku Osoby 1 (dataset.py)
    train_loader, val_loader = get_dataloaders(
        data_dir="data", batch_size=BATCH_SIZE, num_workers=4
    )

    # 2. Ładowanie i zamrożenie Nauczyciela (EfficientNet-B7)
    teacher = None
    if ALPHA > 0.0:
        print("Ładowanie Nauczyciela (EfficientNet-B7)...")
        teacher = efficientnet_b7(weights=EfficientNet_B7_Weights.DEFAULT).to(device)
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad = False

    # 3. Inicjalizacja Ucznia (EfficientNet-B0 od zera, ale na 1000 klas!)
    print("Inicjalizacja Ucznia (EfficientNet-B0)...")
    student = efficientnet_b0(weights=None, num_classes=1000).to(device)

    optimizer = torch.optim.AdamW(student.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_acc = 0.0

    # 4. Główna pętla treningowa
    for epoch in range(1, EPOCHS + 1):
        student.train()
        running_loss = 0.0
        start_time = time.time()

        for imgs_600, imgs_224, labels in train_loader:
            imgs_224 = imgs_224.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                teacher_logits = None
                if teacher is not None:
                    imgs_600 = imgs_600.to(device, non_blocking=True)
                    with torch.no_grad():
                        teacher_logits = teacher(imgs_600)

                student_logits = student(imgs_224)
                loss = distillation_loss(
                    student_logits, teacher_logits, labels, T=TEMPERATURE, alpha=ALPHA
                )

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()
        val_acc = evaluate_val(student, val_loader, device)
        elapsed = time.time() - start_time
        avg_loss = running_loss / len(train_loader)
        max_vram_gb = (
            torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            if device.type == "cuda"
            else 0.0
        )

        print(
            f"Epoka [{epoch:02d}/{EPOCHS:02d}] | "
            f"Loss: {avg_loss:.4f} | "
            f"Val Top-1: {val_acc:.2f}% | "
            f"Czas: {elapsed:.1f}s | "
            f"VRAM: {max_vram_gb:.2f} GB"
        )

        # 5. Zapisujemy tylko wagi z najlepszym wynikiem walidacji
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(student.state_dict(), SAVE_PATH)
            print(f"  -> Nowy rekord! Zapisano model do {SAVE_PATH} ({best_val_acc:.2f}%)")

    print(f"\nZakończono! Najlepszy wynik Val Top-1: {best_val_acc:.2f}%")


if __name__ == "__main__":
    train()