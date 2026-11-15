import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import (
    efficientnet_b7,
    EfficientNet_B7_Weights,
    efficientnet_b0,
)

# --- KONFIGURACJA HIPERPARAMETRÓW ---
BATCH_SIZE = 32          # Liczba zdjęć przetwarzanych jednocześnie na GPU
EPOCHS = 2               # Na razie 2 epoki do testu (docelowo ustawimy 25-30)
LEARNING_RATE = 1e-3     # Szybkość uczenia się Ucznia
TEMPERATURE = 4.0        # Temperatura spłaszczająca pewność siebie Nauczyciela
ALPHA = 0.7              # 70% wagi wiedzy od Nauczyciela, 30% od prawdziwych etykiet
USE_MOCK_DATA = True     # Zmień na False, gdy Osoba 1 wypchnie distillation/dataset.py
SAVE_PATH = "student_b0.pth"


def get_mock_dataloaders(batch_size: int):
    """Tymczasowy generator losowych tensorów do weryfikacji VRAM i pętli treningowej."""
    num_samples = batch_size * 10  # Symulujemy 10 paczek (batchy) na epokę
    imgs_600 = torch.randn(num_samples, 3, 600, 600)
    imgs_224 = torch.randn(num_samples, 3, 224, 224)
    labels = torch.randint(0, 1000, (num_samples,))

    dataset = TensorDataset(imgs_600, imgs_224, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader, loader


def distillation_loss(student_logits, teacher_logits, labels, T: float, alpha: float):
    """
    Łączna funkcja straty dla Knowledge Distillation:
    1. Soft Loss (KL Divergence): porównuje rozkład Ucznia i Nauczyciela w temperaturze T.
    2. Hard Loss (Cross-Entropy): porównuje przewidywania Ucznia (T=1) z prawdziwą etykietą.
    """
    soft_teacher = F.softmax(teacher_logits / T, dim=-1)
    soft_student = F.log_softmax(student_logits / T, dim=-1)
    soft_loss = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (T ** 2)

    hard_loss = F.cross_entropy(student_logits, labels)

    return alpha * soft_loss + (1.0 - alpha) * hard_loss


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Uruchamianie na urządzeniu: {device}")

    # 1. Przygotowanie danych (tymczasowo mock, docelowo z pliku Osoby 1)
    if USE_MOCK_DATA:
        print("TRYB TESTOWY: Używam losowych tensorów (Mock Data)...")
        train_loader, _ = get_mock_dataloaders(BATCH_SIZE)
    else:
        # TODO: Odkomentować po zmergowaniu pliku Osoby 1:
        # from distillation.dataset import get_dataloaders
        # train_loader, _ = get_dataloaders(data_dir="data", batch_size=BATCH_SIZE)
        raise NotImplementedError("Najpierw pobierz dataset.py od Osoby 1!")

    # 2. Ładowanie i zamrożenie Nauczyciela (EfficientNet-B7)
    print("Ładowanie Nauczyciela (EfficientNet-B7)...")
    teacher = efficientnet_b7(weights=EfficientNet_B7_Weights.DEFAULT).to(device)
    teacher.eval()  # Wyłącza warstwy Dropout i aktualizuje BatchNorm w trybie inferencji
    for param in teacher.parameters():
        param.requires_grad = False  # Blokuje liczenie gradientów (oszczędność VRAM)

    # 3. Inicjalizacja Ucznia (EfficientNet-B0 od zera, ale na 1000 klas!)
    print("Inicjalizacja Ucznia (EfficientNet-B0)...")
    student = efficientnet_b0(weights=None, num_classes=1000).to(device)

    optimizer = torch.optim.AdamW(student.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # 4. Główna pętla treningowa
    for epoch in range(1, EPOCHS + 1):
        student.train()
        running_loss = 0.0
        start_time = time.time()

        for step, (imgs_600, imgs_224, labels) in enumerate(train_loader, start=1):
            imgs_600 = imgs_600.to(device, non_blocking=True)
            imgs_224 = imgs_224.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            # Precyzja bfloat16 
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
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
        elapsed = time.time() - start_time
        avg_loss = running_loss / len(train_loader)
        max_vram_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)

        print(
            f"Epoka [{epoch}/{EPOCHS}] | "
            f"Loss: {avg_loss:.4f} | "
            f"Czas: {elapsed:.2f}s | "
            f"Szczytowe zużycie VRAM: {max_vram_gb:.2f} GB"
        )

    # 5. Zapis wag Ucznia (nie wrzucamy testowego pliku na GitHub, tylko lokalnie)
    torch.save(student.state_dict(), SAVE_PATH)
    print(f"Zapisano testowe wagi Ucznia do: {Path(SAVE_PATH).resolve()}")


if __name__ == "__main__":
    train()