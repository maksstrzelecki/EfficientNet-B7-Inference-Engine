import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import (
    EfficientNet_B7_Weights,
    efficientnet_b0,
    efficientnet_b7,
)

from dataset import get_dataloaders


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    use_600_input: bool = False,
) -> tuple[float, float]:
    """Oblicza Top-1 i Top-5 Accuracy na wspólnym zbiorze walidacyjnym ImageNetV2."""
    model.eval()
    correct_top1 = 0
    correct_top5 = 0
    total = 0

    for imgs_600, imgs_224, labels in val_loader:
        inputs = imgs_600 if use_600_input else imgs_224
        inputs = inputs.to(device)
        labels = labels.to(device)

        logits = model(inputs)
        _, top5_preds = logits.topk(5, dim=-1, largest=True, sorted=True)

        total += labels.size(0)
        matches = top5_preds.eq(labels.view(-1, 1).expand_as(top5_preds))
        correct_top1 += matches[:, 0].sum().item()
        correct_top5 += matches.any(dim=-1).sum().item()

    return (100.0 * correct_top1 / total), (100.0 * correct_top5 / total)


def main(student_weights: str, baseline_weights: str | None, data_dir: str) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, val_loader = get_dataloaders(data_dir=data_dir, batch_size=16, num_workers=2)

    print("\n--- WYNIKI EWALUACJI NA WSPÓLNYM ZBIORZE IMAGENET-V2 ---")

    
    teacher = efficientnet_b7(weights=EfficientNet_B7_Weights.DEFAULT).to(device)
    t_top1, t_top5 = evaluate_model(teacher, val_loader, device, use_600_input=True)
    print(f"1. Nauczyciel (EfficientNet-B7, 600x600) -> Top-1: {t_top1:.2f}% | Top-5: {t_top5:.2f}%")

    
    if baseline_weights and Path(baseline_weights).exists():
        baseline = efficientnet_b0(weights=None, num_classes=1000).to(device)
        baseline.load_state_dict(torch.load(baseline_weights, map_location=device, weights_only=True))
        b_top1, b_top5 = evaluate_model(baseline, val_loader, device, use_600_input=False)
        print(f"2. Baseline (B0 bez destylacji, 224x224)  -> Top-1: {b_top1:.2f}% | Top-5: {b_top5:.2f}%")

    
    if Path(student_weights).exists():
        student = efficientnet_b0(weights=None, num_classes=1000).to(device)
        student.load_state_dict(torch.load(student_weights, map_location=device, weights_only=True))
        s_top1, s_top5 = evaluate_model(student, val_loader, device, use_600_input=False)
        print(f"3. Uczeń po destylacji (B0, 224x224)      -> Top-1: {s_top1:.2f}% | Top-5: {s_top5:.2f}%")
    else:
        print(f"Brak pliku wag Ucznia pod ścieżką: {student_weights}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ewaluacja modeli na ImageNetV2")
    parser.add_argument("--student", type=str, default="student_b0.pth")
    parser.add_argument("--baseline", type=str, default="baseline_b0.pth")
    parser.add_argument("--data-dir", type=str, default="data")
    args = parser.parse_args()

    main(args.student, args.baseline, args.data_dir)