# Knowledge Distillation — Wyniki eksperymentu na ImageNetV2 (1000 klas)

## Zakres eksperymentu
Ten eksperyment weryfikuje destylację wiedzy (**Knowledge Distillation**) z pretrenowanego Nauczyciela `EfficientNet-B7` (600×600) do lekkiego Ucznia `EfficientNet-B0` (224×224, trenowanego od zera) na **wszystkich 1000 klasach** ze zbioru **ImageNetV2** (wszystkie 3 warianty: `matched-frequency`, `threshold-0.7`, `top-images`).

> **Rozróżnienie wersji modeli w repozytorium:**
> * `student_b0_imagenetv2_1000c.pth` oraz `baseline_b0_imagenetv2_1000c.pth` — modele z tego eksperymentu (**1000 klas**, ImageNetV2, 30 zdjęć/klasę).
> * `student_b0.pth` — ścieżka pozostawiona dla równoległego wariantu zespołu (100 klas na pełnym zbiorze zdjęć), aby uniknąć konfliktów nazw.

---

## Konfiguracja treningu
| Parametr | Wartość |
| --- | --- |
| Zbiór danych | ImageNetV2 (3 warianty, 1000 klas) |
| Podział train / val | 24 000 / 6 000 (stratyfikowany 80/20: 24 train + 6 val na każdą klasę) |
| Nauczyciel | `EfficientNet-B7` (`IMAGENET1K_V1`, 600×600, zamrożony, 254.7 MB) |
| Uczeń / Baseline | `EfficientNet-B0` (`weights=None`, `num_classes=1000`, 224×224, 21.5 MB) |
| Epoki | 25 |
| Batch size / Num workers | 32 / 4 |
| Optymalizator / Scheduler | `AdamW` (`lr=1e-3`, `weight_decay=1e-4`) + `CosineAnnealingLR` |
| Hiperparametry KD | `alpha=0.7`, `T=4.0` (Baseline: `alpha=0.0`) |
| Sprzęt i precyzja | NVIDIA RTX 5080, `CUDA bfloat16`, `cudnn.benchmark=True`, `seed=42` |

---

## Podsumowanie wyników końcowych (Walidacja: 6 000 zdjęć, 1000 klas)

| Model | Plik wag | Rozmiar | Best Val Top-1 | Val Top-5 (przy best Top-1) | Czas / epokę | Max VRAM |
| --- | --- | --- | --- | --- | --- | --- |
| **Baseline (`B0` bez Nauczyciela)** | `baseline_b0_imagenetv2_1000c.pth` | 21.5 MB | 20.28% *(ep. 25)* | 40.32% | ~35 s | 1.45 GB |
| **Uczeń (`B0` po destylacji z `B7`)** | `student_b0_imagenetv2_1000c.pth` | 21.5 MB | **28.67%** *(ep. 24)* | **49.67%** *(49.92% w ep. 25)* | ~225 s | 5.57 GB |
| **Zysk z Knowledge Distillation** | — | 0 B narzutu | **+8.39 pp (+41.4% wzgl.)** | **+9.35 pp** | — | — |

---

## Przebieg uczenia (wybrane epoki)

| Epoka | Baseline Loss | Baseline Top-1 | Baseline Top-5 | Uczeń (KD) Loss | Uczeń (KD) Top-1 | Uczeń (KD) Top-5 | Przewaga KD (Top-1) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 01 | 6.8524 | 0.35% | 1.37% | 2.3536 | 0.45% | 1.85% | +0.10 pp |
| 05 | 6.0482 | 1.90% | 7.15% | 2.1821 | 3.15% | 10.83% | +1.25 pp |
| 10 | 5.0791 | 7.70% | 20.85% | 1.9598 | 9.15% | 25.12% | +1.45 pp |
| 15 | 4.2930 | 13.30% | 31.20% | 1.7240 | 18.05% | 39.27% | +4.75 pp |
| 20 | 3.6521 | 18.83% | 38.42% | 1.5199 | 25.90% | 48.00% | +7.07 pp |
| 24 | 3.4268 | 20.22% | 40.22% | 1.4448 | **28.67%** | 49.67% | **+8.45 pp** |
| 25 | 3.4015 | **20.28%** | **40.32%** | 1.4434 | 28.23% | **49.92%** | +7.95 pp |

---

## Wnioski

1. **Wyraźna skuteczność Knowledge Distillation:** Przy identycznej architekturze (`EfficientNet-B0`, 21.5 MB — ok. 12× mniejszej niż `EfficientNet-B7`), tym samym zbiorze i tej samej liczbie epok destylacja z Nauczyciela `B7` podniosła wynik o **+8.39 pp Top-1** (**+41.4% względnej poprawy**) oraz **+9.35 pp Top-5** (niemal 50% trafień w pierwszej piątce z 1000 klas). Uczeń przebił końcowy wynik 25-epokowego Baseline’u już w **16. epoce**.
2. **Wpływ rozmiaru zbioru (30 zdjęć/klasę — reżim *low-data*):**
   * Jako **eksperyment badawczy (PoC)** ImageNetV2 sprawdza się bardzo dobrze: przy zaledwie 24 zdjęciach treningowych na klasę bogate „miękkie” rozkłady prawdopodobieństw (*soft targets*) z Nauczyciela `B7` dostarczają znacznie więcej informacji niż same etykiety, a pełny eksperyment kończy się w ~1,5 h na jednym GPU.
   * Jako **model produkcyjny na 1000 klas** ~29% Top-1 (wobec ~82% Nauczyciela) pokazuje naturalne ograniczenie małej liczby danych — pełny ImageNet-1K zawiera ~1300 zdjęć na klasę (ponad 50× więcej).
3. **Uzupełnienie wariantu 100-klasowego:** Ten eksperyment (1000 klas × 30 zdjęć) demonstruje siłę samej destylacji w warunkach niedoboru danych, stanowiąc dobre uzupełnienie dla równoległego wariantu zespołu (100 klas na pełnym zbiorze zdjęć), który pozwala osiągnąć wyższą bezwzględną celność kosztem mniejszej liczby kategorii.

---

## Odtworzenie eksperymentu

```powershell
# 1. Pobranie wszystkich 3 wariantów ImageNetV2 (30 000 zdjęć)
.\.venv\Scripts\python.exe imagenetv2.py

# 2. Trening Baseline (EfficientNet-B0 bez Nauczyciela)
.\.venv\Scripts\python.exe -u distillation\train.py `
  --dataset imagenetv2 --epochs 25 --alpha 0.0 --batch-size 32 --num-workers 4 `
  --save-path baseline_b0_imagenetv2_1000c.pth

# 3. Trening Ucznia (Knowledge Distillation z EfficientNet-B7)
.\.venv\Scripts\python.exe -u distillation\train.py `
  --dataset imagenetv2 --epochs 25 --alpha 0.7 --temperature 4.0 --batch-size 32 --num-workers 4 `
  --save-path student_b0_imagenetv2_1000c.pth
```
