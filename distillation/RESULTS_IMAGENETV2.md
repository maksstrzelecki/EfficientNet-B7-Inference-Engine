# Knowledge Distillation — Raport końcowy (ImageNetV2, 1000 klas)

## O co chodzi w tym eksperymencie? (po ludzku)
Duży, oryginalny model **`EfficientNet-B7`** jest bardzo dokładny, ale też ciężki (ponad **254 MB** i **66,3 mln parametrów**) i wymaga dużych zdjęć `600×600`. Chcieliśmy sprawdzić, ile wiedzy da się z niego „przepisać” do ponad **12 razy mniejszego** modelu **`EfficientNet-B0`** (**20,5 MB**, **5,3 mln parametrów**, zdjęcia `224×224`) za pomocą techniki **Knowledge Distillation (destylacja wiedzy)**.

Zamiast uczyć mały model wyłącznie na suchych etykietach (np. *„to jest golden retriever”*), podczas destylacji duży model (Nauczyciel `B7`) podpowiada małemu (Uczniowi `B0`) cały rozkład prawdopodobieństw dla wszystkich 1000 klas naraz (np. *„golden retriever 70%, labrador 15%, spaniel 5%...”*). Dzięki temu mały model uczy się nie tylko *co* jest na zdjęciu, ale też *do czego to jest podobne*.

Żeby wynik był w 100% uczciwy, wytrenowaliśmy **dwa identyczne małe modele `EfficientNet-B0`** na dokładnie tych samych danych (24 000 zdjęć treningowych, 6 000 walidacyjnych ze wszystkich 3 wariantów **ImageNetV2**, wszystkie **1000 klas**) i w dokładnie tych samych 25 epokach:
1. **Uczeń bez destylacji (Baseline)** — uczył się sam, tylko z etykiet.
2. **Uczeń z destylacją (KD)** — uczył się z podpowiedziami Nauczyciela `EfficientNet-B7`.

> **Uwaga o nazwach plików w repozytorium:**
> * `student_b0_imagenetv2_1000c.pth` oraz `baseline_b0_imagenetv2_1000c.pth` — wagi z tego eksperymentu (**1000 klas** na ImageNetV2, 30 zdjęć na klasę).
> * `student_b0.pth` — nazwa pozostawiona wolna dla równoległego eksperymentu zespołu (**100 klas** na pełnym zbiorze zdjęć), żeby uniknąć konfliktów w Gicie.

---

## Główne porównanie: Oryginał vs Bez destylacji vs Z destylacją

Wszystkie trzy modele zostały zmierzone na **dokładnie tych samych 6 000 zdjęć walidacyjnych ImageNetV2 (1000 klas)** oraz na tym samym sprzęcie (**NVIDIA RTX 5080** i CPU, pomiar pojedynczego zdjęcia `batch=1`, FP32):

| Model | Plik wag | Rozmiar (kompresja) | Parametry | Val Top-1 | Val Top-5 | Czas GPU (RTX 5080) | Czas CPU |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **1. Oryginalny model (Nauczyciel `B7`, 600×600)** | `EfficientNet_B7_Weights` *(pretrenowany na ImageNet-1K)* | 254.7 MB *(1.0×)* | 66.35 mln | **79.97%** | **95.23%** | 17.20 ms *(1.0×)* | 336.52 ms *(1.0×)* |
| **2. Uczeń bez destylacji (Baseline `B0`, 224×224)** | `baseline_b0_imagenetv2_1000c.pth` | **20.5 MB** *(12.4× mniej)* | **5.29 mln** *(12.5× mniej)* | 20.28% *(ep. 25)* | 40.32% | **5.58 ms** *(3.1× szybciej)* | **12.16 ms** *(27.7× szybciej!)* |
| **3. Uczeń z destylacją (KD `B0`, 224×224)** | `student_b0_imagenetv2_1000c.pth` | **20.5 MB** *(12.4× mniej)* | **5.29 mln** *(12.5× mniej)* | **28.67%** *(ep. 24)* | **49.67%** *(49.92% w ep. 25)* | **5.58 ms** *(3.1× szybciej)* | **12.16 ms** *(27.7× szybciej!)* |

> **Co zyskuje sam proces destylacji (wiersz 3 vs wiersz 2)?**
> * **+8.39 punktu procentowego w Top-1** (wzrost z `20.28%` do `28.67%` — czyli aż **+41.4% względnej poprawy**!)
> * **+9.35 punktu procentowego w Top-5** (wzrost z `40.32%` do `49.67%`, czyli niemal co drugie zdjęcie trafione w pierwszej piątce z 1000 klas)
> * **Zero dodatkowego kosztu w działaniu:** po treningu Nauczyciel jest odłączany, więc Uczeń waży tyle samo (**20.5 MB**) i działa równie błyskawicznie (**3.1× szybciej na GPU** i **27.7× szybciej na CPU** od oryginalnego `B7`).

*(Dla pełnego kontekstu: wcześniejsze benchmarki zespołu dla samego `EfficientNet-B7` znajdują się w [README.md](../README.md) — m.in. CPU 790 ms / 82.3% Top-1 na podzbiorze 200 klas oraz MPS 244 ms — oraz w [benchmarks/reports/summary.md](../benchmarks/reports/summary.md) dla serwera FastAPI: 88.16 ms pełnego requestu HTTP przy `concurrency=1`. Powyższa tabela została zmierzona specjalnie w jednolitych warunkach 1:1 na RTX 5080 i pełnym zestawie 6 000 zdjęć walidacyjnych.)*

---

## Jak poszedł trening? (opis krok po kroku)

1. **Przygotowanie danych:** Pobraliśmy wszystkie 3 warianty ImageNetV2 (łącznie **30 000 zdjęć** — po 30 na każdą z 1000 klas) i podzieliliśmy je stratyfikowanie (równo w każdej klasie): **24 zdjęcia na trening** (24 000 łącznie) i **6 na walidację** (6 000 łącznie).
2. **Optymalizacja potoku treningowego:** W trakcie prac przyspieszyliśmy `distillation/train.py` ponad 4-krotnie dla Baseline’u (pomijając zbędne skalowanie zdjęć do `600×600`, gdy Nauczyciel nie pracuje lub trwa walidacja) oraz włączyliśmy natywne `CUDA bfloat16` i `cudnn.benchmark` na RTX 5080.
3. **Trening Baseline’u (bez Nauczyciela):** Całe 25 epok zajęło zaledwie **~15,5 minuty** (~35 s/epokę, zużycie pamięci VRAM tylko **1.45 GB**). Model stopniowo wspinał się od `0.35%` w 1. epoce do końcowych **`20.28%` Top-1** i **`40.32%` Top-5**.
4. **Trening Ucznia (z Nauczycielem `B7`):** 25 epok trwało **~1 godz. 35 min** (~225 s/epokę, **5.57 GB VRAM**). Efekt destylacji był widoczny niemal od razu:
   * W **10. epoce** Uczeń wyprzedzał już Baseline o `+1.45 pp` (Top-1) i `+4.27 pp` (Top-5).
   * W **16. epoce** (`20.12%` Top-1 / `41.15%` Top-5) Uczeń **praktycznie dogonił końcowy wynik 25-epokowego Baseline’u**, mając przed sobą jeszcze 9 epok.
   * W **24. epoce** osiągnął szczytowe **`28.67%` Top-1** (`49.67%` Top-5), a w 25. epoce **`49.92%` Top-5**.

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

1. **Destylacja wiedzy zadziałała książkowo:** Przy dokładnie tym samym rozmiarze modelu (`20.5 MB` — 12.4× mniej niż `B7`) i tej samej szybkości działania (`3.1×` szybciej na GPU, `27.7×` szybciej na CPU), podpowiedzi Nauczyciela podbiły skuteczność Ucznia o **+8.39 pp Top-1 (+41.4% względnie)** oraz **+9.35 pp Top-5**.
2. **Dlaczego Uczeń ma ~29% Top-1 wobec ~80% oryginalnego `B7`?** Nie jest to słabość destylacji, tylko **skala zbioru danych**: oryginalny `EfficientNet-B7` był trenowany na pełnym ImageNet-1K (**1,28 mln zdjęć**, czyli ~1300 zdjęć na klasę), natomiast nasz Uczeń startował całkowicie od zera na zaledwie **24 zdjęciach treningowych na klasę** (ponad 50× mniej danych).
3. **Co to daje w całym projekcie:** Ten eksperyment (1000 klas × 30 zdjęć) idealnie udowadnia **siłę samej destylacji w warunkach małej liczby danych**, stanowiąc bardzo dobre uzupełnienie dla równoległego eksperymentu zespołu (100 klas na pełnym zbiorze zdjęć), gdzie większa liczba przykładów na klasę pozwala uzyskać wyższą bezwzględną celność.

---

## Konfiguracja i odtworzenie eksperymentu

| Parametr | Wartość |
| --- | --- |
| Zbiór danych | ImageNetV2 (3 warianty, 1000 klas: 24 000 train / 6 000 val) |
| Optymalizator / Scheduler | `AdamW` (`lr=1e-3`, `weight_decay=1e-4`) + `CosineAnnealingLR` |
| Hiperparametry KD | `alpha=0.7`, `T=4.0` (Baseline: `alpha=0.0`), `epochs=25`, `batch_size=32` |
| Sprzęt i precyzja | NVIDIA RTX 5080, `CUDA bfloat16`, `cudnn.benchmark=True`, `seed=42` |

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
