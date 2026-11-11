# Model Optimization Team

## 1. Wymagane funkcjonalności

Model powinien udostępniać przynajmniej dwie podstawowe funkcje:

- `load_model()` – ładowanie modelu do RAM lub VRAM.
- `predict_one()` – wykonanie inferencji dla pojedynczego obrazu.
- `predict_batch()` – wykonanie inferencji dla wielu obrazów jednocześnie.

Docelowo implementacja musi ewoluować w kierunku obsługi batch inference na rzecz predict_one.

## 2. Cel zespołu

Waszym zadaniem będzie optymalizacja:

- Zużycia pamięci modelu.
- Czasu pojedynczej inferencji.
- Wykorzystania dostępnego sprzętu.

## 3. Możliwe techniki optymalizacji

### Runtime i kompilacja

- Wykorzystanie różnych runtime, np. PyTorch lub ONNX Runtime.
- Kompilacja modelu pod konkretny sprzęt.
- Optymalizacja grafu obliczeniowego.

### Kompresja modelu

- Quantization – kwantyzacja modelu.
- Pruning – usuwanie niepotrzebnych parametrów lub struktur modelu.
- Knowledge Distillation – przekazywanie wiedzy z większego modelu do mniejszego.
- Layer Reduction – redukcja liczby warstw.
- Weight Sharing – współdzielenie wag.

Każda osoba w grupie wybiera sobie temat, robi research w danym zakresie, wdraża i testuje.
Oczywiście pamiętajcie, że to są tylko przykłady i możecie dowolnie eksperymentować z innymi technikami.

## 4. Organizacja pracy

Każda osoba wprowadzająca nową funkcjonalność tworzy osobny branch według schematu:

`model/<nazwa_funkcjonalności>`

Przykłady:

- `model/quantization`
- `model/pruning`
- `model/onnx-runtime`
- `model/knowledge-distillation`

