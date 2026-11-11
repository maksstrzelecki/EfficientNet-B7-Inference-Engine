# Frontend/Interface Team

## 1. Cel zespołu

Stworzenie interfejsu użytkownika umożliwiającego korzystanie z systemu inferencyjnego oraz monitorowanie jego wydajności.

## 2. Wymagane funkcjonalności

### Obsługa obrazów

- Estetyczny interfejs do przesyłania obrazów (image upload).
- Możliwość dodawania wielu obrazów jednocześnie.
- Podgląd przesłanych obrazów.

### Prezentacja wyników

- Wyświetlanie przewidywanej klasy.
- Prezentacja wyników inferencji.
- Obsługa pojedynczych i wielu obrazów.

### Panel developera

Stworzenie panelu umożliwiającego monitorowanie i testowanie aplikacji.

Przykładowe metryki:

- Latency.
- Throughput.
- Liczba wykonanych requestów.
- Liczba równoczesnych requestów.
- Czas inferencji.
- Status serwera.

## 3. Organizacja pracy

Każda osoba wprowadzająca nową funkcjonalność tworzy osobny branch:

`interface/<nazwa_funkcjonalności>`

Przykłady:

- `interface/image-upload`
- `interface/bulk-upload`
- `interface/results`
- `interface/developer-dashboard`

