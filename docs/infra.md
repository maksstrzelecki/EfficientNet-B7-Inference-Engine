# Backend/Infra Team

## 1. Interfejs zespołu modelowego

Zespół modelowy przekaże infrastrukturze następujące funkcje:

- `load_model()`
- `predict_one()`
- `predict_batch()`

Infrastruktura powinna korzystać z tych funkcji bez konieczności ingerowania w implementację samego modelu.

## 2. Cel zespołu

Waszym zadaniem jest utworzenie aplikacji FastAPI (z serwerem Uvicorn lub innym), obsługującej żądania HTTP.

W szczególności skupiamy się na:

- Skalowaniu pionowym.
- Optymalizacji wydajności.
- Obsłudze wielu równoczesnych klientów.
- Efektywnym wykorzystaniu zasobów CPU.

## 3. Wymagane endpointy

| Endpoint        | Metoda HTTP | Opis                           |
| --------------- | ----------- | ------------------------------ |
| `/health`       | GET         | Sprawdzanie stanu aplikacji    |
| `/predict`      | POST        | Inferencja pojedynczego obrazu |
| `/predict-bulk` | POST        | Inferencja wielu obrazów       |

## 4. Możliwe techniki optymalizacji

### Obsługa współbieżności i I/O

- Programowanie asynchroniczne (`async/await`).
- Samodzielnie zaimplementowana pula wątków (ThreadPool).
- Multipleksowanie I/O, np. `epoll` na Linuxie.
- Przetwarzanie batchowe z użyciem mechanizmu Promise-Future.

### Zarządzanie pamięcią i requestami

- Implementacja pamięci cache, np. LRU.
- Kolejkowanie requestów.
- Dynamic batching.
- Kontrola liczby równocześnie wykonywanych inferencji.

### Skalowanie (tutaj będzie gółwnie rola prowadzącego)

- Implementacja load balancera (serwera reverse proxy).
- Rozdzielanie requestów pomiędzy instancje serwera.
- Testowanie różnych strategii rozdzielania obciążenia.

## 5. Benchmarking i testowanie

Należy stworzyć program umożliwiający benchmarkowanie aplikacji.

Podstawowe metryki:

- Latency – czas obsługi pojedynczego requestu.
- Throughput – liczba obsłużonych requestów na sekundę.
- CPU utilization.
- RAM / VRAM utilization.
- p50, p95 i p99 latency.

Ważne: każda wprowadzona optymalizacja powinna być porównana z wersją bazową.

## 6. Organizacja pracy

Każda osoba wprowadzająca nową funkcjonalność tworzy osobny branch:

`infra/<nazwa_funkcjonalności>`

Przykłady:

- `infra/thread-pool`
- `infra/dynamic-batching`
- `infra/epoll`
- `infra/lru-cache`
- `infra/load-balancer`

