# Git Workflow

## 1. Główne zasady

1. Branch `main` jest główną, stabilną wersją projektu.
2. Nie implementujemy nowych funkcjonalności bezpośrednio na `main`.
3. Każda nowa funkcjonalność powinna być rozwijana na osobnym branchu.
4. Każdy branch powinien dotyczyć jednej konkretnej funkcjonalności.
5. Zmiany trafiają do `main` poprzez Pull Request (PR).
6. Każda funkcjonalność powinna zostać przetestowana przed utworzeniem PR.

## 2. Nazewnictwo branchy

Nazwa brancha powinna określać zespół oraz implementowaną funkcjonalność.

Format:

`<team>/<feature-name>`

Przykłady:

**Model:**

- `model/pruning`
- `model/quantization`
- `model/onnx-runtime`
- `model/knowledge-distillation`

**Infrastructure:**

- `infra/thread-pool`
- `infra/dynamic-batching`
- `infra/epoll`
- `infra/lru-cache`

**Interface:**

- `interface/image-upload`
- `interface/bulk-upload`
- `interface/developer-dashboard`

## 3. Workflow – krok po kroku

### Krok 1. Pobranie najnowszej wersji projektu

Przed rozpoczęciem pracy upewnij się, że masz aktualną wersję `main`.

```bash
git switch main
git pull origin main
```

### Krok 2. Utworzenie nowego brancha

Przykład dla implementacji pruning:

```bash
git switch -c model/pruning
```

Od tej chwili wszystkie zmiany powinny być wykonywane na tym branchu.

Istnieje taka możliwość, że kolega z zespołu utworzył już branch dla danej funkcjonalności. W takiej sytuacji unikamy tworzenia "czegoś w tym stylu" tylko po prostu robimy git pull origin \<nazwa_brancha> i na tej gałęzi pracujemy.

### Krok 3. Implementacja funkcjonalności

Pracujesz nad swoją funkcjonalnością.

Możesz sprawdzać aktualny stan repozytorium:

```bash
git status
```

Dodawanie zmian do staging area:

```bash
git add .
```

Możesz również dodawać konkretne pliki:

```bash
git add src/model.py
```

### Krok 4. Tworzenie commitów

Każdy commit powinien opisywać konkretną zmianę.

W projekcie stosujemy konwencję **Conventional Commits**.

Format:

`<type>(<scope>): <description>`

Gdzie:

- `type` – rodzaj zmiany.
- `scope` – obszar projektu lub funkcjonalność.
- `description` – krótki opis wykonanej zmiany.

#### Rodzaje commitów

| Type       | Zastosowanie                                    |
| ---------- | ----------------------------------------------- |
| `feat`     | Nowa funkcjonalność                             |
| `fix`      | Naprawa błędu                                   |
| `refactor` | Zmiana struktury kodu bez zmiany zachowania     |
| `perf`     | Optymalizacja wydajności                        |
| `test`     | Dodanie lub modyfikacja testów                  |
| `docs`     | Zmiany w dokumentacji                           |
| `build`    | Zmiany w konfiguracji budowania i zależnościach |
| `ci`       | Zmiany w CI/CD                                  |
| `chore`    | Pozostałe prace techniczne                      |

#### Przykłady commitów

```bash
feat(pruning): implement structured pruning
feat(quantization): add INT8 quantization
feat(onnx): add ONNX Runtime support

fix(inference): handle invalid image input
fix(api): resolve timeout during batch prediction

perf(inference): optimize tensor memory allocation
perf(batch): improve batch processing throughput

refactor(model): separate model loading from inference

test(inference): add batch prediction tests
docs(api): document prediction endpoints
build(deps): update PyTorch version
```

Tworzenie commita:

```bash
git add .
git commit -m "feat(pruning): implement structured pruning"
```

**Ważne:**

- Commity powinny być małe i logicznie spójne.
- Nie należy tworzyć jednego ogromnego commita zawierającego wszystkie zmiany.
- Commit message powinien opisywać zmianę, a nie np. `update`, `fix`, `my changes`.

### Krok 5. Wysłanie brancha na GitHub

Po zakończeniu pracy lub wykonaniu istotnej części funkcjonalności wysyłasz branch do zdalnego repozytorium.

```bash
git push -u origin model/pruning
```

Przy kolejnych aktualizacjach:

```bash
git push
```

### Krok 6. Utworzenie Pull Request

Na GitHubie:

1. Otwórz repozytorium.
2. Przejdź do zakładki Pull Requests.
3. Kliknij New Pull Request.
4. Wybierz:
   - Base: `main`
   - Compare: `model/pruning` (nazwa twojego brancha)
5. Utwórz Pull Request.

#### Przykładowy opis Pull Request

**Title:**

`feat(pruning): implement structured pruning`

**Description:**

```markdown
## Description
Implemented structured pruning for the inference model.

## Changes
- Added pruning functionality.
- Added configuration parameters.
- Added tests for pruning.

## Testing
- Tested on EfficientNet-B7.
- Compared model accuracy before and after pruning.
- Measured inference latency and memory usage.

## Related issue (jeśli dotyczny)
Closes #12
```

### Krok 7. Code Review

Po utworzeniu PR osoba odpowiedzialna za przegląd kodu (prowadzący) powinna sprawdzić:

- Czy kod działa poprawnie?
- Czy implementacja jest zgodna z architekturą projektu?
- Czy nie wprowadzono niepotrzebnych zależności?
- Czy dodano odpowiednie testy?
- Czy nie występują konflikty z innymi częściami projektu?
- Czy zmiana nie pogarsza wydajności bez uzasadnienia?

Reviewer może:

- Zaakceptować zmiany (`Approve`).
- Poprosić o poprawki (`Request changes`).
- Zostawić komentarze (`Comment`).

Jeżeli wymagane są poprawki, autor wprowadza je na tym samym branchu i wykonuje kolejne commity oraz `git push`. Pull Request aktualizuje się automatycznie.

### Krok 8. Merge do main

Po zaakceptowaniu Pull Request i przejściu wymaganych testów można połączyć zmiany z `main`.

W projekcie stosujemy **Squash and Merge**.

Pozwala to połączyć wszystkie commity z jednego brancha w jeden commit na głównej gałęzi.

Po wykonaniu merge:

```bash
git switch main
git pull origin main
```

Następnie można usunąć lokalny branch:

```bash
git branch -d model/pruning
```

Można też usunąć branch zdalny, jeżeli GitHub nie zrobił tego automatycznie:

```bash
git push origin --delete model/pruning
```

## 4. Co zrobić, gdy main zmieni się podczas naszej pracy?

Może się zdarzyć, że inna osoba zmerguje swoje zmiany do `main`, zanim skończysz pracę.

W takiej sytuacji należy zaktualizować swój branch.

```bash
git switch main
git pull origin main

git switch model/pruning
git merge main
```

Jeżeli wystąpią konflikty, należy je rozwiązać, przetestować projekt i wykonać commit merge.

Następnie:

```bash
git push
```

Pull Request zostanie automatycznie zaktualizowany.

## 5. Podsumowanie – cały workflow

```bash
# 1. Aktualizacja main
git checkout main
git pull origin main

# 2. Utworzenie brancha
git checkout -b model/pruning

# 3. Implementacja i przygotowanie zmian
git status
git add .
git commit -m "feat(pruning): implement structured pruning"

# 4. Wysłanie brancha
git push -u origin model/pruning

# 5. Utworzenie Pull Request na GitHubie
# Base: main
# Compare: model/pruning

# 6. Code Review i ewentualne poprawki
# Kolejne zmiany wykonujesz na tym samym branchu.

# 7. Po zaakceptowaniu PR – Squash and Merge na GitHubie.

# 8. Aktualizacja lokalnego main
git checkout main
git pull origin main

# 9. Usunięcie lokalnego brancha
git branch -d model/pruning
```

## 6. Najważniejsze zasady projektu

- Nie pracujemy bezpośrednio na `main`.
- Każda funkcjonalność ma własny branch.
- Stosujemy Conventional Commits.
- Nie wrzucamy do repozytorium plików tymczasowych, środowisk `.venv`, cache ani dużych niepotrzebnych plików.
- Nie commitujemy sekretów, API keys ani haseł.
- Każdy Pull Request powinien zawierać opis zmian i informację o testach.
- Nie mergujemy kodu bez Code Review.
- Po zakończeniu pracy aktualizujemy lokalny `main`.

**Cel:** utrzymywać stabilną główną gałąź projektu i umożliwić kilku osobom równoległą pracę bez wzajemnego blokowania się.
