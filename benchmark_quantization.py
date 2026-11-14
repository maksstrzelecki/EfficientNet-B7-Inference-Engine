'''
PRZYKLADOWE URUCHOMIENIE:
python benchmark_quantization.py --image-path test.jpg --runs 10
'''

import argparse
import gc
import importlib
import statistics
import time
from pathlib import Path
from PIL import Image

# Lista modułów do przetestowania
VARIANTS = [
    ("0. Baseline (FP32)", "inference"),
    ("1. Half-Precision (FP16)", "inference_fp16"),
    #("2. Dynamic INT8 (Linear)", "inference_dynamic"),
    #("3. Weight-Only INT8 (All)", "inference_weight_only"),
    #("4. Static FX PTQ (INT8)", "inference_static_fx"),
]


def find_saved_model_size(before_state: dict[Path, float]) -> tuple[float, Path | None]:
    """Wykrywa plik .pth utworzony lub zmodyfikowany przez load_model() i zwraca jego rozmiar w MB."""
    cwd = Path.cwd()
    modified_files = []

    for pth_file in cwd.glob("*.pth"):
        mtime = pth_file.stat().st_mtime
        if pth_file not in before_state or mtime > before_state[pth_file]:
            modified_files.append((mtime, pth_file))

    if modified_files:
        # Bierzemy ostatnio zmodyfikowany plik .pth
        latest_file = max(modified_files, key=lambda x: x[0])[1]
        return latest_file.stat().st_size / (1024 * 1024), latest_file

    # Fallback: sprawdzenie domyślnego efficientnet_b7.pth
    default_pth = cwd / "efficientnet_b7.pth"
    if default_pth.exists():
        return default_pth.stat().st_size / (1024 * 1024), default_pth

    return 0.0, None


def benchmark_module(
    label: str,
    module_name: str,
    image: Image.Image,
    device: str,
    warmup: int,
    runs: int,
    keep_pth: bool,
) -> dict | None:
    print(f"\n>>> Testowanie: {label} (moduł: {module_name}.py)")

    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError:
        print(f"  [POMINIĘTO] Nie znaleziono pliku {module_name}.py w bieżącym katalogu.")
        return None

    # Zapisujemy stan plików .pth przed wywołaniem load_model()
    before_pth = {p: p.stat().st_mtime for p in Path.cwd().glob("*.pth")}

    # 1. Wywołanie load_model() z badanego pliku
    load_start = time.perf_counter()
    target_device = device if device != "auto" else str(module.get_inference_device())
    loaded = module.load_model(device=target_device)
    load_time_s = time.perf_counter() - load_start

    # 2. Sprawdzenie rozmiaru zapisanego pliku .pth
    size_mb, saved_pth = find_saved_model_size(before_pth)

    # 3. Rozgrzewka (Warmup) za pomocą funkcji predict() z badanego pliku
    for _ in range(warmup):
        module.predict(image, loaded, top_k=5)

    # 4. Właściwy pomiar czasu przez funkcję predict()
    inference_times = []
    preprocess_times = []
    postprocess_times = []
    last_predictions = []

    for _ in range(runs):
        predictions, timings = module.predict(image, loaded, top_k=5)
        inference_times.append(timings["inference_time"])
        preprocess_times.append(timings.get("preprocessing_time", 0.0))
        postprocess_times.append(timings.get("postprocessing_time", 0.0))
        last_predictions = predictions

    top1 = last_predictions[0] if last_predictions else {"class_name": "N/A", "probability": 0.0}

    # Czyszczenie pamięci i ewentualne usunięcie pliku .pth
    if saved_pth and not keep_pth:
        saved_pth.unlink(missing_ok=True)

    del loaded
    gc.collect()

    return {
        "label": label,
        "module": module_name,
        "size_mb": size_mb,
        "load_time_s": load_time_s,
        "inf_mean_ms": statistics.mean(inference_times),
        "inf_median_ms": statistics.median(inference_times),
        "inf_min_ms": min(inference_times),
        "pre_mean_ms": statistics.mean(preprocess_times),
        "post_mean_ms": statistics.mean(postprocess_times),
        "top_class": top1["class_name"],
        "top_prob": top1["probability"] * 100,
    }


def main():
    parser = argparse.ArgumentParser(description="Automatyczny benchmark modułów inference_*.py")
    parser.add_argument("--image-path", type=str, default=None, help="Ścieżka do obrazu testowego")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps", "auto"])
    parser.add_argument("--warmup", type=int, default=2, help="Liczba wywołań rozgrzewkowych")
    parser.add_argument("--runs", type=int, default=5, help="Liczba wywołań pomiarowych")
    parser.add_argument("--keep-pth", action="store_true", help="Nie usuwaj wygenerowanych plików .pth po teście")
    args = parser.parse_args()

    if args.image_path and Path(args.image_path).exists():
        img = Image.open(args.image_path).convert("RGB")
        print(f"Obraz testowy: {args.image_path} ({img.size[0]}x{img.size[1]})")
    else:
        img = Image.new("RGB", (600, 600), color=(120, 160, 200))
        print("Brak --image-path: wygenerowano syntetyczny obraz RGB (600x600)")

    results = []
    for label, mod_name in VARIANTS:
        res = benchmark_module(
            label=label,
            module_name=mod_name,
            image=img,
            device=args.device,
            warmup=args.warmup,
            runs=args.runs,
            keep_pth=args.keep_pth,
        )
        if res is not None:
            results.append(res)

    if not results:
        print("Brak wyników do wyświetlenia.")
        return

    # Punkt odniesienia
    base_size = results[0]["size_mb"] or 1.0
    base_inf = results[0]["inf_mean_ms"] or 1.0

    print("\n" + "=" * 118)
    print(
        f"{'Wariant (Moduł)':<28} | {'Rozmiar .pth':<13} | {'Kompresja':<9} | "
        f"{'Inference (śr)':<15} | {'Mediana':<12} | {'Speedup':<8} | {'Top-1 Predykcja'}"
    )
    print("-" * 118)

    for r in results:
        comp = base_size / r["size_mb"] if r["size_mb"] > 0 else 0.0
        speedup = base_inf / r["inf_mean_ms"] if r["inf_mean_ms"] > 0 else 0.0
        pred_str = f"{r['top_class'][:16]} ({r['top_prob']:.1f}%)"

        print(
            f"{r['label']:<28} | {r['size_mb']:>10.2f} MB | {comp:>8.2f}x | "
            f"{r['inf_mean_ms']:>12.2f} ms | {r['inf_median_ms']:>9.2f} ms | "
            f"{speedup:>7.2f}x | {pred_str}"
        )
    print("=" * 118)


if __name__ == "__main__":
    main()