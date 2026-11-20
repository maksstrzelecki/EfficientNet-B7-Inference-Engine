'''
Standard run (default preset "small" - 100 images):
python benchmark_quantization.py

Run with a ready larger preset (e.g. "medium" - 1000 images):
python benchmark_quantization.py --test-size medium

Run with a custom dataset size (e.g. 50 classes with 2 images each = 100 images):
python benchmark_quantization.py --test-size custom --num-classes 50 --samples-per-class 2
'''

import argparse
import gc
import importlib
import statistics
import time
from pathlib import Path
from PIL import Image

from imagenetv2 import get_samples, download_imagenetv2

VARIANTS = [
    ("0. Baseline (FP32)", "inference", "efficientnet_b7.pth"),
    ("1. Half-Precision (FP16)", "inference_fp16", "efficientnet_b7_fp16.pth"),
    ("2. Half-Precision (FP16-ONNX)", "inference_fp16_onnx", "efficientnet_b7_fp16_ort.pth"),
    ("3. Weight-Only (INT8)", "inference_int8", "efficientnet_b7_int8.pth"),
    ("4. Dynamic (INT8)", "inference_dynamic", "efficientnet_b7_dynamic.pth"),
    ("5. Static ONNX", "inference_static_onnx", "efficientnet_b7_static_ort.pth"),
]


def get_test_dataset(
    size: str = "small", 
    num_classes: int = 20, 
    samples_per_class: int = 5, 
    dataset_dir: str = "data/imagenetv2"
) -> list[dict]:
    """Downloads the ImageNetV2 test dataset to validate prediction accuracy."""
    
    print(f"> Checking and downloading ImageNetV2 variants to '{dataset_dir}'...")
    download_imagenetv2(
        output_dir=dataset_dir,
        matched_frequency=True,
        threshold_07=True,
        top_images=True,
    )
    
    print(f"> Loading test samples (preset: {size}, variant: mixed)...")
    return get_samples(
        size=size, 
        variant="mixed", 
        num_classes=num_classes, 
        samples_per_class=samples_per_class, 
        dataset_dir=dataset_dir
    )


def benchmark_module(
    label: str,
    module_name: str,
    expected_filename: str,
    test_samples: list[dict],
    device: str,
    warmup: int,
) -> dict | None:
    print(f"\n>>> Testing: {label} (module: {module_name}.py)")

    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError:
        print(f"  [SKIPPED] File {module_name}.py not found.")
        return None

    target_device = device if device != "auto" else str(module.get_inference_device())
    expected_pth = Path.cwd() / expected_filename
    
    # 1. Load or generate the model
    load_start = time.perf_counter()
    if expected_pth.exists() and hasattr(module, "load_saved_model"):
        print(f"  [INFO] Found ready file {expected_filename}. Loading from disk...")
        # We assume the load_saved_model function takes a file path
        loaded = module.load_saved_model(expected_pth)
    else:
        print(f"  [INFO] Generating/calibrating model from scratch (calling load_model)...")
        loaded = module.load_model(device=target_device)
    
    load_time_s = time.perf_counter() - load_start

    # 2. Check the final file size (requires the file to exist on disk)
    if expected_pth.exists():
        size_mb = expected_pth.stat().st_size / (1024 * 1024)
    else:
        print(f"  [WARNING] The module did not save the '{expected_filename}' file. Size = 0 MB.")
        size_mb = 0.0

    # 3. Warmup on the first image from the dataset
    warmup_img = Image.open(test_samples[0]["path"]).convert("RGB")
    for _ in range(warmup):
        module.predict(warmup_img, loaded, top_k=1)

    # 4. Validation loop over the entire test dataset
    inference_times = []
    correct_predictions = 0

    print(f"  [INFO] Started inference on {len(test_samples)} images...")
    for sample in test_samples:
        img = Image.open(sample["path"]).convert("RGB")
        
        preds, timings = module.predict(img, loaded, top_k=1)
        inference_times.append(timings["inference_time"])
        
        # Accuracy verification (Top-1)
        if preds and preds[0]["class_id"] == sample["class_id"]:
            correct_predictions += 1

    accuracy = (correct_predictions / len(test_samples)) * 100

    del loaded
    gc.collect()

    return {
        "label": label,
        "module": module_name,
        "size_mb": size_mb,
        "load_time_s": load_time_s,
        "inf_mean_ms": statistics.mean(inference_times),
        "inf_median_ms": statistics.median(inference_times),
        "accuracy": accuracy,
    }


def main():
    parser = argparse.ArgumentParser(description="Automatic benchmark of inference_*.py modules")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps", "auto"])
    parser.add_argument("--warmup", type=int, default=2, help="Number of warmup calls")
    
    parser.add_argument(
        "--test-size", type=str, default="small", 
        choices=["single", "small", "medium", "full", "custom"],
        help="Size of the test dataset (use 'custom' to provide your own values)."
    )
    parser.add_argument(
        "--num-classes", type=int, default=20, 
        help="Number of classes to download (required only when --test-size custom)"
    )
    parser.add_argument(
        "--samples-per-class", type=int, default=5, 
        help="Number of images per class (required only when --test-size custom)"
    )
    args = parser.parse_args()

    # Initialize the test dataset
    test_samples = get_test_dataset(
        size=args.test_size,
        num_classes=args.num_classes,
        samples_per_class=args.samples_per_class
    )
    if not test_samples:
        print("Error: The test dataset is empty.")
        return

    results = []
    for label, mod_name, expected_filename in VARIANTS:
        res = benchmark_module(
            label=label,
            module_name=mod_name,
            expected_filename=expected_filename,
            test_samples=test_samples,
            device=args.device,
            warmup=args.warmup,
        )
        if res is not None:
            results.append(res)

    if not results:
        print("No results to display.")
        return

    # Reference point for calculating compression and speedup
    base_size = results[0]["size_mb"] or 1.0
    base_inf = results[0]["inf_mean_ms"] or 1.0

    print("\n" + "=" * 105)
    print(
        f"{'Variant (Module)':<29} | {'Size':<10} | {'Compression':<11} | "
        f"{'Inference (avg)':<15} | {'Speedup':<8} | {'Top-1 Acc'}"
    )
    print("-" * 105)

    for r in results:
        comp = base_size / r["size_mb"] if r["size_mb"] > 0 else 0.0
        speedup = base_inf / r["inf_mean_ms"] if r["inf_mean_ms"] > 0 else 0.0

        print(
            f"{r['label']:<29} | {r['size_mb']:>7.2f} MB | {comp:>10.2f}x | "
            f"{r['inf_mean_ms']:>12.2f} ms | {speedup:>7.2f}x | {r['accuracy']:>6.1f}%"
        )
    print("=" * 105)


if __name__ == "__main__":
    main()
