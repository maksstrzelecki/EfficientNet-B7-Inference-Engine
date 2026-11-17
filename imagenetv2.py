"""Utilities for downloading and sampling images from the ImageNetV2 dataset."""
from collections import defaultdict
import json
from pathlib import Path
import random
from typing import Literal

from datasets import load_dataset
from torchvision.models import EfficientNet_B7_Weights

# Archive filenames on Hugging Face for each ImageNetV2 variant
ARCHIVES = {
    "matched-frequency": "imagenetv2-matched-frequency.tar.gz",
    "threshold-0.7": "imagenetv2-threshold0.7.tar.gz",
    "top-images": "imagenetv2-top-images.tar.gz",
}


# Predefined sampling sizes formatted as (num_classes, samples_per_class)
PRESETS: dict[str, tuple[int, int] | None] = {
    "single": (1, 1),
    "small": (20, 5),
    "medium": (200, 5),
    "full": (None, None),
}


def get_samples(
    size: Literal["single", "small", "medium", "full", "custom"] = "small",
    variant: Literal["matched-frequency", "threshold-0.7", "top-images", "mixed"] = "mixed",
    seed: int = 42,
    num_classes: int | None = None,
    samples_per_class: int | None = None,
    dataset_dir: str | Path = "data/imagenetv2",
) -> list[dict]:
    """Select and return a shuffled list of image samples from the local ImageNetV2 dataset.

    Reads `manifest.json` from `dataset_dir`, filters images by the chosen dataset variant,
    and randomly picks classes and images according to a preset or custom counts.

    Args:
        size: Preset size ('single', 'small', 'medium', 'full') or 'custom'.
        variant: Dataset split to sample from, or 'mixed' to include all variants.
        seed: Random seed for reproducible sampling.
        num_classes: Number of classes to sample (required if size='custom').
        samples_per_class: Number of images per class (required if size='custom').
        dataset_dir: Root directory containing the downloaded images and manifest.json.

    Returns:
        A shuffled list of dicts, each containing 'path', 'class_id', 'class_name', and 'split'.
    """

    root = Path(dataset_dir)
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Not found manifest at {manifest_path}. Please run download_imagenetv2() first."
        )

    if size == "custom":
        if not num_classes or not samples_per_class or num_classes < 1 or samples_per_class < 1:
            raise ValueError(
                "For size='custom', provide positive values for num_classes and samples_per_class."
            )
        target_classes, target_per_class = num_classes, samples_per_class
    elif size in PRESETS:
        preset = PRESETS[size]
        target_classes, target_per_class = preset
    else:
        raise ValueError(f"Unknown size: '{size}'.")

    raw_samples = json.loads(manifest_path.read_text(encoding="utf-8"))["samples"]

    filtered = (
        raw_samples
        if variant == "mixed"
        else [i for i in raw_samples if i["split"] == variant]
    )
    if not filtered:
        raise ValueError(f"No samples found for variant '{variant}' in the manifest.")

    rng = random.Random(seed)

   
    if target_classes is None:
        selected = list(filtered)
    else:
        by_class: dict[int, list[dict]] = defaultdict(list)
        for sample in filtered:
            by_class[sample["class_id"]].append(sample)

        available_classes = sorted(by_class.keys())
        if target_classes > len(available_classes):
            raise ValueError(
                f"Requested {target_classes} classes, but only {len(available_classes)} are available."
            )
        
        chosen_classes = rng.sample(available_classes, target_classes)
        selected = []
        for class_id in sorted(chosen_classes):
            pool = sorted(by_class[class_id], key=lambda x: x["file"])
            if target_per_class > len(pool):
                raise ValueError(
                    f"Class {class_id} has {len(pool)} images (requested {target_per_class})."
                )
            selected.extend(rng.sample(pool, target_per_class))
    rng.shuffle(selected)

    return [
        {
            "path": root / item["file"],
            "class_id": item["class_id"],
            "class_name": item["class_name"],
            "split": item["split"],
        }
        for item in selected
    ]




def download_imagenetv2(output_dir: str | Path = "data/imagenetv2") -> Path:
    """Download all ImageNetV2 variants, save images as JPEGs, and generate manifest.json.

    Fetches the three ImageNetV2 archives from Hugging Face, saves each image to
    `output_dir/<variant>/<class_id>/<index>.jpg`, maps class IDs to standard ImageNet
    category names, and writes an index of all files to `manifest.json`.

    Args:
        output_dir: Directory where the images and manifest.json will be stored.

    Returns:
        Path to the created manifest.json file.
    """

    out_path = Path(output_dir)
    categories = EfficientNet_B7_Weights.DEFAULT.meta["categories"]
    manifest = []

    for variant, archive_name in ARCHIVES.items():
        ds = load_dataset(
            "vaishaal/ImageNetV2",
            data_files=archive_name,
            split="train",
        )

        for idx, row in enumerate(ds):
            class_id = int(row["__key__"].split("/")[-2])

            rel_path = Path(variant) / f"{class_id:03d}" / f"{idx:04d}.jpg"
            file_path = out_path / rel_path
            file_path.parent.mkdir(parents=True, exist_ok=True)

            row["jpeg"].convert("RGB").save(file_path, format="JPEG", quality=95)

            manifest.append({
                "file": rel_path.as_posix(),
                "class_id": class_id,
                "class_name": categories[class_id],
                "split": variant,
            })
    
    manifest_path = out_path / "manifest.json"
    manifest_path.write_text(
        json.dumps({"total": len(manifest), "samples": manifest}, indent=2),
        encoding="utf-8",
    )
    return manifest_path


if __name__ == "__main__":
    download_imagenetv2()