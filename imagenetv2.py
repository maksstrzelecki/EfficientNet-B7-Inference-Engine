import json
from pathlib import Path

from datasets import load_dataset
from torchvision.models import EfficientNet_B7_Weights

ARCHIVES = {
    "matched-frequency": "imagenetv2-matched-frequency.tar.gz",
    "threshold-0.7": "imagenetv2-threshold0.7.tar.gz",
    "top-images": "imagenetv2-top-images.tar.gz",
}


def download_imagenetv2(output_dir: str | Path = "data/imagenetv2") -> Path:
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