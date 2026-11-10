from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import time

import torch
from PIL import Image
from torchvision.models import EfficientNet_B7_Weights, efficientnet_b7

def get_inference_device() -> torch.device:
    
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")

def synchronize(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()

@dataclass
class LoadedModel:
    model: torch.nn.Module
    preprocess: Callable
    categories: list[str]
    device: torch.device


def load_model(device: str = "cpu") -> LoadedModel:

    weights = EfficientNet_B7_Weights.DEFAULT
    model = efficientnet_b7(weights=weights).to(device).eval()

    path = Path.cwd() / "efficientnet_b7.pth"

    try:
        torch.save(model.state_dict(), path)
        print(f"Saved model to: {path}")
    except OSError as e:
        print(f"Failed to save model: {e}")

    return LoadedModel(
        model,
        weights.transforms(),
        weights.meta["categories"],
        torch.device(device)
    )


@torch.inference_mode()
def predict(
    image: Image.Image | str | Path,
    loaded: LoadedModel,
    top_k: int = 5
) -> tuple[list[dict], float]:

    timings = {}
    
    """Return top-k class indices, labels, and softmax probabilities."""

    if not 1 <= top_k <= len(loaded.categories):
        raise ValueError(f"top_k must be between 1 and {len(loaded.categories)}")

    # Preprocessing
    start = time.perf_counter()
    if isinstance(image, (str, Path)):
        with Image.open(image) as source:
            rgb = source.convert("RGB")
    else:
        rgb = image.convert("RGB")

    batch = loaded.preprocess(rgb).unsqueeze(0).to(loaded.device)

    synchronize(loaded.device)
    timings["preprocessing_time"] = (
        time.perf_counter() - start
    ) * 1000

    # Inference
    synchronize(loaded.device)
    start = time.perf_counter()

    logits = loaded.model(batch)
    synchronize(loaded.device)

    timings["inference_time"] = (
        time.perf_counter() - start
    ) * 1000

    # Postprocessing
    start = time.perf_counter()

    probabilities = logits.softmax(dim=1)[0]
    scores, indices = probabilities.topk(top_k)

    predictions = [
        {
            "class_id": index,
            "class_name": loaded.categories[index],
            "probability": score
        }
        for score, index in zip(scores.tolist(), indices.tolist())
    ]

    timings["postprocessing_time"] = (
        time.perf_counter() - start
    ) * 1000

    return predictions, timings
