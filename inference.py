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



@torch.inference_mode()
def predict_batch(
    batch_items: list[tuple[torch.Tensor, int]], 
    loaded: LoadedModel
) -> list[tuple[list[dict], dict]]:
    batch_size = len(batch_items)

    start = time.perf_counter()
    tensors = [tensor for tensor, _ in batch_items]
    batch_tensor = torch.stack(tensors)
    
    batch_tensor = batch_tensor.to(loaded.device)
        
    synchronize(loaded.device)
    total_stack_time = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    logits = loaded.model(batch_tensor)
    synchronize(loaded.device)
    total_inf_time = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    max_top_k = max(top_k for _, top_k in batch_items)
    probabilities = logits.softmax(dim=1)
    batch_scores, batch_indices = probabilities.topk(max_top_k, dim=1)
    
    batch_scores_cpu = batch_scores.cpu().tolist()
    batch_indices_cpu = batch_indices.cpu().tolist()
    total_post_time = (time.perf_counter() - start) * 1000

    shared_timings = {
        "batch_stack_time_ms": total_stack_time,
        "batch_inference_time_ms": total_inf_time,
        "batch_postprocessing_time_ms": total_post_time,
        "amortized_inference_time_ms": total_inf_time / batch_size,
        "batch_size_executed": batch_size,
    }

    results = []
    for i, (_, requested_top_k) in enumerate(batch_items):
        scores = batch_scores_cpu[i][:requested_top_k]
        indices = batch_indices_cpu[i][:requested_top_k]

        predictions = [
            {
                "class_id": idx,
                "class_name": loaded.categories[idx],
                "probability": score,
            }
            for score, idx in zip(scores, indices)
        ]
        results.append((predictions, shared_timings.copy()))

    return results