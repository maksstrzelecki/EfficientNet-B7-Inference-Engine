'''
DYNAMIC QUANTIZATION TO INT8 FORMAT.
MODEL WEIGHTS ARE PRE-QUANTIZED TO INT8, AND INPUT DATA IS QUANTIZED ON THE FLY DURING INFERENCE,
AFTER MULTIPLICATION, THE RESULT IS CONVERTED BACK TO FP32.

LIMITATION: DYNAMIC QUANTIZATION IN PYTORCH PROCESSES ONLY THE FOLLOWING LAYERS:
- torch.nn.Linear,
- torch.nn.LSTM, 
- torch.nn.GRU
WE USE torch.nn.Conv2d LAYERS, THERE IS ONLY ONE LINEAR LAYER AT THE END,
SO INFERENCE SPEEDUP AND SIZE REDUCTION WILL MOST LIKELY BE UNNOTICEABLE.

WE ARE ALSO USING THE torchao LIBRARY HERE, WHICH IS USED TO OPTIMIZE ARCHITECTURES.
'''

from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import time

import torch
from torchao.quantization import quantize_, Int8DynamicActivationInt8WeightConfig
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
    target_device = get_inference_device() if device == "auto" else torch.device(device)
    weights = EfficientNet_B7_Weights.DEFAULT
    model = efficientnet_b7(weights=weights).to(target_device).eval()

    # Apply dynamic quantization (only Linear layers)
    quantize_(model, Int8DynamicActivationInt8WeightConfig())

    path = Path.cwd() / "efficientnet_b7_dynamic.pth"
    torch.save(model.state_dict(), path)
    
    print(f"[Dynamic INT8] Saved model to: {path} ({path.stat().st_size / (1024**2):.2f} MB)")

    return LoadedModel(
        model=model, 
        preprocess=weights.transforms(), 
        categories=weights.meta["categories"], 
        device=target_device
    )


def load_saved_model(model_path: Path) -> LoadedModel:
    """Loads the model structure, quantizes it, and loads the saved weights."""
    target_device = get_inference_device()
    weights = EfficientNet_B7_Weights.DEFAULT
    
    # 1. Create FP32 model
    model = efficientnet_b7().to(target_device).eval()
    
    # 2. Rebuild architecture for dynamic INT8
    quantize_(model, Int8DynamicActivationInt8WeightConfig())
    
    # 3. Load state (Linear weights will be in INT8, the rest in FP32)
    model.load_state_dict(torch.load(model_path, map_location=target_device, weights_only=True))
    
    return LoadedModel(
        model=model,
        preprocess=weights.transforms(),
        categories=weights.meta["categories"],
        device=target_device,
    )


@torch.inference_mode()
def predict(
    image: Image.Image | str | Path, 
    loaded: LoadedModel, 
    top_k: int = 5
) -> tuple[list[dict], dict[str, float]]:
    timings = {}
    
    start = time.perf_counter()
    rgb = Image.open(image).convert("RGB") if isinstance(image, (str, Path)) else image.convert("RGB")
    batch = loaded.preprocess(rgb).unsqueeze(0).to(loaded.device)
    timings["preprocessing_time"] = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    logits = loaded.model(batch)
    timings["inference_time"] = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    probabilities = logits.softmax(dim=1)[0]
    scores, indices = probabilities.topk(top_k)
    
    predictions = [
        {"class_id": idx, "class_name": loaded.categories[idx], "probability": score}
        for score, idx in zip(scores.tolist(), indices.tolist())
    ]
    timings["postprocessing_time"] = (time.perf_counter() - start) * 1000
    
    return predictions, timings
