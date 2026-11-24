'''
THIS IS A HALF-PRECISION TECHNIQUE THAT CONSISTS OF
CHANGING THE WEIGHT REPRESENTATION FROM 32-BIT FLOAT TO 16-BIT FLOAT.
IT REDUCES THE MODEL SIZE BY ABOUT 50%, SPEEDS UP INFERENCE AND REDUCES RAM USAGE.
THIS TECHNIQUE IS TARGETED FOR CUDA AND MPS.
ON CPU IT MIGHT RUN SLOWER THAN FP32 (DUE TO POTENTIAL LACK OF HARDWARE SUPPORT FOR FP16).
AN ALTERNATIVE ON CPU CAN BE BFLOAT16 (BUT THERE MIGHT ALSO BE A LACK OF HARDWARE SUPPORT DEPENDING ON THE PROCESSOR).
'''


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
    dtype: torch.dtype

def load_model(device: str = "cpu") -> LoadedModel:
    weights = EfficientNet_B7_Weights.DEFAULT
    dtype = torch.bfloat16 if device == "cpu" else torch.float16

    model = efficientnet_b7(weights=weights).to(device=device, dtype=dtype).eval()

    path = Path.cwd() / "efficientnet_b7_fp16.pth"
    torch.save(model.state_dict(), path)
    print(f"[FP16] Saved model to: {path} ({path.stat().st_size / (1024**2):.2f} MB)")

    return LoadedModel(
        model,
        weights.transforms(),
        weights.meta["categories"],
        torch.device(device),
        dtype
    )

@torch.inference_mode()
def predict(image: Image.Image | str | Path, loaded: LoadedModel, top_k: int = 5) -> tuple[list[dict], dict]:
    timings = {}
    if not 1 <= top_k <= len(loaded.categories):
        raise ValueError(f"top_k must be between 1 and {len(loaded.categories)}")

    start = time.perf_counter()
    if isinstance(image, (str, Path)):
        with Image.open(image) as source:
            rgb = source.convert("RGB")
    else:
        rgb = image.convert("RGB")

    # Cast the input to the 16-bit type of the model
    batch = loaded.preprocess(rgb).unsqueeze(0).to(device=loaded.device, dtype=loaded.dtype)
    synchronize(loaded.device)
    timings["preprocessing_time"] = (time.perf_counter() - start) * 1000

    synchronize(loaded.device)
    start = time.perf_counter()
    logits = loaded.model(batch)
    synchronize(loaded.device)
    timings["inference_time"] = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    probabilities = logits.float().softmax(dim=1)[0]
    scores, indices = probabilities.topk(top_k)

    predictions = [
        {"class_id": idx, "class_name": loaded.categories[idx], "probability": score}
        for score, idx in zip(scores.tolist(), indices.tolist())
    ]
    timings["postprocessing_time"] = (time.perf_counter() - start) * 1000

    return predictions, timings
