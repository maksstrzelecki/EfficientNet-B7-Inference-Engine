'''
DYNAMICZNA KWANTYZACJA DO FORMATU INT8.
JEST TO TECHNIKA PRZEZNACZONA PRZEDE WSZYSTKIM NA CPU.
WAGI MODELU SA Z GÓRY KWANTYZOWANE DO INT8, A DANE WEJŚCIOWE SA KWANTYZOWANE W LOCIE PODCZAS INFERENCJI,
PO MNOZENIU WYNIK JEST PRZELICZANY Z POWROTEM NA FP32.

OGRANICZENIE: DYNAMICZNA KWANTYZACJA W PYTORCHU PRZETWARZA TYLKO WARSTWY:
    - torch.nn.Linear,
    - torch.nn.LSTM, 
    - torch.nn.GRU
U NAS ZASTOSOWANE SA WARSTWY torch.nn.Conv2d, NA KONCU JEST TYLKO JEDNA WARTSWA LINIOWA,
WIEC PRZYSPIESZENIE INFERENCJI I REDUKCJA ROZMIARU NAJPRAWDOPODOBNIEJ BEDA NIEZAUWAZALNE.

WYKORZYSTUJE TUTAJ DODATKOWO BIBLIOTEKE torchao, KTORA SLUZY DO OPTIMALIZACJI ARCHITEKTUR.
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
    return torch.device("cpu")

def synchronize(device: torch.device):
    pass

@dataclass
class LoadedModel:
    model: torch.nn.Module
    preprocess: Callable
    categories: list[str]
    device: torch.device

def load_model(device: str = "cpu") -> LoadedModel:
    device = "cpu" 
    weights = EfficientNet_B7_Weights.DEFAULT
    model = efficientnet_b7(weights=weights).to(device).eval()

    quantize_(model, Int8DynamicActivationInt8WeightConfig())

    path = Path.cwd() / "efficientnet_b7_dynamic_int8.pth"
    
    torch.save(model.state_dict(), path)
    
    print(f"[Dynamic INT8] Saved model to: {path} ({path.stat().st_size / (1024**2):.2f} MB)")

    return LoadedModel(model, weights.transforms(), weights.meta["categories"], torch.device(device))

@torch.inference_mode()
def predict(image: Image.Image | str | Path, loaded: LoadedModel, top_k: int = 5) -> tuple[list[dict], dict]:
    timings = {}
    start = time.perf_counter()
    rgb = Image.open(image).convert("RGB") if isinstance(image, (str, Path)) else image.convert("RGB")
    batch = loaded.preprocess(rgb).unsqueeze(0).to(loaded.device)
    timings["preprocessing_time"] = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    # Nowoczesne modele torchao można dodatkowo przyspieszyć przez kompilację
    # loaded.model = torch.compile(loaded.model)
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