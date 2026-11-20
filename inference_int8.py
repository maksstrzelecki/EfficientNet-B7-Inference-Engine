'''
THE WEIGHT-ONLY INT8 TECHNIQUE COMPRESSES WEIGHTS TO INT8.
SUCH A MODEL TAKES UP ABOUT 4 TIMES LESS MEMORY.
DURING INFERENCE, WEIGHTS ARE DEQUANTIZED ON THE FLY, WHICH CAUSES SLIGHT SPEED DROPS ON CPU.
ON CUDA IT SHOULD BE FASTER (DEQUANTIZATION IS NOT A PROBLEM AND WE FREE UP VRAM).
'''

from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.models import EfficientNet_B7_Weights, efficientnet_b7

try:
    from torchao.quantization import quantize_, Int8WeightOnlyConfig
    HAS_TORCHAO = True
except ImportError:
    HAS_TORCHAO = False

class WeightOnlyInt8Conv2d(nn.Module):
    """Lightweight wrapper that stores Conv2d weights in INT8 and dequantizes them on the fly."""
    def __init__(self, conv: nn.Conv2d):
        super().__init__()
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups
        w = conv.weight.detach()
        max_val = w.abs().amax(dim=(1, 2, 3), keepdim=True).clamp(min=1e-8)
        scale = max_val / 127.0
        self.register_buffer("weight_int8", torch.round(w / scale).to(torch.int8))
        self.register_buffer("scale", scale.to(torch.float16))
        if conv.bias is not None:
            self.register_buffer("bias", conv.bias.detach())
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_fp = self.weight_int8.to(x.dtype) * self.scale.to(x.dtype)
        return F.conv2d(x, w_fp, self.bias, self.stride, self.padding, self.dilation, self.groups)

class WeightOnlyInt8Linear(nn.Module):
    """Lightweight wrapper that stores Linear weights in INT8."""
    def __init__(self, linear: nn.Linear):
        super().__init__()
        w = linear.weight.detach()
        max_val = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
        scale = max_val / 127.0
        self.register_buffer("weight_int8", torch.round(w / scale).to(torch.int8))
        self.register_buffer("scale", scale.to(torch.float16))
        if linear.bias is not None:
            self.register_buffer("bias", linear.bias.detach())
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_fp = self.weight_int8.to(x.dtype) * self.scale.to(x.dtype)
        return F.linear(x, w_fp, self.bias)

def replace_with_weight_only_int8(module: nn.Module):
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d):
            setattr(module, name, WeightOnlyInt8Conv2d(child))
        elif isinstance(child, nn.Linear):
            setattr(module, name, WeightOnlyInt8Linear(child))
        else:
            replace_with_weight_only_int8(child)

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
    target_device = torch.device(device)
    weights = EfficientNet_B7_Weights.DEFAULT
    model = efficientnet_b7(weights=weights).to(target_device).eval()

    if HAS_TORCHAO and target_device.type == "cuda":
        def filter_fn(module: torch.nn.Module, fqn: str) -> bool:
            return isinstance(module, (torch.nn.Linear, torch.nn.Conv2d))
        quantize_(model, Int8WeightOnlyConfig(), filter_fn=filter_fn)
    else:
        replace_with_weight_only_int8(model)

    path = Path.cwd() / "efficientnet_b7_int8.pth"
    torch.save(model.state_dict(), path)
    print(f"[Weight-Only INT8] Saved model to: {path} ({path.stat().st_size / (1024**2):.2f} MB)")
    
    return LoadedModel(
        model=model, 
        preprocess=weights.transforms(), 
        categories=weights.meta["categories"], 
        device=target_device
    )

def load_saved_model(model_path: Path) -> LoadedModel:
    """Loads the model structure and loads the quantized INT8 weights."""
    target_device = get_inference_device()
    weights = EfficientNet_B7_Weights.DEFAULT
    
    # 1. Create a clean FP32 model
    model = efficientnet_b7().to(target_device).eval()
    
    # 2. Transform the architecture to INT8 equivalents
    replace_with_weight_only_int8(model)
    
    # 3. Load INT8 weights from file
    model.load_state_dict(torch.load(model_path, map_location=target_device, weights_only=True))
    
    return LoadedModel(
        model=model,
        preprocess=weights.transforms(),
        categories=weights.meta["categories"],
        device=target_device,
    )

@torch.inference_mode()
def predict(image: Image.Image | str | Path, loaded: LoadedModel, top_k: int = 5) -> tuple[list[dict], dict[str, float]]:
    # Model compilation using TorchInductor to fuse dequantization and convolution
    '''
    if not hasattr(loaded.model, "_is_compiled"):
        loaded.model = torch.compile(loaded.model)
        loaded.model._is_compiled = True
        
        # Compiler warmup
        dummy = torch.randn(1, 3, 600, 600, device=loaded.device)
        loaded.model(dummy)
    '''
    timings = {}
    start = time.perf_counter()
    rgb = Image.open(image).convert("RGB") if isinstance(image, (str, Path)) else image.convert("RGB")
    batch = loaded.preprocess(rgb).unsqueeze(0).to(loaded.device)
    synchronize(loaded.device)
    timings["preprocessing_time"] = (time.perf_counter() - start) * 1000

    synchronize(loaded.device)
    start = time.perf_counter()
    logits = loaded.model(batch)
    synchronize(loaded.device)
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
