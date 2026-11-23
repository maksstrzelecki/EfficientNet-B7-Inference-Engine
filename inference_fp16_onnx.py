from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import time

import numpy as np
import onnx
import onnxruntime as ort
from onnxconverter_common import float16
import torch
from PIL import Image
from torchvision.models import EfficientNet_B7_Weights, efficientnet_b7


def get_inference_device() -> torch.device:
    # ONNX Runtime for FP16 shows its full potential on graphics cards.
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
    model: torch.nn.Module | Callable
    preprocess: Callable
    categories: list[str]
    device: torch.device


class ONNXRuntimeWrapper(torch.nn.Module):
    """Wraps the ONNX Runtime session in an interface compatible with the benchmark."""

    def __init__(self, model_path: Path, device: torch.device):
        super().__init__()
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Select provider depending on the device
        providers = ["CPUExecutionProvider"]
        if device.type == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=sess_options,
            providers=providers,
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        # Check if the ONNX model expects FP16 format as input
        self.is_fp16_input = self.session.get_inputs()[0].type == 'tensor(float16)'

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        np_batch = batch.detach().cpu().numpy()
        
        # Convert input to FP16 (reduces overhead on memory transfer to GPU)
        if self.is_fp16_input:
            np_batch = np_batch.astype(np.float16)

        ort_outputs = self.session.run([self.output_name], {self.input_name: np_batch})[0]
        out_tensor = torch.from_numpy(ort_outputs)
        
        # Safe conversion of logits back to FP32 before calculating softmax
        if out_tensor.dtype == torch.float16:
            out_tensor = out_tensor.float()
            
        return out_tensor


def load_model(device: str = "cpu") -> LoadedModel:
    target_device = torch.device(device)
    weights = EfficientNet_B7_Weights.DEFAULT
    preprocess = weights.transforms()

    cwd = Path.cwd()
    fp32_onnx_path = cwd / "_temp_effnet_fp32.onnx"
    fp16_pth_path = cwd / "efficientnet_b7_fp16_ort.pth"

    # 1. Export standard FP32 model to ONNX
    print("  [ONNX FP16] Exporting FP32 model to a temporary ONNX file...")
    model_fp32 = efficientnet_b7(weights=weights).to(target_device).eval()
    dummy_input = torch.randn(1, 3, 600, 600, device=target_device)

    torch.onnx.export(
        model_fp32,
        dummy_input,
        str(fp32_onnx_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamo=False,
    )
    del model_fp32

    # 2. Convert graph from FP32 to FP16
    print("  [ONNX FP16] Converting weights and operators to float16 (onnxconverter_common)...")
    onnx_model_fp32 = onnx.load(str(fp32_onnx_path))
    
    # keep_io_types=False forces the model's input and output to also be in FP16 format,
    # which allows for fully testing I/O performance at lower precision.
    onnx_model_fp16 = float16.convert_float_to_float16(
        onnx_model_fp32, 
        keep_io_types=False
    )

    # 3. Save the model (using .pth extension for compatibility with the benchmark function)
    onnx.save(onnx_model_fp16, str(fp16_pth_path))
    fp32_onnx_path.unlink(missing_ok=True)

    print(f"  [ONNX FP16] Saved converted model to: {fp16_pth_path}")

    # 4. Initialize session
    ort_model = ONNXRuntimeWrapper(fp16_pth_path, target_device)

    return LoadedModel(
        model=ort_model,
        preprocess=preprocess,
        categories=weights.meta["categories"],
        device=target_device,
    )


def load_saved_model(model_path: Path) -> LoadedModel:
    """Loads an existing model from disk, without converting it again."""
    target_device = get_inference_device()
    weights = EfficientNet_B7_Weights.DEFAULT
    
    ort_model = ONNXRuntimeWrapper(model_path, target_device)
    
    return LoadedModel(
        model=ort_model,
        preprocess=weights.transforms(),
        categories=weights.meta["categories"],
        device=target_device,
    )


@torch.inference_mode()
def predict(
    image: Image.Image | str | Path,
    loaded: LoadedModel,
    top_k: int = 5,
) -> tuple[list[dict], dict[str, float]]:
    timings = {}

    if not 1 <= top_k <= len(loaded.categories):
        raise ValueError(f"top_k must be between 1 and {len(loaded.categories)}")

    start = time.perf_counter()
    if isinstance(image, (str, Path)):
        with Image.open(image) as source:
            rgb = source.convert("RGB")
    else:
        rgb = image.convert("RGB")

    batch = loaded.preprocess(rgb).unsqueeze(0).to(loaded.device)
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
        {
            "class_id": index,
            "class_name": loaded.categories[index],
            "probability": score,
        }
        for score, index in zip(scores.tolist(), indices.tolist())
    ]

    timings["postprocessing_time"] = (time.perf_counter() - start) * 1000

    return predictions, timings
