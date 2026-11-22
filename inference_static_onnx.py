'''
STATIC QUANTIZATION TO INT8 FORMAT USING ONNX RUNTIME.
THIS TECHNIQUE QUANTIZES BOTH WEIGHTS AND ACTIVATIONS TO INT8.
IT REQUIRES A CALIBRATION PHASE USING A REPRESENTATIVE DATASET TO CALCULATE 
THE SCALES AND ZERO POINTS FOR ACTIVATIONS BEFORE INFERENCE.
IT SIGNIFICANTLY REDUCES THE MODEL SIZE (BY ABOUT 4X) AND SPEEDS UP INFERENCE.

SUPPORTED PLATFORMS:
- CPU (uses CPUExecutionProvider with VNNI instructions)
- CUDA (uses TensorRTExecutionProvider / CUDAExecutionProvider)
- MPS/Apple Silicon (uses CoreMLExecutionProvider)
'''

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable
import sys
import time

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process
import torch
from PIL import Image
from torchvision.models import EfficientNet_B7_Weights, efficientnet_b7

from imagenetv2 import download_imagenetv2, get_samples


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
    model: torch.nn.Module | Callable
    preprocess: Callable
    categories: list[str]
    device: torch.device


class ONNXCalibrationReader(CalibrationDataReader):
    """Provides processed images from ImageNetV2 to the quantize_static function."""

    def __init__(self, input_name: str, batches: list[np.ndarray]):
        self.input_name = input_name
        self._iterator = iter([{self.input_name: b} for b in batches])

    def get_next(self) -> dict[str, np.ndarray] | None:
        return next(self._iterator, None)


class ONNXRuntimeWrapper(torch.nn.Module):
    """
    Wraps the ONNX Runtime session in a torch.nn.Module interface,
    so that the predict() function can call loaded.model(batch).
    """

    def __init__(self, model_path: Path, device: torch.device):
        super().__init__()
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Select execution providers based on the requested hardware
        providers = ["CPUExecutionProvider"]
        if device.type == "cuda":
            providers = [
                "TensorRTExecutionProvider", 
                "CUDAExecutionProvider", 
                "CPUExecutionProvider"
            ]
        elif device.type == "mps":
            providers = [
                "CoreMLExecutionProvider", 
                "CPUExecutionProvider"
            ]

        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=sess_options,
            providers=providers,
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        ort_inputs = {self.input_name: batch.detach().cpu().numpy()}
        ort_outputs = self.session.run([self.output_name], ort_inputs)[0]
        return torch.from_numpy(ort_outputs)


def get_imagenetv2_calibration_paths(
    num_classes: int = 25,
    samples_per_class: int = 1,
    dataset_dir: str | Path = "data/imagenetv2",
) -> list[Path]:
    """Gets paths to calibration images using the get_samples() function."""
    manifest_path = Path(dataset_dir) / "manifest.json"

    if not manifest_path.exists():
        print(f"  [ONNX Static] Dataset not found in '{dataset_dir}'. Downloading ImageNetV2...")
        download_imagenetv2(
            output_dir=dataset_dir,
            matched_frequency=True,
            threshold_07=False,
            top_images=False,
        )

    samples = get_samples(
        size="custom",
        variant="mixed",
        seed=42,
        num_classes=num_classes,
        samples_per_class=samples_per_class,
        dataset_dir=dataset_dir,
    )
    paths = [sample["path"] for sample in samples]

    for i, arg in enumerate(sys.argv):
        if arg == "--image-path" and i + 1 < len(sys.argv):
            p = Path(sys.argv[i + 1])
            if p.exists() and p not in paths:
                paths.append(p)
        elif arg.startswith("--image-path="):
            p = Path(arg.split("=", 1)[1])
            if p.exists() and p not in paths:
                paths.append(p)

    return paths


def load_model(
    device: str = "cpu",
    calibration_images: Iterable[Image.Image | str | Path] | None = None,
    dataset_dir: str | Path = "data/imagenetv2",
) -> LoadedModel:
    target_device = get_inference_device() if device == "auto" else torch.device(device)
    weights = EfficientNet_B7_Weights.DEFAULT
    preprocess = weights.transforms()

    cwd = Path.cwd()
    fp32_onnx_path = cwd / "_temp_effnet_fp32.onnx"
    preproc_onnx_path = cwd / "_temp_effnet_preproc.onnx"
    quantized_pth_path = cwd / "efficientnet_b7_static_ort.pth"

    print("  [ONNX Static] Exporting FP32 model to ONNX format...")
    # The export phase always happens on CPU to avoid device memory limits during conversion
    model_fp32 = efficientnet_b7(weights=weights).to("cpu").eval()
    dummy_input = torch.randn(1, 3, 600, 600, device="cpu")

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

    print("  [ONNX Static] Graph optimization before quantization (quant_pre_process)...")
    try:
        quant_pre_process(
            input_model_path=str(fp32_onnx_path),
            output_model_path=str(preproc_onnx_path),
            skip_symbolic_shape=True,
        )
        model_to_quantize = preproc_onnx_path
    except Exception:
        model_to_quantize = fp32_onnx_path

    if calibration_images is None:
        try:
            calibration_images = get_imagenetv2_calibration_paths(
                num_classes=100,
                samples_per_class=5,
                dataset_dir=dataset_dir,
            )
        except Exception as e:
            print(f"  [ONNX Static] Error loading ImageNetV2 ({e}).")
            calibration_images = []

    calib_batches: list[np.ndarray] = []
    if calibration_images:
        for img_source in calibration_images:
            if isinstance(img_source, (str, Path)):
                with Image.open(img_source) as source:
                    rgb = source.convert("RGB")
            else:
                rgb = img_source.convert("RGB")
            tensor_batch = preprocess(rgb).unsqueeze(0).numpy()
            calib_batches.append(tensor_batch)
        print(f"  [ONNX Static] Prepared {len(calib_batches)} images for calibration.")
    else:
        print("  [ONNX Static] WARNING: No images! Using a synthetic image.")
        synth_img = Image.new("RGB", (600, 600), color=(120, 160, 200))
        calib_batches.append(preprocess(synth_img).unsqueeze(0).numpy())

    reader = ONNXCalibrationReader(input_name="input", batches=calib_batches)

    print("  [ONNX Static] Calibration and static quantization (quantize_static)...")
    quantize_static(
        model_input=str(model_to_quantize),
        model_output=str(quantized_pth_path),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QOperator,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QUInt8,
        op_types_to_quantize=["Conv", "MatMul", "Gemm"],
        per_channel=False,
        reduce_range=False,
        calibrate_method=CalibrationMethod.MinMax,
    )

    fp32_onnx_path.unlink(missing_ok=True)
    preproc_onnx_path.unlink(missing_ok=True)
    for ext_data in cwd.glob("_temp_effnet_*.onnx.data"):
        ext_data.unlink(missing_ok=True)

    print(f"  [ONNX Static] Saved quantized model to: {quantized_pth_path}")

    ort_model = ONNXRuntimeWrapper(quantized_pth_path, target_device)

    return LoadedModel(
        model=ort_model,
        preprocess=preprocess,
        categories=weights.meta["categories"],
        device=target_device,
    )


def load_saved_model(model_path: Path) -> LoadedModel:
    """Loads the compiled ONNX model directly from disk."""
    target_device = get_inference_device()
    weights = EfficientNet_B7_Weights.DEFAULT
    
    print(f"  [ONNX Static] Loading existing model for device: {target_device.type.upper()}")
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
    """Return top-k class indices, labels, and softmax probabilities."""
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
    probabilities = logits.softmax(dim=1)[0]
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
