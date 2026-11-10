# EfficientNet-B7 inference API

Image classification using pretrained TorchVision weights and FastAPI.

## Model

Links: [official TorchVision model documentation](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.efficientnet_b7.html) · [EfficientNet paper](https://arxiv.org/abs/1905.11946).

| Property | Value |
| --- | --- |
| Weights | `EfficientNet_B7_Weights.DEFAULT` (`IMAGENET1K_V1`) |
| Parameters | 66,347,960 |
| Model input | Float tensor `(N, 3, 600, 600)` after preprocessing; one image: `(1, 3, 600, 600)` |
| Model output | Logits `(N, 1000)` for ImageNet-1K classes |
| API output | Top-k class IDs, names, and softmax probabilities |
| Example class names | `tench`, `goldfish`, `great white shark` |

The weights' preprocessing accepts PIL images or image tensors, resizes the
shorter edge to 600 using bicubic interpolation, and center-crops to 600 × 600.
Pixels are scaled to `[0, 1]`, then normalized with mean
`[0.485, 0.456, 0.406]` and standard deviation `[0.229, 0.224, 0.225]`.
This project converts uploaded images to RGB and uses `weights.transforms()`.
Softmax is applied across all 1,000 logits before selecting top-k results.

## Install and start the server

Run these commands from the project directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn server:app --host 127.0.0.1 --port 8000
```

Startup loads the model once per server process. The device is selected
automatically: CUDA if available, otherwise Apple MPS if available, otherwise CPU.
Up to five requests per process can run the inference pipeline concurrently;
additional requests wait for a slot.

Interactive API documentation: <http://127.0.0.1:8000/docs>.

### Where the model is stored

TorchVision downloads the pretrained weights (approximately 255 MB) on first use
and reuses the cached download on subsequent starts. The default cache file is
`~/.cache/torch/hub/checkpoints/efficientnet_b7_lukemelas-c5b4e57e.pth`.
The cache location can change through PyTorch configuration, such as `TORCH_HOME`
(see [PyTorch Hub cache documentation](https://docs.pytorch.org/docs/stable/hub.html#where-are-my-downloaded-models-saved)).
Internet access is needed if the weights are not already cached.

In addition, `load_model()` saves the loaded model's state dictionary to
`efficientnet_b7.pth` in the **current working directory** on every load. Starting
the server from this project directory therefore creates `./efficientnet_b7.pth`.
This local copy is an export; the current implementation loads pretrained weights
through TorchVision rather than reading this file. A failed export is logged and
does not prevent the model from loading.

## API

### Health check

```bash
curl http://127.0.0.1:8000/health
```

Returns `{"status": "ok", "model": "efficientnet_b7"}`.

### Prediction request

`POST /predict` accepts `multipart/form-data` with one required field, `file`,
containing an image readable by Pillow, such as JPEG or PNG. The uploaded file
must be at most 10 MiB (10,485,760 bytes). The image is converted to RGB.

The optional query parameter `top_k` controls how many predictions are returned:
an integer from 1 to 1000, defaulting to 5.

```bash
curl -X POST 'http://127.0.0.1:8000/predict?top_k=3' \
  -F 'file=@image.jpg'
```

Invalid or unsupported images return HTTP 400; files over 10 MiB return 413.
Missing files or invalid `top_k` values return 422.

### Prediction response

A successful request returns HTTP 200 with JSON. Example with `top_k=1`
(values are illustrative):

```json
{
  "predictions": [
    {"class_id": 281, "class_name": "tabby", "probability": 0.87}
  ],
  "timings": {
    "upload_read_time": 0.05,
    "image_decode_time": 3.2,
    "queue_time": 0.01,
    "preprocessing_time": 12.5,
    "inference_time": 150.0,
    "postprocessing_time": 0.8
  }
}
```

Predictions are sorted by decreasing probability. `class_id` is the zero-based
ImageNet-1K class index, `class_name` is its label, and `probability` is its softmax
probability between 0 and 1. Softmax uses all 1,000 classes, so the returned top-k
probabilities need not sum to 1.

All timing values are in **milliseconds**, measured per request:

| Field | Measures |
| --- | --- |
| `upload_read_time` | Checking the uploaded file's size and reading its bytes from the server-side upload file. Excludes network upload and multipart parsing. |
| `image_decode_time` | Opening the image bytes with Pillow and converting the image to RGB. |
| `queue_time` | Waiting to acquire one of the five inference slots. Excludes earlier framework/thread-pool waiting. |
| `preprocessing_time` | RGB conversion inside `predict()`, resizing, cropping, normalization, adding the batch dimension, and moving the tensor to the selected device. Includes device synchronization. |
| `inference_time` | The model forward pass, with device synchronization so GPU execution completes before the timer stops. |
| `postprocessing_time` | Softmax, selecting top-k classes, transferring results to Python values, and building the prediction list. |

The sum of these stages is not the full HTTP request duration: it excludes
framework overhead, multipart parsing, response serialization, and network time.
The server also logs a separate `server latency` measured by HTTP middleware,
from entering the middleware until the response is produced by the downstream
handler. It is logged only, not included in the JSON, and does not measure delivery
of the response body to the client.

## Benchmark client

With the server running, use a second terminal with the virtual environment
activated:

```bash
python client.py --url http://127.0.0.1:8000/predict \
  --image-path /path/to/image.jpg \
  --total-requests 100 --concurrency 10
```

| Argument | Meaning | Default |
| --- | --- | --- |
| `--url` | Prediction endpoint URL; may include a query such as `?top_k=3`. | `http://127.0.0.1:8000/predict` |
| `--image-path` | Path to the image sent with every request. | Required |
| `--total-requests` | Total number of requests; must be positive. | `100` |
| `--concurrency` | Maximum simultaneous client requests; must be positive. | `100` |

The client reads the image once, sends it repeatedly, and prints every response's
`timings` fields plus `client_latency`, all in milliseconds. Lines appear in
completion order; the request number identifies the scheduled request.

`client_latency` measures the HTTP POST from immediately before sending until the
full response is received. It includes HTTP/network overhead and server waiting
and processing, but excludes reading the image from disk, waiting for the client's
concurrency slot, and parsing the response JSON.

The final line reports total requests, elapsed benchmark time in seconds, and
throughput in requests per second. Benchmark time includes client-side waiting
for concurrency slots. Requests use a 60-second HTTPX timeout; HTTP errors or
request failures raise an exception and stop the benchmark.

Show available arguments with `python client.py --help`.

## Python usage

```python
from inference import load_model, predict

loaded = load_model()  # CPU by default for direct Python usage
predictions, timings = predict("image.jpg", loaded, top_k=3)
print(predictions)
print(timings)  # preprocessing, inference, and postprocessing times in ms
```
