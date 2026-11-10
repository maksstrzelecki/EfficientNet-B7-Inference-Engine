from contextlib import asynccontextmanager
from threading import Semaphore
import logging
import time
import io

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from PIL import Image, UnidentifiedImageError

from inference import get_inference_device, load_model, predict


@asynccontextmanager
async def lifespan(app: FastAPI):
    
    device = get_inference_device()
    print(f"Initializing model on {device}")

    app.state.loaded = load_model(device=str(device))
    app.state.semaphore = Semaphore(5)
    print("Model initialized successfully")

    yield

    del app.state.loaded
    del app.state.semaphore


app = FastAPI(title="EfficientNet-B7 Image Classification", lifespan=lifespan)

logger = logging.getLogger(__name__)
logging.getLogger("uvicorn.access").disabled = True
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)

@app.get("/health")
def health():
    return {"status": "ok", "model": "efficientnet_b7"}

@app.middleware("http")
async def measure_request_latency(request, call_next):
    start = time.perf_counter()

    response = await call_next(request)

    server_latency = (time.perf_counter() - start) * 1000

    logger.info(
        "HTTP %s %s | status=%d | server latency=%.2f ms",
        request.method,
        request.url.path,
        response.status_code,
        server_latency
    )
    return response


@app.post("/predict")
def classify(file: UploadFile = File(), top_k: int = Query(5, ge=1, le=1000)):

    timings = {}

    # Upload read
    start = time.perf_counter()
    if file.file.seek(0, 2) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image exceeds 10 MiB")
    file.file.seek(0)
    image_data = file.file.read()
    timings["upload_read_time"] = (
        time.perf_counter() - start
    ) * 1000

    # Image decoding
    start = time.perf_counter()
    try:
        with Image.open(io.BytesIO(image_data)) as image:
            rgb = image.convert("RGB")
    except (
            UnidentifiedImageError,
            OSError,
            ValueError,
            Image.DecompressionBombError
        ):
        raise HTTPException(status_code=400, detail="Invalid or unsupported image")

    timings["image_decode_time"] = (
        time.perf_counter() - start
    ) * 1000

    # Queue / Lock waiting
    start = time.perf_counter()
    with app.state.semaphore:
        timings["queue_time"] = (
            time.perf_counter() - start
        ) * 1000

        # Inference pipeline
        predictions, inference_timings = predict(rgb, app.state.loaded, top_k)

    timings.update(inference_timings)

    # Logging
    logger.info(
        "PREDICT | %s",
        " | ".join(
            f"{key}={value:.2f}ms"
            for key, value in timings.items()
        )
    )

    return {
        "predictions": predictions,
        "timings": timings
    }
