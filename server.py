from contextlib import asynccontextmanager
import logging
import time
import io
import asyncio

import torch
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from PIL import Image, UnidentifiedImageError
from inference import get_inference_device, load_model, predict_batch

# Tutaj należy wrzucić bardziej optymalne wartości
MAX_BATCH_SIZE = 1
BATCH_TIMEOUT = 0.015 
MAX_QUEUE_SIZE = 128


class BatchRequest:
    __slots__ = ("image_tensor", "top_k", "timings", "enqueue_time", "future")

    def __init__(self, image_tensor: torch.Tensor, top_k: int, timings: dict):
        self.image_tensor = image_tensor
        self.top_k = top_k
        self.timings = timings
        self.enqueue_time = time.perf_counter()
        self.future = asyncio.get_running_loop().create_future()


def decode_and_prep(image_data: bytes, preprocess) -> torch.Tensor:
    try:
        with Image.open(io.BytesIO(image_data)) as image:
            rgb = image.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        raise ValueError("invalid_image")

    return preprocess(rgb)


async def collect_batch(queue: asyncio.Queue) -> list[BatchRequest]:
    batch: list[BatchRequest] = []

    while not batch:
        req = await queue.get()
        if not req.future.cancelled():
            batch.append(req)

    deadline = time.perf_counter() + BATCH_TIMEOUT

    while len(batch) < MAX_BATCH_SIZE:
        while not queue.empty() and len(batch) < MAX_BATCH_SIZE:
            req = queue.get_nowait()
            if not req.future.cancelled():
                batch.append(req)

        if len(batch) >= MAX_BATCH_SIZE:
            break

        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            break

        try:
            req = await asyncio.wait_for(queue.get(), timeout=remaining)
            if not req.future.cancelled():
                batch.append(req)
        except asyncio.TimeoutError:
            break

    return batch


async def batch_processor(app: FastAPI):
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = app.state.batch_queue

    while True:
        batch = []
        try:
            batch = await collect_batch(queue)

            dequeue_time = time.perf_counter()
            for req in batch:
                req.timings["queue_wait_time_ms"] = (dequeue_time - req.enqueue_time) * 1000

            items_to_predict = [(req.image_tensor, req.top_k) for req in batch]

            batch_results = await loop.run_in_executor(
                None,
                predict_batch,
                items_to_predict,
                app.state.loaded,
            )

            for req, (predictions, batch_timings) in zip(batch, batch_results):
                if not req.future.done():
                    req.timings.update(batch_timings)
                    req.future.set_result(predictions)

        except asyncio.CancelledError:
            while not queue.empty():
                req = queue.get_nowait()
                if not req.future.done():
                    req.future.cancel()
            break
        except Exception as e:
            logger.exception("Batch inference failed")
            for req in batch:
                if not req.future.done():
                    req.future.set_exception(e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    device = get_inference_device()
    app.state.loaded = load_model(device=str(device))

    app.state.batch_queue = asyncio.Queue(maxsize=MAX_QUEUE_SIZE)
    worker_task = asyncio.create_task(batch_processor(app))

    yield

    worker_task.cancel()
    await asyncio.gather(worker_task, return_exceptions=True)
    del app.state.loaded

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
async def classify(file: UploadFile = File(), top_k: int = Query(5, ge=1, le=1000)):
    if app.state.batch_queue.full():
        raise HTTPException(status_code=503, detail="Server busy, queue is full")

    timings = {}
    start = time.perf_counter()
    image_data = await file.read()
    if len(image_data) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image exceeds 10 MiB")
    timings["upload_read_time_ms"] = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    loop = asyncio.get_running_loop()
    try:
        tensor = await loop.run_in_executor(
            None,
            decode_and_prep,
            image_data,
            app.state.loaded.preprocess,
        )
    except ValueError as e:
        if str(e) == "invalid_image":
            raise HTTPException(status_code=400, detail="Invalid or unsupported image")
        raise HTTPException(status_code=500, detail="Failed to preprocess image")

    timings["decode_and_prep_time_ms"] = (time.perf_counter() - start) * 1000

    req = BatchRequest(image_tensor=tensor, top_k=top_k, timings=timings)
    try:
        app.state.batch_queue.put_nowait(req)
    except asyncio.QueueFull:
        raise HTTPException(status_code=503, detail="Server busy, queue is full")

    try:
        predictions = await req.future
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise HTTPException(status_code=500, detail="Internal inference error")

    return {"predictions": predictions, "timings": timings}