from contextlib import asynccontextmanager
import logging
import time
import io
import asyncio

import torch
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from PIL import Image, UnidentifiedImageError

from inference import get_inference_device, load_model, predict_batch

MAX_BATCH_SIZE = 2
BATCH_TIMEOUT = 0.1 

class BatchRequest:
    def __init__(self, image_tensor: torch.Tensor, top_k: int, timings: dict):
        self.image_tensor = image_tensor
        self.top_k = top_k
        self.timings = timings
        self.enqueue_time = time.perf_counter()
        self.future = asyncio.get_running_loop().create_future()

async def batch_processor(app: FastAPI):
    while True:
        batch = []
        try:
            new_req = await app.state.batch_queue.get()
            batch.append(new_req)

            time_limit = time.perf_counter() + BATCH_TIMEOUT
            while len(batch) < MAX_BATCH_SIZE:
                time_remaining = time_limit - time.perf_counter()
                if time_remaining <= 0:
                    break
                try:
                    new_req = await asyncio.wait_for(app.state.batch_queue.get(), timeout=time_remaining)
                    batch.append(new_req)
                except asyncio.TimeoutError:
                    break

            loop_ref = asyncio.get_running_loop()
            try:
                items_to_predict = [(req.image_tensor, req.top_k) for req in batch]
                
                batch_results = await loop_ref.run_in_executor(
                    None, 
                    predict_batch, 
                    items_to_predict, 
                    app.state.loaded
                )
                for req, (predictions, batch_timings) in zip(batch, batch_results):
                    if not req.future.done():
                        req.timings["queue_time"] = (time.perf_counter() - req.enqueue_time) * 1000
                        req.timings.update(batch_timings)
                        req.future.set_result(predictions)

            except Exception as e:
                for req in batch:
                    if not req.future.done():
                        req.future.set_exception(e)

        except asyncio.CancelledError:
            break
        except asyncio.TimeoutError:
            pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    device = get_inference_device()
    print(f"Initializing model on {device}")

    app.state.loaded = load_model(device=str(device))
    print("Model initialized successfully")

    app.state.batch_queue = asyncio.Queue()
    worker_task = asyncio.create_task(batch_processor(app))

    yield

    worker_task.cancel()
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
    timings = {}

    # 1. Upload read
    start = time.perf_counter()
    image_data = await file.read()
    if len(image_data) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image exceeds 10 MiB")
    timings["upload_read_time"] = (time.perf_counter() - start) * 1000

    # 2. Decode and Preprocess (Moved entirely to a worker thread)
    start = time.perf_counter()
    loop = asyncio.get_running_loop()
    
    def decode_and_prep():
        try:
            with Image.open(io.BytesIO(image_data)) as image:
                rgb = image.convert("RGB")
        except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
            raise ValueError("invalid_image")
        
        return app.state.loaded.preprocess(rgb)

    try:
        tensor = await loop.run_in_executor(None, decode_and_prep)
    except ValueError as e:
        if str(e) == "invalid_image":
            raise HTTPException(status_code=400, detail="Invalid or unsupported image")
        raise HTTPException(status_code=500, detail="Failed to preprocess image")

    timings["decode_and_prep_time"] = (time.perf_counter() - start) * 1000

    # 3. Queue waiting
    req = BatchRequest(image_tensor=tensor, top_k=top_k, timings=timings)
    await app.state.batch_queue.put(req)

    try:
        predictions = await req.future
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise HTTPException(status_code=500, detail="Internal inference error")

    logger.info(
        "PREDICT | %s",
        " | ".join(f"{key}={value:.2f}ms" for key, value in timings.items())
    )

    return {
        "predictions": predictions,
        "timings": timings
    }