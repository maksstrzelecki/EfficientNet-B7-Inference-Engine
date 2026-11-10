import argparse
import asyncio
import mimetypes
from pathlib import Path
import time

import httpx

async def main(args):
    image = Path(args.image_path)
    image_data = image.read_bytes()
    content_type = mimetypes.guess_type(image.name)[0] or "application/octet-stream"
    semaphore = asyncio.Semaphore(args.concurrency)
    limits = httpx.Limits(max_connections=args.concurrency)

    async with httpx.AsyncClient(limits=limits, timeout=60.0) as client:
        async def send_request(index):
            async with semaphore:
                start = time.perf_counter()
                response = await client.post(
                    args.url,
                    files={"file": (image.name, image_data, content_type)},
                )
                latency = (time.perf_counter() - start) * 1000
                response.raise_for_status()
                timings = response.json()["timings"]
                values = [f"{key}={value:.2f} ms" for key, value in timings.items()]
                print(f"Request {index}: " + " | ".join(
                    [*values, f"client_latency={latency:.2f} ms"]
                ))

        start = time.perf_counter()
        await asyncio.gather(*(send_request(i + 1) for i in range(args.total_requests)))
        elapsed = time.perf_counter() - start

    print(f"{args.total_requests} requests in {elapsed:.2f} s "
          f"({args.total_requests / elapsed:.2f} req/s)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark the prediction API.")
    parser.add_argument("--url", default="http://127.0.0.1:8000/predict")
    parser.add_argument("--image-path", required=True)
    parser.add_argument("--total-requests", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=100)
    args = parser.parse_args()
    if args.total_requests < 1 or args.concurrency < 1:
        parser.error("--total-requests and --concurrency must be positive")
    asyncio.run(main(args))
