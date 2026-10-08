"""Synthetic local image admission benchmark. Author: Zeno Ren.

Run in the candidate application image with no network. No provider calls.
This is a bounded workload sample, not a production load test or RSS guarantee.
"""
import asyncio
import base64
from io import BytesIO
import json
import os
import resource
import sys
import time

from PIL import Image, ImageDraw
import main


def fixture():
    with Image.new("RGB", (3840, 2160), "white") as image, BytesIO() as output:
        draw = ImageDraw.Draw(image)
        for row in range(0, 2160, 48):
            draw.rectangle((0, row, 3840, row + 23), fill=(224, 231, 238))
            draw.text((48, row + 2), "Synthetic screenshot: image normalization test", fill="black")
        image.save(output, format="PNG")
        url = "data:image/png;base64," + base64.b64encode(output.getvalue()).decode("ascii")
    return {"model": "synthetic-only", "input": [{"role": "user", "content": [
        {"type": "input_image", "image_url": url} for _ in range(14)]}]}


async def run():
    payload = fixture()
    before_bytes = len(json.dumps(payload).encode())
    started = time.monotonic()
    try:
        if hasattr(main, "prepare_images_async"):
            stats = await main.prepare_images_async(payload, route="synthetic-benchmark")
        else:
            main.check_image_admission(payload)
            stats = await main.compress_images_async(payload)
    except main.ImageAdmissionError as error:
        return {"author": "Zeno Ren", "synthetic": True, "provider_calls": 0,
            "source_count": 14, "source_dimensions": [3840, 2160],
            "source_pixels": 14 * 3840 * 2160, "before_body_bytes": before_bytes,
            "result": "rejected", "message": error.message}
    duration = time.monotonic() - started
    dimensions = []
    for part in payload["input"][0]["content"]:
        raw = base64.b64decode(part["image_url"].split(",", 1)[1])
        with Image.open(BytesIO(raw)) as image:
            dimensions.append(image.size)
    result = {"author": "Zeno Ren", "synthetic": True, "provider_calls": 0,
              "source_count": 14, "source_dimensions": [3840, 2160],
              "source_pixels": 14 * 3840 * 2160,
              "output_count": len(dimensions), "output_dimensions": dimensions,
              "output_pixels": sum(w * h for w, h in dimensions),
              "before_body_bytes": before_bytes, "after_body_bytes": len(json.dumps(payload).encode()),
              "duration_seconds": round(duration, 3), "stats": stats,
              "process_peak_rss_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss /
                  (1024 * 1024 if sys.platform == "darwin" else 1024), 2)}
    return result


async def sample():
    count = int(os.environ.get("BENCHMARK_REQUESTS", "1"))
    results = await asyncio.gather(*(run() for _ in range(count)))
    print(json.dumps(results[0] if count == 1 else results, indent=2))


if __name__ == "__main__":
    asyncio.run(sample())
