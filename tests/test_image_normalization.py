"""Bounded image normalization retains history and owns decode workers.

Author: Zeno Ren
"""
import asyncio
import base64
import copy
import io
import threading
import unittest
from unittest.mock import MagicMock, patch

from fastapi import HTTPException
from PIL import Image

import main


def image_block(width=1600, height=1200, *, color=(255, 255, 255)):
    with Image.new("RGB", (width, height), color) as image, io.BytesIO() as buf:
        image.save(buf, format="PNG")
        encoded = base64.b64encode(buf.getvalue()).decode()
    return {"type": "input_image", "image_url": f"data:image/png;base64,{encoded}"}


def dimensions(block):
    with Image.open(io.BytesIO(base64.b64decode(block["image_url"].split(",", 1)[1]))) as image:
        return image.size


class ImageNormalizationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        main._img_compress_sem = None

    async def test_over_total_budget_is_normalized_without_removing_history(self):
        original = image_block()
        payload = {"input": [{"role": "user", "content": [
            {"type": "input_text", "text": "earlier"}, copy.deepcopy(original),
            {"type": "input_text", "text": "later"}, copy.deepcopy(original),
        ]}]}
        with patch.object(main, "_IMG_MAX_TOTAL_PIXELS", 3_000_000):
            with self.assertRaises(main.ImageAdmissionError):
                main.check_image_admission(payload)
            stats = await main.prepare_images_async(payload, route="/v1/responses")
            main.check_image_admission(payload)
        content = payload["input"][0]["content"]
        self.assertEqual([item["type"] for item in content],
                         ["input_text", "input_image", "input_text", "input_image"])
        self.assertEqual([content[0]["text"], content[2]["text"]], ["earlier", "later"])
        self.assertEqual(stats["count"], 2)
        self.assertTrue(all(max(dimensions(content[i])) <= 1280 for i in (1, 3)))

    async def test_small_encoded_but_large_dimensions_are_resized(self):
        block = image_block(1800, 1600)
        self.assertLess(len(block["image_url"]), 200 * 1024)
        payload = {"input": [{"role": "user", "content": [block]}]}
        stats = await main.prepare_images_async(payload, route="/v1/responses")
        self.assertEqual(stats["count"], 1)
        self.assertLessEqual(max(dimensions(block)), 1280)

    async def test_source_pixel_budget_rejects_before_decode(self):
        payload = {"input": [image_block(), image_block()]}
        with patch.object(main, "_IMG_MAX_SOURCE_PIXELS", 3_000_000, create=True), \
             patch.object(main, "compress_images_async") as compress:
            with self.assertRaises(HTTPException) as cm:
                await main.prepare_images_async(payload, route="/v1/responses")
        self.assertEqual(cm.exception.status_code, 413)
        self.assertIn("Source image pixels", cm.exception.detail["error"]["message"])
        compress.assert_not_called()

    async def test_single_pixel_budget_rejects_before_decode(self):
        payload = {"input": [image_block()]}
        with patch.object(main, "_IMG_MAX_SINGLE_PIXELS", 1_000_000, create=True), \
             patch.object(main, "compress_images_async") as compress:
            with self.assertRaises(HTTPException) as cm:
                await main.prepare_images_async(payload, route="/v1/responses")
        self.assertEqual(cm.exception.status_code, 413)
        self.assertIn("Single image pixels", cm.exception.detail["error"]["message"])
        compress.assert_not_called()

    async def test_failed_normalization_cannot_bypass_final_pixel_budget(self):
        payload = {"input": [image_block(), image_block()]}
        with patch.object(main, "_IMG_MAX_TOTAL_PIXELS", 3_000_000), \
             patch.object(main, "_compress_image_bytes", return_value=None):
            with self.assertRaises(HTTPException) as cm:
                await main.prepare_images_async(payload, route="/v1/responses")
        self.assertEqual(cm.exception.status_code, 413)
        self.assertIn("Total image pixels", cm.exception.detail["error"]["message"])

    async def test_no_images_does_not_acquire_decode_worker(self):
        payload = {"input": [{"role": "user", "content": "unchanged"}]}
        before = copy.deepcopy(payload)
        with patch.object(main, "compress_images_async") as compress:
            stats = await main.prepare_images_async(payload, route="/v1/responses")
        self.assertEqual(stats["count"], 0)
        self.assertEqual(payload, before)
        compress.assert_not_called()

    async def test_cancelled_request_retains_worker_slot_until_thread_finishes(self):
        started = [threading.Event(), threading.Event()]
        release = threading.Event()
        running = 0
        peak = 0
        lock = threading.Lock()

        def compress(which):
            nonlocal running, peak
            with lock:
                running += 1
                peak = max(peak, running)
            started[which].set()
            try:
                if not release.wait(3):
                    raise RuntimeError("test worker release timed out")
                return {"count": 0, "before": 0, "after": 0}
            finally:
                with lock:
                    running -= 1

        with patch.object(main, "_IMG_COMPRESS_CONCURRENCY", 1), \
             patch.object(main, "compress_images_in_payload", side_effect=compress):
            first = asyncio.create_task(main.compress_images_async(0))
            self.assertTrue(await asyncio.to_thread(started[0].wait, 1))
            first.cancel()
            await asyncio.sleep(0)
            first.cancel()  # repeated cancellation must not abandon the worker
            second = asyncio.create_task(main.compress_images_async(1))
            try:
                await asyncio.sleep(0.05)
                self.assertFalse(started[1].is_set(), "decode semaphore released while worker still active")
                self.assertFalse(first.done(), "cancelled request abandoned its live image worker")
            finally:
                release.set()
                await asyncio.gather(first, second, return_exceptions=True)
        self.assertEqual(peak, 1)
        self.assertTrue(first.cancelled())
        self.assertTrue(started[1].is_set())


class ImageDecodeSafetyTests(unittest.TestCase):
    def test_image_memory_is_closed_when_resize_fails(self):
        image = MagicMock(size=(1600, 1200))
        image.__enter__.return_value = image
        image.thumbnail.side_effect = OSError("test decode failure")
        with patch.object(main.Image, "open", return_value=image):
            self.assertIsNone(main._compress_image_bytes(b"fixture"))
        image.close.assert_called_once()

    def test_pillow_decompression_bomb_is_not_treated_as_unknown_dimensions(self):
        with patch.object(main.Image, "open", side_effect=Image.DecompressionBombError("oversized")):
            with self.assertRaises(main.ImageAdmissionError):
                main._peek_image_size(b"fake")

    def test_resize_is_kept_even_when_jpeg_is_larger_than_source(self):
        block = image_block(1500, 1)
        original = base64.b64decode(block["image_url"].split(",", 1)[1])
        compressed = main._compress_image_bytes(original)
        self.assertIsNotNone(compressed)
        self.assertGreater(len(compressed), len(original), "fixture proves safety resize can grow encoded bytes")
        with Image.open(io.BytesIO(compressed)) as image:
            self.assertLessEqual(max(image.size), 1280)

    def test_compressor_rejects_oversized_image_even_if_called_directly(self):
        raw = base64.b64decode(image_block()["image_url"].split(",", 1)[1])
        with patch.object(main, "_IMG_MAX_SINGLE_PIXELS", 1_000_000, create=True):
            with self.assertRaises(main.ImageAdmissionError):
                main._compress_image_bytes(raw)
