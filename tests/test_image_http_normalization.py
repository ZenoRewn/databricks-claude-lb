"""Image normalization at the three public HTTP boundaries. Author: Zeno Ren.

All providers are synthetic. These tests never send inference to a network.
"""
import base64
from contextlib import ExitStack
from copy import deepcopy
from io import BytesIO
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

import httpx
from PIL import Image

import main


def image_block(protocol, size=(100, 100), color="white"):
    with Image.new("RGB", size, color) as image, BytesIO() as output:
        image.save(output, format="PNG")
        encoded = base64.b64encode(output.getvalue()).decode("ascii")
    if protocol == "messages":
        return {"type": "image", "source": {
            "type": "base64", "media_type": "image/png", "data": encoded}}
    url = "data:image/png;base64," + encoded
    if protocol == "responses":
        return {"type": "input_image", "image_url": url, "detail": "high"}
    return {"type": "image_url", "image_url": {"url": url, "detail": "high"}}


def dimensions(block):
    if block["type"] == "image":
        encoded = block["source"]["data"]
    else:
        url = block["image_url"]
        encoded = (url if isinstance(url, str) else url["url"]).split(",", 1)[1]
    with Image.open(BytesIO(base64.b64decode(encoded))) as image:
        return image.size


class ImageNormalizationHttpTests(unittest.IsolatedAsyncioTestCase):
    async def send(self, protocol, content, **limits):
        route = AsyncMock(return_value=main.JSONResponse({"synthetic": True}))
        proxy = SimpleNamespace(verify_api_key=lambda key: key == "synthetic-image-key",
                                proxy_request=route)
        body = {"model": "claude-opus-5" if protocol == "messages" else "gpt-test",
                "input" if protocol == "responses" else "messages": [
                    {"role": "user", "content": deepcopy(content)}]}
        with ExitStack() as stack:
            for name, value in {"proxy": proxy, "copilot_proxy": proxy, "azure_proxy": None,
                                "_route_openai_chat": route, "_route_openai_responses": route,
                                "_img_compress_sem": None, **limits}.items():
                stack.enter_context(patch.object(main, name, value, create=True))
            async with httpx.AsyncClient(transport=httpx.ASGITransport(main.app),
                                         base_url="http://synthetic") as client:
                path = "chat/completions" if protocol == "chat" else protocol
                response = await client.post("/v1/" + path, json=body,
                    headers={"Authorization": "Bearer synthetic-image-key"})
        return response, route

    async def test_source_over_output_budget_is_resized_without_omitting_images(self):
        for protocol in ("messages", "responses", "chat"):
            with self.subTest(protocol=protocol):
                text_type = "input_text" if protocol == "responses" else "text"
                # Exercise the old >200KB path that rejects before compression.
                content = [{"type": text_type, "text": "before" + "." * 210_000}]
                content += [image_block(protocol, color=color) for color in ("red", "green", "blue")]
                content += [{"type": text_type, "text": "after"}]
                response, route = await self.send(protocol, content, _IMG_MAX_DIM=64,
                    _IMG_MAX_TOTAL_PIXELS=25_000, _IMG_MAX_SOURCE_PIXELS=100_000,
                    _IMG_MAX_SINGLE_PIXELS=20_000)
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(route.await_count, 1)
                sent = route.call_args.args[0]["input" if protocol == "responses" else "messages"][0]["content"]
                self.assertEqual([part["type"] for part in sent], [part["type"] for part in content])
                self.assertEqual(sent[0], content[0])
                self.assertEqual(sent[-1], content[-1])
                self.assertTrue(all(max(dimensions(part)) <= 64 for part in sent[1:-1]))
                self.assertIn("images.compressed", response.headers.get("x-lb-transformed-parameters", ""))
                self.assertNotIn("images.trimmed", response.headers.get("x-lb-dropped-parameters", ""))
                if protocol == "responses":
                    self.assertTrue(all(part["detail"] == "high" for part in sent[1:-1]))

    async def test_small_encoded_body_still_enforces_single_image_and_count_limits(self):
        for protocol in ("messages", "responses", "chat"):
            for reason, content, limits in (
                ("single", [image_block(protocol)], {"_IMG_MAX_SINGLE_PIXELS": 5_000}),
                ("count", [image_block(protocol, (2, 2))] * 3, {"_IMG_MAX_COUNT": 2}),
            ):
                with self.subTest(protocol=protocol, reason=reason):
                    response, route = await self.send(protocol, content, **limits)
                    self.assertEqual(response.status_code, 413, response.text)
                    self.assertEqual(route.await_count, 0)
                    self.assertTrue(response.headers["content-type"].startswith("application/json"))

    async def test_failed_normalization_cannot_forward_over_budget_images(self):
        for protocol in ("messages", "responses", "chat"):
            with self.subTest(protocol=protocol):
                with patch.object(main, "_compress_image_bytes", return_value=None):
                    response, route = await self.send(protocol, [image_block(protocol)] * 3,
                        _IMG_MAX_DIM=64, _IMG_MAX_TOTAL_PIXELS=25_000,
                        _IMG_MAX_SOURCE_PIXELS=100_000, _IMG_MAX_SINGLE_PIXELS=20_000)
                self.assertEqual(response.status_code, 413, response.text)
                self.assertEqual(route.await_count, 0)
