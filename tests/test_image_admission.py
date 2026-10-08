"""图片准入门（`check_image_admission`）的行为覆盖。

字节数上限管不住位图：一张 4000×3000 图 base64 才 ~2MB，解码和重采样却需要几十 MB。
此文件保留默认总像素/张数准入契约；有界缩图及取消所有权见 test_image_normalization.py。

设计要点（测试即契约）：
  - 只 peek header 拿 `(w, h)`，不 `.load()`，所以门本身几乎不占内存
  - 张数与总像素两道预算，任一超限抛 `ImageAdmissionError` → 客户端拿 JSON 413
  - 源图单图/累计预算必须在压缩前检查，默认总像素预算在压缩后仍须满足
  - 小于 200KB 的图片仍检查尺寸，纯色大图也须受预算保护
"""
import base64
import io
import inspect
import os
import pathlib
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import main

try:
    from PIL import Image
    _PIL = True
except Exception:  # pragma: no cover
    _PIL = False


def png_b64(w: int, h: int, color=(255, 255, 255)) -> str:
    """造一张真实 PNG 的 base64。纯色 PNG 压缩率极高，所以「字节小、像素大」。"""
    buf = io.BytesIO()
    Image.new("RGB", (w, h), color).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def anthropic_block(b64: str) -> dict:
    return {"type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": b64}}


def openai_data_url(b64: str) -> dict:
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}


@unittest.skipUnless(_PIL, "Pillow 不可用时准入门自动跳过，无从验证")
class AdmissionBudgetTests(unittest.TestCase):

    def test_count_budget_rejects_anthropic_blocks(self):
        tiny = png_b64(8, 8)
        payload = {"messages": [{"role": "user",
                                 "content": [anthropic_block(tiny) for _ in range(4)]}]}
        with patch.object(main, "_IMG_MAX_COUNT", 3):
            with self.assertRaises(main.ImageAdmissionError) as cm:
                main.check_image_admission(payload)
        self.assertIn("Too many images", cm.exception.message)

    def test_count_budget_rejects_openai_data_urls(self):
        """两种载荷形态都要被看见 —— 只认一种等于对另一种完全没有防护。"""
        tiny = png_b64(8, 8)
        payload = {"messages": [{"role": "user",
                                 "content": [openai_data_url(tiny) for _ in range(4)]}]}
        with patch.object(main, "_IMG_MAX_COUNT", 3):
            with self.assertRaises(main.ImageAdmissionError):
                main.check_image_admission(payload)

    def test_pixel_budget_rejects_small_bytes_but_huge_bitmap(self):
        """这是这道门真正防的东西：字节数上限拦不住的位图炸弹。

        纯色 PNG 的 4000×3000（1200 万像素）编码后只有几 KB，靠
        `MAX_RAW_REQUEST_SIZE` 永远拦不住；解码成位图却要几十 MB。
        """
        big = png_b64(4000, 3000)
        self.assertLess(len(big), 100_000, "纯色 PNG 必须字节很小，否则这个用例失去意义")
        payload = {"messages": [{"role": "user", "content": [anthropic_block(big)]}]}
        with patch.object(main, "_IMG_MAX_TOTAL_PIXELS", 1_000_000):
            with self.assertRaises(main.ImageAdmissionError) as cm:
                main.check_image_admission(payload)
        self.assertIn("Total image pixels", cm.exception.message)

    def test_pixel_budget_accumulates_across_images(self):
        """单张都不超、加起来超 —— 预算是累加的。"""
        one = png_b64(1000, 1000)  # 100 万像素
        payload = {"messages": [{"role": "user",
                                 "content": [anthropic_block(one) for _ in range(3)]}]}
        with patch.object(main, "_IMG_MAX_TOTAL_PIXELS", 2_500_000), \
             patch.object(main, "_IMG_MAX_COUNT", 50):
            with self.assertRaises(main.ImageAdmissionError):
                main.check_image_admission(payload)

    def test_within_budget_passes(self):
        one = png_b64(100, 100)
        payload = {"messages": [{"role": "user",
                                 "content": [anthropic_block(one), openai_data_url(one)]}]}
        with patch.object(main, "_IMG_MAX_COUNT", 50), \
             patch.object(main, "_IMG_MAX_TOTAL_PIXELS", 100_000_000):
            main.check_image_admission(payload)  # 不抛即通过

    def test_disabled_switch_bypasses_everything(self):
        big = png_b64(4000, 3000)
        payload = {"messages": [{"role": "user", "content": [anthropic_block(big)]}]}
        with patch.object(main, "_IMG_ADMISSION_ENABLED", False), \
             patch.object(main, "_IMG_MAX_TOTAL_PIXELS", 1):
            main.check_image_admission(payload)

    def test_corrupt_payloads_do_not_crash_but_still_count(self):
        """坏 base64 / 非图片数据不能让准入本身抛非预期异常。

        张数仍然计入（`count += 1` 在 peek 之前），所以「用一堆坏数据绕过张数预算」
        这条路是堵住的；像素数无从统计，但 PIL 同样解不开它，压缩阶段也不会解码。
        """
        payload = {"messages": [{"role": "user", "content": [
            anthropic_block("!!!not-base64!!!"),
            anthropic_block(base64.b64encode(b"not an image at all").decode()),
            openai_data_url("@@@@"),
        ]}]}
        with patch.object(main, "_IMG_MAX_COUNT", 50):
            main.check_image_admission(payload)  # 不抛
        # 三个节点都要计入张数，包括 base64 解不开的那个（`!!!not-base64!!!`）。
        # 从前 `b64decode` 在 `account` 之前且在同一个 try 里，解码抛异常就整体跳过
        # 计数 —— 于是畸形 base64 可以无限绕过张数预算。
        with patch.object(main, "_IMG_MAX_COUNT", 2):
            with self.assertRaises(main.ImageAdmissionError):
                main.check_image_admission(payload)

    def test_undecodable_images_cannot_bypass_the_count_budget(self):
        """整批都是解不开的 base64 时，张数预算仍须生效。"""
        payload = {"messages": [{"role": "user", "content": [
            anthropic_block("!!!not-base64!!!") for _ in range(10)]}]}
        with patch.object(main, "_IMG_MAX_COUNT", 3):
            with self.assertRaises(main.ImageAdmissionError) as cm:
                main.check_image_admission(payload)
        self.assertIn("Too many images", cm.exception.message)

    def test_traverses_nested_structures(self):
        """真实 payload 里图片埋在 messages[].content[] 甚至更深，遍历不能漏。"""
        tiny = png_b64(8, 8)
        payload = {"input": [{"role": "user", "content": [
            {"type": "text", "text": "x"},
            {"nested": {"deeper": [anthropic_block(tiny) for _ in range(5)]}},
        ]}]}
        with patch.object(main, "_IMG_MAX_COUNT", 3):
            with self.assertRaises(main.ImageAdmissionError):
                main.check_image_admission(payload)

    def test_peek_reads_dimensions_without_decoding(self):
        raw = base64.b64decode(png_b64(4000, 3000))
        self.assertEqual(main._peek_image_size(raw), (4000, 3000))
        self.assertIsNone(main._peek_image_size(b"definitely not an image"))
        self.assertIsNone(main._peek_image_size(b""))


@unittest.skipUnless(_PIL, "Pillow 不可用时压缩/裁剪自动跳过")
class TrimExcessImagesTests(unittest.TestCase):
    """`trim_excess_images` 会**静默丢掉用户最早的图**并替换成占位文本。

    它是准入被拒后的优雅降级路径（不是死代码）：先 trim 再重试准入，trim 不到才 413。
    这类「用户可见地丢数据」的行为此前零覆盖 —— 占位文本的 `type` 用错就会让整个
    请求被上游按 schema 拒绝，而那种失败看起来完全不像「我们丢了图」。
    """

    def test_keeps_the_newest_and_replaces_the_oldest(self):
        tiny = png_b64(8, 8)
        content = [anthropic_block(tiny) for _ in range(5)]
        for i, blk in enumerate(content):
            blk["source"]["data"] = png_b64(8, 8, (i, i, i))  # 可区分
        payload = {"messages": [{"role": "user", "content": content}]}
        removed = main.trim_excess_images(payload, max_count=2)
        self.assertEqual(removed, 3)
        kept = payload["messages"][0]["content"]
        self.assertEqual([c.get("type") for c in kept],
                         ["text", "text", "text", "image", "image"],
                         "保留的必须是最后 max_count 张（最新的）")
        self.assertIn("image omitted", kept[0]["text"])

    def test_placeholder_type_matches_each_protocol(self):
        """占位块的 `type` 必须与原协议一致，否则上游按 schema 拒整个请求。"""
        tiny = png_b64(8, 8)
        cases = [
            (anthropic_block(tiny), "text"),                        # Anthropic
            (openai_data_url(tiny), "text"),                        # Chat Completions
            ({"type": "input_image", "image_url": f"data:image/png;base64,{tiny}"},
             "input_text"),                                          # Responses API
        ]
        for block, expected_type in cases:
            with self.subTest(orig=block.get("type")):
                payload = {"messages": [{"role": "user", "content": [dict(block), dict(block)]}]}
                removed = main.trim_excess_images(payload, max_count=1)
                self.assertEqual(removed, 1)
                self.assertEqual(payload["messages"][0]["content"][0]["type"], expected_type)

    def test_no_op_when_within_budget(self):
        tiny = png_b64(8, 8)
        payload = {"messages": [{"role": "user", "content": [anthropic_block(tiny)]}]}
        before = repr(payload)
        self.assertEqual(main.trim_excess_images(payload, max_count=5), 0)
        self.assertEqual(repr(payload), before, "未超预算时不得改动 payload")

    def test_index_shifts_do_not_corrupt_siblings(self):
        """同一个 list 里删多个元素时，索引位移不能把不该动的元素替换掉。"""
        tiny = png_b64(8, 8)
        content = [{"type": "text", "text": "keep-0"},
                   anthropic_block(tiny),
                   {"type": "text", "text": "keep-1"},
                   anthropic_block(tiny),
                   {"type": "text", "text": "keep-2"},
                   anthropic_block(tiny)]
        payload = {"messages": [{"role": "user", "content": content}]}
        self.assertEqual(main.trim_excess_images(payload, max_count=1), 2)
        texts = [c.get("text") for c in payload["messages"][0]["content"]]
        self.assertIn("keep-0", texts)
        self.assertIn("keep-1", texts)
        self.assertIn("keep-2", texts)
        self.assertEqual(payload["messages"][0]["content"][5]["type"], "image",
                         "最后一张图必须留下")


@unittest.skipUnless(_PIL, "Pillow 不可用时压缩自动跳过")
class CompressionTests(unittest.TestCase):
    """压缩链路：按字节或尺寸触发，缩到 ≤1280px；安全缩图允许编码变大。"""

    @staticmethod
    def _photo_b64(w, h):
        """噪声图，保证 base64 超过 200KB 门槛（纯色 PNG 太小，压不动）。"""
        import random
        random.seed(7)
        img = Image.new("RGB", (w, h))
        img.putdata([(random.randint(0, 255), random.randint(0, 255),
                      random.randint(0, 255)) for _ in range(w * h)])
        buf = io.BytesIO(); img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    def test_downscales_and_reencodes_anthropic_block(self):
        big = self._photo_b64(1600, 1200)
        self.assertGreaterEqual(len(big), main._IMG_COMPRESS_THRESHOLD,
                                "夹具必须真的超过压缩门槛")
        payload = {"messages": [{"role": "user", "content": [anthropic_block(big)]}]}
        stats = main.compress_images_in_payload(payload)
        self.assertEqual(stats["count"], 1)
        self.assertLess(stats["after"], stats["before"])
        src = payload["messages"][0]["content"][0]["source"]
        self.assertEqual(src["media_type"], "image/jpeg", "必须改写 media_type")
        out = Image.open(io.BytesIO(base64.b64decode(src["data"])))
        self.assertLessEqual(max(out.size), 1280, "长边必须缩到 ≤1280")
        self.assertEqual(out.format, "JPEG")

    def test_small_images_are_left_untouched(self):
        small = png_b64(64, 64)
        payload = {"messages": [{"role": "user", "content": [anthropic_block(small)]}]}
        stats = main.compress_images_in_payload(payload)
        self.assertEqual(stats["count"], 0)
        self.assertEqual(payload["messages"][0]["content"][0]["source"]["media_type"],
                         "image/png", "门槛以下不得改写 media_type")

    def test_corrupt_base64_is_left_as_is(self):
        payload = {"messages": [{"role": "user",
                                 "content": [anthropic_block("!" * 300_000)]}]}
        stats = main.compress_images_in_payload(payload)
        self.assertEqual(stats["count"], 0)
        self.assertEqual(payload["messages"][0]["content"][0]["source"]["data"],
                         "!" * 300_000, "解不开就原样保留，不能吞掉数据")

    def test_openai_data_url_string_is_compressed_in_place(self):
        big = self._photo_b64(1600, 1200)
        payload = {"messages": [{"role": "user", "content": [openai_data_url(big)]}]}
        stats = main.compress_images_in_payload(payload)
        self.assertEqual(stats["count"], 1)
        url = payload["messages"][0]["content"][0]["image_url"]["url"]
        self.assertTrue(url.startswith("data:image/jpeg;base64,"))
        out = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))
        self.assertLessEqual(max(out.size), 1280)

    def test_no_images_is_cheap_and_lossless(self):
        payload = {"messages": [{"role": "user", "content": [{"type": "text", "text": "x" * 500_000}]}]}
        before = repr(payload)
        stats = main.compress_images_in_payload(payload)
        self.assertEqual(stats["count"], 0)
        self.assertEqual(repr(payload), before)


class AdmissionWiringTests(unittest.TestCase):
    """三个入口共用同一源预算/压缩/输出复核，不能因请求体较小绕过。"""

    def setUp(self):
        self.src = pathlib.Path(main.__file__).read_text(encoding="utf-8")

    def test_all_three_entry_points_use_shared_preparation(self):
        for route in ("/v1/messages", "/v1/responses", "/v1/chat/completions"):
            with self.subTest(route=route):
                self.assertEqual(self.src.count(f'prepare_images_async(body, route="{route}")'), 1)

    def test_preparation_gates_before_decode_and_rechecks_output(self):
        source = inspect.getsource(main.prepare_images_async)
        initial = source.index("check_image_admission(payload, source=True)")
        trim = source.index("trim_excess_images(payload)")
        trimmed = source.index("check_image_admission(payload, source=True)", initial + 1)
        compression = source.index("await compress_images_async(payload)")
        output = source.index("check_image_admission(payload)")
        self.assertLess(initial, trim)
        self.assertLess(trim, trimmed)
        self.assertLess(trimmed, compression)
        self.assertLess(compression, output)


if __name__ == "__main__":
    unittest.main()
