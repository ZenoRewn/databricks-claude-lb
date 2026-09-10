"""PoolTimeout 诊断的**文档↔代码**契约守卫。

为什么需要这一层：`_describe_pool_timeout` 的行为早就被
`tests/test_stream_response_ownership.py::test_pool_timeout_neutral_despite_business_count_or_probe`
钉死了 —— classification 恒为中立的 `pool_acquire_timeout`、`httpx_pool_observed_full`
是纯观测、废弃计数器恒 0、旧文案不再出现。代码一直是对的。

**漂移的是文档。** 2026-09-10 审计发现：那次「撤回因果分类」的迁移只写进了
`docs/STREAM_OWNERSHIP.md`，而 `CLAUDE.md`、`docs/TROUBLESHOOTING.md` 和
`/admin/copilot/reset-pool` 的 docstring 全都还在描述迁移前的世界：

  - TROUBLESHOOTING §9 的**症状字符串**是旧文案 —— 运维拿真实报错去搜，搜不到这一节
  - 三处把 `classification` 说成 `local_pool_saturated` / `upstream_connect_stalled`
    二选一 —— 按这个写 KQL 过滤恒零行
  - TROUBLESHOOTING 把 `copilot_pool_timeout_upstream_stall_total` 标成
    「上游握手挂的次数（**用户报错这一条**）」—— 那个计数器**结构性恒 0**，
    照它写的告警永远不会触发，而运维会读成「没有上游 stall」

最后一条最危险：它和 `orphaned…{recovered}` 恒 0 是同一种失败 —— 指标看着可测，
实际不可达。行为测试抓不到这类缺陷（代码是对的），所以守卫必须落在文档上。
"""
import os
import pathlib
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import main

ROOT = pathlib.Path(main.__file__).parent
SOURCE = pathlib.Path(main.__file__).read_text(encoding="utf-8")

# 撤回的因果分类字面量。撤回理由见 _describe_pool_timeout 的 docstring：
# PoolTimeout 只说明「等池分配超时」，不能证明 TCP/TLS 建连失败。
WITHDRAWN_LITERALS = ("upstream_connect_stalled", "local_pool_saturated")
NEUTRAL_CLASSIFICATION = "pool_acquire_timeout"
DEPRECATED_COUNTER = "pool_timeout_upstream_stall_total"

# 允许文档提到旧字面量 / 废弃计数器，但同一文件里必须带上作废说明，
# 否则读者会当成现行指引。
WITHDRAWAL_MARKERS = ("撤回", "已不再出现", "废弃", "Deprecated", "deprecated")


def _docs():
    """CLAUDE.md + docs/*.md。"""
    paths = [ROOT / "CLAUDE.md"] + sorted((ROOT / "docs").glob("*.md"))
    return [(p.relative_to(ROOT).as_posix(), p.read_text(encoding="utf-8"))
            for p in paths if p.exists()]


class CodeIsTheSourceOfTruthTests(unittest.TestCase):
    """先确认代码这一侧的事实，文档守卫才有基准。"""

    def test_withdrawn_literals_are_gone_from_the_code(self):
        for literal in WITHDRAWN_LITERALS:
            with self.subTest(literal=literal):
                self.assertNotIn(literal, SOURCE,
                                 "撤回的分类字面量不得回到代码里")

    def test_classification_is_a_single_neutral_value(self):
        assigns = re.findall(r'fields\["classification"\]\s*=\s*"([^"]+)"', SOURCE)
        self.assertEqual(assigns, [NEUTRAL_CLASSIFICATION],
                         "classification 必须只有一个中立取值")

    def test_no_control_flow_branches_on_classification(self):
        """撤回的核心是「不许拿它做判断」。单端点快速失败的判据只能是端点数。"""
        self.assertNotIn("classification ==", SOURCE)
        self.assertNotIn('classification"] ==', SOURCE)

    def test_the_deprecated_counter_has_no_increment_site(self):
        """结构性恒 0 —— 这正是文档不能拿它当诊断的原因。"""
        self.assertEqual(SOURCE.count(f"self.{DEPRECATED_COUNTER} = 0"), 1,
                         "应当只有初始化一处")
        self.assertEqual(SOURCE.count(f"self.{DEPRECATED_COUNTER} += 1"), 0,
                         "一旦有了自增点，就必须同步更新文档与本测试")

    def test_the_deprecated_counter_is_still_exposed_but_marked(self):
        """保留是为了不破坏既有抓取；HELP 必须自带 Deprecated 说明。"""
        idx = SOURCE.find(f'emit("copilot_{DEPRECATED_COUNTER}"')
        self.assertNotEqual(idx, -1, "删掉会破坏既有 Prometheus 抓取")
        self.assertIn("Deprecated", SOURCE[idx:idx + 300],
                      "恒 0 的计数器必须在 HELP 里说明自己已废弃")


class DocsMustNotResurrectTheWithdrawnFramingTests(unittest.TestCase):

    def test_docs_naming_the_deprecated_counter_carry_a_caveat(self):
        """最关键的一条：不许再有文档把主诊断指向一个恒 0 的计数器。"""
        offenders = [name for name, text in _docs()
                     if DEPRECATED_COUNTER in text
                     and not any(m in text for m in WITHDRAWAL_MARKERS)]
        self.assertEqual(offenders, [],
                         f"这些文档提到 {DEPRECATED_COUNTER} 却没说它已废弃/恒 0")

    def test_docs_naming_withdrawn_literals_carry_a_caveat(self):
        offenders = []
        for name, text in _docs():
            if any(lit in text for lit in WITHDRAWN_LITERALS) and not any(
                    m in text for m in WITHDRAWAL_MARKERS):
                offenders.append(name)
        self.assertEqual(offenders, [],
                         "提到撤回的分类字面量时必须说明它已不再产出")

    def test_troubleshooting_quotes_the_message_the_code_actually_emits(self):
        """症状字符串是排查手册的**入口**。它一旦过期，运维拿真实报错搜不到。"""
        text = (ROOT / "docs" / "TROUBLESHOOTING.md").read_text(encoding="utf-8")
        # 与 _describe_pool_timeout 的 sse_msg 模板逐字对齐的稳定片段
        for fragment in ("Copilot connection pool acquisition timed out for",
                         "this does not establish a TCP/TLS failure"):
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, SOURCE, "代码里应有这段模板")
                self.assertIn(fragment, text, "排查手册要引用现行文案")

    def test_the_observation_field_is_documented_as_the_replacement(self):
        """撤回二分类后，能用来判断的是 httpx_pool_observed_full。"""
        self.assertIn('fields["httpx_pool_observed_full"]', SOURCE)
        for name in ("CLAUDE.md", "docs/TROUBLESHOOTING.md"):
            with self.subTest(doc=name):
                text = (ROOT / name).read_text(encoding="utf-8")
                self.assertIn("httpx_pool_observed_full", text,
                              "撤回了旧分类就必须告诉运维改看什么")


if __name__ == "__main__":
    unittest.main()
