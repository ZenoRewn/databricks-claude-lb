# Claude LB Logo 候选与选定图标

Author: Zeno Ren

2026-10-01 使用 Azure OpenAI 的 `gpt-image-2.5-sunburst`，通过 imagegen 技能提供的 CLI 生成 6 张 1024 × 1024 PNG；6 次生成均成功。用户指定名称中的 `suburst` 按可用模型目录拼写 `sunburst` 处理。没有记录凭据或私有 endpoint。

- [01 汇流](01-confluence.png)：多路进入、中心聚合、统一出口。**用户已选择，已接入 Dashboard。**
- [02 网关](02-gateway.png)：入口与通道。
- [03 均衡](03-balanced-nodes.png)：三节点平衡。
- [04 流带](04-flow-ribbon.png)：LB 字母与连续路径。
- [05 桥接](05-bridge.png)：可靠连接。
- [06 智能路由](06-routing-spark.png)：中心调度与四向连接。

[六款预览](contact-sheet.png)包含 34px 明暗主题模拟。[最终透明 PNG](selected-confluence.png)为 01 的派生资产：去除白色底色，保留彩色几何形状，居中缩放至 256 × 256。Dashboard 将该 PNG 内嵌到 HTML，显示为 34 × 34，无额外静态资源路由。

[完整提示词](prompts.jsonl)逐条保存；每款独立提示词，参数为 `quality=high`、`size=1024x1024`。提示词约束为蓝/青配色、紧凑几何形状、白色背景、适合 34px、无文字及第三方品牌标识。原始 PNG 保留，不覆盖生成结果。
