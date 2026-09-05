# 上下文压缩评测

评测代码只保留三个 Python 文件：

| 文件 | 作用 |
| --- | --- |
| `lme_timeladder.py` | 单题回放；分层直接用 `ContextEngine`，滚动把旧压缩区与新前缀一起重摘要 |
| `batch.py` | 选题、批量并发、断点续跑 |
| `report.py` | 汇总完整 real 结果 |

最近一次真实对照实验的详细记录见 [`EVAL_REPORT_2026-09-06.md`](EVAL_REPORT_2026-09-06.md)。

数据集是 `_data/lme_s.json`，结果写入 `_out/`。评测直接读取每道题的完整历史，不做截断。

单题配置直接写在 `lme_timeladder.py` 顶部：

```python
CASE_SLICE = slice(166, 167)
MODES = ("hier", "roll")
BUDGET = 32_000
FAKE = False
```

先把 `FAKE = True` 做零成本检查，确认后改回 `False` 使用真实摘要器：

```bash
uv run python experiments/lme_timeladder.py
```

批量选题只看完整历史 token 长度，并按题型做固定种子的比例抽样。直接运行即可：

```bash
uv run python experiments/batch.py
uv run python experiments/report.py
```

默认读取完整历史、32K 预算、3 路并发。`FAKE = True` 只用于检查执行形状，不进入结论。
