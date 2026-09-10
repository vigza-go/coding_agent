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

## 没落盘的那几次

**锚点保留率对照**（合并到 L3 后路径类锚点归零、"等预算下不优于只留最近"）的原始数字**不在这
个目录**，它当时只写进了 `readme.md`，后来从正文拿掉、结论改成一句话。数字本身在 git 历史里：
`git show e41b8f2:readme.md`（那句"路径 100%→12%→5%→0%、标识符 100%→42%→15%→6%，16 格里只赢
2 格输 10 格"）；口径与成因分析在 `DESIGN.md`「上下文布局与压缩」——摘要提示词那条"删除可从代码
重新获得的细节"是锚点清零的直接原因，附录式修复只测过单价、**尚未实现**。想把这组数字变成可复
跑的东西，得重做一次对照（`lme_timeladder.py` 换 `MODES` 即可），而不是从这里引用。
