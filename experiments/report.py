"""读取批量评测结果，输出一张容易检查的配对表。

只比较同一题、同一预算、同一历史长度下的完整 real 结果。
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

EXP = Path(__file__).resolve().parent
VERSION = 5


def _identity(path: Path) -> tuple[str, str, int, str, str] | None:
    parts = path.stem.split("_")
    if len(parts) == 7 and parts[:2] == ["lme", "tl"]:
        _, _, mode, case, budget, limit, flavor = parts
    elif len(parts) == 6 and parts[:2] == ["lme", "tl"]:
        # 只接受 batch.py 生成的 c<index> 文件；旧单题文件可能来自截断评测，不能混入。
        _, _, mode, case, budget, flavor = parts
        if not case.startswith("c"):
            return None
        limit = "full"
    else:
        return None
    if mode not in {"hier", "roll"} or flavor != "real":
        return None
    return mode, case, int(budget), limit, flavor


def load_pairs(directory: Path) -> tuple[dict, int]:
    runs: dict[tuple[str, int, str], dict[str, dict]] = defaultdict(dict)
    skipped = 0
    for path in sorted(directory.glob("lme_tl_*.jsonl")):
        identity = _identity(path)
        if identity is None:
            continue
        mode, case, budget, limit, _flavor = identity
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        if not rows or rows[0].get("version") != VERSION or not rows[-1].get("finished"):
            skipped += 1
            continue
        runs[(case, budget, limit)][mode] = rows[-1]
    pairs = {key: value for key, value in runs.items() if set(value) == {"hier", "roll"}}
    return pairs, skipped


def show(pairs: dict) -> None:
    print("题目       历史长度   分层输入   滚动输入   差值     分层调用 滚动调用   最高层 H/R   最大重压 H/R")
    savings: list[float] = []
    for (case, budget, limit), group in sorted(pairs.items()):
        hier, roll = group["hier"], group["roll"]
        delta = roll["input_tokens"] - hier["input_tokens"]
        percent = delta / roll["input_tokens"] * 100 if roll["input_tokens"] else 0
        savings.append(percent)
        print(
            f"{case:<10} {limit:>8} {hier['input_tokens']:>10,} {roll['input_tokens']:>10,}"
            f" {percent:>+6.1f}% {hier['calls']:>9} {roll['calls']:>9}"
            f" L{hier['top_level']}/L{roll['top_level']:>3}"
            f" {hier['press_max']:>8}/{roll['press_max']:<8}"
        )
    if savings:
        print(f"\n配对 {len(savings)} 组｜滚动比分层多喂：中位 {statistics.median(savings):+.1f}%")
    else:
        print("\n没有找到完整配对结果。")


def main() -> int:
    pairs, skipped = load_pairs(EXP / "_out")
    print(f"结果目录：{EXP / '_out'}｜完整配对：{len(pairs)}｜跳过：{skipped}")
    show(pairs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
