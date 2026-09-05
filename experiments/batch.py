"""批量评测配置与执行。修改顶部常量即可，不需要命令行参数。"""
from __future__ import annotations

import json
import random
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    from experiments.lme_timeladder import EXP, build_case, run_case
except ModuleNotFoundError:  # 兼容直接执行 batch.py
    from lme_timeladder import EXP, build_case, run_case

DATA_FILE = EXP / "_data" / "lme_s.json"
CASE_LIST = EXP / "cases.txt"
REFRESH_CASES = True
FAKE = False
BUDGET = 32_000
STEP = 8
SLOTS = 3
DRY = 0
TARGET_CASES = 30
MIN_HISTORY_TOKENS = 100_000
RANDOM_SEED = 20260906


def select_cases(data: list[dict]) -> list[int]:
    """只按完整历史长度筛选，再按题型比例做可复现抽样。"""
    groups: dict[str, list[int]] = defaultdict(list)
    for index, item in enumerate(data):
        case = build_case(item)
        if case.total_tokens >= MIN_HISTORY_TOKENS:
            groups[item.get("question_type", "unknown")].append(index)
    if not groups:
        raise RuntimeError("没有找到足够长的历史")

    rng = random.Random(RANDOM_SEED)
    for indexes in groups.values():
        rng.shuffle(indexes)
    total = sum(len(indexes) for indexes in groups.values())
    target = min(TARGET_CASES, total)
    quotas = {kind: 0 for kind in groups}

    # 题型足够多时，每类至少留一道；剩余名额按各类候选题数量比例分配。
    if target >= len(groups):
        for kind in groups:
            quotas[kind] = 1
    remaining = target - sum(quotas.values())
    while remaining:
        available = [
            kind for kind, candidates in groups.items() if quotas[kind] < len(candidates)
        ]
        kind = max(
            available,
            key=lambda name: (
                target * len(groups[name]) / total - quotas[name],
                len(groups[name]) - quotas[name],
                name,
            ),
        )
        quotas[kind] += 1
        remaining -= 1
    indexes = sorted(
        index for kind, candidates in groups.items() for index in candidates[: quotas[kind]]
    )
    CASE_LIST.write_text(" ".join(map(str, indexes)) + "\n", encoding="utf-8")
    distribution = "，".join(
        f"{kind}={quotas[kind]}" for kind in sorted(quotas) if quotas[kind]
    )
    print(f"候选 {total} 道，选出 {len(indexes)} 道：{distribution}")
    print(f"题目清单：{CASE_LIST}")
    return indexes


def case_indexes(data: list[dict]) -> list[int]:
    if REFRESH_CASES or not CASE_LIST.exists():
        return select_cases(data)
    return [int(value) for value in CASE_LIST.read_text(encoding="utf-8").split()]


def result_path(mode: str, index: int) -> Path:
    flavor = "fake" if FAKE else "real"
    return EXP / "_out" / f"lme_tl_{mode}_c{index}_{BUDGET}_{flavor}.jsonl"


def finished(path: Path) -> bool:
    if not path.exists():
        return False
    rows = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return bool(rows and json.loads(rows[-1]).get("finished"))


def run_one(data: list[dict], index: int, mode: str, settings) -> None:
    path = result_path(mode, index)
    if finished(path):
        print(f"跳过题目 {index} [{mode}]：已有完整结果")
        return
    run_case(data[index], BUDGET, mode, settings, STEP, FAKE, path)


def main() -> int:
    data = json.loads(DATA_FILE.read_text(encoding="utf-8"))
    indexes = case_indexes(data)
    if DRY:
        indexes = indexes[:DRY]
    from coding_agent.config import load_settings

    settings = load_settings()
    jobs = [
        (index, mode)
        for index in indexes
        for mode in ("hier", "roll")
        if not finished(result_path(mode, index))
    ]
    print(f"题目 {len(indexes)} 道｜待执行 {len(jobs)} 路｜并发 {SLOTS}｜fake={FAKE}")
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=SLOTS) as pool:
        futures = [pool.submit(run_one, data, index, mode, settings) for index, mode in jobs]
        for future in as_completed(futures):
            future.result()
    print(f"全部完成，用时 {(time.monotonic() - started) / 3600:.1f} 小时")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
