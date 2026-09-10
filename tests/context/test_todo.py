"""计划（todo）的校验层：整表替换、状态机硬约束、回执与渲染。

这一层不碰数据库，纯函数进出——工具那层只负责把 ``TodoError`` 翻成给模型的 ``Error:`` 文本。
"""

from __future__ import annotations

import pytest

from coding_agent.context.todo import (
    MAX_ITEMS,
    MAX_TITLE_CHARS,
    TodoError,
    normalize,
    open_items,
    receipt,
    render,
)


def test_normalize_keeps_the_submission_order_instead_of_sorting():
    """顺序有语义：先做后做不能被打乱（这点与 work_state 按键名排序刻意相反）。"""

    items = normalize(
        [
            {"id": "b", "title": " 第二件事 ", "status": "pending"},
            {"id": "a", "title": "第一件事", "status": "in_progress"},
        ]
    )

    assert [item["id"] for item in items] == ["b", "a"]
    assert items[0]["title"] == "第二件事", "两头空白顺手去掉"


def test_at_most_one_item_can_be_in_progress():
    with pytest.raises(TodoError, match="in_progress"):
        normalize(
            [
                {"id": "1", "title": "甲", "status": "in_progress"},
                {"id": "2", "title": "乙", "status": "in_progress"},
            ]
        )


@pytest.mark.parametrize(
    "item, reason",
    [
        ({"id": "1", "title": "甲", "status": "doing"}, "status 只有四个合法值"),
        ({"id": "", "title": "甲", "status": "pending"}, "id 不能空"),
        ({"id": "1", "title": "", "status": "pending"}, "title 不能空"),
        ({"id": "1", "title": "两\n行", "status": "pending"}, "一行一项"),
        ({"id": "1", "title": "甲"}, "缺 status"),
        ({"id": "1", "title": "甲" * (MAX_TITLE_CHARS + 1), "status": "pending"}, "title 超长"),
    ],
)
def test_bad_items_are_rejected(item, reason):
    with pytest.raises(TodoError):
        normalize([item])


def test_duplicate_ids_are_rejected():
    with pytest.raises(TodoError, match="duplicate"):
        normalize(
            [
                {"id": "1", "title": "甲", "status": "pending"},
                {"id": "1", "title": "乙", "status": "pending"},
            ]
        )


def test_a_plan_is_capped():
    items = [
        {"id": str(index), "title": f"第 {index} 步", "status": "pending"}
        for index in range(MAX_ITEMS + 1)
    ]
    with pytest.raises(TodoError, match=str(MAX_ITEMS)):
        normalize(items)


def test_an_empty_list_clears_the_plan():
    assert normalize([]) == []
    assert render([]) == ""


def test_receipt_reports_the_changes_and_the_whole_table():
    before = [{"id": "1", "title": "读代码", "status": "in_progress"}]
    after = normalize(
        [
            {"id": "1", "title": "读代码", "status": "completed"},
            {"id": "2", "title": "写代码", "status": "in_progress"},
        ]
    )

    text = receipt(before, after)

    assert text.startswith("plan updated | 2 items: 1 in_progress, 1 completed")
    assert "completed 1 读代码" in text and "added 2 写代码" in text
    assert "- [x] 1 读代码" in text, "全表要一起回：它就是模型下一步的工作台"
    assert "- [~] 2 写代码" in text


def test_receipt_says_unchanged_when_nothing_moved():
    same = [{"id": "1", "title": "甲", "status": "pending"}]

    text = receipt(same, normalize(same))

    assert text.startswith("plan unchanged | 1 items: 1 pending")
    assert "changes:" not in text


def test_rewording_and_dropping_are_reported():
    before = [
        {"id": "1", "title": "旧说法", "status": "pending"},
        {"id": "2", "title": "放弃我", "status": "pending"},
    ]

    text = receipt(before, [{"id": "1", "title": "新说法", "status": "pending"}])

    assert "reworded 1 新说法" in text
    assert "dropped 2 放弃我" in text


def test_open_items_counts_pending_and_in_progress_only():
    items = normalize(
        [
            {"id": "1", "title": "甲", "status": "completed"},
            {"id": "2", "title": "乙", "status": "cancelled"},
            {"id": "3", "title": "丙", "status": "pending"},
        ]
    )

    assert [item["id"] for item in open_items(items)] == ["3"]
    assert open_items([]) == []


def test_render_tolerates_hand_written_rows_read_back_from_storage():
    """库里的旧行可能缺字段：渲染要宽容（状态认不出时标 ``?``），别在渲染路径上炸。"""

    assert render([{"id": "1", "title": "甲"}]) == "- [?] 1 甲"
    assert render(["不是字典"]) == ""
