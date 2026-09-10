from __future__ import annotations

import json
from pathlib import Path

import pytest

from coding_agent.config import Settings, load_settings
from coding_agent.integrations.agents_md import AgentsMdError, load_agents_md


def make_settings(tmp_path, *, limit: int = 15_000, global_text=None, project_text=None) -> Settings:
    """全局规则放 tmp 的子目录、项目规则放 tmp 的仓库根，两层都摆在可预期的地方。"""

    global_path = tmp_path / "home" / ".coding_agent" / "AGENTS.md"
    workspace = tmp_path / "repo"
    workspace.mkdir(exist_ok=True)
    if global_text is not None:
        global_path.parent.mkdir(parents=True, exist_ok=True)
        global_path.write_text(global_text, encoding="utf-8")
    if project_text is not None:
        (workspace / "AGENTS.md").write_text(project_text, encoding="utf-8")
    return Settings(
        agents_md_path=global_path,
        agents_md_limit_tokens=limit,
        workspace_root=workspace,
    )


def test_no_rule_files_is_not_an_error(tmp_path):
    """没有规则文件是常态：不报错、也不留一个空的空标签。"""

    rules = load_agents_md(make_settings(tmp_path))

    assert rules.text == ""
    assert rules.files == ()


def test_empty_rule_file_is_skipped(tmp_path):
    rules = load_agents_md(make_settings(tmp_path, project_text="   \n\n"))

    assert rules.text == ""
    assert rules.files == ()


def test_missing_project_file_still_gets_global_rules(tmp_path):
    rules = load_agents_md(make_settings(tmp_path, global_text="全局：不要编数据"))

    assert "不要编数据" in rules.text
    assert [item.path for item in rules.files] == [tmp_path / "home" / ".coding_agent" / "AGENTS.md"]


def test_global_rules_come_before_project_rules(tmp_path):
    """远的先、近的后：项目那份贴得更近，模型读得更晚，也就更有话语权。"""

    rules = load_agents_md(
        make_settings(tmp_path, global_text="全局规则", project_text="项目规则")
    )

    assert rules.text.index("全局规则") < rules.text.index("项目规则")
    assert "项目 > 全局" in rules.text, "冲突怎么裁决要写在标签里"
    assert rules.text.startswith("\n\n<agents_md>") and rules.text.endswith("</agents_md>")
    assert [
        (item.path.name, item.path.parent.name) for item in rules.files
    ] == [("AGENTS.md", ".coding_agent"), ("AGENTS.md", "repo")]


def test_rule_block_is_byte_identical_between_loads(tmp_path):
    """它坐在请求最前面，内容一抖整条历史就得重编码，所以必须逐字节稳定。"""

    settings = make_settings(tmp_path, global_text="全局", project_text="项目")

    assert load_agents_md(settings).text == load_agents_md(settings).text


def test_oversized_rules_fail_loudly(tmp_path):
    """超上限直接报错，不截断：宁可开不了工，也不静默少一条规则。"""

    settings = make_settings(tmp_path, limit=100, global_text="a" * 397, project_text="b" * 400)

    with pytest.raises(AgentsMdError) as error:
        load_agents_md(settings)

    message = str(error.value)
    assert "100" in message and "超过上限" in message
    assert ".coding_agent" in message and "repo" in message, "报错要说清是哪两份文件"


def test_rules_exactly_at_the_limit_are_allowed(tmp_path):
    """边界：正好等于上限不算超（`a`*397 估成 100 token）。"""

    rules = load_agents_md(make_settings(tmp_path, limit=100, project_text="a" * 397))

    assert rules.files[0].tokens == 100


def test_unreadable_rule_file_fails_loudly(tmp_path):
    """"没有"能放过，"有却读不了"必须炸出来——写了的规则没生效是最坏的失效模式。"""

    settings = make_settings(tmp_path)
    settings.workspace_root.mkdir(exist_ok=True)
    (settings.workspace_root / "AGENTS.md").write_bytes(b"\xff\xfe\x00\x01")

    with pytest.raises(AgentsMdError):
        load_agents_md(settings)


def test_defaults_and_overrides_are_wired_into_settings(monkeypatch, tmp_path):
    monkeypatch.delenv("AGENTS_MD_PATH", raising=False)
    monkeypatch.delenv("AGENTS_MD_LIMIT_TOKENS", raising=False)

    defaults = load_settings(tmp_path / "missing.json")
    assert defaults.agents_md_path == Path.home() / ".coding_agent" / "AGENTS.md"
    assert defaults.agents_md_limit_tokens == 15_000

    config = tmp_path / "config.json"
    config.write_text(
        json.dumps({"agents_md_path": "~/rules/AGENTS.md", "agents_md_limit_tokens": 2048}),
        encoding="utf-8",
    )
    configured = load_settings(config)
    assert configured.agents_md_path == Path("~/rules/AGENTS.md").expanduser().resolve()
    assert configured.agents_md_limit_tokens == 2048

    config.write_text(json.dumps({"agents_md_limit_tokens": 0}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_settings(config)

    monkeypatch.setenv("AGENTS_MD_LIMIT_TOKENS", "4096")
    assert load_settings(config).agents_md_limit_tokens == 4096, "环境变量优先于 config.json"


def test_rules_are_injected_into_the_shared_prompt_suffix(tmp_path):
    """装配层真把规则拼进 system 后缀（父子共用），不是读了就丢。"""

    from dataclasses import replace

    from coding_agent.config import AgentSettings
    from coding_agent.integrations.langchain_agent import shared_agent_tools

    settings = replace(
        make_settings(tmp_path, project_text="这个仓库的规矩"),
        agent=AgentSettings(bash_enabled=False, search_enabled=False),
    )

    tools, prompt_suffix = shared_agent_tools(
        settings=settings,
        context_engine=None,  # only 建 work_state 工具，用不到 engine 实例
        bash_executor=None,
        search_client=None,
    )

    assert "这个仓库的规矩" in prompt_suffix
    assert "work_state" in {t.name for t in tools}


def test_the_passed_in_rules_win_over_the_disk(tmp_path):
    """父进程传进来那份就是提示词里那份。

    装配时再读一遍盘的话，`/status` 显示的和模型正在遵守的就可能对不上——规则文件改了要重启
    才生效，面板不能偷偷看新版本。
    """

    from dataclasses import replace

    from coding_agent.config import AgentSettings
    from coding_agent.integrations.agents_md import AgentsMd
    from coding_agent.integrations.langchain_agent import shared_agent_tools

    settings = replace(
        make_settings(tmp_path, project_text="磁盘上的旧规矩"),
        agent=AgentSettings(bash_enabled=False, search_enabled=False),
    )
    injected = AgentsMd(text="\n<agents_md>\n启动时读进来的规矩\n</agents_md>\n", files=())

    _tools, prompt_suffix = shared_agent_tools(
        settings=settings,
        context_engine=None,
        bash_executor=None,
        search_client=None,
        agents_md=injected,
    )

    assert "启动时读进来的规矩" in prompt_suffix
    assert "磁盘上的旧规矩" not in prompt_suffix
