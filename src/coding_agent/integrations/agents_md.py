"""AGENTS.md：跨会话的持久规则，落在 system 前缀的固定段里。

两个来源，远的先、近的后：全局 ``~/.coding_agent/AGENTS.md`` 与项目 ``workspace_root/AGENTS.md``。
两份都注入、不互相覆盖，冲突交给模型按"更具体者优先"判断——这一点写在标签里，不靠代码裁决。

**启动时读一次，进程内不再变**。这段文本在请求第 0 个 token 上（``langchain_anthropic`` 会把
SystemMessage 提到 ``system`` 字段），中途改一次就是整条历史重编码，和 work_state 当初那个
bug 是同一个位置。所以这一版不轮询、不热重载：换规则就重启进程。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..config import Settings
from ..context.tokens import estimate_tokens

PROJECT_RULE_FILENAME = "AGENTS.md"


class AgentsMdError(RuntimeError):
    """规则文件读不了，或体积合计超过上限。

    在装配 Agent 时抛出：宁可开不了工，也不要静默地少一条规则——"看不见的缺失"是最坏的失效
    模式。它每轮都要带上，所以体积超了必须由人先砍，不能由代码替你截断。
    """


@dataclass(frozen=True)
class LoadedRuleFile:
    path: Path
    tokens: int


@dataclass(frozen=True)
class AgentsMd:
    """渲染好的规则段（可直接拼进 system prompt）+ 载入清单。"""

    text: str
    files: tuple[LoadedRuleFile, ...]


def rule_file_paths(settings: Settings) -> tuple[Path, ...]:
    """按注入顺序返回两层来源：全局在前，项目在后（贴得越近，模型读得越晚）。"""

    return (settings.agents_md_path, settings.workspace_root / PROJECT_RULE_FILENAME)


def load_agents_md(settings: Settings) -> AgentsMd:
    loaded: list[LoadedRuleFile] = []
    sections: list[str] = []
    total = 0
    for path in rule_file_paths(settings):
        # 没有这份规则是常态，不是错误；但"有却读不了"必须炸出来。
        if not path.is_file():
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as error:
            raise AgentsMdError(f"读不了规则文件 {path}：{error}") from error
        if not content.strip():
            continue
        tokens = estimate_tokens(content)
        total += tokens
        loaded.append(LoadedRuleFile(path=path, tokens=tokens))
        sections.append(f"## {path}\n\n{content.strip()}")
    if total > settings.agents_md_limit_tokens:
        detail = "、".join(f"{item.path} {item.tokens} token" for item in loaded)
        raise AgentsMdError(
            f"规则文件合计 {total} token，超过上限 {settings.agents_md_limit_tokens}（{detail}）。"
            "规则段在请求最前面、每轮都要带上，请先砍到上限之内再启动。"
        )
    if not sections:
        return AgentsMd(text="", files=())
    body = "\n\n".join(sections)
    text = (
        "\n\n<agents_md>\n"
        "以下是本机与这个工作目录的长期规则，必须遵守。两份冲突时更具体的那份优先"
        "（项目 > 全局）；与框架提示冲突时以框架提示为准。\n\n"
        f"{body}\n</agents_md>"
    )   
    return AgentsMd(text=text, files=tuple(loaded))
