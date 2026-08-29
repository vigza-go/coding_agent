# 运行说明

## 准备

1. 安装依赖：`uv sync --extra dev`
2. 复制 `config.example.json` 为本地 `config.json`，或使用 `.env.example` 中的环境变量。
3. 确保两个 MySQL URI 指向已存在的数据库：
   - `DATABASE_URL` 使用 SQLAlchemy 格式 `mysql+pymysql://...`
   - `CHECKPOINT_DATABASE_URL` 使用 PyMySQLSaver 格式 `mysql://...`
4. API key 建议只放在 `LLM_API_KEY` 环境变量。`config.json` 已被 Git 忽略。

首次启动会通过 SQLAlchemy `create_all` 创建六张业务表，并通过 PyMySQLSaver `setup` 创建
LangGraph checkpoint 表。已有数据库启动时会执行幂等兼容迁移；旧版本的
`memory_blocks.is_frontier` 字段及其索引会被删除，当前块集合改由 greedy cover 动态计算。

## 启动

```text
uv run --env-file .env coding-agent --thread demo
```

不使用 `.env` 时也可以运行 `uv run coding-agent --thread demo`。增加 `--debug` 会在错误时显示
完整 traceback。

TUI 命令：

- `/history [N]`：查看最近 N 条有效 canonical 消息；`/list` 是兼容别名。
- `/threads`：查看最近会话。
- `/thread ID`：切换 thread。
- `/status`：查看当前 head、记忆块层级时间线、压缩区/工作区 token 占用和 work state。
- `/undo [N]`：预览并确认后撤销 `user_seq >= N`；省略 N 时撤销当前 head。
- `/help`：查看输入帮助。
- `/exit`：退出。

输入支持本次进程内的方向键历史；Enter 发送，Alt+Enter 插入换行。模型和工具执行期间会显示
进度。调用失败或被 Ctrl-C 中断时，TUI 会报告失败轮次并询问是否立即 undo，防止不完整的工具
协议历史影响下一轮。交互界面使用“你”提示符标识用户输入，提交后用分隔线划开运行区域，
并用带 `Agent` 标题的蓝色面板展示模型最终回答；`/history` 也会按消息角色使用不同标题和
边框颜色。

## 验证

```text
uv run ruff check src tests main.py
uv run pytest -q
```

V1 只保证 `write_file`、`edit_file` 和 `delete` 的文件撤销。代理通过 `execute` 执行 shell
命令造成的文件变化不被 mutation log 捕获。
