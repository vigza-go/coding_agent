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
uv run coding-agent --thread demo
```

也可以运行 `uv run python main.py --thread demo`。

TUI 命令：

- `/list`：查看当前有效的 canonical 消息。
- `/undo`：撤销当前有效的最后一个用户轮次。
- `/undo N`：撤销 `user_seq >= N`。
- `/thread ID`：切换 thread。
- `/exit`：退出。

## 验证

```text
uv run ruff check src tests main.py
uv run pytest -q
```

V1 只保证 `write_file`、`edit_file` 和 `delete` 的文件撤销。代理通过 `execute` 执行 shell
命令造成的文件变化不被 mutation log 捕获。
