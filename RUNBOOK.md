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

Bash 工具默认开启。可以在 `config.json` 的 `agent` 中调整或显式关闭：

```json
{
  "bash_enabled": true,
  "bash_executable": "/bin/bash",
  "bash_timeout_seconds": 120,
  "bash_max_output_bytes": 100000
}
```

也可以通过 `AGENT_BASH_ENABLED=false` 临时关闭。`/status` 会显示 Bash 状态。Bash 命令从
workspace root 执行；`src/app.py` 这样的相对路径基于 workspace，而 `/tmp/a` 这样的绝对路径
直接访问宿主机。它不是沙箱，也可能读取当前进程环境，因此不能用于不可信输入或多租户部署。

TUI 命令：

- `/history [N]`：查看最近 N 条有效 canonical 消息；`/list` 是兼容别名。
- `/threads`：查看最近会话。
- `/thread ID`：切换 thread。
- `/status`：查看当前 head、记忆块层级时间线、压缩区/工作区 token 占用和 work state。
- `/usage [N]`：查看当前会话最近 N 条模型回复的 API 用量和加权缓存命中率，默认 20。
- `/undo [N]`：预览并确认后撤销 `user_seq >= N`；省略 N 时撤销当前 head。
- `/help`：查看输入帮助。
- `/exit`：退出。

输入支持本次进程内的方向键历史；Enter 发送，Alt+Enter 插入换行。模型和工具执行期间会显示
进度。调用失败或被 Ctrl-C 中断时，TUI 会报告失败轮次并询问是否立即 undo，防止不完整的工具
协议历史影响下一轮。中断时仍在运行的 Bash 会被终止，缺失的工具结果会自动补成 error
ToolMessage；即使选择不 undo，下一轮也不会因孤立 tool call 被模型 API 拒绝。交互界面使用
“你”提示符标识用户输入，提交后用分隔线划开运行区域，
并用带 `Agent` 标题的蓝色面板展示模型最终回答；`/history` 也会按消息角色使用不同标题和
边框颜色。

轮次退出后，属于该轮的迟到进度回调会被丢弃，不会在新的输入提示符后继续打印工具结果。
撤销后 `active_head_seq` 从剩余有效消息重新计算，因此连续撤销或历史存在空洞时，无参数
`/undo` 仍会指向真正的最新有效轮次。

## 工具窗口

工具历史平时不滚动裁剪。工作区超过压缩阈值时，先按 `context.recent_tool_interactions`
保留最近 N 条工具结果的完整内容（默认 10），更早结果只替换 `content` 为占位文本，
不删除消息或调用，再进行分块摘要。剩余消息直接保存在现有缓存中，不新增窗口状态或表。
重启、撤销等缓存重建会恢复未压缩尾部的原始内容；下次达到阈值时再应用窗口。
已生成且仍有效的记忆块继续使用。
原始记录仍可通过 `/history` 查看。单条工具结果的 5k 截断及 artifact 落盘规则不变。

## 完成通知与防休眠

默认开启。可在 `config.json` 中添加以下配置，修改后重启 TUI：

```json
{
  "tui": {
    "notifications_enabled": true,
    "prevent_sleep": true,
    "usage_recent_messages": 20,
    "system_command_timeout_seconds": 3
  }
}
```

`AGENT_NOTIFICATIONS_ENABLED`、`AGENT_PREVENT_SLEEP`、`AGENT_USAGE_RECENT_MESSAGES` 环境变量
分别覆盖前三项。

- 通知表示一轮调用结束并返回控制权，不表示代理已验证整个项目完成；执行失败和中断也会通知。
  macOS 通过 `osascript` 发送通知，仅含会话名、状态和耗时，不包含提问、模型回答或工具输出。
  如果发送失败或不是 macOS，则尝试终端提示音；终端静音时可能听不到。
  通知横幅受 macOS「通知」权限与专注模式影响；命令成功不代表用户一定看到横幅。
- 防休眠在模型/工具执行前启动 `caffeinate -i -w <应用进程 PID>`，正常结束、失败或 Ctrl-C
  后释放，并且在等待撤销确认时已经释放；应用进程退出时也会释放。不阻止显示器熄屏，
  不更改系统电源设置，也不保证合盖、手动睡眠或断电后任务继续执行。
  当前只支持 macOS；其他系统或缺少工具时给出一次提示，不阻止任务运行。

## 缓存命中率口径

`/usage` 直接读取已持久化助手消息中的 LangChain `usage_metadata`，无需增加表或调用模型；
旧会话若记录了这些字段，也可以查询。按全局消息 id 取当前会话最近 N 条助手消息，可能
包含多个用户轮次或不同模型，不是最近 N 条用户输入；缺少用量的回复仍占样本位置并明确显示覆盖数。

缓存命中率 = 可统计样本的 `cache_read` 总和 / 同批样本的 `input_tokens` 总和。
标准化后的 `input_tokens` 已包含缓存读取和创建量，不能再次相加；不使用工作区估算 token，
也不对每次请求的命中百分比直接取平均。缺失/无效缓存字段不当作 0% 计入，输入为 0 时显示
无法计算。未命中输入包含缓存创建，不代表这一部分价格都相同。

已撤销历史仍计入（撤销不会退还用量）。当前仅覆盖已落库的主模型响应，不包含摘要请求、
没有用量记录的失败请求和 SDK 内部重试，因此不是完整计费账单。

## 验证

```text
uv run ruff check src tests main.py
uv run pytest -q
```

文件并发回归测试默认使用 SQLite。设置 `TEST_MYSQL_ADMIN_URL` 后运行
`uv run pytest -q tests/integrations/test_file_concurrency.py`，还会验证 MySQL 并发插入和
REPEATABLE READ 下的 blob 复用。该连接需有创建/删除数据库权限；测试仅使用自动生成的
`coding_agent_test_<随机标识>` 临时库，结束后删除，不在 URI 指定的业务库中写测试数据。

文件快照并发修复不需要数据库迁移。更新代码后需重启 TUI 才生效，旧会话可继续使用。
同一应用内，针对同一路径的受控写工具串行执行，不同文件仍可并行；Bash 和外部进程不受
此锁保护。修复不会自动重放之前失败的编辑，也不会撤销之前成功的文件变更。

V1 只保证 `write_file`、`edit_file` 和 `delete` 的文件撤销。代理通过 `bash` 执行 shell
命令造成的文件变化不被 mutation log 捕获。
