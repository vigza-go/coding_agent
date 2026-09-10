# 运行说明

## 准备

1. 安装依赖：`uv sync --extra dev`
2. 复制 `config.example.json` 为本地 `config.json`，或使用 `.env.example` 中的环境变量。
3. 确保 MySQL 数据库存在，`DATABASE_URL` 使用 SQLAlchemy 格式 `mysql+pymysql://...`。
   只有一个数据库连接需要配置：应用不启用 LangGraph checkpointer，原因见 DESIGN.md「撤销」
   一节与下方排障条目。
4. API key 建议只放在 `LLM_API_KEY` 环境变量。`config.json` 已被 Git 忽略。
5. （可选）写规则文件：全局 `~/.coding_agent/AGENTS.md`、项目 `<workspace_root>/AGENTS.md`。
   启动时各读一次、拼在提示词最前面；两份合计超过 `agents_md_limit_tokens`（默认 15k token）
   直接报错，不截断。见 DESIGN.md「跨会话规则」。

首次启动会通过 SQLAlchemy `create_all` 创建六张业务表。已有数据库启动时会执行幂等兼容迁移；旧版本的
`memory_blocks.is_frontier` 字段及其索引会被删除，当前块集合改由 greedy cover 动态计算。

历史上启用过 PyMySQLSaver 的库会留下 `checkpoints` / `checkpoint_blobs` / `checkpoint_writes` /
`checkpoint_migrations` 四张表（实测单库 9.8GB，占整库 99.6%），它们不再被读写，但存量不会自愈。
收尾三步，顺序不能换：

```text
# 1) 体检 + 导出：找出「只在 checkpoint 里、业务库没有」的消息，落成 JSON
PYTHONPATH=src:. uv run python experiments/retire_checkpoint_tables.py --export

# 2) 退出所有 coding-agent 进程 —— 改动前启动的进程里还挂着 checkpointer，
#    表一删，它们的下一轮就会报 Table ... doesn't exist

# 3) DROP
PYTHONPATH=src:. uv run python experiments/retire_checkpoint_tables.py --drop --yes
```

`--drop` 自带守卫：还有活的 `coding-agent` 进程、或有未导出的 checkpoint-only 内容时直接拒绝。
之所以要先导出：业务库 `messages` 是 canonical source，正式线程的快照实测全是冗余（差集只剩
`memory-*` / `work-state-*` 这类每轮现搭的投影脚手架，脚本会忽略），但早期试跑线程可能只有
checkpoint 里留了记录。`innodb_file_per_table=ON` 时 DROP 才真的把空间还给操作系统。

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

## 子代理（subagent）

主代理可以把一份"要读很多文件、跑很多命令、但不必占用主对话上下文"的活儿派给一个子代理
去做，拿回一份报告。子代理是**另起的独立进程**，跑完整的主代理循环——工具能力（bash、文件、
搜索、work state）与主代理一致，只是不能再生子代理（防套娃）。在 `config.json` 的 `agent` 中
开关：

```json
{
  "subagent_enabled": true
}
```

子代理的工具集和额度**不单独配置**，直接沿用父的 `model_call_limit` / `tool_call_limit` /
`tool_retry_max` 等（同一套参数，不复制一份改数）。步数上限也不单开一份：跟主代理一样交给
LangGraph 默认值，实际的闸只有额度。

也可用 `AGENT_SUBAGENT_ENABLED=false` 临时关闭。几条要记住的性质：

- **不污染主对话**：子代理的中间过程既不投影父历史、也不写进 `messages`/`conversations`
  表，用户的时间轴看不到它，也不占用可见会话名额。
- **改动可撤销**：子代理对文件的修改记在**派它那一轮**（`user_seq`）名下，主代理 `/undo`
  那轮时会一并退回。（子代理用独立数据库连接、独立会话锁记账，与父进程互不干扰。）
- **有界**：子代理受与父相同的模型/工具调用上限；报告优先把结论放最前，避免被截断。
  子进程崩溃只回一条带退出码的错误，主回合不中断。
- **本轮收工前一定等到它**：`run_subagent` 立刻返回任务号，父模型可以边派边干别的活；但
  父模型说完话**不等于这一轮结束**——本轮会等它派出去的所有子代理退出、把报告作为一条
  消息喂回去让父续跑，直到队列清空才真正收工、才把输入框还给用户。所以报告绝不会"迟到到
  下一次按键"，用户也就不会看见代理在自己不说话时偷偷干活。
- **因此 Ctrl-C 停的是父子一起**：等子代理的这段时间输入框根本没在等输入，那一下中断打的
  就是这一轮，正在飞的子代理会被一并终止，不会留下半死的进程往作废的回合里写文件。
- **暂不并发**：子进程本身可以并行派（多次调用工具即可），但父子同时改同一个文件没有跨进程
  锁，遵循后写获胜。合并语义留待后续版本。

TUI 命令：

- `/history [N]`：查看最近 N 条有效 canonical 消息；`/list` 是兼容别名。
- `/threads`：查看最近会话。
- `/thread ID`：切换 thread。目标正被另一个会话跑 turn 时**拒绝切换**，
  以免两边的历史搅在一起。
- `/status`：查看当前 head、记忆块层级时间线、压缩区/工作区 token 占用和 work state。
- `/usage [N]`：查看当前会话最近 N 条模型回复的 API 用量和加权缓存命中率，默认 20。
- `/undo [N]`：预览并确认后撤销 `user_seq >= N`；省略 N 时撤销当前 head。
- `/clear`：预览并确认后清空当前线程的上下文——历史消息、压缩块、work state 全部停用，下一轮从零开始。
  它**不还原任何文件**，也不停用文件账：那些改动留在自己那一轮上，之后照样能 `/undo <那一轮>` 退回。
  换句话说 `/clear` = `/undo 1` 减去「还原文件」；想连文件一起退，用 `/undo`。
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

工具历史平时不滚动裁剪。工作区超过压缩阈值时，按 `context.recent_tool_interactions`
保留最近 N 条工具结果的完整内容（默认 10），更早结果只替换 `content` 为占位文本，
不删除消息或调用；同一次动作里还把冷思维链削进预算，剪完若仍占超过触发线一半才分块摘要，
剪出余量就不摘要。剩余消息直接保存在现有缓存中，不新增窗口状态或表。
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

## 排障：`/undo` 或每轮开头静等数秒

现象是点确认撤销后静止 10 秒上下，普通对话则是首 token 特别慢（容易被误判成模型慢）。两者走的是
同一个调用：`langgraph-checkpoint-mysql` 的 `PyMySQLSaver.get_tuple`。它的 `SELECT_SQL` 用
`json_table` 展开 `checkpoint.channel_versions` 再去 join `checkpoint_blobs`，MySQL 优化器不会把
相关条件下推，于是先用主键前缀 `(thread_id, checkpoint_ns_hash)` 捞出该线程**全部** blob 再逐行
比对版本——线程历史越长读得越多，成本随历史线性增长；`innodb_buffer_pool_size` 还是默认 128MB 时
基本等于冷读磁盘。注意 `PregelLoop.__enter__` 取"线程最新 checkpoint"也走同一条查询，所以症状
不只出现在撤销上。

确认（只读，不写业务数据）。`langgraph-checkpoint-mysql` 已随本条目一起从项目依赖退役，
所以这两个探针要临时把它带回来：

```text
CKPT="langgraph-checkpoint-mysql[pymysql]>=3.0"
PYTHONPATH=src:. uv run --with "$CKPT" python experiments/undo_latency_profile.py <thread> <back>  # 分段计时，看 P6 占比
PYTHONPATH=src:. uv run --with "$CKPT" python experiments/checkpoint_read_probe.py <thread>         # 真实 SQL 的 EXPLAIN ANALYZE
```

现在的应用不配置 checkpointer，因此不再产生也不读取这些表；存量按「准备」一节里的
`experiments/retire_checkpoint_tables.py` 三步清掉。若将来因为审批（`interrupt()`）需要重新启用，
得先把这个依赖加回 `pyproject.toml`（`setup()` 会重建这几张表），
并且必须换成「每线程只留最新一份」的 `ShallowMySQLSaver`，或按
`experiments/checkpoint_rewrite_patch_probe.py` 里实测过 680x 的写法覆写 `_select_sql` 改成逐通道
主键点查——照原样挂回默认 saver 就是把这 10 秒装回来；同时把 `innodb_buffer_pool_size` 提到可用
内存的 50-70%。

## 排障：提示「thread 正在被另一个会话使用」

三种入口都会报这句，都是常态不是故障（一次都没写，历史没被污染）：启动时 `启动失败：thread 'x' 正被
另一个会话使用`、切换时 `不切换。…`、以及极少见的 `本轮未开始：…`（只在你的会话空闲期间被别的进程接管
时才出现）。占用是**会话级**的：只要另一个进程还停在这条 thread 里 —— 哪怕它一个字都没问 —— 你就进不
去。让对方 `/exit`（或 Ctrl-C）即可；急着干活就用 `/thread <别的 id>` 换一条轨道。故意没做强制接管：
宁可显式退出，也不要两个会话同时写一条轨道。

关掉窗口不等于退出会话。Terminal.app 关标签页时只做一件事：**撤销**（revoke）那块 pty（顺带让 shell
自己死于挂断），**不给作业发挂断**——拿内核进程表实测定案：那一秒会话的 `e_tdev` 变 -1、`e_tpgid` 变
"无终端"，进程照活，而它对 SIGHUP 是默认处置，说明信号压根没送到。所以以前会留下一个"没了终端、还在占
座、又没有窗口可以 `/exit`"的进程。现在 TUI 在**整条会话期间**每秒自检控制终端：`/dev/tty` 开不开得
（管"主端已关、从端还开着"那个中间态）+ `os.tcgetpgrp(0)` 还认不认我们（管终端彻底销毁），约 2 秒判定
"终端没了"，先给自己一发 SIGINT 走本轮的中断收尾（杀子进程、结清时间轴、放座），收尾不配合 8 秒后硬退。
zsh 的 `nohup`/`hup`、进程组、Ctrl-Z 都与此无关，别再去改 shell 配置。

**跑 turn 途中关窗口也一样**（不是残留情形）：那一刻的停手 = 用户显式意图，走的就是平时 Ctrl-C 那条
收尾路径，时间轴上会留下一个被打断的轮次。要确认现场是不是"终端没了的会话"：

```text
pgrep -f coding-agent | xargs ps -o pid,ppid,stat,tty,etime -p
```

`TTY` 显示 `??` 的就是丢了终端的会话，`kill -9` 它即可 —— 不需要清锁，锁挂在它那条连接上。

不需要手工清锁：锁是 MySQL 的**连接级**咨询锁，Ctrl-C、`kill -9`、断电、网络断开，服务端都会随会话
把它收回，下一个会话立刻拿得到 —— 没有 TTL、没有 holder 列、没有要清的残留表。

如果报的是 `在本轮期间被别的会话推进或回滚`：本轮被安全终止，它之前已提交的轮次仍在库里（可
`/undo` 退回），重新发起即可。这类终止**刻意
不自动重试** —— 空窗期里状态不可信，静默续写才是真正会搅乱历史的行为。

同一棵工作树允许并行开多个会话（各自一条 thread），树级和路径级都不互斥；并发写同一文件靠
`edit_file` 自然失败暴露冲突，而不是靠锁预防。需要真隔离用 Git worktree。

## 验证

```text
uv run ruff check src tests main.py
uv run pytest -q
```

文件并发回归测试默认使用 SQLite。设置 `TEST_MYSQL_ADMIN_URL` 后运行
`uv run pytest -q tests/integrations/test_file_concurrency.py`，还会验证 MySQL 并发插入和
REPEATABLE READ 下的 blob 复用。该连接需有创建/删除数据库权限；测试仅使用自动生成的
`coding_agent_test_<随机标识>` 临时库，结束后删除，不在 URI 指定的业务库中写测试数据。

thread 占用锁的 MySQL 行为同理 gated：
`TEST_MYSQL_ADMIN_URL=... uv run pytest -q tests/integrations/test_thread_lock_mysql.py`
覆盖跨连接互斥、`KILL` 掉持锁连接后本轮终止、空窗期被别的会话推进时终止（该连接还需
PROCESS/KILL 权限）。SQLite 没有跨会话咨询锁，所以单测只覆盖 seq 分歧检测与调用点。

文件快照并发修复不需要数据库迁移。更新代码后需重启 TUI 才生效，旧会话可继续使用。
同一应用内，针对同一路径的受控写工具串行执行，不同文件仍可并行；Bash 和外部进程不受
此锁保护。修复不会自动重放之前失败的编辑，也不会撤销之前成功的文件变更。

V1 只保证 `write_file`、`edit_file` 和 `delete` 的文件撤销。代理通过 `bash` 执行 shell
命令造成的文件变化不被 mutation log 捕获。
