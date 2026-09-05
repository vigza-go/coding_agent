# 分层上下文 Coding Agent 设计

## 代码边界

```text
coding_agent/
├── context/       # token 预算、原子分块、摘要、合并与覆盖重建
├── persistence/   # SQLAlchemy 模型、连接、repository 与消息编码
├── workspace/     # 工具结果 artifact 和文件 mutation undo
├── services/      # 跨领域 rollback 编排
├── integrations/  # LangChain/LangGraph 中间件与 agent 组装
├── application.py # TUI 可调用的应用用例
└── ui/            # 终端界面
```

依赖方向是领域/持久化组件 → service → integration/application → UI。UI 不直接操作
repository；框架中间件也不进入 context、workspace 等核心模块。

## 不变量

- `messages.id` 是数据库全局、递增且不复用的消息时间轴。
- `user_seq` 是 thread 内用户调用编号，从 1 开始，只增不减；撤销不会复用编号。
- `conversations.active_head_seq` 是当前有效历史的最大用户轮次，空历史为 0。
- `memory_blocks.active` 只表示块仍属于有效历史。
- 同一条 thread 在同一时刻只有一个会话待在里面（含空着不说话的会话）；该占用挂在数据库连接上，连接消失即失效。
- 释放座位不允许依赖"有人给我们发信号"：终端消失时收不到任何信号，会话必须自检（见「thread 占用」末段）。
- 持锁连接只由拿锁的那个线程碰（进入、离开、每轮开头）；轮次中间并行工具线程走的 `verify` 只读游标。
- 当前压缩块集合不单独持久化：缓存首次加载或撤销后，从左端开始选择最长 active block，
  由 greedy cover 动态推导。父块生成后，父子都保持 active；父块因覆盖范围更长而自然
  取代子块进入当前上下文。
- 所有阈值来自配置，业务代码中不散落上下文 magic number。

## 业务表

所有主键都是数据库自增整数。

- `conversations`：`thread_id`（唯一）、`active_head_seq`、`next_user_seq`、时间戳。
- `messages`：`thread_id`、`user_seq`、`type`、`content_json`、
  `langchain_message_id`（thread 内幂等键）、`active`、时间戳。
- `work_state_snapshots`：`thread_id`、`user_seq`、`state_json`、`active`、时间戳。
- `memory_blocks`：`thread_id`、`text`、`begin_message_id`、`end_message_id`、`level`、
  `token_count`、`active`、时间戳。
- `file_blobs`：`sha256`（唯一）、`storage_uri`、`content`、时间戳；content 和 URI 至少有一个。
- `file_mutations`：`thread_id`、`user_seq`、`tool_call_id`、`operation_type`、`path`、
  `before_blob_id`、`before_hash`、`after_hash`、`active`、`status`、时间戳。

`next_user_seq`、`langchain_message_id` 和 `after_hash` 是在原始草案上补充的字段：
分别解决 seq 不复用、步骤重试幂等以及文件审计问题。

## 上下文布局与压缩

默认 1M token 上下文：压缩区上限 25%，工作区触发线 50%，其余 25% 留给框架提示、
动态提示和输出缓冲。工作区超过触发线时，先将最近 10 条之外的工具消息内容替换为占位文本，
再按裁剪后消息的 token 权重保留约 20% 尾部原文，前缀尽量均衡拆成 4 个 L0 块。
越过触发线后先做一轮冷思维链剪裁（见下），**若剪裁已使工作区回到线下，本次不再摘要**；
随后的工具结果裁剪则不同——一旦越过触发线，本次不因它把数值裁到线下而跳过摘要。没有可分割
前缀时只更新消息缓存，不拆开完整工具交互，也不创建空摘要。消息数量、顺序和 ID 不变，记忆块边界算法不变。

每次摘要的上限是输入 token 的 50%。两条路径的"输入"口径不同：L0 量的是**被压缩消息的原始
JSON**（与触发线同一个估算器），而实际送进模型的是 `sanitize_transcript()` 消毒后的纯文本——
它丢掉 reasoning 块和 JSON 包壳，量级远小于这个分母。实测同一段五条消息：带 CoT 时算出的上限
是其**真实输入的约 19 倍**，去掉 CoT 仍约 2 倍。也就是说这条 50% 只在"原始 JSON"口径下成立，
别把它读成"摘要不超过模型所见文本的一半"；合并路径量左右两块入库时存的 `token_count`，与其
实际输入一致。输出为空或超限时最多重试三次，
提示词逐步变得更激进；三次仍超限则**接受最短的那版超限输出**——上限是软目标，略微超限的摘要
依然在缩小记忆，为它拖垮整轮用户请求代价更大。只有全部尝试都返回空、或含协议标签残骸时才抛
`SummaryError`。一次 L0 压缩的各个 chunk 相互独立，默认最多并发发送 4 个摘要请求；结果仍按
chunk 原始顺序收集。抛 `SummaryError` 时不写任何 memory block，但缓存**不**保持原样：越线那
一次的冷思维链剪裁已写进缓存视图且失败不回滚，这是刻意的——剪 CoT 不花钱、不丢记忆，回滚只会
让下一轮在同一位置重剪一次；库里仍是原文，缓存失效后边界归零。所有摘要生成并验证后才在一个
事务里写入记忆块，再更新缓存中的剩余消息。

**思维链（CoT）在三处分别处理，互不重叠**：①摘要输入侧 `compaction.py` 的
`sanitize_transcript()` 无条件丢弃 `thinking` / `reasoning` / `redacted_thinking` 块，摘要器
永远看不到草稿；②摘要模型自身以 `summary_llm.thinking="disabled"` 关闭扩展思考（否则它会把
摘要写进推理里、正文只剩几十 token）；③主模型投影侧 `engine.py` 的
`_retain_reasoning_within_budget()` 只在越过工作区触发线那一次，按“最老先丢”把投影里的 CoT
削到 `context.reasoning_retain_ratio × working_trigger` 预算内，被削处留一行指针文本。
③只改投影不改存储（数据库仍存原文，可审计），边界只前进不后退（否则每轮重写中段前缀会打爆
前缀缓存），且永不削最新一段——当前一轮不能丢自己刚写的草稿。三处都不回填历史。
预算用与触发线同一个估算器逐条量差额，因此“无 CoT 的消息”零误伤。

压缩区超限时，合并当前缓存 cover 中最早的一对相邻同级块并循环到低于上限；首次加载和
撤销重建时才从 active blocks 重新运行 greedy cover——合并本身不重跑，父块的左右坐标直接
继承两个孩子，天然在 cover 里顶掉它们。不变量错误有**两个**抛出点：①超限却找不到同级相邻对，
宁可让整轮失败也不丢记忆；②取行锁重读时发现子块已失活（并发撤销所致），防止给作废的块造父块。
两者都从 `before_model` 冒出去终止本轮；因为压缩在每次模型调用前重跑，**不撤销就会每轮在同一
位置再失败一次**，直到撤销或改配置。抛错前已提交的合并不回滚（每轮合并各自一个事务）。

## 运行时上下文缓存

应用数据库仍是唯一 canonical source。进程内为每个 thread 懒加载一个派生缓存，只保存不可变
消息快照、当前 greedy cover 选中的记忆块、原文工作区尾部和最新 work state，不保存跨 session 的
SQLAlchemy ORM 对象。正常运行时，用户、模型、工具消息以及 work state 都先提交数据库，再增量
追加缓存；压缩块也先事务写库，再替换缓存视图。

缓存只在追加消息、更新 work state、压缩和失效等短操作期间按 thread 加锁，不锁住完整 turn，
以允许 LangGraph 并行执行同一批工具。并行工具消息按数据库全局 id 有序插入并去重。当前 TUI
要求 rollback 只能在 turn 结束后触发；rollback 提交数据库变更后，必须丢弃对应缓存并从 active
rows 重建。进程崩溃或缓存更新失败时也直接丢弃缓存，下次访问从数据库恢复；应用不配置 LangGraph
checkpointer，所以该缓存没有第二份持久化副本，也不需要与任何派生快照对齐。多进程的**写冲突**
由下一节的 turn 占用锁挡住；跨进程的缓存失效通知仍然没有，所以同一条 thread 还是建议固定一个
会话，长空闲后靠写轴前的分歧检测兜住。

## thread 占用（跨进程单写者）

一条 thread 同一时刻只允许一个会话待在里面。锁不落在数据层：MySQL 连接级咨询锁
`GET_LOCK('ca:t:'+sha1(thread_id)[:16], 3)` 挂在一条自己独占的 NullPool 连接上，锁寿命等于连接寿
命，所以 Ctrl-C、`kill -9`、断电都由服务端随会话收回，新会话立刻接管 —— 没有 holder 列、没有 TTL、
没有心跳线程，也不加表加字段。（必须单独建 NullPool engine：业务池的 `connect()` 退出只是还池、
不断 TCP，锁会跟着池里的连接继续活着。）

占用是**会话级**的，活过 turn：进入 thread（启动时的 `--thread`，或 `/thread <id>` 成功）时
acquire 并一直持有，切走或退出才 release。所以一个开着终端不说话的会话也算占着这条轨道，别的会话
进不来（`本轮未开始` 那种提示基本只在下面第四条的接管场景出现）。`run_turn` 只对占用做幂等续期。
这条口径的代价是"别人得等你退出"，换来的是"不会有人跟你同时写一条轨道"，而且不需要 TTL、心跳、
清理界面；也刻意没做 `/takeover` 强制接管——真被卡住就 `ps` 找到那个会话把它退出。不报占用者是谁，
想知道是哪个进程 `ps` 一眼就够，为此多养一条查询和一堆透传字段不值。

座位的释放**不指望任何人发信号**。实测 Terminal.app 关标签页只做两件事：杀掉 shell、**撤销**
（revoke）那块 pty，**不给前台作业组发挂断**——拿内核里的进程表看过一次现成的鬼：那一秒它的
`e_tdev` 从设备号变成 -1、`e_tpgid` 变成"无终端"标记，进程照活，而它对 SIGHUP 的处置是默认终止
（既不在 ignore 也不在 catch 名单里），说明信号根本没送到。zsh 的 `nohup`/`hup`、进程组、Ctrl-Z
全都与本案无关；VS Code 终端没这问题是因为它连进程一起杀。关标签页是用户的显式意图（"我不想看了"），
所以**整条会话期间**都得停手，不挑状态：`TerminalUI._watch_terminal`（daemon 线程，只在
`stdin.isatty()` 时启动）每秒问一次"`/dev/tty` 还开得开吗、`tcgetpgrp(0)` 还认我们吗"（两条各管一段
坏状态，见下面第二条）。都不吞输入，比读 fd 安全。连续两次判定"没了"就**先给自己一发
SIGINT**，走 `run_turn` 既有的中断收尾：杀掉本轮 bash 的整个进程组、终止在飞的子代理、把半截工具批次
结清在时间轴上，一路 unwind 顺带放出座位。收尾不一定走得完（可能正堵在模型的网络读上），8 秒后
`os._exit(0)` 兜底；硬退也不脏——进程一死 socket 就断，未提交的事务由服务端回滚，座位当秒收回。
**全程一个字都不许往屏幕写**：终端已经没了，写它就是 `OSError(EIO)`，第一版就因为退出前那行提示把本线程
摔死在 `os._exit` 前面，座位照样没还（真机 22:27 / 22:30 那两只鬼）。判据两条都要：`tcgetpgrp` 管终端
彻底销毁那一段，`/dev/tty` 打不开（ENXIO）管"主端已关、从端还开着"那个中间态——`isatty` 与 `tcgetpgrp`
在中间态都说"终端好着"，只问它们是瞎的。反过来，若进程**生来就没有**控制终端（被脚本用 `setsid` 直接拉起、
没有 shell 认领这块 pty，`tcgetpgrp` 一上来就抛 ENOTTY），判据没资格说话：自检观察到"终端曾在"之后才
允许判死，否则自己收工，绝不给一场好好的会话发信号。

取证方法（踩过的假阳性，别再犯）：

- 判据必须在**真终端**里看。`Popen` 出来的进程默认不在任何会话里（没有控制终端），得在 `preexec_fn`
  里 `os.setsid()` + `fcntl.ioctl(0, TIOCSCTTY, 0)`，让它像个真的 Terminal 会话主。
- **只关 master，别顺手杀 shell**。杀会话主时内核会给那块终端的前台作业组发挂断，会话 0.0~0.3 秒就死，
  测到的是信号、不是自检。
- 也别拿管道当 stdin 演"关窗口"：`stdin` 换成管道（`isatty()` 为假）会让 prompt_toolkit 永久睡在 read
  上，那是我们造出来的假鬼。
- 后台起的作业会被 `SIGTTIN` 停住再死于别的原因（0.2 秒内），同样比轮询快，看着像"自检生效"。
- 真正的验收还是得在 Terminal.app 里手点：新窗口 `agent --thread zz` → 关标签页 → 3 秒后
  `pgrep -f coding-agent` 不该有这个 thread，`SELECT IS_USED_LOCK('ca:t:'+SHA1(thread)[:16])` 该是 NULL。

每次写时间轴前 `TurnGuard.verify` 只比游标：`(active_head_seq, next_user_seq) == (user_seq,
user_seq + 1)`。这个等式由本轮开头的 `reserve_user_seq` 写下，本轮之内没有别的代码再动这两列，
所以等式还成立就等于没有别的会话写入或回滚过。**verify 刻意不去碰那条持锁连接**：一轮里工具是
并行执行的，每个分支各自在自己的线程里调 `verify`，而 SQLAlchemy 的连接不是线程安全的 —— 真机
踩过两个线程同时 `GET_LOCK` 互相踩事务，把持锁连接弄死、锁随之消失，之后每一轮都误判"占用已失效"，
整条会话再也跑不动。锁的存活只在每轮开头的 `acquire`（幂等续期）确认一次，那里必然是拿锁的线程
自己。少了轮中复核也不等于会双写：接管者要写就得先占游标，等式立刻不成立。

跨会话的重取只发生在 `acquire`（每轮开头一次）：会话空闲太久时
`wait_timeout` 会收走连接、锁随之消失而本机不知情，此时重取一次是合法的 —— 拿得回来说明没人动过，
拿不回来就报 `ThreadBusyError`（真被接管了），而空窗期里若有人写过，紧随其后的 seq 比对仍然会拦住。

刻意不做：不锁工作树、不锁路径 —— 同一棵树下并行发起多个会话是产品前提，树级互斥会把并行调研变成
排队，冲突交给 `edit_file` 的自然失败和 `/undo` 承担；`/undo` 也不取这把锁、不做任何前置拦截。需
要真隔离时用 Git worktree（见 readme）。

SQLite 没有跨会话咨询锁，锁的部分直接不做（单进程不会自己跟自己抢），seq 检测照常生效。真实行为
由 gated 的 `tests/integrations/test_thread_lock_mysql.py` 验证：跨连接互斥、切换时先抢新的再让出
旧的、空闲被 `wait_timeout` 收走后下一轮开头重取（取不回就报错，绝不换连接静默接管）、空窗期被
别的会话推进时终止本轮，以及并行工具线程同时 `verify` 不得把锁搞丢。

项目只注册一个 AgentRuntimeMiddleware，并由它显式编排上下文投影、调用额度、工具执行、文件
mutation、artifact 和消息持久化等普通 service。FilesystemMiddleware 仍单独保留用于注册和执行
文件工具。模型消息和工具最终结果都沿正常返回链持久化，不扫描 state 查找 unseen 消息。

工具顺序固定为：额度检查 → 原文件快照 → 整体重试循环 → mutation 完成 → artifact → 消息持久化。
同一个逻辑工具调用无论重试多少次都只创建一条 mutation。ls/read_file/glob/grep/write_file 可以
重试；edit_file/delete_file/bash 只执行一次。所有最终工具异常都转换为 error ToolMessage。

受控写工具按解析后的真实文件路径互斥：在读取原文前取锁，保存快照、执行（含重试）和
记录最终状态后释放，异常退出也释放。同一路径的 mutation id 顺序因此与实际修改顺序一致；
不同文件仍可并行。锁属于当前应用的 recorder，不提供跨进程或 Bash/外部编辑器的互斥；
并行批次中存在前后依赖的编辑仍应拆成先后两批，互斥不保证按模型列出的顺序执行。
blob 按 sha256 使用数据库原子插入/冲突复用，不覆盖已有内容；MySQL 取回记录时使用当前读，
避免 REPEATABLE READ 的旧快照看不到并发提交的 blob。无需修改表结构或清理旧数据。

Bash 默认开启，可通过配置显式关闭。它使用明确的 `/bin/bash -lc`（可配置 executable），从真实
workspace root 运行，stdin 固定关闭，单次命令受超时和输出字节上限约束。相对路径基于 workspace
root；绝对路径是宿主机真实路径，不是文件工具的虚拟 `/path`。非零退出码和超时返回 error
ToolMessage，让模型读取输出后自行修正，不触发自动重试。Bash 不关闭 LangGraph 的批量工具并发；
模型不得把存在读写依赖的 Bash 和文件操作放进同一批调用。V1 接受模型违反该约束时产生的文件
读写竞态，以免 Bash 默认开启后让所有独立只读工具退化为串行。

## 工具结果

正常模型调用不滚动裁剪工具消息，历史只追加。仅在工作区触发 L0 压缩时应用
`recent_tool_interactions` 窗口（默认 10，0 表示替换全部工具结果内容）：仅将更早的
ToolMessage 的 `content` 替换为短占位符，保留消息、`tool_call_id` 和 assistant 的完整调用。
数据库原消息及其 active 状态不变。摘要使用此次裁剪后的内容。

裁剪后的尾部直接保存在现有消息缓存中，不新增窗口状态或数据库表。新工具结果累积到
下一次压缩才再次裁剪。缓存重建（包括重启、撤销）时，从数据库恢复尚未被记忆块覆盖的
原始尾部，允许旧工具结果恢复完整内容；达到阈值后再裁剪、压缩。已持久化且仍有效的
记忆块不受影响。`/status` 直接统计缓存中的消息；token 估算公式不变。

单条工具结果超过 5k token 时，完整内容写到 `.artifacts/tool-results/`，模型输入保留前 5k
token 并附绝对路径，提示代理用 `read_file` 按需读取。

## TUI 交互

TUI 使用 prompt-toolkit 处理多行输入和会话内输入历史，Rich 负责 Markdown、状态和表格渲染。
用户输入先经过命令路由，未知斜杠命令不会发送给模型。模型与工具 callback 转换成结构化
TurnEvent，仅展示工具名称、路径等短参数和执行状态，不展开完整工具结果。

每轮在调用 Agent 前持久化用户消息并获得 user_seq。调用异常或 Ctrl-C 中断后，应用抛出包含
该 user_seq 的 TurnExecutionError。异常边界会先终止仍在运行的 Bash 进程组，再检查本轮尾部
tool-call batch；每个缺失结果都用确定性 `tool-{tool_call_id}` 消息 ID 补入 error ToolMessage，
保证下一次模型调用的工具协议完整。迟到的真实结果使用同一 ID，被 canonical 持久化幂等吸收。
TUI 展示补齐数量；如果中断收尾自身失败，则明确要求撤销，不用收尾错误覆盖原始调用错误。
用户仍可决定是否 rollback。手工 /undo 必须先展示消息、文件 mutation、
不同文件和 work state 数量，再由用户确认。

每轮 TUI 进度事件有独立生命周期。轮次结束或中断时先关闭事件入口并等待正在渲染的事件结束，
迟到的工具线程回调不得写入下一轮输入区域。

TUI 的桌面辅助行为通过 `tui` 配置控制：macOS 仅在 `run_turn` 期间持有防空闲休眠断言，
无论成功、异常或中断均释放，等待用户确认时不持有；退出进程也会自动释放。完成/失败通知
只携带会话、状态和耗时，不发送对话内容，脚本参数与代码分离。辅助程序缺失或失败不会
改变任务结果，不修改系统全局设置。

`/usage [N]` 只读查询最近 N 条当前 thread 的助手消息中的标准化 API 用量 JSON 字段，
不加载正文，不额外建表；统计包含 inactive 历史。缓存命中率按可用样本的输入 token 加权，
未知字段不按零处理。页面同时显示用量/缓存覆盖样本数；该统计不包括未落库的摘要请求和
失败重试，也不改变 ContextEngine 的 token 估算与压缩规则。

## 撤销语义

`rollback(thread_id, N)` 精确定义为撤销 `user_seq >= N`：

1. `file_mutations` 按全局 id 倒序恢复；成功 mutation 才恢复，失败 mutation 不恢复。
2. `messages.user_seq >= N` 标记 inactive。
3. `work_state_snapshots.user_seq >= N` 标记 inactive。
4. 根据撤销消息 id 的范围，停用与之相交的 memory block；不使用 user_seq 猜测块范围。
5. 从剩余 active 消息重新计算最大 `user_seq` 作为 `active_head_seq`，空历史为 0；
   `next_user_seq` 不回退。
6. 从所有 active memory block 中执行最长左块贪心覆盖，缺口回退到原始消息。
7. 恢复最近的有效 work state。

应用数据库是 canonical source，文件恢复逐条幂等提交。应用不配置 LangGraph checkpointer：
每次模型调用前 `AgentRuntimeMiddleware.before_model` 都会用业务库投影整体覆盖 `messages` 通道，
checkpoint 里的快照既不进入 prompt 也没有读者，却会按「每个图步骤一份全量 messages 快照」无界
增长，并把 `get_tuple` 拖成秒级（详见 RUNBOOK 的排障条目）。因此撤销只改数据库，无派生快照需要对齐。

## 文件 mutation

`write_file`、`edit_file`、`delete` 在执行前保存文件原始 bytes 到 SHA-256 去重 blob，记录 pending
mutation；工具成功后写 after hash 和 succeeded，失败则记 failed。撤销**刻意不比对**当前文件
hash：撤销是用户显式触发的意图，即使文件已被别的会话或手工改动带偏也必须照做——拦下来只会让
人在最需要退回去的时候退不回。因此这里不是漏检，而是"显式触发即视为授权覆盖"（thread 占用锁
只保证同一条 thread 不同时写，同一棵工作树允许并行，见「thread 占用」一节）。崩溃遗留的
pending mutation 也保守恢复；恢复 before 状态是幂等的。
`bash` 造成的文件变化不在 V1 追踪范围内，目录删除在 V1 中拒绝执行。Bash 在宿主机直接运行，
工作目录不是安全沙箱，也不能阻止命令访问 workspace 外路径；只应在可信的本地开发环境中显式
启用。需要隔离和可撤销 Shell 时，应改用容器/VM、overlay 或 Git worktree 级执行后端。
