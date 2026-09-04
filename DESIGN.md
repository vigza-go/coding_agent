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

每次摘要的硬上限是输入 token 的 50%。输出为空或超限时，最多重试三次，提示词逐步变得
更激进。一次 L0 压缩的各个 chunk 相互独立，默认最多并发发送 4 个摘要请求；结果仍按 chunk
原始顺序收集。任一请求最终失败都不写 memory block，消息缓存也保持原样；
所有摘要生成并验证后才在一个事务里写入记忆块，再更新缓存中的剩余消息。

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
撤销重建时才从 active blocks 重新运行 greedy cover。若超限却没有同级相邻块，抛出不变量
错误，不静默丢记忆。

## 运行时上下文缓存

应用数据库仍是唯一 canonical source。进程内为每个 thread 懒加载一个派生缓存，只保存不可变
消息快照、当前 greedy cover 选中的记忆块、原文工作区尾部和最新 work state，不保存跨 session 的
SQLAlchemy ORM 对象。正常运行时，用户、模型、工具消息以及 work state 都先提交数据库，再增量
追加缓存；压缩块也先事务写库，再替换缓存视图。

缓存只在追加消息、更新 work state、压缩和失效等短操作期间按 thread 加锁，不锁住完整 turn，
以允许 LangGraph 并行执行同一批工具。并行工具消息按数据库全局 id 有序插入并去重。当前 TUI
要求 rollback 只能在 turn 结束后触发；rollback 提交数据库变更后，必须丢弃对应缓存并从 active
rows 重建。进程崩溃或缓存更新失败时也直接丢弃缓存，下次访问从数据库恢复；应用不配置 LangGraph
checkpointer，所以该缓存没有第二份持久化副本，也不需要与任何派生快照对齐。多进程部署仍需同一
thread 固定路由，或增加分布式缓存失效通知。

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
mutation；工具成功后写 after hash 和 succeeded，失败则记 failed。撤销不检查当前文件 hash，按
产品定义直接恢复。崩溃遗留的 pending mutation 也保守恢复；恢复 before 状态是幂等的。
`bash` 造成的文件变化不在 V1 追踪范围内，目录删除在 V1 中拒绝执行。Bash 在宿主机直接运行，
工作目录不是安全沙箱，也不能阻止命令访问 workspace 外路径；只应在可信的本地开发环境中显式
启用。需要隔离和可撤销 Shell 时，应改用容器/VM、overlay 或 Git worktree 级执行后端。
