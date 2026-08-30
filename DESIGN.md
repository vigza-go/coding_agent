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
动态提示和输出缓冲。工作区超过触发线时，按完整消息保留约 20% 尾部原文，前缀按 token
尽量均衡拆成 4 个 L0 块。

每次摘要的硬上限是输入 token 的 50%。输出为空或超限时，最多重试三次，提示词逐步变得
更激进。一次 L0 压缩的各个 chunk 相互独立，默认最多并发发送 4 个摘要请求；结果仍按 chunk
原始顺序收集。任一请求最终失败都不写 memory block，所有摘要生成并验证后才在一个事务里写入。

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
rows 重建。进程崩溃或缓存更新失败时也直接丢弃缓存，下次访问从数据库恢复；LangGraph
checkpoint 不作为该缓存的数据来源。多进程部署仍需同一 thread 固定路由，或增加分布式缓存
失效通知。

项目只注册一个 AgentRuntimeMiddleware，并由它显式编排上下文投影、调用额度、工具执行、文件
mutation、artifact 和消息持久化等普通 service。FilesystemMiddleware 仍单独保留用于注册和执行
文件工具。模型消息和工具最终结果都沿正常返回链持久化，不扫描 state 查找 unseen 消息。

工具顺序固定为：额度检查 → 原文件快照 → 整体重试循环 → mutation 完成 → artifact → 消息持久化。
同一个逻辑工具调用无论重试多少次都只创建一条 mutation。ls/read_file/glob/grep/write_file 可以
重试；edit_file/delete_file/bash 只执行一次。所有最终工具异常都转换为 error ToolMessage。

Bash 默认开启，可通过配置显式关闭。它使用明确的 `/bin/bash -lc`（可配置 executable），从真实
workspace root 运行，stdin 固定关闭，单次命令受超时和输出字节上限约束。相对路径基于 workspace
root；绝对路径是宿主机真实路径，不是文件工具的虚拟 `/path`。非零退出码和超时返回 error
ToolMessage，让模型读取输出后自行修正，不触发自动重试。Bash 不关闭 LangGraph 的批量工具并发；
模型不得把存在读写依赖的 Bash 和文件操作放进同一批调用。V1 接受模型违反该约束时产生的文件
读写竞态，以免 Bash 默认开启后让所有独立只读工具退化为串行。

## 工具结果

送模型前只保留最近 10 条完整 ToolMessage；更早的结果替换成短占位符，数据库原消息不变。
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

## 撤销语义

`rollback(thread_id, N)` 精确定义为撤销 `user_seq >= N`：

1. `file_mutations` 按全局 id 倒序恢复；成功 mutation 才恢复，失败 mutation 不恢复。
2. `messages.user_seq >= N` 标记 inactive。
3. `work_state_snapshots.user_seq >= N` 标记 inactive。
4. 根据撤销消息 id 的范围，停用与之相交的 memory block；不使用 user_seq 猜测块范围。
5. 从剩余 active 消息重新计算最大 `user_seq` 作为 `active_head_seq`，空历史为 0；
   `next_user_seq` 不回退。
6. 从所有 active memory block 中执行最长左块贪心覆盖，缺口回退到原始消息。
7. 用 `RemoveMessage(REMOVE_ALL_MESSAGES)` 加重建结果替换 LangGraph checkpoint 消息。
8. 恢复最近的有效 work state。

应用数据库是 canonical source。文件恢复逐条幂等提交；LangGraph checkpoint 是派生状态，若进程
恰好在数据库提交后、checkpoint 更新前崩溃，启动时可从 canonical rows 再次重建。

## 文件 mutation

`write_file`、`edit_file`、`delete` 在执行前保存文件原始 bytes 到 SHA-256 去重 blob，记录 pending
mutation；工具成功后写 after hash 和 succeeded，失败则记 failed。撤销不检查当前文件 hash，按
产品定义直接恢复。崩溃遗留的 pending mutation 也保守恢复；恢复 before 状态是幂等的。
`bash` 造成的文件变化不在 V1 追踪范围内，目录删除在 V1 中拒绝执行。Bash 在宿主机直接运行，
工作目录不是安全沙箱，也不能阻止命令访问 workspace 外路径；只应在可信的本地开发环境中显式
启用。需要隔离和可撤销 Shell 时，应改用容器/VM、overlay 或 Git worktree 级执行后端。
