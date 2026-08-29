数据库表的设计

messages





id



thread_id



user_seq 【用户发起调用时的 seq】



type // user/assistant/tool



content_json



active [撤销后，active=0，启用为 1]



create_at



converstations





id



thread_id



active_head_seq 【用户最新调用的 seq，起始为 1】



updated_at



created_at



work_state_snapshots





id



thread_id



user_seq



state_json



active [撤销后，active=0，启用为 1]



create_at



memory_blocks





id



thread_id



text 



begin_message_id



end_message_id



level



token_count



active [撤销后，active=0，启用为 1。active=1 表示该块属于当前有效历史，而不是“当前一定被放进模型”。]



create_at



file_mutations

- id

- thread_id

- user_seq

- tool_call_id

- operation_type

- path

- before_blob_id

- before_hash





active [撤销后，active=0，启用为 1]

- status [用于标记该工具是否执行成功，执行失败则无需撤销]



file_blobs

- id

- sha256

- storage_uri / content

注：数据库主键递增



关于上下文管理的说明。
记忆块分层合并算法，就是优先贪心合并记忆时间上较早的同级块。

首先，我们的上下文模型结构为：框架提示词 + 压缩区 +工作区 + 动态构造提示词 + 缓冲区，我们设置即 1/4 的上下文的压缩区空间，每当工作区的长度超过了 1/2 上下文的空间，就尝试保留工作区尾部的 1/5 原文，工作区前缀 4/5按照按照消息完整性拆成 4 块 L0 记忆块，扔进压缩区。对于压缩区，如果发现的 token 占用当前超过了 1/4 的上下文容量，则产生一个循环，将目前所拥有的记忆块，按照其维护的消息范围的左坐标排序，然后按顺序读，将读到的连续相邻两块同层记忆块合并。如果还是超过上限，则重新循环。类似于 2048 小游戏的算法。它的特色是，越早的记忆压缩次数越多，越模糊，越近的记忆次数少，越清晰。





然后关于这个压缩的时候，是否一定能找到同级块进行压缩？





本质我们每次是插入了 L0，压缩是遇到同级才晋级，就是 2 进制，简而言之，最终的得到的结果就是 Lx 就是产生的原始 L0 总量 cnt 的第x个二进位代表的数量。假如，100k 压缩后一个块的大小是 50k，容量 1/4 * 1 m = 250k，那么可以放下 5 个块，假如每次只放进去 1 个块，然后同层合并，6个块的时候才会崩溃，也就是 L0 总量是 111111 = 2^6 - 1 = 63 个才会崩溃，每次对应的是 100k，也就是 63 00k = 6.3 m。然而，我们是一次放入 4 个块进入。相当于一次放一个 L2 块，那么临界最大数量是= L7 + L6 + L5 + L4 + L3 + L2即 11111100 = 252 -> 25.2m。



然后考虑我们的理论效益，f(k)是压缩到第 k 层需要几次压缩：





f(k) = f(k-1) * 2 + 1，S(k) = f(k) + f(k-1) + ... + f(0)，f(0) = 1， S(k) =



f(k) + 1 = 2^(k+1)



S(k) + k + 1 = f(0) + 1 + f(1) + 1 + f(2) + 1 .... f(k) + 1 = 2 + 4 + 8 + .... + 2^(k + 1) = 2^(k + 2) - 2



S(k) = 2^(k + 2) - k - 3



S(7) = 2 ^ 9 - 10 = 502



由于11111100后面是 00，代表没有出现，所以还要减去 (1 + 3) = 498 次



498 * 100k = 49.8m 的摘要调用，压缩了实际 25m 的上下文，数学上也符合单次 0.5 的压缩比。并且一个旧信息，最多被压缩了 8 次。



如果常规的前缀压缩算法，也假设压缩比为 0.5，1m 的上下文，那么在达到了 750k 的上限后，取 100k 后缀为原文不动，对 650k 内容，进行压缩，产生了一个 325k的 L0。



如果达到 25m 的压缩量，就是 25000 k / 325 k = 76.9 次



成本是 77 * 650k = 50m



也就是是说在，理论压缩比一致、压缩量一致的情况下，两者的累计输入文本相同。



但是前一个算法，旧信息最多被压缩了 8 次（log 级），而后一个算法几乎是 77 次。



缺点是调用次数多，可能占用的时间会更多，优点是对记忆产生了良好的分辨力



问题是控制这个压缩比为 0.5 要注意





所以，如果压缩失败，就以更激进的提示词，让 llm 压缩。

关于这个工具消息剪裁说明





首先，我们只保留最近的 10 条完整工具交互在近期上下文里。





这个在实现的时候，需要投送给 langchain invoke 之前过滤工具消息





然后，工具的返回结果我们设置上限5k，超过 5k 就落盘，然后路径附加在 5k 的末尾。指出要求 agent 用工具 read 读。

我们考虑下基本逻辑运转逻辑：

langchain 启动的时候，由于我们走的是持久化的 checkpointer，直接复用 thread_id 作为 config 就可以继承上下文。 





前面提到的工具管理、上下文管理，我们通过中间件来实现



当用户触发了撤销，就需要重建上下文，RemoveMessage(id=REMOVE_ALL_MESSAGES),就跟据上下文时序压缩算法的递归组装算法，跟据当前最新的消息的 seq 为右边界限制 ，1 为左边界，循环找到最少子块覆盖集，也就是先找到一个最长左子块，然后以它的右边界+1，开始再找一个最长左子块。如果找不到记忆压缩子块了，就直接返回原始消息区段。然后通过这个组装算法，组装记忆。然后还有这个 work state，它是个 json，存放着当前的正在进行的工作状态。





现在考虑这个撤销机制，假如用户说 seq xxx 以及之后的消息都撤销掉了





如果是上下文的撤销，直接按照上文说的循环找最小子块覆盖集去构造就行了。





构造成功后，然后把 seq 之后的记忆块、work state、原消息标记为 active = 0



首先，用户在工具调用的时候，我通过中间件，对于两个写工具write ,edit hook，并通过读参数，把操作原文件的原文储存到了数据库里的某张表里？然后给他们标记了 thread_id 和 用户本轮消息的 seq，和全局序号 id。然后保存原文件的时候，看下那个文件的 hash 是否已经在数据库存在了，如果存在直接复用 blob，否则创建新的 blob。如果撤销时，用户修改了源文件，这个我们不管，直接还原。



然后用户现在打算撤销，我从数据库里读出，seq 之后的工具调用，按照全局序号倒序反向还原。





然后将 seq 之后的工具调用全部标记为 active = 0



编辑工具，我打算通过FilesystemMiddleware(backend=FilesystemBackend(root_dir="./", virtual_mode=True) ),来提供。shell的绕过我们不管。



workstate 提供一个工具，先让 agent 自己来维护



一些规格：





数据库 系统 mysql,SQLAlchemy



框架 langchain



messages.id = 全局消息时间轴，数据库递增，不复用 user_seq = thread 内用户轮次，从1开始，只增不减 active_head_seq = 当前有效的最大 user_seq



压缩事务顺序

建议写死：

生成摘要
↓
验证输出完整 + token_count <= hard_limit
↓
事务 INSERT memory_blocks
↓
重建当前 context
↓
更新 LangGraph state/checkpoint





UI 实现 TUI



撤销精确定义

例如：

rollback(thread_id, user_seq=N)

语义：

撤销 user_seq >= N

处理顺序：

① file_mutations 按 id DESC 回滚
② messages >= N active=0
③ work_state >= N active=0
④ 所有覆盖被撤销消息的 memory block active=0
⑤ active_head_seq = N-1
⑥ greedy cover 重建 Context
⑦ RemoveMessage(REMOVE_ALL_MESSAGES) + rebuilt
⑧ 恢复最近有效 work_state

这里尤其要写明：



memory block 是否受影响，根据 begin_message_id/end_message_id 与被撤销 message id 区间判断，不是直接拿 user_seq 比。





参数全部配置化

不要散落 magic number





