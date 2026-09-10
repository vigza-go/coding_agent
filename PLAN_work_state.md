# work_state 便签化改造计划

## 一句话目标

剪裁（思维链 + 旧工具结果）合并成一步做完；剪完把**当前 work_state 便签**贴在 working 段最前面；
便签只在"剪裁那一下"换新，平时冻着；去掉 pin 和每轮无条件的 system 注入。

## 改动前 vs 改动后

**改动前（每轮都跑）**
```
build()
 └─ compact_if_needed()
     └─ _create_l0_if_needed()
         ├─ 剪思维链 → 够就 return 0（工具结果没剪）
         └─ 剪工具结果 → 压缩
 └─ render_pieces()
 └─ 无条件把 work_state 渲染成 SystemMessage 放最前   ← 一变就断全前缀
```

**改动后**
```
build()
 └─ compact_if_needed()
     └─ _slim_and_compact()          ← 合并后的唯一剪裁入口
         ├─ 超线？没超就 return 0（什么都不动）
         ├─ 剪思维链 + 剪旧工具结果（一次做完，中间不再 return）
         ├─ 剪出了余量（<= trim_sufficient_line）→ 收手；没剪出 → 压缩成 L0 块
         ├─ 把最新 work_state 渲染成便签，存进 state.pin_work_state
         └─ 返回
 └─ pieces() 渲染：[blocks] + [便签] + [working 原文]
```

---

## 步骤

### 1. `context/work_state.py`：删掉 pin

- 删 `PIN_PREFIX` 常量、`ordered()` 里的 pin 排序（退回普通按键名字典序）、`render_index()` 整段、以及两处 `(pinned)` 文案。
- 保留 `render()`、`apply_op()`、`flatten()`、`as_text()`。pin 想干的事（"忘了会违规的约束"）改由便签头部 + AGENTS.md 承担，不再靠 `!` 前缀。

### 2. `integrations/langchain_agent.py`：重写 work_state 工具描述

- 现在描述里"键名以 `!` 开头表示钉住（常驻注入）"这句删掉。
- 新描述要点（agent 的笔记本口吻）：
  - 这是我的**备忘录**，记自己需要跨轮记住的东西（用户偏好、确认过的事实、踩过的坑、临时约束）；
  - 内容**不会每轮自动出现**，需要回顾时用 `get` / `list` 主动翻；
  - 剪裁/压缩之后框架会把最新一版作为便签贴在最前面提醒我，所以**写完不用担心丢**；
  - 简洁，不是所有东西都往这里塞。

### 3. `context/engine.py`：剪裁合并成一个函数

- 把 `_create_l0_if_needed()` 里"剪思维链 → 判超线 → return 0 → 剪工具结果"的**中间 return 去掉**，改成：
  ```
  if working_tokens <= trigger: return 0     # 唯一的提前返回
  working = _retain_reasoning_within_budget(...)
  working = _trim_old_tool_results(...)       # 无条件接着做
  # 剪完判"剪出余量没有"：<= trim_sufficient_line 就收手，否则压缩
  ```
- 理由：分步碎剪"这次剪思维链、下次剪工具结果"，等于**每次触发都断一次前缀**；
  一次剪到位 → 触发次数变少 → 断前缀次数变少。（代价：可能多剪一点，划得来。）
- **判据是"剪出多少余量"，不是"是否压到线下"**：剪到刚好压线，下一轮一个工具结果就能再
  顶过线，于是又剪一次、又断一次前缀——等于拿剪裁当滑动窗口。剪完还剩一大截说明大头是
  对话正文，剪裁刮不出多少，该压缩。见 `ContextSettings.trim_sufficient_ratio`
  （1.0 = 退化成滑动窗口，别用；调小则趋近"总是压缩"）。
- 函数改名 `_slim_and_compact()`（或保留原名），内部顺序不变：`_retain_reasoning_within_budget` → `_trim_old_tool_results` → 切块压缩。

### 4. `context/engine.py` + `context/cache.py`：便签字段 + 贴上去

- `ThreadContextState` 加一个字段 `pin_work_state: MessageSnapshot | None = None`。
  （**不放**进 `working_messages`：那样剪裁、`atomic_message_units`、切块压缩、`pieces()` 的 begin id 全要加"绕开它"的守卫，容易漏。独立字段 = 机器天然碰不到它。）
- 在 `_slim_and_compact()` **确实动过东西之后**（剪掉过思维链/工具结果，或压出了块），
  渲染最新快照：`render(latest_work_state.state_json)`，包成一条 `name="work_state"` 的
  HumanMessage 快照，写进 `state.pin_work_state`。
  - 没动东西（提前 return 的那条路径）→ 便签**不动**，`build()` 逐字节不变，缓存全中。
- 便签是**派生视图**：不落 DB，不进 `messages` 表，不进 `memory_blocks`。

### 5. `context/cache.py`：冷启动恢复

- `_load()` 里：若 `selected_blocks` 非空（说明历史上压过），就把 `pin_work_state` 也渲染出来，
  避免进程重启后便签凭空消失。重启本身就是冷启动，断前缀无所谓。

### 6. `services/context_projection.py`：删每轮注入，改渲染便签

- 删掉 `build()` 里整段 `if snapshot is not None: ... SystemMessage(...)`，以及硬塞的 `!user_note`。
- `render_pieces()` 改成：`[memory 块...] + [便签] + [raw...]`，便签插在 memory 块之后、原文之前。
- 便签用 `name="work_state"` 的 HumanMessage（**绝不能用 SystemMessage** —— langchain_anthropic 会把它提到请求最前，又变成老 bug）。
- `!user_note` 那句"如无必要，勿增实体"挪进**固定 system 文案**（AGENTS.md 方向），不再跟着动态快照走。

### 7. 配置 / 常量

- `reasoning_budget`、`recent_tool_interactions` 仍是这两个开关，合并后一起用，不改 config 键名。
- 便签渲染若要设个体积上限（防止 work_state 长成大杂烩），在这儿加一个上限常量。

---

## 验收标准

1. **未超线**：连续多轮 `build()` 输出逐字节相同，缓存命中 100%。
2. **超线一次**：只发生**一次**剪裁（思维链 + 工具结果同时），便签出现一次；
   之后若没再超线，便签**逐字节不变**（不刷新）。
3. **便签不进库**：`messages` 表和 `memory_blocks` 表里搜不到便签文本。
4. **重启**：已有 blocks 的线程，`_load()` 后便签自动恢复。
5. **不误剪**：便签不会被 `_retain_reasoning_within_budget` / `_trim_old_tool_results` /
   切块压缩碰到（独立字段天然满足，测试里断言一次）。

## 风险 / 要盯的点

- **别让它变成每轮刷新**：刷新只挂在"剪裁那一下"，否则逐写断前缀的老毛病立刻回来。
- **模型会不会把便签当指令**：位置在 memory 块之后、原文之前，比放尾部安全，但仍要实测（真实长 ReAct 里看它会不会"回应"便签）。提示词不兜底。
- **便签占位**：它命中缓存所以便宜，但一直占着。work_state 太长时要有上限，工具描述也要劝住"别啥都往里塞"。
- **写入侧**：这套只回放"已经写进去的"。所以 agent 得在信息新鲜时写（write-through）；好在它是"当场记"，不是"几百轮后想起来翻"，可靠得多。
