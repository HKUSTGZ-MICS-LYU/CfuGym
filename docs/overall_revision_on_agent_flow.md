截至 **2026-09-28**，我看下来有一个比较明显的趋势：主流 coding agent 正在从“设计一张很复杂的 Agent Graph”转向 **更薄的 agent loop + 更强的 harness/context/tooling**。而 CfuGym 这种领域 Agent 恰好相反——它真正有价值的部分不是“会不会调用 shell、改代码”，而是 **CFU Contract、HW/SW 联合验证、QoR、EDA 预算、独立 oracle 这些领域约束**。

所以我不建议把 CfuGym 改造成另一个 Codex/Claude Code；更适合的方向是：

> **保留一个薄的领域状态机作为 control plane，把代码理解、搜索、修改、debug、恢复等能力下沉成一个通用 coding-agent execution plane。**

这其实会比现在“一节点一职责”的 LangGraph 结构更接近当前成熟 coding agent 的实现路线。

先说明一下证据范围：我能确认当前 `mico_soc` 分支根目录同时包含 `agents/`、`benchmarks/` 和 `skills/cfu-designer/`，README 也已经支持把 `cfu-designer` 作为 Codex Skill 使用，说明你们事实上已经开始尝试“领域 workflow + 通用 coding agent”这一方向。:chatgpt-content-reference{index="0"} 本次 GitHub 网页访问对嵌套 `agents/` tree 有限制，所以我对其中具体 Agent 实现的分析同时基于你之前保存的 AutoCFU 架构设计文档；这些文档描述了现有 `load/profile/analyze/contract/patch/validate/Yosys/review` 闭环及后续改造方向。:chatgpt-content-reference{index="1"}

## 1. 现在 Coding Agent 大致走了哪几条实现路线

| 路线 | 典型系统 | 核心思想 | 控制流在哪里 |
|---|---|---|---|
| **Tool-loop 型** | Codex、Claude Code、Gemini CLI | LLM 不断观察环境→调用工具→得到结果→继续推理 | **模型驱动** |
| **Harness / Context Engineering 型** | Cursor Agent | Agent loop 本身不复杂，重点优化 context retrieval、tool interface、缓存、错误反馈、模型适配 | **模型 + harness** |
| **Event-driven Runtime 型** | OpenHands | action/observation、workspace、security、context condenser 都标准化 | **runtime 驱动** |
| **ACI 极简型** | SWE-agent / mini-SWE-agent、Aider | 不追求复杂 orchestration，而是把 edit/search/test 接口做得特别适合 LLM | **模型驱动** |
| **Coordinator-worker 型** | Cursor Projects 等 | 主 Agent 管任务分解，不直接写代码；多个隔离 worker 处理子任务 | **层级式** |

Codex 官方对自己的核心描述其实非常简单：用户输入和历史上下文送给模型，模型产生 tool call，harness 执行工具并把结果重新加入上下文，直到模型给出最终回答；真正复杂的是工具、sandbox、权限、planning 和 context 生命周期，而不是 graph 本身。:chatgpt-content-reference{index="2"}

Cursor 在 2026 年的路线更加明显。他们公开提到，过去大量静态注入的 context 和 guardrail 被逐渐删除，改成 **Agent 在执行过程中动态取得需要的信息**；还专门针对不同模型调整 edit tool，例如某些模型更适合 patch，另一些更适合 string replacement。最近他们还在进一步减少静态 system prompt 和 tool schema token，并把工具改成按需加载。:chatgpt-content-reference{index="3"}

这点对 CfuGym 很重要：**coding agent 的进步并没有主要来自“更多节点”，而来自 Agent Computer Interface。**

SWE-agent 的工作尤其能说明这一点。它把很多性能提升归因于 agent-computer interface，例如限制一次查看的代码范围、提供简洁 repo search、edit 后立即运行 linter、显式告诉模型命令没有输出，而不是依赖更复杂的 multi-agent graph。新版甚至进一步推荐更简单的 mini-SWE-agent。:chatgpt-content-reference{index="4"}

Aider 的 repo map 也是类似思路：不把整个 repository 塞给模型，而是先构建包含重要 class/function/signature 的紧凑结构地图，再按需要读取具体实现。:chatgpt-content-reference{index="5"}

另一方面，Gemini CLI 已经把一些很实用的“Agent OS”功能标准化，例如 context compression、tool-output summarization、loop detection、checkpoint/shadow git、hierarchical project context、hooks 等。:chatgpt-content-reference{index="6"}

OpenHands 则比较接近“Agent runtime”的思路：agent core 尽量保持 event-driven/stateless，把 typed action/observation、workspace、context condenser、security、skills 和 MCP 等功能抽成独立组件。:chatgpt-content-reference{index="7"}

---

## 2. 这和 CfuGym Agent 的根本差异在哪里

你们现有设计实际上比一般 coding agent **更 workflow-centric**。AutoCFU 的计划结构是：

```text
ingest / baseline
       ↓
dynamic profile + static analysis
       ↓
hotspot rank
       ↓
CFU Contract
       ↓
archetype / candidate generation
       ↓
patch
       ↓
build / simulation
       ↓
semantic verify
       ↓
Yosys / cycle measurement
       ↓
diagnose / repair / QoR optimization
```

并且 state 里面不是只有 chat history，而是明确存 `SourceSnapshot / ProfileEvidence / KernelFacts / CfuContract / BuildEvidence / SemanticEvidence / MetricBundle / Diagnosis / ParetoArchive / BudgetState`。:chatgpt-content-reference{index="8"}

这其实不是缺点。反而是 CfuGym 最重要的领域优势。

| 维度 | 通用 Coding Agent | CfuGym / AutoCFU |
|---|---|---|
| 任务目标 | “让 repo 达到用户要求” | “找到可加速 region，并生成满足语义与成本约束的 HW/SW 实现” |
| Control flow | LLM 自由决定下一工具 | Graph 预定义阶段 |
| State | message/context/tool history | typed domain state |
| 中间表示 | 通常没有严格 IR | **CFU Contract IR** |
| Correctness | tests/build/linter | scalar oracle + CFU simulation + whole-program |
| Optimization | 通常做到 tests pass 即结束 | correctness 后还要优化 cycles / area / PPA |
| Failure | stderr 返回给 Agent | 可显式分类 SCALA / ELAB / SIM / TIMEOUT / QoR |
| “不做” | 很少是一等结果 | **ABSTAIN / 不值得加速应该是合法结果** |
| 修改对象 | 普通 source tree | C + SpinalHDL + SoC integration + ISA + simulation |
| 成本 | 主要 token / wall time | token + simulation + synthesis + EDA budget |

你此前对当前原型的审计其实已经抓住了几个关键问题：contract 还是自由文本、patch 容易过大、profile 最大热点未必是最适合 CFU 的 region、失败后重新 implement 时缺少结构化 diagnosis、Yosys/top/flag 对已知案例有硬编码。:chatgpt-content-reference{index="9"}

这些问题和现在 coding-agent 社区正在解决的问题高度一致，但 **解决层次不同**。

---

# 3. 我认为最值得做的一次架构调整

我会把当前十几个 LangGraph node **收缩成 5 个领域 Gate**：

```text
                 ┌─────────────────────────────┐
                 │      Domain Coordinator     │
                 │        (LangGraph)          │
                 └─────────────────────────────┘
                              │
       ┌──────────────┬───────┼────────┬─────────────┐
       ↓              ↓       ↓        ↓             ↓
   DISCOVER        CONTRACT IMPLEMENT VERIFY      OPTIMIZE
 profile/static     CFU IR     coding    oracle     QoR search
 rank/abstain       validate   agent     build      Pareto
                                loop      sim
                                  │
                                  ▼
                    ┌───────────────────────┐
                    │ Generic Coding Harness│
                    ├───────────────────────┤
                    │ context manager       │
                    │ repo/co-design map    │
                    │ tool registry         │
                    │ edit abstraction      │
                    │ checkpoint/worktree   │
                    │ loop detector         │
                    │ history condenser     │
                    └───────────────────────┘
                                  │
          ┌───────────────────────┼──────────────────────┐
          ↓                       ↓                      ↓
      clang/LLVM             RepoToolbox           Sim / Yosys
```

这里最重要的变化是：

**LangGraph 不再负责指导 Agent “下一步该读哪个文件、怎么改代码”。**

它只负责规定：

> 当前处于什么领域阶段、什么 artifact 必须存在、哪些 invariant 必须满足、什么时候允许进入下一阶段。

至于：

> “我要查哪个 Scala class？”
>
> “这个 VexiiRiscv API 怎么用？”
>
> “compiler error 是不是 import 错了？”
>
> “应该查看哪个类似 CFU？”

这些应该全部交给一个普通 coding-agent loop。

这是我认为你们当前架构和 Codex/Cursor/OpenHands 最值得融合的点。

---

# 4. 哪些东西应该坚决保留在 CfuGym，而不是学通用 Agent

你们已经设计的 **CfuContract** 应该继续加强，而不是弱化。

例如：

```yaml
semantics:
  operation: dot_product
  element_type: int8
  accumulator_type: int32
  overflow: twos_complement_wrap

isa:
  archetype: stateful
  commands: [reset, mac4, read]

microarch:
  lanes: 4
  accumulator_width: 32

acceptance:
  min_speedup: 1.2
  max_area_cells: 5000
```

只有 validator pass 后才能生成代码。这个做法明显比普通 coding agent 的“写完跑测试”更严格，也更适合科研系统。:chatgpt-content-reference{index="10"}

同样应该保留的还有 **独立 oracle**。Agent 自己写的 test 不能作为最终 correctness evidence；最终数字只能来自 deterministic tools，而不能让 LLM 自己填写。你们之前的设计已经明确这一点，我认为这是论文里很值得强调的地方。:chatgpt-content-reference{index="11"}

换句话说：

> Coding Agent 可以控制 implementation，  
> 但绝不能控制 evaluation criterion。

---

# 5. 我会优先改的几个地方

| 优先级 | 改动 | 为什么 |
|---|---|---|
| **P0** | 把 `implement/repair` 改成真正的 **tool-loop coding executor** | 不再一次 prompt → 一次 patch |
| **P0** | 增加 **checkpoint/worktree per candidate** | patch 失败可瞬间 rollback；QoR candidate 可并行 |
| **P0** | 把所有 build/sim/Yosys 输出统一成 typed observation | Agent 不应该直接消化几千行 stderr |
| **P0** | 加入 `(patch hash, failure signature)` loop detection | 避免重复生成同一种错误补丁 |
| **P1** | 做一个 **HW/SW Co-design Repo Map** | 比把整个 repo/RAG 塞给模型更适合代码 Agent |
| **P1** | 把 `skills/cfu-designer` 作为动态 domain knowledge 层 | 避免 prompts.py 与 skill 重复维护 |
| **P1** | tool schema 做 provider/model abstraction | Cursor 的经验表明不同模型适合不同 edit interface |
| **P1** | Graph node → 5 个 macro gates | 减少 workflow 对模型行为的过度约束 |
| **P1** | context condenser / stage summary | 长时间 synthesis-debug 不让历史 log 挤爆 context |
| **P2** | QoR 阶段再启用 isolated subagents | 多 candidate 才真正值得 multi-agent |
| **P2** | 加 harness evaluation metrics | 除最终 speedup 外，还测 repair rate、tool error、loops、tokens |

尤其是 **checkpoint**，我认为成本极低、收益很大。Gemini CLI 已经采用 shadow git 保存修改状态和会话 checkpoint。:chatgpt-content-reference{index="12"}

对 CfuGym 可以进一步做得更领域化：

```text
candidate_0/
candidate_1/
candidate_2/
```

每个 candidate 对应：

```text
git/worktree commit
CFU Contract hash
patch hash
build artifact hash
simulation evidence
cycle result
Yosys result
```

这样 `ParetoArchive` 不只是几个数字，而是**可重新 checkout 的设计点**。

---

# 6. Repo Map 在你们这里应该比 Aider 更进一步

普通 Aider 是：

```text
repo
 ├── class
 ├── method
 └── signature
```

我建议 CfuGym 做：

```text
CoDesignMap

CFU
 └── I8DotCfu
      ├── SpinalHDL component
      ├── Fiber/container
      ├── SoC parameter
      ├── CLI flag
      ├── custom opcode
      ├── C intrinsic
      ├── smoke test
      ├── simulation top
      └── Yosys top

Software
 └── i8_dot()
      ├── call graph
      ├── loop L0
      ├── type = int8
      ├── trip count
      └── candidate CFUs
```

这会成为一种真正的 **hardware/software cross-layer codebase representation**。

也就是说，不只是：

> “哪个 symbol 在哪个文件？”

而是：

> “一条 custom instruction 从 C 调用点到 ISA encoding，再到 SoC integration、SpinalHDL component、simulation 和 Yosys measurement 的完整 dependency chain 是什么？”

我甚至认为这个组件本身就可以成为 AutoCFU 相对于普通 coding agent 的一个特色。

---

# 7. Repair Loop 也可以进一步借鉴 coding agent

你们原计划已经是：

```text
failure
 → diagnose
 → repair
```

而且要求 repair prompt 只放 `contract + previous patch + typed diagnosis + minimal logs`，这是对的。:chatgpt-content-reference{index="13"}

但我建议把它进一步变成：

```text
Tool Error
   ↓
deterministic parser
   ↓
FailureObservation
{
    stage,
    category,
    primary_error,
    blamed_files,
    relevant_symbols,
    previous_failure_hash,
    repeated_count,
    suggested_tools
}
   ↓
Coding Agent
```

例如不要给模型：

```text
5000 lines sbt output
```

而是：

```text
category = SCALA_COMPILE
primary_error =
  "value payload is not a member of Stream[Bits]"

location =
  BnCfu.scala:137

related_symbols =
  [Stream, payload, rsp]

previous_same_failure = false
```

原始 log 仍然存在，Agent 想看时自己调工具读取。

这和 Cursor 最近强调的一个经验一致：**tool error 本身必须是高质量 context**，因为大量低质量 tool errors 会不断积累，产生 context rot。:chatgpt-content-reference{index="14"}

---

# 8. Multi-Agent 不应该现在全面引入

这一点我和你之前的架构判断仍然一致。

现在没有必要搞：

```text
C Expert Agent
RTL Expert Agent
Verification Agent
PPA Agent
Manager Agent
Critic Agent
```

这样大概率只是：

> 同一批 context 被复制 5 次 + 更多 token + 更多通信错误。

当前 coding-agent 系统里的 subagent 更大的价值其实是 **context isolation**，不是“角色扮演”。Cursor 也把 subagent 描述成让任务在 fresh context 中执行的一种机制，而最新的 token efficiency 工作同时指出 subagent 会引入 coordination tax。:chatgpt-content-reference{index="15"}

CfuGym 真正适合 subagent 的位置是在 **Contract 已冻结之后的 candidate exploration**：

```text
                 fixed CfuContract
                       │
           ┌───────────┼───────────┐
           ↓           ↓           ↓
        worker A    worker B    worker C
         lanes=2     lanes=4     lanes=8
           │           │           │
        worktree    worktree    worktree
           └───────────┼───────────┘
                       ↓
                 deterministic eval
                       ↓
                  Pareto Archive
```

这里 candidate 之间本来就是天然独立的，所以 multi-agent 有实际并行价值。

---

# 9. 一个我认为很关键的理念变化

当前 CfuGym Agent 容易被描述为：

> **LLM generates a CFU through a LangGraph workflow.**

我觉得最终应该变成：

> **CfuGym provides a domain-constrained agentic co-design environment in which a coding agent searches implementation space under an executable HW/SW contract and independent correctness/QoR feedback.**

两者差异很大。

前者容易被 reviewer 看成：

> LangGraph + prompts + EDA tools。

后者强调的是：

```text
           Domain Contract
                ↓
Program ── Agentic Search ── HW/SW Implementation
                ↑
        executable feedback
       /          |          \
 semantics      cycles       area
```

这也更容易解释为什么不能直接拿 Codex/Claude Code 来完成同样的事情：**通用 Agent 有代码修改能力，但没有你们这个 problem-specific search space、contract、evaluation semantics 和 QoR objective。**

---

## 最终我会怎么定位这套架构

我不会把目前的 LangGraph 推翻。

我会把它**降级**。

从：

> LangGraph = Agent 本身

变成：

> **LangGraph = domain control plane**

再在下面放：

> **Coding-agent harness = execution plane**

最终形成：

```text
                 AutoCFU
        ┌────────────────────┐
        │ Domain Control Plane│
        │                    │
        │ Discover           │
        │ Contract           │
        │ Verify             │
        │ Optimize           │
        │ Budget / ABSTAIN   │
        └─────────┬──────────┘
                  │
        ┌─────────▼──────────┐
        │ Coding Agent Plane │
        │                    │
        │ Search / Read      │
        │ Edit / Patch       │
        │ Debug / Recover    │
        │ Plan / Explore     │
        └─────────┬──────────┘
                  │
        ┌─────────▼──────────┐
        │ CfuGym Environment │
        │ clang / Vexii      │
        │ SpinalHDL / GCC    │
        │ MiCoSocSim / Yosys │
        └────────────────────┘
```

这样一来，你们既吃到了 **Codex/Cursor/Gemini/OpenHands 这两年在 coding-agent harness 上积累的经验**，又不会牺牲 AutoCFU 真正有科研价值的领域约束。

而且有一个很有意思的信号：当前仓库已经同时存在 `agents/` 和 `skills/cfu-designer/`。:chatgpt-content-reference{index="16"} 我反而会建议未来逐渐让两者职责清晰化：

**`skills/cfu-designer` = knowledge/procedure/interface，`agents` = domain orchestration/evaluation。**

这样同一个 CFU domain skill 未来既可以被你自己的 LangGraph executor 使用，也可以直接挂到 Codex/Gemini/其他通用 Agent 上。这个设计还能自然产生一个很不错的实验：**相同 skill + 相同 tools + 相同 benchmark，比较 Generic Coding Agent、现有 Fixed Workflow、以及 Hybrid AutoCFU 三者。** 这个 baseline 会比仅仅比较几个 prompt/LLM 更有研究说服力。

如果你愿意继续往实现层推进，我下一步比较建议直接针对 `agents/cfu_design_agent` 做一版 **“现有文件 → 新架构模块”的逐文件 refactor proposal**，具体到 `graph.py / state.py / tools.py / prompts.py` 哪些保留、哪些删除、哪些改造成 coding-agent loop；如果连接 GitHub，我也可以直接按当前 HEAD 做代码级审计，而不是只基于现有架构文档。