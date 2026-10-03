# CFU Agent：Workload → LLVM IR 特征与 per-loop 画像实现计划

## 目标与边界

把现有的 **动态 per-loop 周期证据** 与 **LLVM IR 静态结构特征** 关联，形成可追溯的 workload 分析输入，帮助挑选 CFU 候选、明确数据/状态/访存契约。IR 是分析证据，不是周期模拟器，也不取代真实 benchmark 验证。

- 首期支持 Embench 的 C 源码；之后在编译命令可重放的前提下扩展到 C-project。自然语言生成的 benchmark 可沿同一入口复用。
- 保留当前 Embench `reference` / `profile` / `cfu` 三份隔离源码：只读地从**未插桩的 reference 源码**额外生成分析 IR；现有 profile 仍测循环整体（一次进入计一次），reference/CFU 的正确性与加速比分数不变。
- 不以 IR 指令数冒充运行次数、耗时或 Roofline 数据；未知的 trip count、字节数、访存边界保持 `unknown`。嵌套循环与函数计数是 inclusive，不可相加。
- **进度：P0 试点与 P1 静态特征提取已接入**：`cfu_design_agent/ir/` 通过默认关闭的 `--ir-features off|auto|require|approx` 导出 reference 源 `.ll`；LLVM 14 `opt`（LoopInfo/ScalarEvolution/DependenceAnalysis）+ 文本 opcode 计数生成 `<stem>.features.xml`（数据搬移、带位宽/向量lane 的运算计数、SSA/loop-carried/内存依赖、SCEV 常量 trip count），经 `inspect_observation(view='features')` 按需供给 Agent，缺失/失败 best-effort 降级。P2 loop 对齐及 P3 独立缓存/C-project 仍待实现。不改历史实验记录，不依赖在线服务。

## 已有接口与落点

- `agents/cfu_design_agent/graph.py` 的 `profile_embench → extract_hotspot → collect_ir_evidence → analyze_workload → design_contract` 链路；纯 Python runner 必须与 LangGraph 同步接线。`05_profile/embench.yaml` 已有 `observation.points/applied/skipped` 与每点 cycles/calls；被选中的 `WorkloadSpec.hot_loop` 包含 `id/file/function/ordinal/line/loop_kind`。
- `agents/cfu_design_agent/benchmark/embench_instrument.py`、`benchmark/embench_loops.py` 的源位置扫描可作为**现有观测点**的来源；不要把其简易 C lexer 当作 LLVM loop 分析。`profile_cache.py` 当前保存 reference/profile 周期测量，若 IR 证据另行缓存，须有独立版本与输入指纹，不能默默重用旧内容。
- `agents/cfu_design_agent/benchmark/generic.py` 的 C-project 分支目前复制项目并主要包装 `main`，不能在没有独立 per-loop 动态数据时把 IR 的静态循环排序叫作“热点”。
- `agents/cfu_design_agent/state.py` 的 `WorkloadSpec` / `CfuDesignState`、`prompts.py` 的 `workload_prompt`/`contract_prompt`、`07_analysis/workload_analysis.yaml` 是消费端。把冗长原始 `.ll` 留在 artifact，提示词只注入受限的结构化摘要。

## 数据契约（建议新建 `ir_features.py` 与版本化 schema）

`07_analysis/ir_features.yaml` 保存机器可读的**静态**证据；必要时同目录保存 `reference.ll`（及每 TU 独立文件），禁止把大段 IR 原样送进 LLM。建议最小格式：

```yaml
schema_version: '1'
status: ok  # ok | partial | unavailable | failed
reason: ''
toolchain: {clang: 'version string', target: 'riscv32-unknown-elf', opt_level: O1}
inputs: {source_hash: '...', compile_flags_hash: '...', ir_hash: '...'}
functions:
  - {id: 'file.c::kernel', source_file: 'file.c', source_line: 35,
     blocks: 12, instructions: 90, loads: 8, stores: 4, calls: 2}
loops:
  - id: 'ir:<module hash>:<function>:<header>:<depth>'
    source: {file: 'file.c', function: 'kernel', line: 42, column: 3}
    structure: {depth: 2, parent_id: '...', blocks: 3, exits: 1,
                trip_count: unknown, backedge_count: 1}
    operations: {integer_add: 3, integer_mul: 1, shifts: 0,
                 compares: 2, loads: 2, stores: 1, calls: 0}
    qualifiers: {volatile: false, atomic: false, inline_asm: false}
    confidence: static_ir
links:
  - {observation_id: 'obs_loop_...', ir_loop_id: 'ir:...', status: matched,
     method: debug_location_and_nesting, candidates: 1}
measurements:
  - {observation_id: 'obs_loop_...', cycles_inclusive: 1200, calls: 1,
     config: profile, provenance: '05_profile/embench.yaml'}
```

上例中的数字均为**格式示意，不是当前实测值**。实际提取可加整数宽度、常量范围、分支/PHI、GEP 步长、已知/未知外部调用、向量化与规约候选等字段；每个字段都标注 `static_ir`、`dynamic_profile` 或 `inferred` 来源。没有可靠内存对象大小/别名信息时不计算 bytes/op。`unknown` 应是显式值，不能填 0。源路径采用 workload-relative 形式，ID 必须稳定且避免同名函数/跨 TU 冲突；动态 observation ID 继续沿用现有体系，不要求 LLVM 生成同一个 ID。

## 分阶段实施

### P0 — 能力探测与可复现 IR（不改 profile 默认行为）

1. 新增 `agents/cfu_design_agent/ir_toolchain.py`：发现并记录 `clang`（必要时 `opt`、LLVM 解析库）版本，检查与 GCC 编译的 include/define/target/sysroot 是否匹配。当前环境能找到 Clang 14、RISC-V GCC，但 `opt`、`llvm-dis`、`llvm-config` 不在 PATH：先做探测与清晰的 `unavailable` 降级，不在运行中下载工具。LLVM pass 如需 `opt`，版本必须与产出 bitcode 的 Clang 对齐；也可由显式配置的版本匹配 LLVM 库直接分析。
2. 从 **实际 build flags** 确定 `-target riscv32-unknown-elf`、`-march`、ABI、`-I`、`-D`、语言标准与优化级；逐 translation unit 生成 `-S -emit-llvm` 的可读 `.ll` 和 `-g` 调试位置。使用隔离的只读输入/工作区，不覆盖 GCC 构建产物；头文件、编译器扩展、内联汇编不受支持时记录错误并继续原 profile 路径。不能只从 Makefile 文本猜测完整 argv；需可审计的 compile command 或构建适配器。
3. 初期固定明确的优化级（例如 `-O1`）并另测 `-O0` 用于映射调试；报告 opt-level。**不得**把 Clang 的 IR 计数与 GCC 编译运行周期解释为一一对应的机器指令；两条编译链的差异必须作为限制显示。
4. 产出工具链命令、退出码、stderr、输入 hash、IR hash 和 `status`；对缺失工具、编译失败、权限/预算失败实行 fail-open（只关闭 IR 增强，保留原 profile），但在产物中显式留痕。

**验收：** 在至少 crc32、matmult-int、md5sum 上验证可重现的 IR 或明确的失败原因；reference/profile/CFU 原有 ELF、verify 和 cycle 测量不因 IR 导出而改变。

### P1 — 结构化静态特征提取

1. 优先使用 LLVM 的 LoopInfo、DominatorTree、ScalarEvolution、DebugLoc 等分析 API（或固定版本的 `opt` pass 输出），而非对文本 `.ll` 用正则猜测自然循环。按函数和 loop 层级统计 basic block、opcode 类别、读/写、调用、exit、依赖/归约线索；遇到不可判定的别名或 trip count 输出 `unknown`。
2. 区分源级 loop 与优化后 IR loop：展开、合并、向量化、内联可能改变数量和层级。保留 source location、模块/TU、函数符号、inline chain、LLVM loop header/depth；没有有效 `DebugLoc` 时记录 `unmapped`，不要造一条匹配。
3. 对结果按 schema 验证、设大小上限与确定性排序；IR 字节串留 artifact，摘要中记录单位与 provenance。新增针对属性声明、宏/内联、嵌套/unbraced/do-while、优化消除循环、无调试信息及多 TU 同名函数的 fixture 测试。

**验收：** 相同输入与工具链产生相同结构化特征；不支持的 opcode/结构不会被误报为已知计算量。

### P2 — 动态观测点 ↔ IR loop 对齐

1. 从 `observation_report` 取已经 applied 的 loop 点，以 workload-relative 文件、函数、源行/列、嵌套关系匹配 LLVM DebugLoc。候选唯一且位置可信才标记 `matched`；0 个为 `unmapped`，多个为 `ambiguous` 并保存候选与原因。宏展开、一个源循环拆成多个 IR loop 时允许明确的 1:N 关系，但不把一个动态计数复制成多个互斥测量。
2. 以 `obs id` 作为动态主键，挂接 profile 的 `cycles` / `calls` 与关联静态特征。排序沿用**实测**周期，不以静态 IR 长度覆盖 `hot_spot`；有 0 calls、缺失/溢出计数、未应用点不得参与动态排序。提供 reference/profile 总周期与开销提醒，但不把被插桩的 profile 周期当作正式 speedup 分母。
3. `extract_hotspot` 保留源函数为 software edit target，原始 loop ID/行号保持在 `hot_loop`；analysis 中附 IR 证据状态/映射置信度。对齐失败时照旧使用现有 per-loop 源码映射，不能阻塞 ISA/验证链路。

**验收：** matmult-int 等嵌套 loop 的链接稳定、inclusive 计数不相加；错位/歧义有显式测试；IR 开关 on/off 对原有热点选择与参考/cfu 验证无回归。

### P3 — 接入 Agent、缓存与 C-project 试点

1. 在 profile 后、`analyze_workload` 前增加一个可选的 IR 提取/关联步骤（或在 `extract_hotspot` 后调用纯函数），同时修改 LangGraph 和 `run_agent_python`；记录 `ir_result`、降级原因与 artifact。`prompts.py` 只给选中 loop 及少量对照候选的摘要、来源、置信度与未知项；`design_contract` 消费分析结论，严禁因 IR 猜测虚构 Roofline 精确数值。
2. 将 IR 的工具版本、target、编译 argv、源码/头文件与 flags hash、schema/extractor 版本纳入**独立 IR 缓存键**。`profile_cache.py` 的动态测量缓存维持原语义：IR extractor 变化不应强制重跑两次昂贵仿真；IR 变化只失效 IR artifact。缓存命中需要校验输入、schema 和证据存在。
3. C-project 仅在可重放 compile commands（如 `compile_commands.json`）且源码复制隔离、构建权限受控时启用静态 IR；缺少 per-loop 动态插桩时只标记 `static_candidate`，不可称 `measured_hot_loop`。后续若要把 per-loop 动态插桩扩展到任意 C-project，需独立解决构建命令、宏/模板、不受控 side effect、ABI 和回退策略，不与 IR MVP 捆绑。
4. 已添加 `--ir-features off|auto|require|approx`：目前默认 `off`，`auto` 为严格目标的 best-effort，`require` 仅用于显式实验，失败有独立原因；`approx` 显式接受简化 ISA 和交叉工具链头文件，不能标记为严格目标匹配。为批量实验记录 IR status、覆盖率、匹配率与额外构建时长。

**验收：** 无 LLVM 时 `auto` 可跑通原有 agent；有 LLVM 时 artifact、提示词、命中/失效、dry-run 与 Python/LangGraph 结果一致；目标测试与完整 `agents/tests` 套件通过。

## 验证矩阵与停止条件

| 样本 | 要验证的行为 |
| --- | --- |
| crc32 / md5sum | 单个与多个源 loop，观测 ID ↔ IR 映射、周期排序 |
| matmult-int | 嵌套 loop、inclusive 计数、外/内层特征区分 |
| sglib-combined / tarfind | 带属性函数声明、复杂控制流、循环定位 |
| 人工 fixture | 跨 TU 同名函数、宏展开、do/while、continue/break/return、优化消除与歧义 |
| C-project | compile commands 可得/缺失两种情况、静态候选不得冒充动态热点 |
| 无 LLVM/不匹配版本 | `auto` 显式降级；`require` 可审计失败；原 benchmark 验证不变 |

每阶段均用单元测试、RISC-V 源码语法/构建检查及可用时的 simulator smoke test 收口。若 IR 的 target/header 不能复现实际 workload、匹配歧义率高、或提取影响 reference/CFU 可比性，则暂停将其用于自动 ISA 决策，只保留 `partial/unavailable` 产物与现有 per-loop 路径；不把“有 IR 文件”视为功能完成。

## 交付顺序

1. P0 工具探测、命令捕获、只读 IR 导出与回退测试。
2. P1 版本化 schema、LLVM loop/函数特征、稳定摘要与 provenance。
3. P2 动态 per-loop 关联、歧义处理、热点选择不变性。
4. P3 提示词/状态/缓存/产物集成、C-project 可选试点和端到端回归。

完成定义：针对已测循环可以从报告中的 `obs id` 追到源代码、IR loop、工具链/编译 flags 与动态周期；静态/动态字段可区分、缺失证据不被填造，原 Embench correctness/speedup 和失败归因维持原有语义。
