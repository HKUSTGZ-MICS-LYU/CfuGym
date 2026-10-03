# CFU Design Agent：LLM × Benchmark 的 pass@1 / pass@3 / pass@5 实验矩阵设计

状态：已实现 —— `agents/experiments/run_matrix.py`（配套单测
`agents/tests/test_experiment_matrix.py`，35 项，不调用付费模型）。
范围：在现有 agent flow（`agents/run_cfu_agent.py` 全校验梯）下，对比 3 个 LLM 在 3 个
benchmark 上的成功率曲线 pass@{1,3,5}，每个实验单元重复 5 次（可调 3–5）。

---

## 1. 研究问题与判定口径

**问题**：在固定的 CFU 设计 agent flow（相同 prompt、相同校验梯、相同工具预算）下，
不同 LLM 把一个 workload 推进到 `status=complete`（全部 rung 通过：ISA 契约 → HW →
SoC 生成 → 一致性测试 → SW → 编译 → 仿真正确性 + ≥1.05x 加速 → Yosys 面积）所需的
尝试次数分布如何？

**一次 run 的成功判定**：只有 CLI `status=complete`（exit code 0）算 pass。
`planned` / `failed` / `blocked` 一律不算。

**attempt（尝试）定义**：agent 的一次"完整穿越校验梯"机会。
`--max-iters N` 的语义是 *N−1* 次 repair 预算（`graph.py: max_repairs = max_iters - 1`），
即 `--max-iters 5` = 首次尝试 + 4 次修复 = **5 次尝试**，恰好覆盖 pass@1…pass@5。
这就是"用 max-iters 5 覆盖 pass@1-5"的含义：

```
attempt_index = repairs_used + 1        # 成功时落在 1..5；失败时记 repairs_used + 1 并标记 fail
pass@k（单次 run 的贡献）= 1 当且仅当 run 成功 且 attempt_index <= k
```

- `pass@1` = 一次通过（零修复），是"首试成功率"；
- `pass@3` = 在 3 次尝试预算内收敛；
- `pass@5` = 在 5 次尝试预算内收敛（= 本次 run 的 complete 率）。

**每个 (model, benchmark) 单元 R 次重复后的两个估计量都算**（同一份数据，两种口径）：

1. **预算口径（主指标）**：`pass@k_budget = (1/R) * Σ 1[t_i ≤ k]`，
   `t_i` 是第 i 次重复的 `attempt_index`（失败 = ∞）。衡量"给 k 次尝试预算的成功概率"。
2. **采样口径（副指标，标准 pass@k）**：把 R 次完整独立 run 当作 R 次采样，
   `pass@k_sample = 1 - C(R−c, k) / C(R, k)`，`c` = complete 的 run 数。
   衡量"独立采样 k 个完整 solution 至少一个可用的概率"。

> 口径说明（写进报告）：run 内的 repair 不是 i.i.d. 采样（第 j 次尝试以第 j−1 次失败为条件），
> 所以主指标是"预算内收敛率"，采样口径才是文献标准 pass@k。另外 repair prompt 里带有
> `(repair i/N)` 总预算字样（`graph.py` repair 文案），因此从 max-iters=5 的轨迹反推
> k=1/3 的结果是近似值。若要严格，可另做一组 `--max-iters 1/3` 的对照（见 §12 备选）。

---

## 2. 实验因子矩阵

| 因子 | 水平 | 说明 |
| --- | --- | --- |
| LLM（3） | `deepseek-pro`、`glm-5.2`、`gpt-5.6-terra` | 三个不同厂商/家族 |
| Benchmark（3） | `crc32`、`matmult-int`、`md5sum`（均 embench-iot） | 三种热点内核形态 |
| attempt 预算 k（3） | 1 / 3 / 5 | **不额外跑**：由单次 `--max-iters 5` run 的 `attempt_index` 推出 |
| 重复 R | 5（配置项，下限 3） | 每个单元独立 run 5 次 |

**总规模**：3 × 3 × 5 = **135 次全梯 run**（R=3 时 81 次）。
试点阶段先跑 3 × 3 × 1 = 9 次。

### 2.1 模型选择（3 个）

| matrix 名 | `--llm-config` | 模型 | api_mode | 选择理由 |
| --- | --- | --- | --- | --- |
| `deepseekpro` | `deepseek-pro` | DeepSeek-V4-Pro | auto | 国产推理模型代表，DeepSeek 家族 |
| `glm52` | `glm-5.2` | GLM-5.2 | chat | 智谱家族；chat 全程回传 `reasoning_content`，失败归因可视性最好 |
| `gpt56terra` | `gpt-5.6-terra` | gpt-5.6-terra | auto | OpenAI 前沿基线 |

- 三个模型取自三个不同家族，避免同家族重复；`agents/llm_configs/` 里还有
  `kimi-k3`（Moonshot 家族、同样全程有 `reasoning_content`）可作为 `gpt-5.6-terra`
  的替换/追加水平，矩阵是配置驱动的，换水平只改一行。
- 已排除 `deepseek-flash2.json.dontuse`（作者显式标记弃用）。
- 已知观测性差异（不影响评分，影响事后分析）：DeepSeek 网关既不回 chat reasoning
  也无 `/responses`（thinking 抓不到）；GPT-5.6 只有 Responses summary。
  **失败环节记录不依赖 reasoning**（见 §9/§10：全部来自 gate/log 证据），reasoning
  只是分析素材。

### 2.2 Benchmark 选择（3 个，全 embench-iot）

选择标准：(a) 热点内核形态互补（难度梯度）；(b) 走同一套 harness（`--embench`，
Embench 自带 verify oracle + 参考周期），保证跨模型可比；(c) 已有可行性证据。

| matrix 名 | `--embench` | 内核形态 | 选择理由 |
| --- | --- | --- | --- |
| `crc32` | `crc32` | 查表 + 串行位运算，偏访存 | **难**档。本机已有 3 次完整 run 证据（1 complete @1.05x 擦线、2 次死于 `hw_conformance`），难度标定最好 |
| `matmultint` | `matmult-int` | 整数 MAC、计算密集 | **易**档。CFU 最经典的加速形态，能拉开模型差距（比拼数据重用/向量驻留） |
| `md5sum` | `md5sum` | 位运算轮函数、寄存器级重用 | **中**档。计算密集但结构不规则，考验 ISA 设计能力 |

- 备选（若试点发现 build/仿真时间或难度不合适可换）：`depthconv`（卷积 MAC）、
  `nettle-sha256`（位运算，sha 家族）、`huffbench`（分支/表混合）、
  `statemate`（控制流为主，最难档替代 crc32）。
- 刻意不混入 `--task`（自然语言）和 `--project-root`（C 工程）两种输入形态：三种
  input mode 的 harness、判定口径不同，混进来会把"模型差异"和"输入形态差异"混淆。
  输入形态可作为后续第二阶段的独立矩阵轴。

---

## 3. 固定控制变量（所有 run 完全一致）

| 项 | 取值 | 备注 |
| --- | --- | --- |
| 运行形态 | `--apply-patch --run-commands`（autonomous 全梯） | 不加 `--skip-sim` / `--skip-yosys` |
| `--max-iters` | 5 | = attempt 预算 5（1 首试 + 4 repair） |
| `--loop` | **关闭** | 否则成功后继续优化，pass 时点不干净 |
| `--visibility` | `standard` | 历史 run 用过 `extended`，必须冻结一个 |
| `--repair-routing` | `auto`（agent 路由） | 这是 agent flow 的组成部分 |
| `--software-change` | `auto` | 同上 |
| 工具预算 | 默认 `--max-tool-rounds 6 --max-tool-calls 24` | 不动 |
| `--profile-cache` | `on` | 参考/画像周期跨 run 复用（实测会 hit），把 harness 噪声从对比中去掉 |
| `temperature` | **0.7（实验配置强制写入）** | 现有 config 未设 temperature → 默认 0；temp=0 下重复实验退化成同一个解，pass@k 无意义。这是本实验对现有 config 的唯一改动 |
| 其余 thinking/reasoning_effort | 沿用各 config 的 `thinking: true, reasoning_effort: high` | 三模型保持同一 effort |
| 环境 | **所有外部调用统一在 conda 的 `mico_env` 环境内执行**：`conda run -n mico_env --no-capture-output python3 -B agents/run_cfu_agent.py …` | harness 默认 `--conda-env mico_env`（sbt/工具链/仿真器都在这个环境里）；`--conda-env ""` 可退回裸 python3 |
| 版本锚点 | 记录 `git rev-parse HEAD` + dirty 标记 + 三个实验 config 的 sha256 | 写入 `experiment.yaml` |

温度配置落地方式：`agents/llm_configs/` 已被 gitignore（只跟踪 `.md`），由 harness 生成
三个实验专用 config（不碰原文件）：

```
agents/llm_configs/<exp_id>-deepseek-pro.json   # 原 deepseek-pro.json + "temperature": 0.7
agents/llm_configs/<exp_id>-glm-5.2.json
agents/llm_configs/<exp_id>-gpt-5.6-terra.json
```

---

## 4. Run 命名方案（显式、有意义、可反推实验坐标）

框架侧 `--run-id` 只允许 `[A-Za-z0-9_-]+`（`artifacts._safe_run_id`），run 产物落在
`agents/generated/runs/<run_id>/`。harness 显式传 `--run-id`，命名模板：

```
<exp_id>-<model>-<benchmark>-r<RR>-mi<K>
```

| 段 | 含义 | 取值 |
| --- | --- | --- |
| `exp_id` | 实验批次 | `pk5-<yyyymmdd>`，如 `pk5-20260212` |
| `model` | 模型 slug | `deepseekpro` / `glm52` / `gpt56terra` |
| `benchmark` | benchmark slug | `crc32` / `matmultint` / `md5sum` |
| `r<RR>` | 重复序号（1 起） | `r01`…`r05` |
| `mi<K>` | attempt 预算（max-iters） | `mi5` |

示例：`pk5-20260212-glm52-matmultint-r03-mi5`。
基础设施重跑追加 `-infra<N>`（如 `...-mi5-infra1`），与模型失败严格区分（§10.3）。
harness 侧实验输出目录按同样的 run name 组织，实现"目录名 ↔ 实验坐标"双向可读：

```
agents/generated/experiments/<exp_id>/
  experiment.yaml                    # 矩阵定义、git commit、config sha256、判定口径
  runs/<run_name>/
    cmd.txt  stdout.log  stderr.log  exit_code
    record.json                      # 单 run 采集结果（§9）
    failure.json                     # 仅失败 run（§10）
  results/runs.csv                   # 每 run 一行
  results/failures.csv               # 每个失败环节一行
  results/pass_at_k.csv              # 每单元一行，主/副指标
  results/summary.md                 # 人读汇总（矩阵表 + 失败环节分布）
```

---

## 5. 脚本设计：`agents/experiments/run_matrix.py`

单文件即可（配套 `agents/tests/test_experiment_matrix.py` 单测，不需要付费模型）。

```bash
python3 agents/experiments/run_matrix.py \
  --exp-id pk5-20260212 \
  --models deepseek-pro,glm-5.2,gpt-5.6-terra \
  --benchmarks crc32,matmult-int,md5sum \
  --repeats 5 --max-iters 5 --temperature 0.7 \
  --conda-env mico_env --python "python3 -B" \
  --jobs 2 --timeout 10800 \
  [--dry-run] [--collect-only] [--aggregate-only] [--only glm52,matmultint,r03]
```

**流程**：

1. **预检**：对每个实验 config 跑框架自带连通性检查 `--test-api`（1 次小请求），
   不过就整批不启动；结果写入 `experiment.yaml`。
2. **矩阵展开 + 固定种子打乱**：展开 135 个 run，用固定 seed 打乱后**按 round-robin
   调度**（外层 repeat 序号、内层单元格）——同一单元的 5 次重复分散在不同时间段，
   把网关负载/时段漂移摊平到所有单元，而不是让某个单元独占某个时段。
3. **逐 run 执行**：拼装完整命令（§8），`subprocess` 捕获 stdout/stderr 到
   `runs/<run_name>/`，记录 exit code 与墙钟时间；单 run 设 `--timeout`（默认 3h），
   超时杀掉并按 infra 处理。
4. **断点续跑**：`record.json` 已存在且 `terminal=true` 的 run 跳过；`--only` 支持补跑
   指定单元格/重复。失败的模型 run **不**由 harness 重试（它就是数据）。
5. **采集 + 聚合**：每跑完一个 run 立即采集（崩溃不丢数据），批次结束后聚合出
   `pass_at_k.csv` / `failures.csv` / `summary.md`。

**并发**：默认 `--jobs 2`；run workspace 彼此隔离（`agents/generated/runs/<run_id>/workspace/`），
只有 sbt/sim 侧的共享状态需要互斥（见下节）。

### 5.1 并发与锁：sbt / RTL 仿真的互斥（关键实现点）

**锁的来源**：agent 的所有重活都以 `sbt` 为可执行入口（策略 `agent_access_policy.yaml`
的 `soc_generate_agent` / `soc_sim` / cost 的 `GenAgentCfu` 全是 `executable: sbt`），
`sbt` 进程之间会争：

- 仓库根共享的 `target/`（增量编译状态）与 `project/target`、sbt boot 锁；
- **仓库根共享的 `simWorkspace/`**（`MiCoSocSim` 在 sbt 里跑 SpinalSim/Verilator；
  访问策略里 `**/simWorkspace/**` 被 deny 也印证它不是 run 隔离的）；
- ivy / coursier 缓存锁。

**解法：`sbt` PATH shim + `flock` 全局互斥锁，粒度是"每次 sbt/仿真调用"**，
而不是整个 run——LLM 调用、`make` 软件编译、Yosys（各自 run 内输出目录）继续跨 run
并行，只有 sbt/RTL 仿真串行：

1. harness 在 `<exp_dir>/bin/` 生成 `sbt` shim（`write_sbt_shim`），内容要点：
   - 防递归哨兵（`CFU_EXP_SBT_SHIM_ACTIVE`，异常 PATH 时以 97 明确报错而不是 fork 炸弹）；
   - 从 PATH 里**逐条目过滤掉所有** shim 目录出现后 `command -v sbt` 解析出**真实 sbt 绝对路径**
     （系统或 conda 环境内的都行——harness 自己可能根本看不到 mico_env 里的 sbt），
     并显式拒绝解析回 shim 自身。**注意：这里不能用 `${PATH//:$SHIM_DIR:/:}` 模式替换**——
     harness 与 conda wrapper 会各前置一次 shim 目录（`<shim>:<shim>:…` 相邻重复），
     模式替换在相邻重复时会残留一份（第一个匹配吃掉了第二个需要的 `:`），sbt 解析回 shim、
     哨兵误报 recursion、退出 97，每次 sbt/仿真全灭（实测踩坑，已修 + 回归测试覆盖）；
   - `exec flock -x -w <timeout> <lock> <real-sbt> "$@"` 持锁执行。
   - shim 自身报错（`sbt lock shim:`）被归入 `infra_error`（走 `-infra<N>` 重试），
     永远不会记成模型失败。
2. **conda 激活的 PATH 遮蔽问题（必须处理，否则锁静默失效）**：`conda run -n mico_env`
   激活环境时把 `<env>/bin` 插到 PATH 最前，会把 shim 顶下去，`sbt` 直接解析到真 sbt，
   锁被绕过。因此 harness 组装命令时在 conda **内部**再包一层 `bash -c`，在激活完成之后
   重新把 shim 目录置顶：
   ```bash
   conda run -n mico_env --no-capture-output \
     bash -c 'export PATH="$1:$PATH"; shift; exec "$@"' cfu-exp-sbt-shim <shim_dir> \
     python3 -B agents/run_cfu_agent.py …
   ```
3. 锁文件默认 `<out-root>/.locks/sbt.lock`，**跨实验批次共享**，两个 harness 进程
   同时开跑也不会并发 sbt；等待上限 `--sbt-lock-timeout`（默认 4h，覆盖长仿真排队）。
4. `--no-sbt-lock` 可关掉（此时应 `--jobs 1`）；`flock(1)` 缺失时 harness 拒绝以
   `--jobs > 1` 启动。

代价（可接受、已知）：两个并发 run 交替使用不同 overlay 触发 sbt 增量编译状态互换，
会多付一些重编译时间（正确性不受影响）；换来的是 LLM 调用与非 sbt 环节的并行。

---

## 6. 单 run 命令模板

```bash
conda run -n mico_env --no-capture-output \
  bash -c 'export PATH="$1:$PATH"; shift; exec "$@"' cfu-exp-sbt-shim <shim_dir> \
  python3 -B agents/run_cfu_agent.py \
  --embench crc32 \
  --llm-config pk5-20260212-deepseek-pro \
  --run-id pk5-20260212-deepseekpro-crc32-r01-mi5 \
  --max-iters 5 \
  --apply-patch --run-commands \
  --visibility standard \
  --repair-routing auto \
  --software-change auto \
  --profile-cache on
```

（无 `--loop`、无 `--skip-*`；`--workdir` 为仓库根；`bash -c` 包装即 §5.1 的 shim 置顶，
`--no-sbt-lock` 时退化为直接 `conda run … python3 -B agents/run_cfu_agent.py …`。）

---

## 7. pass@k 汇总口径（`pass_at_k.csv`）

每行 = 一个 (model, benchmark) 单元：

| 列 | 含义 |
| --- | --- |
| `runs` | 完成的重复数 R |
| `complete_runs` | `status=complete` 的次数 c |
| `pass_at_1/3/5_budget` | 主指标 `(1/R)·Σ1[t_i ≤ k]` |
| `pass_at_1/3/5_sample` | 副指标 `1 − C(R−c,k)/C(R,k)`（k > R 时置 1 或 NA） |
| `attempt_index_median` | 成功 run 的 `repairs_used+1` 中位数 |
| `first_try_rate` | = pass@1_budget |
| `pass_at_k_ci95` | 对 R 次重复 bootstrap（10000 次）的 95% CI（R=5 时很宽，如实报告） |
| `tokens_per_complete_mean`、`wall_h_mean` | 成本维度 |
| `speedup_x100_median`、`yosys_cells_median` | 成功 run 的质量维度（副结果） |
| `failure_classes` | 该单元失败环节分布摘要，如 `hw_conformance:2, sim_timeout:1` |

---

## 8. 数据采集：单 run `record.json` 与 `runs.csv`

**采集来源映射**（按优先级，全部留证据路径，解析失败写 `parse_warnings` 而不是丢字段）：

| 字段组 | 来源 |
| --- | --- |
| `status`、`stopped_at`、`summary:`（cycles/speedup/yosys）、`token_usage:`、逐 stage 进度 | CLI stdout（`status=`、`stopped_at=`、`summary:` 块，`cli.py` 稳定打印） |
| `llm_request.config/model/base_url`、`visibility`、`profile_cache`、`tool_policy`、artifact 清单（含 sha256） | `<run>/run.yaml` |
| 每 rung 的完整 gate dict（`ok/errors/attribution/repair_decision/software_decision/verdict`）、`repairs used: N/M (iteration=N)` | `<run>/report.md` 的 `## Gates` / `## Validation Ladder`（`ast.literal_eval` 解析） |
| sim 证据（reference/cfu cycles、`speedup_x100`、`min_speedup_x100`、`attribution`） | `<run>/10_validation/embench.yaml`、`10_validation/sim.yaml` |
| 编译/仿真失败日志 | `10_validation/hw_build.yaml`、`sw_build.yaml`、`09_implementation/{hw,sw}_repair_feedback.txt` |
| conformance 明细（`verdict`、`oracle_failure`、cases） | `09_implementation/conformance/`、gate `hw_conformance` |
| 面积 | `11_cost/result.yaml` |
| repair 路由决策 | `12_review/sim_repair_decision.yaml`、gate `repair_decision` |

**`runs.csv` 列**（每 run 一行）：

```
exp_id, run_name, run_id, model_config, model, benchmark, suite, repeat, max_iters,
status, exit_code, stopped_at, failure_class, failure_rung,
repairs_used, attempt_index, pass_at_1, pass_at_3, pass_at_5,
sim_ok, reference_cycles, cfu_cycles, speedup_x100,
yosys_cells, yosys_wire_bits, token_calls, token_in, token_out, token_total,
wall_seconds, profile_cache_status, visibility, infra_retries, parse_warnings,
run_dir, report_path, started_at, finished_at
```

其中 `attempt_index = repairs_used + 1`（成功 run 用于 pass@k；失败 run 也记录，表示
"预算耗尽时的尝试序号"）；`pass_at_k` 三列预计算成 0/1/空（失败 run 为空）。

---

## 9. 失败环节记录（"记录失败环节"的落地）

### 9.1 失败环节分类（failure_class ← 判定规则）

| failure_class | 判定（`stopped_at` / gate 证据） | 语义 |
| --- | --- | --- |
| `input_or_policy` | `stopped_at=prepare_benchmark` 或 `load_context.denied` 非空 | 输入/权限问题（实验事故，不是模型失败，见 9.3） |
| `isa_contract` | `stopped_at=isa`（含 `status=blocked`） | ISA/契约 lint 不过 |
| `no_deliverable` | `implementation_hw/sw` gate `errors` 含 `no EDIT_BEGIN/FILE_BEGIN/PATCH_BEGIN deliverable` 或 `tool-intent preamble` | 模型只输出了计划/空 patch |
| `hw_build` | `stopped_at=hw_build` | Scala 编译 / SpinalHDL elaboration / Verilog 生成失败 |
| `hw_conformance` | `stopped_at=hw_conformance`（子类：`oracle_failure=true` → 指令语义错误；false → 编码/覆盖失败） | 差分微测试失败 |
| `sw_build` | `stopped_at=sw_build` | 内核编译失败 |
| `sim_correctness` | `stopped_at=sim` 且 `attribution` 含 verify fail / fail symbol | 结果算错 |
| `sim_timeout` | `attribution` 含 `timed out` | 挂死/死循环 |
| `sim_speedup_floor` | sim gate ok=false 且 `speedup_x100 < min_speedup_x100`（105） | 正确但没加速 |
| `sim_no_change` | sim gate 报 "no design change applied" | 根本没改设计 |
| `cost` | `stopped_at=cost` | 独立 CFU RTL 生成/Yosys 失败 |
| `infra_timeout` / `infra_error` | harness 判定（9.3） | 非模型原因 |

每条失败都带 `budget_exhausted: true/false`（预算耗尽才停，还是 `--max-iters 1` 式的一次即停）。

### 9.2 失败 run 的 `failure.json`（结构化失败环节档案）

```json
{
  "run_name": "pk5-20260212-glm52-crc32-r04-mi5",
  "terminal": {
    "status": "failed",
    "stopped_at": "hw_conformance",
    "failure_class": "hw_conformance",
    "failure_rung": "hw_conformance",
    "budget_exhausted": true,
    "repairs_used": 4,
    "gate": {"ok": false, "verdict": "fail", "oracle_failure": true, "errors": ["..."]},
    "attribution": "Embench verify failed ...",
    "errors_tail": "…最后 60 行日志…"
  },
  "episodes": [
    {"round": 1, "kind": "repair", "trigger_rung": "hw_conformance", "target": "implement_hw",
     "decision_source": "fallback", "feedback_file": "09_implementation/hw_repair_feedback.txt",
     "outcome": {"rung": "hw_conformance", "ok": false}}
  ],
  "evidence": {"report": "...", "run_yaml": "...", "sim": "...", "conformance": "...",
               "llm_raw": ["09_implementation/llm_raw/hardware_2.txt"]},
  "spent": {"token_total": 512340, "wall_seconds": 2740}
}
```

`episodes` 覆盖每次 repair / `route_repair` 路由决策（`12_review/sim_repair_decision.yaml`：
`target/root_cause/reason/source=llm|fallback|fixed`），使"修了什么、谁决定修哪儿、结果如何"
完整可回溯；`failures.csv` 由这些档案摊平而成（run × 失败环节/修复回合）。

### 9.3 模型失败 vs 基础设施失败

harness 重试只针对 infra：API 5xx/连接失败/网关抽风、sbt/Verilator 进程崩溃、磁盘/超时。
特征判定 + 人工复核标记 `infra_suspect`。infra 失败**换新 run_id（`-infra<N>` 后缀）重跑
最多 2 次**；模型产出的失败（编译错、验证错、超时……）永不重跑。两类在 `runs.csv` 里
分开统计，`summary.md` 单列一张"infra 事故表"，防止把网关抖动算成模型能力。

---

## 10. 执行计划

| 阶段 | 内容 | 规模 |
| --- | --- | --- |
| P0 预检 | 3 个实验 config `--test-api`；每个 benchmark 用最便宜模型跑 1 次 `--max-iters 1 --skip-yosys` 的短烟测（只验 harness/编译/仿真链路，不进统计） | 3 + ≤3 |
| P1 试点 | 完整矩阵 × 1 次重复，跑通采集/聚合，实测单 run 墙钟与 token 成本 | 9 |
| P2 正式 | 补齐到 R=5（round-robin 调度，断点续跑） | +126 → 135 |
| P3 聚合 | `pass_at_k.csv` + `summary.md` + 失败环节分布；（可选）对 `crc32` 这类擦线单元加做 R=10 收窄 CI | — |

墙钟粗估（P1 实测后修订）：单 run 全梯 20–60 min → 135 run 顺序执行约 45–135 h，
`--jobs 2` 减半。token 成本以 P1 的 `token_usage` 外推，先出预算再放量。

**备选增强**（不在本期范围）：
- 严格反推对照组：对同一批单元补 `--max-iters 1` / `--max-iters 3` 各 R 次，
  验证"由 mi5 轨迹推 k=1/3"的近似误差（§1 口径说明）。
- temp=0.2 敏感性对照（验证结论对温度不敏感）。
- 第二矩阵轴：`--task`（自然语言）/`--project-root`（C 工程）输入形态。

---

## 11. 风险与缓解

| 风险 | 影响 | 缓解 |
| --- | --- | --- |
| 现有 config 默认 temperature=0 | 重复之间几乎同解，pass@k 失真 | 实验 config 强制 temperature=0.7（§3），并在报告注明 |
| repair prompt 含 `(repair i/N)` 总预算 | mi5 轨迹反推 k=1/3 是近似 | 报告注明；可选严格对照组（§10 备选） |
| benchmark 难度不均（crc32 可能全员低 pass） | 单元格差异被难度吞掉 | 难/中/易三档是刻意设计；P1 看分布后可换备选 benchmark |
| 仿真墙钟长、Yosys/sbt 重 | 135 run 时间成本 | `--profile-cache on`、`--jobs 2`、断点续跑、P1 实测后定放量 |
| **并发 run 争 sbt / `simWorkspace/` / `target/` 锁** | 构建损坏、仿真互踩、结果不可信 | sbt lock shim：调用粒度 `flock` 全局互斥（§5.1）；conda 激活 PATH 遮蔽由 `bash -c` 置顶修复 + 防递归哨兵；锁文件跨批次共享 |
| 网关抖动被算进模型失败 | 结论偏差 | infra 分类 + `-infra<N>` 重跑 + 事故表（§9.3） |
| R=5 的置信区间宽（每次重复 20pp） | 只能看大差异 | 如实报 bootstrap CI；关键单元格追加 R=10 |
| `report.md` 的 Gates 是 Python dict 文本 | 解析脆 | `ast.literal_eval` + 多来源兜底 + `parse_warnings` 字段，不静默丢数据 |
| 并行写 profile cache | 数据竞争 | 框架已有内容寻址 key；P0 烟测验证并行写一次 |
