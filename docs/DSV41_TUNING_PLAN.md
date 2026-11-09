# DS4.1/sm_75 implementation playbook

Status: source cross-check at `c62d5e3a`, 2026-11-07. Documentation only. No GPU work was run for this
revision. This file supersedes the earlier tuning plan and cross-checks `docs/SM75_TECHNIQUE_AUDIT.md`.
The audit is evidence to check, not instructions to copy. Where they disagree, the resolution below follows the
current source.

The target is DeepSeek-V4.1-Flash on 4 x RTX 2080 Ti 22 GB (sm_75). A few audit items belong to the Qwen
engine, the Terminal, serving, or packaging instead; their ownership is named explicitly. Numbers quoted below
are old measurements, not acceptance evidence for a new run. The old DeepSeek measurements used the machine,
driver, toolkit and pack named in `docs/SM75_TECHNIQUE_AUDIT.md`; keep the 2.00bpw and 2.25bpw results apart.

## 0. Contract for the implementing agent

Work one item at a time, in the order in section 4. Read the whole item before editing. Its **Gate** says whether
it may start. Its **Stop** rule is mandatory. Do not turn a deferred or rejected item into an extra experiment.

Every item has the same contract:

- **Sites** use symbol names first and line numbers at `c62d5e3a` second. Re-locate the symbol before editing;
  do not paste by line number.
- **Change** is the largest allowed implementation. Do not change a default, public signature, or additional file
  unless the item says so.
- **Checks** separate no-GPU checks from checks that need the target card. A build is not a performance result.
- **Accept** states the evidence required to keep the change. **Stop** states when to revert the item and record
  the failure. **Next decision** states what the executor does after the outcome — which item follows a pass,
  which item follows a fail, and which outcome closes the line. Every runbook has one; when it says "close",
  nothing further is owed on that item. Revert only the item's own edits; never discard another person's work.
- New numerics-changing behaviour is opt-in. Unset environment and arguments must remain byte-identical to the
  previous release unless an item explicitly implements an already measured architecture default.
- Do not edit `third_party/llama.cpp`. Use a build-tree overlay or an upstream patch for the MMQ experiment.
- Do not run GPU work until the user has switched the box to that work. `run-tests.sh` kills `strata-dsv41`
  processes; ask before using it on a box that may be serving a user.
- Record the commit, diff, exact commands, engine build, pack, hardware, clocks, run directories, medians, ranges,
  quality result and failures in `~/StrataTuneNotes.md`. Static analysis is labelled **unconfirmed static
  analysis** until executed.

Before each implementation item:

```sh
cd /home/jurek/Strata
git status --short
git rev-parse --short HEAD
rg -n "THE_SYMBOL_NAMED_BY_THE_ITEM" src include tests tools serve strata_control packaging CMakeLists.txt
```

Stop and ask if the named symbol has moved in meaning, the worktree would be overwritten, the required card or
runner is absent, or a command would mutate machine settings without explicit approval.

Dispositions used below:

- **code now**: implementation and no-GPU tests may start; target-card acceptance can still be gated.
- **measure first**: run the named cheap probe before writing the feature.
- **experiment**: implement only after its parent gate passes; always opt-in.
- **defer**: a hardware, runner, dependency, or workload prerequisite is absent.
- **closed**: dead on the reference box, already implemented, or rejected as written. Do not implement without a
  new fact that invalidates the stated reason.

## 1. Coverage and dispositions

### 1.1 Original tuning-plan items

| Plan item | Audit | Disposition | Runbook | Source-checked resolution |
| --- | --- | --- | --- | --- |
| 2.1 split-KV decode attention | 1 | code now; GPU acceptance gated | 5.1 | Gate unset default on **sm_75 only**, not every major-7 card. sm_70 is unmeasured. Keep HIP and Metal off by default and explicit `DS4_ATTN_SPLIT=0|1` authoritative. Use per-device state. |
| 2.2 FAT and attention-slab profile | 12 | code now as `DS4_DEFAULTS=075`; default unchanged | 5.2 | Implement in `ds4_experts_rule.hpp` and its two call sites. Do not change `defaults_026()` or Qwen-only `src/core/nv_defaults.cpp`. |
| 2.3 sm_75 release SASS | 15 | code now; packaged run gated | 5.3 | Change both `packaging/build_dsv41_cuda.sh` and the DS4 default in `CMakeLists.txt`; changing only packaging leaves hand builds uncovered. |
| 2.4 `--threads on` | none | measure first on target box | 5.4 | Configuration experiment only. Async decode is already active; this tests device-to-device pipelining. |
| 2.5 expert-stage profile split | supports 2, 3, 7 | code now | 5.5 | The old mark-insertion sketch would corrupt stage accounting. Use independent event spans; keep stage 3 as the total and print the breakdown separately. |
| 2.6 F16 accumulator | 3 | experiment after 5.5 | 5.6 | A runtime switch needs two template instantiations and typed output scratch; changing only the fragment type does not compile. Default off. |
| 2.7 int8 MMA set kernel | 2 | experiment after 5.5 and lab preflight | 5.7 | Implement in DS4's `launch_set` family. Do **not** call whole-MoE `fused::experts_native` there; its API, layout, and Turing type gate do not match `SetArgs`. Reuse its Turing MMA techniques. |
| 2.8 NVLink peer hand-off | part of 8/9 diagnostics | defer | 5.8 | Needs at least 377.6 MB a card plus headroom, peer capability on each real hop, and a fallback. The reference box has about 280 MiB free. |

### 1.2 All nineteen audit items

| Audit | Plan | Disposition | Resolution and next action |
| --- | --- | --- | --- |
| 1 split-KV attention | 2.1 | code now | Section 5.1. Kernel evidence exists; end-to-end acceptance still needs same-day measurement. |
| 2 i-quant expert GEMM | 2.7 | experiment | Section 5.7. The audit's proposed direct `experts_native` call is rejected; port the arithmetic into DS4's set API. |
| 3 F16 accumulator | 2.6 | experiment | Section 5.6. Profile first, then quality-gated opt-in implementation. |
| 4 Turing vendored-MMQ tile | none | defer | Section 6.1. First prove that the vendored MMQ fallback is material. Then measure a build-tree overlay; never edit the vendored file. |
| 5 narrow-M 2–24 kernel | dead-items table | closed on this box | Section 7. DSpark is off. Current `MvJob` also rejects `ncol > 16`, so the audit's 24-column statement is not today's dispatch. Revisit only with an active verify workload and a profile. |
| 6 all-reduce `__restrict__` | dead-items table | defer | Section 7. `nm == 1` on this pack/box; no all-reduce call executes. Needs a compatible tensor-parallel box and output-equivalence gate. |
| 7 CUDA graphs | none | measure first | Section 6.2. Profile host launch gaps first. Do not capture asynchronous host-flag waits or changing expert-cache addresses. |
| 8 topology at launch | none | code now, diagnostics only | Section 6.3. Add read-only compute-capability, PCIe and topology capture to the tuner report, with mocked tests. |
| 9 persistence and locked clocks | none | explicit benchmark procedure only | Section 6.3. Report current state. Never set persistence or clocks silently. Clock changes require user approval and restoration. |
| 10 interleaved GEMV 5–8 table | none | doc fix now; table deferred | Section 6.4. `kIlRows_75` already exists and `docs/MMVQ_IL_TABLE.md` is stale. Add `kIlRows58_75` only from parity measurements on an active 5–8-column workload. |
| 11 sm_75 tuner knowledge | none | code now, conservative | Section 6.5. Propagate NVIDIA compute capability into `Context`; offer only the bitwise Qwen sm_75 series as an explicit exact arm. Do not invent DS4 tuner flags. |
| 12 per-card DS4 defaults | 2.1, 2.2 | covered | Sections 5.1–5.2. Split is per-device; FAT/slab are an explicit 075 profile. No new global default table and no Qwen `nv_defaults.cpp` edit. |
| 13 warp-specialised experts | none | rejected as written | Section 7. `moe_ws` is the GP100 **weight-stationary XMAD** family, not a Turing warp-specialisation port. The real `native_ws_kernel` is a different sm_80 path. No gate relaxation. |
| 14 compute-sanitizer CI | none | local runbook now; CI deferred | Section 6.6. There is no CUDA/self-hosted sm_75 workflow runner. Add CI only when a real runner exists; never add an always-skipping green job. |
| 15 sm_75 release SASS | 2.3 | code now | Section 5.3. Build inspection and a real 2080 Ti package smoke test are both required. |
| 16 GDN tensor-core scan | none | experiment, large | Section 6.7. Turing needs two k8 MMA steps **and** a new shared-memory plan. Existing plans need 97,808–160,784 B; sm_75 has 65,536 B opt-in. Use a separate opt-in so the serve stack cannot enable it silently. |
| 17 capability-aware plan/chunk rule | none | split decision | Section 6.8. Reject compute capability in byte-only `make_plan`: it models Qwen memory, not DS4 speed. A DS4 chunk branch waits for a measured candidate; add no speculative signature churn. |
| 18 serve batch capability line | none | code now, log only | Section 6.9. Reuse the existing `compute_cap` probe and print the active card class. Do not change which stacks are enabled. |
| 19 fused all-reduce + post-norm | dead-items table | defer | Section 7. Same `nm == 1` blocker as audit 6, plus the hyper-connection ordering invariant. |

## 2. Reference-box facts that control scope

Do not re-derive these on the 4 x 2080 Ti DeepSeek box. Re-check them on any other box.

1. **`--tp 4` becomes tp 1.** `src/dsv41/ds4_model.cu:2200-2218` requires
   `(n_ff / tp) % blk == 0`. Here `n_ff = 2304`, the IQ2 block has 256 values, and `2304 / 4 = 576` is not a
   multiple of 256. `DS4_TP_ALIGN` is a separate later test and cannot fix this. Startup prints the fallback.
   This is a layer pipeline, not tensor parallelism.
2. **Therefore `nm == 1` for every layer.** All five all-reduce call sites are behind `if (nm > 1)`:
   `src/dsv41/ds4_model.cu:8335, 8461, 8489, 10282, 10406`. All-reduce, reduce-scatter, `DS4_AR_F16`,
   `DS4_RING_MB` tuning for collectives, three-way expert sharding, and fused post-normalisation are dead here.
3. **Decode is serial unless `--threads on` is added.** `par = o.threads || tp > 1` at
   `src/dsv41/ds4_model.cu:2223`; dispatch is at `:9737-9739`. Prompt chunks can still use workers because
   `prompt_par` examines raw `o.tp` at `:2227-2230`.
4. **DSpark auto does not load on this pack/box.** The gate at `src/dsv41/ds4_model.cu:2338-2348` compares every
   routed-expert byte with 1.5 times free room. The recorded case is 156.3 GB against about 88 GB, so the drafter
   is dropped. Decode has `T = 1`; `DS4_DSPARK_*`, `--dspark-block`, `--dspark-depth`, `DS4_SPEC_*`, and the
   5–16-column verify kernels have no live workload. `--dspark on` is a new unmeasured experiment, not a fix for
   this plan.
5. **Async decode is already active.** On CUDA, `sync_decode` is true only when `DS4_SYNC` is present
   (`src/dsv41/ds4_model.cu:1129-1132`). `wp.async = par || (!sync_decode && T <= async_max_t)` at `:9567-9568`.
   The source default for `async_max_t` is 8; the reference config sets `DS4_ASYNC_T=4`. Do not claim `--threads
   on` enables async.
6. **Split-KV production reach is bounded.** `src/dsv41/ds4_kernels.cu:2445-2457` requires FP4 CKV, `a.gen == 0`,
   `1 <= T <= 5`, heads divisible by 4, and at most `24 * 40 = 960` rows. The reference decode shape has 128
   window + 512 selected = 640 rows. Chunked prompt sets `ar.gen = 1` and does not take this branch.
7. **Current tests are not uniformly registered with CTest.** The DS4 binaries are built under
   `STRATA_BUILD_TESTS=ON`, but `ds4_iq_test` and `ds4_experts_rule_test` must be invoked directly. Check
   `ctest -N` instead of claiming that a bare `ctest` covered them.

Recorded baseline to beat only as historical context: `bench/results/2026-11-06-t-225b`, five `--long` runs:
prompt 538.8 / 939.8 / 1064.7 tok/s at 4K / 16K / 32K, chat decode 28.7, agent decode 27.7, agent-turn TTFT
12.25 s. The old split run reached chat 31.2 and agent 31.6. The old best config is
`strata-t-best-now.json`; it carries 2.25bpw weights and box-local `DS4_CACHE_THREADS=4`. Do not compare a
2.00bpw candidate with a 2.25bpw baseline.

## 3. Common build and bench procedures

Every runbook below refers to these procedures instead of repeating them. Where a procedure needs the target
box, it is labelled GPU. Nothing in this section runs GPU work during the documentation phase.

### 3.1 Configure and build (no GPU needed for the build)

The existing `build-dsv41` directory was configured with `CMAKE_CUDA_ARCHITECTURES=75`
(`build-dsv41/CMakeCache.txt`). A fresh equivalent:

```sh
cmake -S . -B build-dsv41 -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=ON -DSTRATA_BUILD_DSV41=ON \
  -DSTRATA_NATIVE_EXPERTS=ON -DCMAKE_CUDA_ARCHITECTURES=75 \
  -DSTRATA_GGML_DIR=third_party/llama.cpp
```

Per-item targets (build only what the item touched; a full build costs much longer):

```sh
cmake --build build-dsv41 -j 8 --target strata-dsv41                     # any src/dsv41 change
cmake --build build-dsv41 -j 8 --target ds4_experts_rule_test            # ds4_experts_rule.hpp
cmake --build build-dsv41 -j 8 --target ds4_iq_test ds4_expert_bench     # ds4_iq.cu
cmake --build build-dsv41 -j 8 --target mmvq_il_parity                   # native_mmvq.cu
cmake --build build-dsv41 -j 8 --target gdn_rec_parity                   # gdn_tc.cuh / kernels.cu
```

Rules:

- After editing an `src/dsv41` file, build `strata-dsv41` even when the item's own test builds it transitively;
  the runbooks' acceptance is against the engine binary.
- A change to `src/dsv41/ds4_kernels.cu`, `ds4_iq.cu` or `ds4_model.cu` compiles into HIP builds too. If a HIP
  toolchain is present, also run `tools/dsv41/build_hip_linux.sh` (defaults: `BUILD_DIR=build-dsv41-hip`,
  `STRATA_HIP_ARCHS=gfx1151`). If it is not present, state "HIP build not run: no toolchain" in the notes; the
  runbooks' `#if defined(STRATA_USE_HIP)` guards exist so the answer is never "I forgot HIP".
- Spill guard (no GPU, needs `cuobjdump` and a demangler): run it on the binary that actually holds the kernels.
  `ds4_iq.cu`, `ds4_kernels.cu` and `ds4_model.cu` compile into the **executable** `strata-dsv41`
  (`CMakeLists.txt:918-920`), not into `libstrata_dsv41_kernels.a` (that library holds only
  `src/dsv41/vendor/*`). So:

  ```sh
  python3 tools/check_cuda_spills.py build-dsv41/strata-dsv41
  ```

  For `src/prefill` work (`gdn_tc.cuh`, `kernels.cu`), give it the library those sources build into
  (`build-dsv41/libstrata_prefill.a`) plus `build-dsv41/libstrata_mmq.a` when MMQ is touched. The script judges
  every kernel Strata wrote, including anonymous-namespace ones (`{anonymous}`), and skips vendored ggml kernels
  whose demangled name has no `::`. A new kernel with STACK or LOCAL bytes fails unless it is listed with a
  measured allowance; do not add an allowance without the measurement that justifies it.

### 3.2 Host-only rule tests (no GPU)

```sh
./build-dsv41/ds4_experts_rule_test        # exits 0; prints FAIL lines on any broken rule
.venv/bin/python -m pytest tests/tuning -q     # when an item touches strata_control/ (sections 6.3, 6.5)
.venv/bin/python -m pytest tests/terminal -q    # when an item touches strata_control/ or serve/
.venv/bin/python -m unittest serve.test_parallel serve.test_engine_arch_note   # for section 6.9
```

These run without a card. A failing one is a stop condition for its item, not a detail to average over. There is
no host test for `ds4_model.cu` or `ds4_kernels.cu`; say so in the notes rather than inventing one.

### 3.3 Engine bench (GPU; only when the box is switched to it)

```sh
# TAG is the WHOLE suffix after "strata-": the file is strata-$TAG.json, and run-tests.sh derives
#   config  = /home/jurek/Strata/strata-$TAG.json
#   results = bench/results/<date>-$TAG
# so a tag of "t-075base" needs the file strata-t-075base.json. Do not add a "strata-t-" prefix yourself.
TAG=t-075base
cp strata-t-fat1024.json  strata-$TAG.json     # 2.00bpw family      -> strata-t-075base.json
# cp strata-t-best-now.json strata-$TAG.json   # 2.25bpw family instead
./run-tests.sh --long $TAG                     # runs exactly strata-t-075base.json
```

Rules that have already cost runs:

- After editing `args` programmatically, set `args_written` equal to `args`.
- Never append a duplicate flag pair to `args`.
- `run-tests.sh` runs `pkill -f strata-dsv41` before and after every config. Do not run it while anything else
  uses the box.
- A result counts only against a same-day baseline of the same pack: run the baseline tag first on the day,
  then the candidate. Spread on this box is roughly ±2% prompt, ±3–4% decode; a move inside that is not a
  result.
- **Keep the two packs apart, and know which recorded runs are which.** `/home/jurek/Strata/strata-t-best-now.json`
  and `strata-t-225b.json` point at the **2.25bpw** pack; `bench/results/2026-11-06-t-225b` is their baseline
  (prompt 538.8 / 939.8 / 1064.7, chat 28.7, agent 27.7). The **2.00bpw** family is
  `strata-t-split.json`, `strata-t-fat.json`, `strata-t-fat1024.json`, `strata-t-slab*.json`,
  `strata-t-fat-slab.json`; its unmodified baseline is `bench/results/2026-11-04-abc-notc-A`
  (prompt 423.9 / 671.4 / 764.9, chat 28.5, agent 26.5).
- **The recorded 2.00bpw runs are not a clean factorial.** Read the configs' `env` and `args` before comparing:
  `2026-11-05-t-fat1024` has no `--reserve-mib` and no `DS4_ATTN_SPLIT`/`STRATA_GLM_EXL3_DECODE`, while
  `2026-11-05-t-slab2048` has all three. So the slab 2048 delta in the record is confounded with split, EXL3
  decode and the reserve. That is exactly why 5.2 asks for its own A/B/C/D matrix rather than arithmetic on old
  directories.
- **Build a clean baseline for your own run.** Do not inherit a candidate config. Start from
  `strata-t-fat1024.json` (2.00bpw pack, `--tp 4 --experts 5`, FAT 64/1024, no split), copy it to
  `strata-$TAG.json` with `TAG=t-075base`, and set `env` to exactly `{"DS4_ASYNC_T": "4", "DS4_FAT_ROWS": "0",
  "DS4_ATTN_SLAB": "0", "DS4_ATTN_SPLIT": "0"}`. Run that first; every candidate tag differs from it in one
  `env` entry. `args` must be identical across the pair — including the `--reserve-mib` decision, which the
  recorded configs disagree about (rule below).
- **Args must be byte-identical across a baseline/candidate pair; only `env` may differ.** Keep
  `--max-context 131072`, `--vision on`, `--tp 4`, `--experts 5` and `--dspark auto` unless the item is about one
  of them. About `--reserve-mib`: the recorded 2.00bpw configs are **inconsistent** — `strata-t-fat1024.json` has
  no `--reserve-mib` while `strata-t-slab2048.json` has `--reserve-mib 300` — and that difference is one of the
  confounders noted below. Pick one and apply it to every config in your matrix: either add
  `"--reserve-mib", "300"` to `args` (and say so in the notes; it is what the 2.25bpw family uses) or leave it
  out of all of them. Never compare a run with it against a run without it.

### 3.4 Greedy-identity and profile probes (GPU)

A short functional check without the full bench: run the engine binary directly with fixed ids and compare
greedy output between the default and the candidate.

```sh
# one load per variant; <ids> = the prompt ids (a few hundred), --n the tokens to generate
./build-dsv41/strata-dsv41 --pack <pack> --ids @ids.txt --n 64 --max-context 32768 --reserve-mib 300 \
    > run-default.log 2>&1
DS4_ATTN_SPLIT=0 ./build-dsv41/strata-dsv41 --pack <pack> --ids @ids.txt --n 64 --max-context 32768 \
    --reserve-mib 300 > run-variant.log
# compare the generated id lists; record them either way (the split path is NOT bit-identical by design,
# so "identical or a recorded, expected difference" is the outcome, never silence)
```

`DS4_PROF=1` prints the per-stage table at exit (decode stages 0–5; a chunked prompt run adds the finer table
and the expert-kernel aggregation). `DS4_PROF_SYNC=1` forces host-clock marks with a drained stream — slower,
but it shows host-side launch gaps that event timing hides. Cheap threads probe: one `--ids` run with
`--threads on` and `DS4_PROF=1`; a hang or a `between layers` stage that explodes is the failure mode to close
the item on.

### 3.5 Machine facts to record with every GPU run

The audit's items 8–9 exist because run reports lacked these. Record, once per session, before the first run:

```sh
nvidia-smi --query-gpu=index,name,compute_cap,memory.total,driver_version --format=csv
nvidia-smi topo -m
nvidia-smi -q -d CLOCK,PERFORMANCE | egrep 'Persistence|Clocks|SM|Perf' | head -40
```

Read-only. Do not set persistence mode or lock clocks for a benchmark unless the user asked; if they are set,
record the values and restore them after.

### 3.6 Recording rule

Every run, passing or failing, gets one block in `~/StrataTuneNotes.md`: date, commit, diff summary, pack,
config tag, run directory, medians with ranges, and one sentence of verdict. A change without a note is treated
as not measured.

### 3.7 Quality gate for any numerics-changing item (GPU)

`tools/quant_quality.py run` is written for the Qwen engine's `--serve` interface and its flags; it is **not**
the DS4.1 tool. The DS4.1 equivalent is the engine's own scoring mode, which reads the same ids file and writes
the same TSV shape the comparator expects:

```sh
# reference: the same build, switch OFF; candidate: switch ON. One load each.
./build-dsv41/strata-dsv41 --pack <pack> --max-context 32768 --reserve-mib 300 \
  --score data/quant_quality/v1/set.ids.json --score-out notes/q-ref.tsv > notes/q-ref.log 2>&1
<SWITCH>=1 ./build-dsv41/strata-dsv41 --pack <pack> --max-context 32768 --reserve-mib 300 \
  --score data/quant_quality/v1/set.ids.json --score-out notes/q-cand.tsv > notes/q-cand.log 2>&1
.venv/bin/python tools/quant_quality.py compare notes/q-ref.tsv notes/q-cand.tsv --segments
```

Facts that make this the right command: `score()` (`src/dsv41/ds4_main.cpp:1100`) reads
`data/quant_quality/v1/set.ids.json` (committed, 5 segments x 1000 tokens), writes `<out>.tsv` plus
`<out>.tsv.seg.json`, and `quant_quality.py::read_run` requires exactly that `.seg.json` companion. `--window`
defaults to 16 tokens, so every position is read through the decode windows rather than the batched prompt path.

How to read it: `compare` prints top-1 agreement and mean / p99 KL of the candidate against the reference. A
numerics-changing switch is acceptable only when its KL stays at the class the engine already accepts for such a
switch (the recorded gates are KL ≈ 1e-3 or below with identical greedy ids for the rounding-level switches, and
≈ 0.006 is the top-1/KL gate quoted for the Qwen fast-attention default in `docs/SM75_TECHNIQUE_AUDIT.md`). Quote
the numbers you measured; do not reuse another item's gate as if it were this item's result. Run this gate for
5.1 (split merge order), 5.2 (FAT summation order), 5.6 (F16 accumulator) and 5.7 (int8 MMA) before any speed
claim, and record the tsv files' path in the notes.

Bit-identity items (5.3, 5.5, 6.2 probe, 6.4 doc fix) skip this gate and instead need the recorded greedy-id
comparison from 3.4 to show no change.

### 3.8 Hardware and workload gates

An item that needs one of these does not start until it is present; its absence is a **defer**, never a
simulated result.

| Gate | How to confirm | If absent |
| --- | --- | --- |
| an sm_75 card | `nvidia-smi --query-gpu=index,name,compute_cap --format=csv` shows `7.5` | code-only items may proceed; all GPU acceptance waits |
| the engine can run on it | `cuobjdump -lelf <binary> \| grep -o 'sm_[0-9]*' \| sort -u` includes `sm_75` | do 5.3 first |
| a working verifier workload (T > 1) | `--dspark on` load succeeds and `DS4_PROF=1` shows a verify batch | keep the verify-window items closed (audit 5 narrow-M in section 7, 6.4's table) |
| peer access on the real hop | `ds4_xfer_bench --pair A B` (built per 3.10) prints peer access 1/1 and verifies the copy | 5.8 stays deferred |
| tensor-parallel members > 1 | engine startup prints no `tensor parallel N -> 1` line | all-reduce family stays deferred |
| >= 400 MB spare VRAM a card | `nvidia-smi --query-gpu=memory.free` during the loaded model | 5.8 stays deferred |
| an owned sm_75 CI runner | a workflow exists with a CUDA runner that executes the bounded tests | 6.6 stays a local runbook |

### 3.9 Rollback rule

When an item's Stop condition fires, revert only that item's edits:

```sh
# list the files the item changed, then restore exactly those (never `git checkout -- .`)
git status --short -- <the files the item touched>
git restore -- <the same paths>
```

Remove any allowances or environment entries the item added to shared files (`tools/check_cuda_spills.py`,
`strata_control/tune_knobs.py`, `serve/server.py`) as part of the same revert. Leave the item's notes block in
place: the negative result is the deliverable. Do not delete other people's uncommitted work to clean the tree.

### 3.10 Kernel labs (GPU; compile lines verified at `c62d5e3a`)

The two labs used by the kernel items are **not** CMake targets. `tools/dsv41_lab/build.sh` builds only `tqk_lab`
and `ds4_lab`, and its `NVCC` default points at a 12.9 toolkit, so pass the toolkit explicitly. Run from
`tools/dsv41_lab`:

```sh
# ds4_lab: parity + graph-replayed timings of the kernels in src/dsv41/ds4_kernels.cu (section 6.6's bounded target)
/usr/local/cuda/bin/nvcc -O3 -std=c++17 -arch=sm_75 -lineinfo -I../../include \
  -o out/ds4_lab ds4_lab.cu ../../src/dsv41/ds4_kernels.cu

# ds4_xfer_bench: the hand-off cost, engine kernels against one cudaMemcpyPeerAsync (section 5.8)
/usr/local/cuda/bin/nvcc -O3 -std=c++17 -arch=sm_75 -I../../include -I../../src/dsv41 \
  -o out/ds4_xfer_bench ds4_xfer_bench.cpp ../../src/dsv41/ds4_kernels.cu
```

The second needs `-I../../src/dsv41` in addition to `-I../../include`; without it the build stops at
`fatal error: ds4_kernels.cuh: No such file or directory`. Both lines were compiled for `sm_75` while this file was
written; neither was run.

`ds4_lab` usage: `./out/ds4_lab --tokens 1,2,4,5 --only attn --ctx 8192 --reps 50`. `ds4_xfer_bench` usage:
`./out/ds4_xfer_bench --pair 0 1 --tokens 4096 --reps 10`.

## 4. Execution order

1. Code-only, no GPU: 5.3 (packaging), 5.1 (split default), 5.2 (075 defaults), 5.5 (profiling breakdown),
   6.3 (tuner machine facts), 6.4 doc row (`MMVQ_IL_TABLE.md`), 6.5 (tuner arch plumbing), 6.9 (serve log
   line). Each ends at its own build + host tests; nothing in this step changes a default for any card.
   Exception to check explicitly: 5.1 does change a default (sm_75 only, split on), and 6.5 adds an opt-in arm,
   not a default. Nothing here changes behaviour on a non-Turing card.
2. First GPU session on the target box: same-day baseline of the pack you will test (2.00bpw per 3.3, or the
   2.25bpw shape if that pack is the subject), then 5.4 (`--threads on` probe), 5.1/5.2 acceptance benches,
   5.5's profile run (which decides 5.6 and 5.7).
3. Experiments, gated: 5.6 (F16 accumulator) then 5.7 (int8 set kernel) only if the 5.5 profile leaves the
   expert kernels worth attacking; 6.2 (CUDA graphs probe) after 5.4's decision; 6.1 (MMQ Turing tile) only if
   the MMQ fallback proves material; 6.7 (GDN scan port) is a separate large effort, not part of this order.
4. Hardware-gated: 5.8 (peer hand-off) and the all-reduce family (section 7) wait for a different box.

## 5. Runbooks: the original items 2.1–2.8

### 5.1 Split-KV decode attention on sm_75 by default (plan 2.1, audit 1)

**Gate:** code now. Acceptance needs the target box.

**Sites:** `split_enabled()` in `src/dsv41/ds4_kernels.cu` (at `c62d5e3a`: lines 2027–2033). The dispatch gate
is the `ckv_fmt == kKvFp4 && a.gen == 0 && ...` branch at `:2445`. Per-device pattern to mirror: `tc_ok()` in
`src/dsv41/ds4_iq.cu` (`:1085-1100`). `<atomic>` is already included (`ds4_kernels.cu:19`).

**Change:** replace the body of `split_enabled()` with the per-device rule below. Keep the existing comment's
R9700 sentence; add that sm_75 is measured (the lab table of `docs/DSV41_SM75_ANALYSIS.md`, candidate 1) and
sm_70 is not, which is why the gate is minor-exact.

```cpp
bool split_enabled() {
    const char* e = std::getenv("DS4_ATTN_SPLIT");
    if (e != nullptr) return std::atoi(e) != 0;          // the flag wins both ways, on every card
#if defined(STRATA_USE_HIP) || defined(STRATA_METAL)
    return false;                                        // HIP and Metal: unmeasured, keep today's default (off)
#else
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 16) return false;
    static std::atomic<int> state[16];                    // 0 unknown, 1 on, 2 off
    int v = state[dev].load(std::memory_order_relaxed);
    if (v != 0) return v == 1;
    int major = 0, minor = 0;
    if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess ||
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev) != cudaSuccess) return false;
    v = major == 7 && minor == 5 ? 1 : 2;                 // sm_75 only: measured; sm_70 unmeasured, stays off
    state[dev].store(v, std::memory_order_relaxed);
    return v == 1;
#endif
}
```

Do not keep the old function-local `static bool`: on a mixed box the first caller would decide for every card.
Do not touch the `STRATA_USE_HIP` gate of the `DS4_FAST_ATTN` branch further down; it is a different path. The
`STRATA_METAL` term is there because this file is also built for Metal and nothing measured the split there;
leaving it out would add a CUDA per-device query to a backend nobody measured, which the contract in section 0
forbids.

**Checks (no GPU):** build `strata-dsv41` (3.1). Build the HIP variant if the toolchain exists (3.1): the
`#if defined(STRATA_USE_HIP)` branch must compile.

**Checks (GPU, target box):** per 3.3–3.4. Choose one pack and stay on it for the whole comparison; the split
effect was measured on **2.00bpw**, so use that unless you deliberately re-measure 2.25bpw.

1. Baseline tag: the config from 3.3 with `DS4_ATTN_SPLIT=0` (serial path).
2. Candidate tag: the same config with `DS4_ATTN_SPLIT` **removed entirely**, so the new card rule decides. On an
   sm_75 card this must take the split path; if it does not, the per-device query is wrong — check that before
   looking at timings.
3. Control tag: the same config with `DS4_ATTN_SPLIT=1`. It must match the candidate's numbers; a gap between
   them means the flag and the card rule disagree.

Historical values from the recorded 2.00bpw runs (`2026-11-05-t-split`, five `--long` runs): chat 31.2, agent
31.6 with the split on, prompt 528.1 / 879.1 / 1001.2, against the serial 2.00bpw baseline
`2026-11-04-abc-notc-A` at chat 28.5 and prompt 423.9 / 671.4 / 764.9. Those two directories also differ in FAT
and slab settings, so they are orientation, not a controlled split A/B — yours is.
4. One short `--ids` run per variant; record the greedy ids. The split merge sums in another order, so ids may
   legitimately differ from the serial path; a difference is recorded, not an automatic failure. A crash, a
   hang, or a NaN in the log is a failure.

**Accept:** candidate's decode exceeds the off-switch run by more than the ±3–4% spread on the same day, prompt
unchanged within ±2%, no new error lines in the engine log. The change is a default change for sm_75 only;
upstream acceptance is the maintainer's call, per `CONTRIBUTING.md`'s default rule.

**Stop:** decode gain inside spread, any crash or hang, or prompt moves beyond ±2%: revert the edit, record the
numbers, keep the env-var opt-in as the shipped answer.

**Next decision:** pass → 5.1 is done for sm_75 and the same-day baseline becomes the reference for 5.2 and 5.5.
Fail → revert, and the split stays env-only; do not retry it with different chunk sizes or split counts, which
nobody measured. Either way, the next item is 5.2, on a separate bench day.

### 5.2 `DS4_DEFAULTS=075`: FAT rows and attention slab (plan 2.2, audit 12)

**Gate:** code now. Acceptance needs the target box.

**Sites:** `is_026_token` / `defaults_026()` in `src/dsv41/ds4_experts_rule.hpp` (`:22-28`).
`attn_slab_rows(...)` in the same header (`:117-131`). `fat_rows()` in `src/dsv41/ds4_model.cu` (`:3474-3481`).
The slab call site is `src/dsv41/ds4_model.cu:3354`. Host test: `tests/dsv41/experts_rule_test.cpp` (already
covers `attn_slab_rows` and the 026 switches).

**Change:**

1. Add to `ds4_experts_rule.hpp`, next to the 026 pair:

```cpp
inline bool is_075_token(const char* v) {
    return v != nullptr && (std::strcmp(v, "075") == 0 || std::strcmp(v, "0.7.5") == 0 || std::strcmp(v, "v0.7.5") == 0);
}
inline bool defaults_075() {
    static const bool on = is_075_token(std::getenv("DS4_DEFAULTS"));
    return on;
}
```

2. Extend `attn_slab_rows` with a trailing `bool v075 = false`: when set and `env < 0`, return
   `clampn(2048)` (clamped to the chunk as always) and skip the `spare_bytes` test. Keep `env == 0` and
   `env > 0` exactly as today — `DS4_ATTN_SLAB=0` and `=N` remain the escape hatches. Extend the header comment:
   the 075 row exists because 64-row FAT GEMMs plus a 2048-row slab measured faster on 4 x 2080 Ti: the recorded
   2.00bpw baseline is 423.9 / 671.4 / 764.9 tok/s at 4K / 16K / 32K and FAT 64/1024 with slab 2048 is
   545.1 / 912.4 / 1044.2 (five `--long` runs each). Those two directories differ in more than these knobs, so
   quote the pair, never an "increment" — 5.2's one-knob matrix is what attributes the gain. The forced 2048 costs a card
   whose spare VRAM is all experts (the rule's own 5070 record: -3% decode), which is why it is behind
   `DS4_DEFAULTS=075` and not automatic.
3. `fat_rows()` lambda in `ds4_model.cu`: the unset case becomes
   `strata::ds4::defaults_075() ? 64 : 0`. Keep the `std::max(v, 17)` floor and the `DS4_FAT_ROWS` override
   exactly as today. Do not touch `DS4_FAT_PIECE` (already 1024).
4. Slab call site: pass `defaults_075()` as the new trailing argument.
5. `defaults_026()` and `DS4_DEFAULTS=026` stay byte-identical to today: the 026 token must not turn any 075
   branch on. Add `experts_rule_test.cpp` cases: `attn_slab_rows(-1, ...)` with `v075` true returns 2048
   regardless of spare; with `v075` false and the same arguments returns today's values (the existing cases
   already pin that); `is_075_token` accepts the three spellings and rejects "026".

**Checks (no GPU):** build `ds4_experts_rule_test` and `strata-dsv41`; run `./build-dsv41/ds4_experts_rule_test`.
Byte-identical check of the default path: with neither `DS4_DEFAULTS` set nor `DS4_FAT_ROWS`/`DS4_ATTN_SLAB`,
`grep` the engine log of a load for the busy-experts line and the attention-slab line; they must be absent /
report the base 256 rows, exactly as before the change. `DS4_DEFAULTS=026` must produce the same lines.

**Checks (GPU):** use the 2.00bpw config procedure in 3.3. For clean attribution, run four configurations on the
same day, each differing from the previous in exactly one place:

Concrete tags (each is also the `strata-$TAG.json` file name and the result folder suffix, per 3.3):

| Tag | `env` (all also carry `DS4_ASYNC_T=4`) | What it isolates |
| --- | --- | --- |
| `t-075base` | `DS4_FAT_ROWS=0`, `DS4_ATTN_SLAB=0`, `DS4_ATTN_SPLIT=0` | the clean 2.00bpw baseline |
| `t-075fat` | `DS4_FAT_ROWS=64`, `DS4_FAT_PIECE=1024`, slab 0, split 0 | FAT alone |
| `t-075slab` | the above plus `DS4_ATTN_SLAB=2048` | the slab, on top of FAT |
| `t-075profile` | `DS4_DEFAULTS=075` only (no FAT/slab/split variables) | that the resolver reaches the same numbers as `t-075slab` |

Recorded historical values, for orientation only — **not** a clean factorial: baseline 423.9 / 671.4 / 764.9
(`2026-11-04-abc-notc-A`), FAT 64/1024 516.3 / 857.7 / 969.8 (`2026-11-05-t-fat1024`), slab 2048 545.1 / 912.4 /
1044.2 (`2026-11-05-t-slab2048`). Those two candidate directories differ from each other and from the baseline in
more than the named knob (see 3.3), so their ratio is not the slab's contribution. That is the reason for the
matrix above. Compare each tag against `t-075base`, and `t-075profile` against `t-075slab` to catch resolver
mistakes. Decode must not move beyond ±3–4%. Do not mix this test with 5.1's split-default A/B.

**Accept:** prompt gain beyond ±2% spread with decode flat, `DS4_DEFAULTS` unset and `=026` logs byte-identical
to before the patch.

**Stop:** prompt gain inside spread, any decode regression beyond spread, or a 026 regression in the host
tests: revert, record, keep `DS4_FAT_ROWS`/`DS4_ATTN_SLAB` envs as the documented opt-ins (they are unchanged).

**Do not** fold `DS4_CACHE_THREADS=4` into this profile: it is box-local (4 GPUs, ≥8 cores) and measured
+1.6% at 16K only.

**Next decision:** pass → `DS4_DEFAULTS=075` is documented as the measured profile and 5.5's profile run is the
next GPU step (it decides 5.6 and 5.7). Fail → revert the resolver, keep `DS4_FAT_ROWS`/`DS4_ATTN_SLAB` as the
per-feature envs, and go to 5.5 anyway: the profile is needed regardless. If `t-075profile` and `t-075slab`
disagree, fix the resolver before anything else — that is a bug, not a performance result.

### 5.3 sm_75 SASS in the DS4.1 release build (plan 2.3, audit 15)

**Gate:** code now; the packaged-binary run needs a 2080 Ti.

**Sites:** `packaging/build_dsv41_cuda.sh:18` (`ARCH=${ARCH:-86-real;89-real;120}`, documented in the header at
`:6-11`). The hand-build default in `CMakeLists.txt` (the `STRATA_BUILD_DSV41` branch of the
`CMAKE_CUDA_ARCHITECTURES` default, `:224-227` at `c62d5e3a`).

**Change:**

1. `packaging/build_dsv41_cuda.sh`: default `ARCH` becomes `75-real;86-real;89-real;120`; update the header
   comment to say sm_75 (RTX 20 / T4) is included and which machine ran the packaged sm_75 binary (until one
   does: "built, not yet run on hardware").
2. `CMakeLists.txt`: the DS4 default architecture list gains `75-real` in the same way, with its comment
   updated. This keeps a hand build and a package build identical.
3. No change to the non-DS4 default (120) and none to `packaging/build_cuda_linux.sh` (already `75;86;89;120`).

**Checks (no GPU):**

```sh
SRC=$PWD B=/tmp/dsv41-arch CUDA=/usr/local/cuda-13.0 GG=$PWD/third_party/llama.cpp \
  packaging/build_dsv41_cuda.sh
# list the SASS architectures actually in the binary (not the ELF class lines)
cuobjdump -lelf /tmp/dsv41-arch/strata-dsv41 | grep -o 'sm_[0-9]*' | sort -u
# must print sm_75 beside sm_86, sm_89, sm_120
```

Build time and binary size grow; record both. If `cuobjdump` lists no `sm_75`, the change is wrong — do not
proceed to the run step.

**Checks (GPU):** on a 2080 Ti, run the packaged binary's `--help` and one short `--ids` decode. The failure
this item exists to prevent is "no image for sm_75"; a JIT from `compute_120` PTX cannot cover it.

**Accept:** `cuobjdump` lists `sm_75`; the packaged binary starts and decodes on a 2080 Ti. Note: this is the
gate audit items 1–7 all share for being shippable at all.

**Stop:** the package build breaks or the binary will not load on the card: revert; a hand-built sm_75 engine
remains the only path and the docs say so.

**Next decision:** pass → every later item may be measured on the packaged binary, and the sm_75 row of 3.8 is
satisfied. Fail → the sm_75 release path is closed until a maintainer changes the packaging decision; continue
with hand-built `build-dsv41` and note that every result then depends on a local build.

### 5.4 `--threads on` for layer-split decode (plan 2.4)

**Gate:** measure first, on the target box. No code.

**Sites (read-only):** parse at `src/dsv41/ds4_main.cpp:1346`; `par` at `ds4_model.cu:2223`; dispatch at
`:9737-9739` (`forward_par` vs `forward_serial`); workers created when `par || chunk > 0` at `:3178`; the
hand-off pair `xfer_get` `:9903/9926` and `xfer_put` `:9932`; the `wp.par && mi == 0 && sr.remote` branch at
`:10236`.

**Hypothesis:** one host thread per device lets device N+1 enqueue its layers while device N still finishes,
instead of the single serial thread doing every hand-off inline. Only worth testing because `--tp 4` collapses
to 1 (fact 1). Async decode is already on (fact 5); this tests pipelining, not async.

**Procedure (GPU):**

1. Cheap probe: `--threads on` in a copy of the best config, one short `--ids` run with `DS4_PROF=1`. Pass: the
   run finishes, ids recorded, profile table parses. Fail: hang, crash, or `between layers` dominating — close
   the item and record.
2. Full: `./run-tests.sh --long t-threads` against the same-day baseline. Decode and agent-turn TTFT are the
   metrics.
3. If decode moves more than 4% either way, run three more times before believing it.

**Accept:** decode gain beyond spread with identical greedy ids to the baseline's same-day run (the path change
should not alter arithmetic — different ids are a finding, not a feature). TTFT regression beyond spread counts
against it even if decode wins.

**Stop / close:** any failure above. The result, positive or negative, goes into the notes and ends the item;
there is no "tune threads harder" follow-up in this plan.

**Next decision:** pass and decode wins → adopt `--threads on` in the reference config for later benches (say so
in every notes block that uses it). Pass with no gain → leave it off and close. Fail (hang/crash) → close and
report; a wrong `wp.par` is an engine bug for a separate issue, not a tuning knob. Either way, the answer decides
6.2's gate: a still-large `between layers` share after this item is what makes the graph probe worth its profile.

### 5.5 Expert-stage profile breakdown (plan 2.5)

**Gate:** code now. The deciding profile run needs the target box.

**Sites:** `prof_mark` / `prof_collect` at `src/dsv41/ds4_model.cu:1361-1428`; the stage-3 marks at `:8464`,
`:8492` (chunked prompt) and `:10358` (serial decode); `run_set` at `:4181-4278`; the `EkMark` machinery at
`:1341-1346` and its print at `:9445-9464`; the `nm_[16]` table and print loop at `:9439-9445`.

**Correction to the old sketch:** do not insert extra `prof_mark(d, N)` calls inside `run_set`. `prof_collect`
bills the span since the previous mark into that mark's stage, so inserting marks mid-stage would move the
covered time **out of** stage 3 and corrupt every comparison with old profiles. Stage 3 must keep meaning
"the whole experts stage".

**Change:** extend the existing `EkMark` aggregation, which already brackets gate/up (`e0→e1`) and down
(`e1→e2`) per `run_set` call and already knows which experts are FAT:

1. Extend `EkMark` with `fat_exp`, `fat_rows_sum`, `e3`, `e4`, `e5`, `e6`. `e3→e4` brackets only the
   FAT gate/up loop; `e5→e6` brackets only the FAT down loop. Create/recycle those four events with the same
   device-aware `ek_ev()` / `ek_pool` rules as `e0..e2`.
2. In `run_set`, where `fat` is built, fill `fat_exp`/`fat_rows_sum`. In the MXFP4 branch record `e3` just
   before the `fat_gate_up` loop and `e4` just after; record `e5` just before the `fat_down` loop and `e6`
   just after (all on `d.st`). Do not allocate those events when `fat.empty()`. On the non-MXFP4 branch record
   nothing (FAT never runs there — `fat_ready` requires MXFP4).
3. In the print block, after the existing per-type lines, print FAT gate/up and down milliseconds, calls,
   experts; then report tiled gate/up and down as the existing `EkMark` totals minus the corresponding FAT
   times. Do not call the leftover "waits": it is only unclassified time and the stage includes stream waits,
   launch gaps and combine work. Keep `experts (incl waits)` stage 3 as the authoritative total. Print one line
   per quantity with the same `ds4 prof` prefix. Clamp tiny negative subtraction to zero and flag a larger
   negative as a bookkeeping error.
4. Return every recorded event to the correct device's pool after collection, including early-error paths.
   Leave `nm_[16]`, the 11-row loop, and every existing mark untouched.

**Checks (no GPU):** build `strata-dsv41`; no host test exists for this (state it in the notes).

**Checks (GPU):** one prompt-heavy run with `DS4_PROF=1` in the best config (FAT is on there). Expected output:
the stage table unchanged from before the patch, plus the new lines. Sanity: tiled + FAT + unclassified sums to
stage 3 within event-resolution noise; if it does not, the event placement is wrong — fix before trusting it.
Also run the same prompt once with `DS4_FAT_ROWS=0`: the FAT lines must read zero and the tiled line must come
close to the whole stage, which is the cross-check that the subtraction is attributed to the right loop.

**Accept:** the breakdown answers "is the busy-expert FP16 GEMM path a large share of the experts stage" with
numbers, and the unchanged stage-3 line matches the pre-patch profile of the same run shape.

**Stop:** the sanity sum fails after one fix attempt: keep the patch out, record, and fall back to comparing
`DS4_FAT_ROWS=0` vs 64 profiles (coarser, still answers the question).

**Next decision:** FAT share large → 5.6 (accumulator in the tiled path still matters) and 5.7's preflight;
tiles share large → 5.7 first; "other/waits" large → neither; close 5.6/5.7 and say so.

### 5.6 F16 accumulator in `set_tc_kernel` (plan 2.6, audit 3) — experiment, opt-in

**Gate:** only after 5.5 shows the tiled expert kernels are still a large share, and at most one day of work.
Old ceiling estimate: expert kernels 12.1% of a 33K prompt profile, so a 14% kernel win is ~1.7% end to end.

**Sites:** `set_tc_kernel<T, GATE_UP>` at `src/dsv41/ds4_iq.cu:821-930`. The accumulator fragments are the
`wm::fragment<wm::accumulator, 16, 16, 16, float> fg[2], fu[2]` at `:848`. Dispatch: `launch_set` at
`:1186-1203`. Quality harness: `tools/dsv41_lab/ds4_expert_bench.cpp` (its `check_ref` prints relative RMS of
`set_tc` against the `f16_rows` + double CPU reference when `DS4_BENCH_CHECK=1`).

**Change:**

1. Add a third template parameter or a parallel kernel: `set_tc_kernel<T, GATE_UP, ACC>` with `ACC = float`
   (today) and `ACC = __half`. For the half variant: fragments are
   `wm::fragment<wm::accumulator, 16, 16, 16, __half>`; the `Out` scratch's `g`/`u` arrays become `__half`
   (store with `wm::store_matrix_sync` into them directly); the final loop converts once —
   `swiglu(__half2float(O.g[r][c]), __half2float(O.u[r][c]), a.limit)` for gate/up, `__half2float(O.g[r][c])`
   for down. Do not change `Tiles` or the load loops: the MMA operands stay `__half` either way.
2. Select in `launch_set` behind `DS4_TC_HALFACC=1` (read once, CUDA only): `tc_ok()` first, then the env.
   Unset or `0` is today's path, bit-identical.
3. Note in the kernel's comment: sm_75 has no `m16n8k16` F16; the repo's two-k8 pattern is
   `src/kernels/cuda/qsa_prompt_attn.cu:52-56` — WMMA hides this, which is also why the expected gain is the
   accumulator throughput difference, not a new instruction.

**Checks (no GPU):** build `ds4_iq_test`, `ds4_expert_bench`, `strata-dsv41`.

**Checks (GPU, before any bench):**

```sh
DS4_BENCH_CHECK=1 ./build-dsv41/ds4_expert_bench                      # baseline: record set_tc rel RMS (old: 3.8-6.2e-4)
DS4_BENCH_CHECK=1 DS4_TC_HALFACC=1 ./build-dsv41/ds4_expert_bench     # candidate: rel RMS must be <= that
./build-dsv41/ds4_iq_test                                             # parity, both env settings
```

**Accept (all three):** relative RMS at or below the current path's; kernel time better beyond the bench's own
run-to-run spread; end-to-end, one `--long` bench shows prompt beyond ±2% with decode flat. If RMS is worse,
the flag stays documented as rejected-for-quality and the item closes.

**Stop:** RMS worse, one day spent, or end-to-end inside spread. Record and close.

**Next decision:** pass → ship opt-in behind `DS4_TC_HALFACC` and go to 5.7 only if 5.5 still shows the tiled path
as the largest expert cost. Rejected for quality → close the accumulator line permanently for this pack and say so
in the notes; 5.7 is then the only remaining kernel lever.

### 5.7 int8 MMA set kernel for the i-quant experts (plan 2.7, audit 2) — experiment, opt-in

**Gate:** only after 5.5 shows the tiled `set_tc` path is the large share **and** a lab preflight passes.

**Correction to the audit's "wire up `experts_native`" framing:** `fused::experts_native`
(`include/strata/prefill/moe_fused_iq.hpp:23-45`, used at `src/prefill/prefill.cpp:4313-4316`) is a whole-MoE
batch API over the Qwen prefill's own layouts, and its Turing setup covers `T_Q2_0` only
(`src/prefill/moe_fused_iq.cu:1817-1826`; the `t75` type gate at `:2013`). DS4's `launch_set` consumes
`x3::SetArgs` with per-expert bounds and DS4's blob layout. So the item is a new kernel in `ds4_iq.cu` that
borrows `moe_fused_iq.cu`'s Turing techniques — the four-way `m8n8k16` int8 decomposition (`:201-229`), the
`__ldg` copy substitute for `cp.async` (`:202-229`), `ASTAGES_DEV = 2` below sm_80 (`:71-77`) — not a call into
that API.

**Preflight (GPU, before writing the kernel):** the audit's "codebook formats do not fit 64 KB" comment reason
does not reproduce from `smem_bytes()` (its own numbers: IQ2_XXS/XS need 61,696 B at `ww = 4`, 47,616 B at
`ww = 2`, both under 65,536). Before designing around a wall, find out if there is one: build the
`STRATA_PF_FUSED_SM75=1` setup on the card and log which `setup_one` calls fail (unconfirmed static analysis
until then). Also run `./build-dsv41/ds4_iq_test --bench-tiles 16,17 8 32,64` and
`./build-dsv41/ds4_expert_bench` for the `set_tc` numbers the new kernel must beat, and record the MMQ column
(3.1x faster in the old lab, at 20–40x worse RMS — that trade is why this kernel is its own work, not "use
MMQ").

**Change (shape):** `set_int8_kernel<T, GU>` in `src/dsv41/ds4_iq.cu`, consuming `SetArgs`, dispatched inside
`launch_set` behind `DS4_SET_INT8=1` after `tc_ok()`. Int32 sums are exact in any order, so gate/up can be
bit-checked against MMQ; the down product needs the RMS gate. Same `EkMark` coverage as the tiled path so 5.5
sees it.

**Checks:** `ds4_iq_test` parity (both env settings), `ds4_expert_bench` RMS against the double reference at or
below the `set_tc` band (3.8-6.2e-4 old), then the standard bench pair (3.3).

**Accept:** RMS in band, kernel faster than `set_tc` beyond spread at the 5.5-profiled shapes, end-to-end prompt
gain beyond ±2%. Ships opt-in regardless of size of win.

**Stop:** preflight shows the shared-memory or type-coverage wall is real, RMS out of band, or a week spent.
Record; the fallback answer stays 5.6 + FAT.

**Next decision:** pass → declare the i-quant kernel opt-in (`DS4_SET_INT8=1`), re-run 5.5 to show the expert
share moved, and hand the result to review as a separate PR. Fail at preflight → close the int8 line for this pack
and record which `setup_one` call failed, because that is the fact the audit lacked. Either way the performance
line ends here; nothing after 5.7 in this plan attacks the expert stage again.

### 5.8 NVLink peer hand-off (plan 2.8) — deferred

**Gate:** do not start on the reference box. It needs a device buffer of one boundary per card — 377.6 MB at
T=4096 (`chunk_hand_bytes()`, `ds4_model.cu:3258-3260`; the hand-off buffers are host-mapped at `:3424`) — and
the cards have ~280 MiB free. Old lab numbers (`tools/dsv41_lab/ds4_xfer_bench.cpp`, GPUs 0+1 NV2): engine
`xfer_put` 28.66 ms + `xfer_get` 33.03 ms = 61.69 ms against one `cudaMemcpyPeerAsync` at 8.04 ms,
byte-verified; but only 2 of 3 hops are NV2 and the hand-off is 7.7% of the 33K profile, so the ceiling is
~2.5% at 33K, ~1.4% at 4K.

**Revive when:** a box has ≥400 MB spare VRAM per card after the best config loads, or more NV links. First
steps then: `ds4_xfer_bench --pair` each real hop to confirm peer access; a device-side hand-off buffer with
the host-mapped path kept as fallback; byte-verification of the first hand-off; then the standard bench pair.
Note the engine enables peer access nowhere in `src/dsv41` today (`cudaDeviceEnablePeerAccess` exists only in
`src/core/peer_experts.cpp:149-152`).

**Next decision:** nothing on this box. The next action is the hardware gate in 3.8 (≥400 MB free a card, peer
access verified on each real hop); when a box meets it, this section is the specification and the first command is
`ds4_xfer_bench --pair` on each hop. Until then the item is closed, not pending.

## 6. Runbooks: surviving audit-only items

### 6.1 Turing tile for vendored MMQ (audit 4) — defer until material

**Gate:** no work until a profile or pack-path trace shows the vendored MMQ fallback consumes a measurable share
of DS4.1 time. `ds4_iq.cu::launch_set` uses DS4's own kernels; do not assume it is the MMQ path.

**Sites:** vendored selection is `third_party/llama.cpp/ggml/src/ggml-cuda/mmq.cuh:230-259`; it routes NVIDIA
Volta/Turing builds through the Ampere config. `common.cuh:284-289` defines Turing MMA availability. Repository
rule: do not edit `third_party/llama.cpp`; there is an existing overlay pattern in `cmake/strata_ggml_tq_overlay.cmake`.

**Cheap probe:** on an sm_75 box build with the vendored default, capture `DS4_PROF=1` and engine stage timing;
profile the actual MMQ fallback caller (identify it from `src/prefill/moe_mmq.cu` call sites). If that path is not
called or is below 5% of the prompt time, close the item with the profile evidence. The audit's tile concern is
static analysis, not a performance result.

**If material:** make a build-tree overlay copy of only the needed MMQ headers. Change a Turing-specific tile
config in the overlay; do not edit generated vendor source or the pinned third-party checkout. Add a compile test
that proves the Turing overlay is selected on cc 75 and the old path is selected on cc 80+; add a per-card
microbench that tests every format/shape that uses the changed tile against the unchanged config.

**Checks:** CUDA sm_75 build; overlay test; existing `prefill_mmq_scatter_test` and `prefill_fused_iq_test` as
relevant; benchmark at DS4's actual type/row shapes. Record compiler resources with
`cuobjdump --dump-resource-usage` and run the spill guard (3.1). Check output against the old MMQ bitwise where
MMQ promises bitwise output. Then run same-day prompt/decode bench.

**Accept:** no correctness differences outside the old MMQ contract, no new spills, and end-to-end prompt improves
more than ±2% spread without decode regression. **Stop:** cannot isolate the MMQ caller, only microbench moves,
or tile occupancy/spill cost erases the gain. Keep the overlay out of the final build.

**Next decision:** MMQ not on DS4's path or below 5% → close, and the audit's item 4 is closed with evidence
gathered here (that is a useful result in itself). Material → the next action is an upstream report/PR to
llama.cpp with the measurement, and only then a local overlay if a maintainer asks for one.

### 6.2 CUDA graphs in DS4.1 decode (audit 7) — profile first

**Gate:** only if `DS4_PROF_SYNC=1` shows host launch gaps are a measurable share of the token latency after the
threads choice (5.4). No graph code before this profile.

**Sites:** reusable graph wrapper is `include/strata/core/graph.hpp` / `src/core/graph.cpp`; DS4 currently contains
no graph capture. Decode's async CPU/GPU flag protocol is `DS4_SYNC`/`sync_decode` at
`src/dsv41/ds4_model.cu:1129-1136`, the waits in `ds4_kernels.cu:1699-1704` and `:1744-1760`, and changing
expert/cache buffers at `ds4_model.cu:1155` and `:3424`.

**Probe (GPU):** run a fixed short decode at T=1 with `DS4_PROF=1 DS4_PROF_SYNC=1`, then `DS4_SYNC=1` with the
same ids and config. Record host time and per-stage timings. If sync mode changes greedy ids, deadlocks, or costs
more than it saves, graphing the whole layer is closed; do not infer graph value from other engines.

**If still material:** prototype capture of a stable, synchronous, single-layer body only, with fixed buffers and
explicit graph invalidation at every residency/buffer change. Do not capture a host flag spin or CPU lane wait.
Use `CapturedGraph`'s bounded wait and correct destruction order. Add a parity harness that alternates uncaptured
and captured layers across all layer types and T=1; test reset/re-record after buffer change. Add a hard timeout.

**Accept:** IDs identical, capture and replay work through cache movement, same-day decode gain outside spread,
prompt/TTFT not worse. **Stop:** any stale-pointer risk that cannot be invalidated, deadlock, wrong ID, or less
than 3% host-launch share in the probe. No graph feature ships from a microbench alone.

**Next decision:** probe shows the host share is small → close CUDA graphs for DS4.1 decode and say so with the
numbers. Share large → build the synchronous single-layer prototype next, and treat the cache-invalidation
harness as part of that prototype rather than a follow-up.

### 6.3 Machine diagnostics for the tuner; report only (audit 8–9)

**Gate:** code now, no performance effect. Owner: `strata_control` tuner only, not DS4 engine defaults.

**Sites:** `strata_control/tune_machine.py::gpus()` queries NVIDIA index/name/memory/driver; add compute capability
with `nvidia-smi --query-gpu=index,name,compute_cap,memory.total,memory.used,driver_version --format=csv,noheader,nounits`.
Read topology as a separate optional bounded probe (Linux + `nvidia-smi`): `nvidia-smi topo -m`; collect PCIe
link fields while a benchmark is loaded with `nvidia-smi -q -d PCIE`; clocks/persistence from
`nvidia-smi -q -d CLOCK,PERFORMANCE`. Do not persist raw command failure as zero; use `None` and a report note.

**Change:** add `compute_cap` to NVIDIA records. Add a `diagnostics()` helper in `tune_machine.py` that returns
read-only topology, PCIe and clock/persistence summaries, each value `None` when its probe failed; add
`machine: dict[str, Any] = field(default_factory=dict)` to `TuneResult` (it has no such field today) and fill it
once in `Tuner.run()`. Do not add setters, persistence changes, clock locks, or a gate that changes candidate
selection in this item. The point is reproduction and diagnosing an asymmetric link, not recommendations.

**Tests (no GPU):** add `tests/tuning/test_tune_machine.py` (new file) driving `tune_machine._run` with
`unittest.mock` — NVIDIA row with `compute_cap`, missing field, malformed CSV, `nvidia-smi` absent, topology
timeout — and assert every unreadable probe is `None`, never `0` or a guessed value; assert no command line
contains a setter (`-pm`, `-lgc`, `-rgc`, `-c`). Add regression cases that AMD `gpus()` output is unchanged. Run
`python -m pytest tests/tuning -q`.

**Accept:** tuner report names GPU model, `compute_cap`, topology/PCIe/clock state when available and explicitly
marks each unavailable probe; a fake `nvidia-smi` proves no setter commands were issued. **Stop:** if readings
cannot be acquired read-only or perturb benchmark load, report unavailable and do not block tuning.

Persistence and locked clocks remain a **procedure**, not code: if the operator changes them, record state and
restore with the same user-approved commands. Never silently run `nvidia-smi -pm 1` or `-lgc`.

**Next decision:** land it and move to 6.5, which consumes the same `compute_cap` field — doing 6.3 first avoids
two parsers. If a probe cannot be made reliable, ship the rest and record the missing one in the report rather than
blocking on it.

### 6.4 Interleaved GEMV 5–8 table (audit 10)

**Gate:** documentation correction can land now; table entries wait for measured active workload.

**Source correction:** `src/kernels/cuda/native_mmvq.cu` already has `kIlRows_75[]`, registers `{75, ...}` in
`kIlArch[]`, and uses it for 2–4 columns. This makes the sentence in `docs/MMVQ_IL_TABLE.md` that "sm_75 and older"
are off stale. Its 5–8 table `kIlRows58_<cc>` has no cc 75 entry. The reference box disables DSpark and runs
T=1, so the 5–8 verify table has no workload there.

**Doc-only step:** edit only `docs/MMVQ_IL_TABLE.md` to add a row for sm_75 `kIlRows_75`, measured on an RTX
2080 Ti, 2–4 columns; state that 5–8 columns still use plain kernels. Verify the code's `kIlArch[]` and
`native_mmvq_il_rows_for()` before writing. Check the doc says output is bitwise equal (a table is speed only).
Two more doc facts to state on the same row, since the file currently omits them: the sm_75 table is reached
through `sm75_il_on()` (`STRATA_SM75_IL`, following `STRATA_SM75_PERF`), so on a 2080 Ti the row is on only when
that switch is on; and `STRATA_MMVQ_IL_ROWS` can override single cells without a rebuild, which is the cheap way
to A/B a suspicion before rebuilding.

**Table step, only if DSpark/another caller is live:** build the target with `cmake --build build-dsv41 -j 8
--target mmvq_il_parity`; on an actual 2080 Ti run `./build-dsv41/mmvq_il_parity --bench --emit-table` and
require `mmvq_il_parity: OK` before looking at timings. The
current program emits existing 2–4 columns; it does not promise a 5–8 sm_75 sweep yet. First extend the parity
harness to test 5–8 shape parity and collect those timings, then add `kIlRows58_75` plus lookup in
`native_mmvq_il_rows_for`. Test the table under `STRATA_MMVQ_IL_WIDE=0|1` and prove 0 returns the plain path.

**Accept:** each non-zero cell improves at least 3% over plain across repeated runs of that cell, output is
bitwise equal, and end-to-end verify improves. **Stop:** no active workload, parity failure, or noisy cells; keep
the doc correction only and leave the table absent.

**Next decision:** doc row lands → done for now; the 5–8 table waits on the verifier-workload gate in 3.8 and is
not a pending task. If a verify workload appears, extend `mmvq_il_parity` to 5–8 shapes first and only then add
`kIlRows58_75` — a table written without parity evidence is worse than none.

### 6.5 sm_75 tuner awareness (audit 11) — conservative and opt-in

**Gate:** code now as metadata and an explicit tuner arm only. DS4 config suggestions are not in scope.

**Sites:** `tune_machine.gpus()` currently has no NVIDIA `compute_cap`; `autotune.context_from_config()` only
sets `Context.arch` for HIP using `engine_component.hip_archs_present()`; `tune_knobs.Context.arch` already
exists and currently names AMD strings (e.g. `gfx1201`). Tuner candidate policy is the `ARMS`, `BOUNDS`, and
`NEVER` tables in `strata_control/tune_knobs.py`; tuner report schema is `TuneResult.classes` and `facts` in
`autotune.py`.

**Change, in this order:**

1. `tune_machine.gpus()`: extend the NVIDIA query with `compute_cap` and store it as an int (`75`, `86`, `120`),
   parsed from `"7.5"`; unparseable or missing → `None`. Do not reuse the `arch` key: it names AMD arches there.
2. `tune_knobs.Context`: add `nvidia_cc: int | None = None`. It must have a default — `tests/tuning/test_presets.py`
   constructs a bare `Context()`, and `tests/terminal/test_sh_stream_windows.py` passes only AMD `arch`.
   `condition_holds` gains one key, `"sm75": ctx.backend == "cuda" and ctx.nvidia_cc == 75`, so conditions stay
   declarative like the existing `gfx12` / `vram_ge_24g` ones.
3. `autotune.context_from_config()`: fill `nvidia_cc` from the `system["gpus"]` argument it already takes (the
   selected `used` cards, not every card in the box; mixed NVIDIA generations → `None`, because one boolean cannot
   describe them). That parameter is the injection point: a test passes synthetic GPU rows and needs no probe.
4. One `Arm` only if it is worth offering: `Arm("sm75_series", "Use this card's Turing kernels", "exact",
   env={"STRATA_SM75_PERF": "1"}, backends=("cuda",), needs=("sm75",))`. `STRATA_SM75_PERF` is the bitwise
   series (`include/strata/core/sm75_series.hpp:7`), so `exact` is the honest class. Never put the quality
   switches (`STRATA_PF_FUSED_SM75`, `STRATA_BF16_TC`, `STRATA_SELECT_SIMT`) in an automatic arm, and do not add a
   DS4.1 architecture default to this Qwen tuner.

Note what this arm cannot do on a mixed box: `sm75_default()` is per device (`src/core/verify.cpp:411`), so the
setting is safe there but the arm's `needs` still declines to offer it — report `nvidia_cc = None` rather than
inventing a value.

**Tests (no GPU):** in `tests/tuning/test_autotune.py` build contexts through `context_from_config(cfg, ...,
engine_backend="cuda", system={"gpus": [...], "memory": {...}, "cpu": {...}})`: cc 7.5 → the arm is planned;
cc 8.6 and `compute_cap` missing/malformed → it is in `skipped` with a reason; `backend="hip"` with `arch=
"gfx1201"` and `backend="cpu"` → unchanged classes and no arm. Also assert the existing `Context()` and
`Context(arch="gfx1030", ...)` constructions still work unchanged. Run `python -m pytest tests/tuning -q`.

**Accept:** sm_75 capability is visible, only eligible CUDA card(s) see the exact/opt-in arm, and no previous
machine class or answer policy changes. **Stop:** capability is unknown or mixed cards cannot be represented
per device; report it but offer no arm.

**Next decision:** land the metadata and, if the arm is offered, measure it with `strata tune` on the card. A win
is an argument for changing a default; that change is a separate, maintainer-agreed decision under
`CONTRIBUTING.md`, not a follow-up commit. If mixed cards cannot be represented, ship the metadata only and close
the arm as declined.

### 6.6 `compute-sanitizer` coverage (audit 14) — local test plan; CI waits for a runner

**Gate:** local target-card testing can start when a 2080 Ti runner is available. CI addition is deferred because
no CUDA self-hosted runner appears in `.github/workflows/`.

**Targets:** first choose small, bounded binaries that exercise synchronization: `ds4_lab` attention split /
packed paths and the smallest flag-protocol harness that can run without loading the model. `xfer_put/get` and
all-reduce flag waits need a dedicated harness input or a multi-GPU lab; do not run a full `strata-dsv41`
model under racecheck as the initial check. Relevant locations: `ds4_kernels.cu:1463-1483`, `:1699-1704`,
`:1732-1760`; `__syncwarp`-using kernels in that file; `tools/check_cuda_spills.py` is not a race checker.

**Procedure (GPU):** build the target with `75-real`; close other GPU work. Run the same binary and input on a
previously validated sm_86 GPU to establish tool noise; then on sm_75:

```sh
compute-sanitizer --tool memcheck --error-exitcode 99 <bounded-test-command>
compute-sanitizer --tool racecheck --error-exitcode 99 <bounded-test-command>
compute-sanitizer --tool synccheck --error-exitcode 99 <bounded-test-command>
```

Save complete output. Compare reports to the sm_86 run. Intended host-mapped volatile-flag protocols can trigger
tool limitations; identify each report at a source line and classify it as a tool limitation only with a matching
baseline and code review. Do not suppress a finding by broad `--ignore`.

**CI gate:** after an owned sm_75 runner exists and can run these tests deterministically, add a narrowly scoped
workflow and exact command; otherwise do not add a job that skips as success. **Accept:** zero unexplained
reports on both devices, bounded runtime, reproducible exit status. **Stop:** no target card, tool absent, or
unbounded flag wait — record unavailable, do not claim sanitizer coverage.

**Next decision:** clean on both devices → record the coverage claim narrowly (which binaries, which tools) and
close. Findings → file each with its full report before any optimisation lands on top of that kernel; a sanitizer
finding outranks every performance item in this plan. No runner → leave the CI part deferred and say
"not run" in the notes.

### 6.7 GDN tensor-core scan on sm_75 (audit 16) — large experiment

**Gate:** no change until an owner accepts a dedicated kernel project. Its current implementation cannot launch
on Turing by merely lowering the major-version check.

**Sites:** `src/prefill/gdn_tc.cuh` is compiled only for sm_80+ (`STRATA_GDN_TC_SM80` at `:32-35`); it uses
`ldmatrix`, `m16n8k16`, and full/split scan buffers. `gdn_rec_tc()` in `src/prefill/kernels.cu:1308-1368`
returns `cudaErrorNotSupported` when `cc_major_of(major) < 8` (`:1322`) and then checks SM count and shared
memory. `gdn_rec_parity.cu` includes the exact code and states its 1e-4 relative-error limits in the header.
There are **three** gates to move together, not one: the `__CUDA_ARCH__` guard, the runtime `major < 8` test, and
`gdn_rec_parity`'s own `--selftest-frag` early return (`if (p.major < 8) { ...return 77; }` at `:1727`) — leave
that last one and the new kernel can never be self-tested on the card it was written for.

The audit compared only the MMA shapes; it missed the shared memory. The four `static_assert`ed plans in
`gdn_tc.cuh` are 97,808 B (`kTcScanSmemSplit`), 98,832 B (`kTcScanSmemSplitNorm`, split + fused norm),
124,944 B (`kTcScanSmem`, the full path) and 160,784 B (`kTcScanSmem` with the fused norm). Every one exceeds
sm_75's 65,536 B opt-in per block, so the smallest existing plan is already 1.5x too large. Because they are
`static_assert`ed, a Turing variant fails the build rather than silently over-requesting shared memory — keep it
that way.

**Preflight (no GPU):** design a separate sm_75 scan tiling/shared staging plan that fits 65,536 B, or stop.
Two `m16n8k8` calls can replace an `m16n8k16` product only if fragment mapping and summation order are proven;
existing `qsa_prompt_attn.cu:52-56` is a reference for one product, not proof the whole scan fits or is correct.
Do not reduce `__CUDA_ARCH__` guards globally, do not invoke the trap wrappers, and do not enable the Qwen
server's `STRATA_GDN_TC=1` on Turing: `serve/server.py::gdn_stack_env` sets it for Linux CUDA Qwen, while the
kernel currently guards sm_80.

**Implementation if preflight accepted:** add a distinct `STRATA_GDN_TC_SM75` opt-in; use a new sm_75 kernel
variant or explicit `if constexpr` that does not instantiate unsupported instructions; add static assertions for
every shared buffer ≤ 65,536 B; preserve existing sm_80 variants byte-identically; require the actual runtime
capability to be 7.5 and enough SMs. Add the same opt-in to `gdn_rec_parity.cu` so it tests the production
kernel, not a copied analogue.

**Checks:** CUDA 13 sm_75 build; run `gdn_rec_parity --selftest-frag`, full `gdn_rec_parity`, `--long` only if
its memory precondition fits; stress tests already listed at the file header (small/large/zero state, large V,
slow gates, coherent keys). Require output and state relative error ≤1e-4 against the existing FP64/reference
and F32 chunk path. Then `gdn_rec_parity --bench-tc` alternating baseline/candidate and same-day prompt bench;
report KL/needle tests (`tools/quant_quality.py`, `tools/needle_bench.py`) before recommending any user-facing
setting.

**Accept:** shared memory fits by static assertion and runtime attribute, all parity/stress cases ≤1e-4, and
measurable end-to-end gain with an explicit opt-in. **Stop:** no 64KB design, any parity failure, or no gain beyond
spread. Keep the default off even after success until separately measured on sm_75.

**Next decision:** preflight fails → close the GDN port and record the binding constraint (shared memory, most
likely); do not attempt a partial port. Preflight passes → the port becomes its own project with this section as
its spec, and it is not part of this plan's execution order.

### 6.8 Capability in memory planner vs DS4 chunk rule (audit 17)

**Disposition:** reject architecture in the Qwen byte planner; defer a DS4 rule until a candidate proves need.

**Source:** `include/strata/plan/plan.hpp::make_plan()` computes byte costs, currently with a pool constant and
no GPU architecture in `Geometry` or `Costs`; `src/plan/plan_main.cpp` accepts byte sizes. Architecture cannot
change bytes for a fixed pack/context, so adding cc there is not justified. The separate DS4
`prefill_chunk_default()` in `src/dsv41/ds4_experts_rule.hpp:64-81` uses VRAM, pack placement, TP and backend.

**If DS4 needs it later:** first measure the same VRAM and pack on a Turing and a newer card, then prove chunk
sizes differ beyond spread and that the performance mechanism is compute-bound, not just expert placement. Add
one explicit cc input to `prefill_chunk_default` only after that evidence; keep an old overload/default preserving
all existing call sites. Add pure host tests for cc 75 vs 86 with identical args plus existing no-cc cases. Then
run same-day prompt/decode A/B on each card. No card pair, no change.

**Next decision:** rejected in the planner → nothing further; that half is settled. The DS4 chunk half reopens
only with the two-card measurement above — until then it is closed, and no signature change is made speculatively.

### 6.9 Serve log for batch-stack hardware class (audit 18)

**Gate:** diagnostics only, code now; no change to enablement.

**Sites:** `PARALLEL_STACK_ENV`, `parallel_stack_platform()`, and `parallel_stack_env()` at
`serve/server.py:3675-3713`; startup log at `:9231` (`engine_arch_note`). The audit's statement that these flags
reach DeepSeek is contradicted by current code: `parallel_stack_env` exits for `cfg.arch == "deepseek41"` and
HIP. It enables the nine flags only for Linux CUDA Qwen with >=2 batch slots. Keep that existing gate.

**Change:** add `parallel_stack_note(cfg)` that reports which of the nine switches the stack actually set and on
which card class, or "Linux CUDA batch stack active; GPU compute capability unknown". Do not write a second
capability probe: `strata_control/engine_component.py::cuda_archs_present()` already returns `sm_XX` per NVIDIA
card from `nvidia-smi --query-gpu=compute_cap`, and `serve/server.py::engine_arch_note()` already has an inline
`index,compute_cap` query to copy the parsing rules from. Reuse one of them; a third parser is a third bug.

The note's gate must be the same predicate chain as `parallel_stack_env()`, in the same order — `parallel_slots
(cfg) >= 2`, `(cfg.get("arch") or "") != "deepseek41"`, `cfg.get("backend")` not in `("sycl", "metal",
"vulkan")`, `cfg.get("backend") != "hip"`, `parallel_stack_platform()`. Factor the shared part into one helper
and call it from both, so the log can never claim the stack is on when it is off (or the reverse). Report the
count actually applied: `parallel_stack_env` uses `env.setdefault`, so a user-set value wins and the number of
newly set variables can be fewer than nine. Honour the selected GPU indexes when reading capabilities; when the
query fails or a selected card is not NVIDIA, say "unknown" rather than a guessed architecture.

**Tests:** extend `serve/test_parallel.py::ParallelStackDefaults` for active CUDA, no slots, DeepSeek, HIP,
Windows/macOS and explicit env overrides; add mocked selected GPU metadata with 7.5/8.6 and missing fields.
Run `python -m unittest serve.test_parallel -v`. The message must describe what is set, not assert a speed claim.

**Accept:** active-stack startup log identifies measured/unavailable hardware, all existing environment outputs
remain byte-for-byte as tests pin them. **Stop:** any enablement behavior changes or unknown hardware is guessed.

**Next decision:** log line lands → the item is done; enabling or disabling the stack per card is a different,
unmeasured change and is out of scope. If the log cannot name the card without a new probe, ship the "unknown"
wording and close — a wrong architecture in a log is worse than an honest unknown.

## 7. Closed and hardware-blocked ideas: do not start

This section completes audit items not otherwise made into runbooks. It is an instruction not to implement them on
the reference box.

| Idea | Disposition | Evidence / exact gate to reopen |
| --- | --- | --- |
| 5–24 column narrow-M verify kernel | closed here | DSpark auto is dropped (fact 4); `src/dsv41/ds4_iq.cu:1719-1722` rejects jobs above 16 columns. Reopen only with a measured active verifier workload; first profile `MvJob` columns and verify parity harness. |
| `__restrict__` on all-reduce kernels | defer | `nm==1`, no call runs. Reopen only on a >=3-member TP box with a real all-reduce message size; prove pointers cannot alias and run output-equivalence plus same-day end-to-end. |
| fused all-reduce + `hc_post` | defer | same `nm==1` blocker. Additionally preserve "all streams read before any write" and the peer-flag wait order at `hc_post`/allreduce. Needs a dedicated multidevice parity test. |
| warp-specialised expert kernel for sm_75 | rejected as described | `src/prefill/moe_ws.cu::mode()` explicitly requires every visible GPU cc 6.0; `moe_ws_gu_iq2.cu`, `moe_ws_gu_iq3.cu`, and `moe_ws_down.cu` are weight-stationary XMAD kernels, not `native_ws_kernel`. `prefill_ws_parity_test` is compiled only with `STRATA_EXPERIMENTAL_SM60` and returns 77 off cc 6.0. Do not change that gate or widen formats based on the audit's name similarity. A distinct Turing port would need its own implementation and parity project. |
| Dspark-only variables, `--dspark-block`, `--dspark-depth`, `DS4_SPEC_*` | closed on reference box | Drafter is dropped by the recorded expert-bytes/free-room inequality (fact 4). `--dspark on` forces a different, unmeasured 3.3x byte-read regime; no bench in this goal. |
| allreduce `DS4_AR_F16`, reduce-scatter tuning, 3-way expert sharding, `DS4_TP_ALIGN`, `DS4_TP_SPLIT` | closed on reference box | TP fails expert-block divisibility first; `nm==1`; see facts 1–2. Recheck startup TP line on a different box before reviving. |
| GPU-to-GPU peer hand-off | defer | 5.8; at least 400 MB/card and required peer topology. |
| graphing async decode / layers with flag waits | closed until probe | 6.2; never capture a host spin or unstable cache pointer. |
| lock persistence/clocks automatically | rejected | diagnostic procedure in 3.5 only; machine state is changed. User approval and restoration required. |
| Qwen parallel-stack flags for DS4.1 | rejected | `serve/server.py::parallel_stack_env()` explicitly exits for DeepSeek; audit 18 is a diagnostic log item, not an enablement request. |
| GDN TC by dropping the sm_80 guard | rejected | 6.7: shared memory exceeds sm_75 limit and unsupported wrapper bodies trap. |
| MMQ edit in `third_party/llama.cpp` | rejected | 6.1: use a build-tree overlay after proving the code path matters. |
| the audit's own "Considered and rejected" table (INT4 tensor cores, W4A16 Marlin, W8A8/QServe, Triton sweep, FlashAttention, other engines' flags, ECC, BF16, unified memory, the "14× trick") | closed | Reason survives in `docs/SM75_TECHNIQUE_AUDIT.md`. That file still exists; nothing here overrides it. |
| the original plan's dead items and every other already-measured-negative switch | closed | Listed explicitly in 7.1 and 7.2, because the earlier version of this file was never committed and cannot be consulted. |

### 7.1 Dead items carried over from the original plan (explicit list)

The previous revision of this file had a "Dead items" table. That revision was never committed, so the table
cannot be looked up — it is reproduced here so no implementer has to guess. All of these were measured or
established on the reference box; do not implement and do not re-measure them there.

| Item | Why it is closed |
| --- | --- |
| all-reduce `__restrict__`, `DS4_AR_F16`, reduce-scatter tuning | `nm == 1` always (facts 1–2); no call executes |
| fused all-reduce + `hc_post` | nothing to fuse at `nm == 1` |
| `NC = 8 / 12` GEMV windows in `mv_nc` (`src/dsv41/ds4_iq.cu:1278-1284`) | the 5–15 column case is the drafter's verify, and the drafter is off (fact 4); dispatch also rejects `ncol > 16` (`:1719`) |
| `DS4_TP_ALIGN`, `DS4_TP_SPLIT` | the `blk` divisibility test fails first; tp cannot be 4 with 256-value blocks |
| `DS4_DSPARK_*`, `--dspark-block`, `--dspark-depth`, `DS4_SPEC_*` | the drafter never loads (fact 4) |
| rope cos/sin table in `finish_kernel` | measured dead: rope is 33.8–51.3 µs of a 1019.1 µs `finish` at 131 072 rows, ~0.4% of a 10.5% stage |
| a CUDA wmma port of sparse attention | implemented, measured, reverted: 0.2–2.1% at slab sizes 16–2048, and 30–36% slower at T=2–4 where the production scratch path runs |
| `DS4_HEAD_PER` | no-op; the head is not EXL3 on this pack |
| `DS4_SYNC=1`, `--prefill-chunk 8192`, `--experts 6`, `DS4_SETTLE_MS=0`, hugepages + pin, `DS4_SETTLE_STORE`, `DS4_CACHE_BONUS=0`, `DS4_IQ_CTA_PER_SM=2`, `DS4_EG_DIRECT` | all measured, all negative or flat; `DS4_EG_DIRECT` is −8% by its own comment (`src/dsv41/ds4_model.cu:2870`) |
| `DS4_CPU_SMT`, `DS4_READ_SPLIT`, `DS4_PF_ALL`, `DS4_IO_PRIO` | compiled out on Linux (`src/dsv41/ds4_model.cu:170`, `src/dsv41/ds4_os.hpp:125,417`, `src/dsv41/ds4_pool.hpp:192`) |
| `DS4_NO_Q8_BATCH`, `DS4_NO_AUTO_VX` | inside `#if defined(STRATA_METAL)` (`src/dsv41/ds4_gguf.cu:258`, `src/dsv41/ds4_model.cu:4619`) |
| `DS4_IQ_OWN`, `DS4_GG_IQ256` | already self-tuning and bit-identical; nothing to win |
| `EXL3_GEMM_H_ACC` | that symbol does not exist in `src/dsv41/vendor/exl3.cu`; only `STRATA_GLM_EXL3_DECODE` does. (The audit's F16-accumulator item is 5.6, in DS4's own `set_tc_kernel`, and is not this flag.) |
| `DS4_CHUNK` / `--prefill-chunk 2048` | 4096 is the rule's own choice for this shape (`prefill_chunk_default`, `ds4_experts_rule.hpp:64-81`); 8192 measured −11…−31%, 1024 measured worse in the rule's own R9700 record |

### 7.2 Items that need a different box

| Item | Why it cannot run here | Gate to reopen |
| --- | --- | --- |
| all-reduce `__restrict__`, `DS4_AR_F16`, 3-way reduce-scatter | dead code at `nm == 1`; value exists only for a ≥3-way tensor-parallel run | startup prints TP members > 1 (3.8) |
| fused all-reduce + `hc_post` | same, plus the hyper-connection ordering invariant | same |
| peer hand-off over NVLink (5.8) | needs 377.6 MB spare VRAM a card; the box has ~280 MiB free | ≥400 MB free a card, peer access verified (3.8) |
| `DS4_TP_ALIGN` tuning | no alignment makes `2304 / 4` a multiple of 256 | a pack/GPU pair where `n_ff / tp` divides by the block size (e.g. `n_ff = 2560`, `tp = 4` → 640) |

If a future box meets one of these gates, re-check the startup line first: it is the single check that decides
whether the whole all-reduce family is live.

## 8. Source and claim provenance

- `docs/DSV41_SM75_ANALYSIS.md`: machine, pack and kernel measurements; especially split attention, i-quant GEMM
  arithmetic and complexity, existing Turing WMMA, and dead ends.
- `docs/SM75_TECHNIQUE_AUDIT.md`: 19 cross-project proposals and their stated code sites. The coverage tables in §1
  are the disposition; item-level corrections above were checked against source.
- `docs/DEEPSEEK41_ENGINE_MAP.md`: target layout, build defaults and registered/unregistered test targets.
- `docs/MMVQ_IL_TABLE.md`, `docs/OLDER_GPUS.md`, `include/strata/core/sm75_series.hpp`: current sm_75 behavior and
  the Qwen tuner/default patterns.
- `CONTRIBUTING.md`: default-path, measurement, and quality requirements.

Numbers in this playbook are either explicitly marked old measurement with a hardware/pack pointer or are source
constants used to define a test. Any new figure must have a new run directory and notes entry before it is called
measured.

