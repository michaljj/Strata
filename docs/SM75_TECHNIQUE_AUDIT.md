# What the three sm_75 research documents give Strata: an audit

This page answers one question: of everything the three research documents propose for Turing cards (sm_75, the
RTX 20 series and the T4), what can Strata use for its own benefit today, where would each item land in this tree,
and what is it worth?

The three sources (kept outside this repository, in the reader's home directory):

| Source | What it is | Size |
| --- | --- | :-- |
| `sm75optiResearch.md` | Two pasted research reports: a system-level one (engine choice, residency, interconnect, quantisation, batching) and a kernel-level one (SM75 ISA, expert-local shapes, load pipelines, attention, collectives) | 3,440 lines |
| `sm75optiResearch_2.md` | A second system report (§1-§13, Turing decision tree, the hybrid-MoE "14× trick") plus its own kernel-level deep dive | 497 lines |
| `cudaToolkitDoc.md` | NVIDIA's *Tuning CUDA Applications for Turing* | 96 lines |

**This is an audit, and nothing here is a claim of a speed-up.** Every item carries either a measurement that exists
in this repository (with what it was measured on) or the label **needs measurement**, or **unconfirmed static
analysis** where the finding comes from reading code. No engine code was changed to write it.

**Measured on (for every number quoted from this repo):** 4 x RTX 2080 Ti 22 GB (sm_75), Xeon E5-2680 v4, 121.4 GiB
RAM, Ubuntu 24.04, driver 595.91.07, CUDA 13.4, project 0.1.55 (commit c62d5e3a), engine ds4-0.1, pack
`deepseek-v4.1-flash-ist-das-2.00bpw`, unless the row says otherwise. Where a number came from a different machine,
the row says which.

## The short answer

Nineteen items survive. The rest of this page says which proposals the tree already implements (ten) and which do not
transfer (twelve), each with the reason. Ranked by expected value, with the order of the work in mind:

The Area column is the objective's three-way split: **engine kernel** and **engine launch** touch the CUDA/HIP engine,
**launch** is what the Terminal's launcher should ask the machine, **plan/scheduler** is `src/plan` and the chunk rule,
**serve** is `serve/`, and **packaging** and **engine CI** are the two things that decide whether any of it ships.

| # | Area | Item | Expected value | Evidence | Where it would go |
| --: | --- | --- | --- | --- | --- |
| 1 | engine kernel | The split-KV attention on by default for sm_75 | 3.8× at the kernel, one query; up to ~12 % of decode **if** attention is on the critical path | **measured here** (kernel), needs measurement (end to end) | `src/dsv41/ds4_kernels.cu:2027` |
| 2 | engine kernel | The i-quant expert GEMM on Turing: wire up what `moe_fused_iq` already contains | 4.0× instruction rate against the `__dp4a` dot; covers 27 % of this pack's expert bytes **if** the port works | unconfirmed static analysis + needs measurement | `src/prefill/moe_fused_iq.cu:1817`, called from `src/dsv41/ds4_iq.cu:1191` |
| 3 | engine kernel | An F16-accumulator MMA path for the prompt tiles | llama.cpp's own analogy is ~14 % at batch 1; ~1 % of output RMS error at k=4096 | needs measurement | `src/dsv41/ds4_iq.cu:848`, `src/dsv41/ds4_gguf.cu:791` |
| 4 | engine kernel | A Turing tile config for the vendored MMQ instead of the Ampere one | the 1.34× step-time figure is Ampere's, not ours | needs measurement | `third_party/llama.cpp/ggml/src/ggml-cuda/mmq.cuh:252` |
| 5 | engine kernel | A narrow-M kernel for the 2-24 column window | 1.34× step time on Ampere with speculation; our draft path is unmeasured | needs measurement | `src/dsv41/ds4_iq.cu:911` (`mv_kernel`'s `NC`) |
| 6 | engine kernel | `__restrict__` on the three all-reduce kernels | 12.49 → 6.51 µs on a 10 KB reduce is llama.cpp's measurement from that one qualifier | needs measurement | `src/dsv41/ds4_kernels.cu:1773`, `src/dsv41/ds4_kernels.cu:1834`, `src/dsv41/ds4_kernels.cu:1883` |
| 7 | engine launch | CUDA graphs for the decode step | every Turing decode pays per-kernel launch overhead; no number here | needs measurement | `src/core/graph.cpp:49` exists; `src/dsv41` uses none |
| 8 | launch | Print the link topology and the link state at launch | converts "four GPUs is slower" from a mystery into a number | needs measurement | `strata_control/tune_machine.py` |
| 9 | launch | Persistence mode and locked clocks before any benchmark | doc-2's §13 puts it first; ECC does not exist on a 2080 Ti, so only these two apply | needs measurement | `strata_control/tune_machine.py` |
| 10 | engine kernel | A 5-8 column table for the Turing interleaved GEMV (the missing `kIlRows58_75`), and the stale doc row | 2-4 columns already measured at 81.1 → 37.4 µs; 5-8 is the verify window and has no Turing table | **measured here** (2-4), needs measurement (5-8) | `src/kernels/cuda/native_mmvq.cu:3735`, `docs/MMVQ_IL_TABLE.md:21` |
| 11 | launch | A `sm_75` row in the tuner's bounds and never-lists | the tuner currently has none, so `strata tune` cannot be card-aware | unconfirmed static analysis | `strata_control/tune_knobs.py:137` |
| 12 | engine default | Per-card defaults for DS4.1 (a `ds4_arch_defaults()` beside `defaults_026()`) | no speed by itself; makes items 1-5 shippable as one-line defaults | unconfirmed static analysis | `include/strata/core/defaults_026.hpp:12` is the pattern; `src/core/nv_defaults.cpp:26` has no 7.5 case |
| 13 | engine kernel | Warp-specialised expert kernels for Turing | exists, but gated to GP100 only | unconfirmed static analysis | `src/prefill/moe_ws.cu:51` |
| 14 | engine CI | `compute-sanitizer` in CI for the sm_75 kernels | catches ITS races the 85 `__syncwarp` sites were written to avoid | unconfirmed static analysis | no script or workflow mentions it |
| 15 | packaging | Ship sm_75 SASS in the DeepSeek release package | every DS4.1 item above needs a hand-built engine first | unconfirmed static analysis | `packaging/build_dsv41_cuda.sh:18` |
| 16 | engine kernel | The GDN tensor-core scan (`STRATA_GDN_TC=1`) on sm_75 | the engine's own measurement is +4.4 % at 4K, +5.5 % at 16K, +5.7 % at 32K prompt, at KL 0.0019 — on another card class | needs measurement | `src/prefill/gdn_tc.cuh:32-35` (the sm_80 gate), `src/prefill/kernels.cu:2986-2989` |
| 17 | plan/scheduler | Give the plan and the chunk rule the card's compute capability | the planner's pool is a hard-coded `5943000000ull` and every `prefill_chunk_default` branch keys on VRAM size, so a 22 GB Turing and a 22 GB Ada card are the same bucket | unconfirmed static analysis | `include/strata/plan/plan.hpp:126-150`, `src/dsv41/ds4_experts_rule.hpp:64-79` |
| 18 | serve | A capability line (or at least a log line) for the serve batch stack | nine flags measured together on one machine are enabled for any Linux CUDA Qwen config with 2+ slots | needs measurement | `serve/server.py:3675-3713` |
| 19 | engine kernel | Fuse the all-reduce with the post-normalisation | one launch and one row read a layer saved; 20,480 B a layer at one token | needs measurement | `src/dsv41/ds4_model.cu:10406-10407`, `src/dsv41/ds4_kernels.cu:409-420` |

Items 12-15 are not optimisations. They are the reason items 1-11 cannot be measured by a user who installs Strata.
Items 16-19 are smaller: a measured win gated one card class too high, a scheduler and a serving stack that cannot see
the card, and one kernel fusion.

The labels mean: **measured here** — a number exists in this repository, on the card named above;
**needs measurement** — the mechanism is checked in code, the number is not; **unconfirmed static analysis** — a
finding from reading code only, nobody ran it.

## The items in detail

### 1. The split-KV attention, on by default for sm_75

**What the docs say.** Split the KV dimension for decode, merge the per-chunk softmax states
(`sm75optiResearch.md` §2226), and pick the split count from context length, head geometry and the parallel work
already present. Both documents also warn that a kernel win is not an answer win, because the merge changes the
summation order.

**Where it lands.** The kernel is already in the tree and already off:
`attnsplit::sparse_attn_split<kSplitG, kSplitR>` at `src/dsv41/ds4_attn_split.cuh`, dispatched from
`src/dsv41/ds4_kernels.cu:2445-2458`. The gate is `split_enabled()` at `src/dsv41/ds4_kernels.cu:2027`, one
environment variable (`DS4_ATTN_SPLIT`), and its comment says why it is off: "no gain measured on 4x R9700
(21.8 vs 21.9 tok/s)".

**What it is worth.** Measured here with `tools/dsv41_lab/ds4_lab.cu` built for sm_75
(`nvcc -O3 -std=c++17 -arch=sm_75 -I../../include --ctx 8192 --reps 50`), 64 heads x 512, 128 window + 512 selected
rows, one device, one stream, CUDA graphs so the launch gap is out:

| Kernel | 1 query | 2 queries | 4 queries | max error vs double |
| --- | ---: | ---: | ---: | --- |
| `sparse_attn (fp4 ckv)` — what the engine runs at decode | 142.14 µs | 164.47 µs | 421.99 µs | 4.6e-7 |
| `attn_split G8 R40` | **37.36 µs** | **69.63 µs** | **187.59 µs** | 1.0e-6 |
| `attn_split G4 R40` | 44.37 µs | 85.20 µs | 223.64 µs | 1.0e-6 |
| `attn_split G2 R32` | 65.66 µs | 124.59 µs | 333.38 µs | 1.0e-6 |

105 µs off a layer, 40 layers, is 4.2 ms a token; at 30 tok/s a token is 33 ms, so this is up to 12 % of decode
**if** the attention stage is on the critical path. Whether it is needs `DS4_PROF=1` before and after — the engine
overlaps the expert reads with the attention.

**The change.** Two lines and a comment: on CUDA, `major == 7`, `split_enabled()` returns true unless
`DS4_ATTN_SPLIT=0`. `DS4_ATTN_SPLIT=1|0` keeps working. The Qwen engine's answer to exactly this situation is
`sm75_default()` in `include/strata/core/sm75_series.hpp:37`.

**Risk.** The merge sums the chunks in another order: 4e-7 against a double reference in the kernel's own comment,
1.0e-6 in the run above, and the shipped path's own error is 4.6e-7. The split path is not bit-identical to the
packed path. `kSplitMaxT = 5` (`src/dsv41/ds4_kernels.cu:2026`) bounds it to the decode window, so a prompt batch
never takes it.

### 2. The i-quant expert GEMM: a Turing path that is written and never called

**What the docs say.** `sm75optiResearch.md` §1656-1730 lays out the arithmetic paths (packed-weight GEMV, W4A16
Tensor Core, W4A8 integer) and §1868-1907 says to use the native instruction shape, inspect SASS rather than PTX, and
compare accumulator policies. `cudaToolkitDoc.md` §1.4.2 adds the fact that matters most: Turing's integer MMA exists
for 8-bit operands with 32-bit accumulation, in 16x16x16, 32x8x16 and 8x32x16 fragment sizes.

**Where it lands.** `src/prefill/moe_fused_iq.cu` is a fused int8 prompt-expert GEMM and it contains a complete,
compiling Turing path:

- an int8 MMA written for both shapes, `m16n8k32` where it exists and a four-way `m8n8k16` decomposition below
  sm_80 (`src/prefill/moe_fused_iq.cu:165`, `src/prefill/moe_fused_iq.cu:171`, `src/prefill/moe_fused_iq.cu:178`, and the `#if __CUDA_ARCH__ >= 750` branch at `src/prefill/moe_fused_iq.cu:201-229`);
- the `cp.async` substitute Turing needs, `cp16 = *(uint4*)dst = __ldg(...)` with no-op commit and wait, plus the
  kept `prefetch.global.L2` hints (`src/prefill/moe_fused_iq.cu:202-229`; the same pattern at
  `src/prefill/moe_fused.cu:181-214`);
- pipeline depth reduced for Turing's shared memory: `ASTAGES_DEV = 2` below sm_80 with the reason in the comment
  (`src/prefill/moe_fused_iq.cu:71-77`);
- `ldmatrix` fragment loads shared by both architectures (`src/prefill/moe_fused_iq.cu:232`);
- the `cudaFuncSetAttribute` opt-in that makes a 60 KB request legal (`src/prefill/moe_fused_iq.cu:1705`).

**Why it does nothing for DS4.1 today.** Two independent reasons, both mechanical:

1. On Turing, `setup_dev()` sets up the Q2_0 pack only.
   `src/prefill/moe_fused_iq.cu:1817-1826` — `if (major < 8 && !(major == 7 && minor == 5 && sm75)) return d;` then
   `d.ok = setup_one<T_Q2_0, true>(occ) && setup_one<T_Q2_0, false>(occ);`. Every i-quant type (IQ2_XXS, IQ2_XS,
   IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS) and `T_IQ4_NL` down is set up on the sm_80+ path only, and
   `native_supported()` refuses a Turing card outright (`src/prefill/moe_fused_iq.cu:2013`,
   `if (d.t75 && (gu_type != T_Q2_0 || d_type != T_Q2_0)) return false;`).
2. DS4.1 links `strata_prefill` (`CMakeLists.txt:937`) but never calls the fused expert path:
   `grep -rn "prefill::\|fused::" src/dsv41/` finds nothing. The only symbol it takes from that library is the RDNA3
   dense GEMM header (`src/dsv41/ds4_model.cu:35`).

So the port is not a kernel: it is (a) adding the i-quant types to the Turing setup, and (b) calling it from the
DS4.1 launch site `launch_set` at `src/dsv41/ds4_iq.cu:1186-1201` (`if (tc_ok())` at `src/dsv41/ds4_iq.cu:1191`), next to
`set_tc_kernel<T,GU>` (`src/dsv41/ds4_iq.cu:821`).

**What it is worth.** The instruction-rate ratio is measured: 244-252 TOP/s for int8 tensor cores against 62-63 TOP/s
for the `__dp4a` the decode dots use, on this machine, with `tools/dsv41_lab/mma_peak.cu` built
`nvcc -O3 -arch=sm_75` (the table is in [DSV41_SM75_ANALYSIS.md](DSV41_SM75_ANALYSIS.md)). Against the F16 tiles that
already run the prompt it is about 2× (244-252 TOP/s against 124-126 TFLOP/s for
`m16n8k8.row.col.f32.f16.f16.f32`). The i-quant formats are 27 % of this pack's expert bytes — 33.7 GiB of
123.0 GiB, the ten layers whose gate/up *and* down are both covered (10, 18, 22, 23, 25, 26, 28, 33, 38, 39). The
end-to-end effect of the port has **not** been measured.

**One correction to the tree.** The comment at `src/prefill/moe_fused_iq.cu:1821` says "the codebook formats' do not
fit 64 KB at 4 warps a row". The file's own `smem_bytes()` (`src/prefill/moe_fused_iq.cu:112`) does not reproduce that: at `ww = 4` (64 weight
rows a stage) IQ2_XXS and IQ2_XS need 61,696 B, and at `ww = 2` they need 47,616 B — both under the 65,536 B that
`include/strata/core/emulate.hpp:25-34` grants an sm_75 device. The stated reason is therefore at most a
`NT = 256` or warp-specialised-path limit, not a capacity wall at `ww = 4`. **Unconfirmed static analysis:** someone
with a card has to run the setup and see which of the eight `setup_one` calls fails before the gate is loosened.

**Risk.** The int32 sums are exact in any order (the file says so at `src/prefill/moe_fused_iq.cu:203-220` and `src/prefill/moe_fused_iq.cu:90-98`), so the int8 path
should match the MMQ reference bit for bit on the gate/up product; the down projection lands in a different order and
needs the usual KL gate. Treat it as opt-in (`STRATA_PF_FUSED_SM75=1`) exactly as it is today.

### 3. An F16-accumulator MMA path for the prompt tiles

**What the docs say.** This is the single most transferable kernel claim in the three documents.
`sm75optiResearch_2.md` §2 reports a SASS measurement of the 16x16x16 F16 MMA on Turing: FP32 accumulation is about
99 cycles against about 74 for F16 accumulation, and cites ExLlamaV3's `EXL3_GEMM_H_ACC` as "~14 % faster at bsz 1"
with "max observed error vs fp32 accumulation ... ~1 % of output RMS at k=4096". That flag is gated to sm_86 and is
explicitly unvalidated on architectures where FP32-accumulator HMMA is also half rate — which is Turing. The
documents' three-stage advice is: F16 accumulators where K blocks are short, promote into FP32 registers once per
k-slice, never globally.

**Where it lands.** No `mma.sync` in this tree uses an F16 accumulator. The instruction is written out at 19 sites in
`src` and `include`, and every one is `f32.<t>.<t>.f32` or `s32.<t>.<t>.s32`:

| Immediates | Sites | Where |
| --- | --: | --- |
| `f32.f16.f16.f32` | 8 | `src/kernels/cuda/qsa_prompt_attn.cu:61`, `src/kernels/cuda/qsa_prompt_attn.cu:64`, `src/kernels/cuda/qsa_prompt_attn.cu:68`, `src/kernels/cuda/qsa_prompt_attn.cu:1487`, `src/dsv41/vendor/exl3.cu:282`, `src/dsv41/vendor/exl3.cu:926`, `src/prefill/gdn_tc.cuh:143`, `src/prefill/moe_fused_iq.cu:178` |
| `f32.tf32.tf32.f32` | 2 | `src/kernels/cuda/native_qsa_score.cu:52`, `src/kernels/cuda/qsa_select.cu:194` |
| `f32.bf16.bf16.f32` | 1 | `src/prefill/kernels.cu:2591` |
| `s32.s8.s8.s32` | 7 | `src/kernels/cuda/qsa_prompt_attn.cu:1097`, `src/prefill/moe_fused.cu:194`, `src/prefill/moe_fused.cu:208`, `src/prefill/moe_fused_iq.cu:165`, `src/prefill/moe_fused_iq.cu:171`, `src/prefill/moe_fused_iq.cu:210`, `src/prefill/moe_fused_iq.cu:214` |
| `s32.u8.s8.s32` | 1 | `src/kernels/cuda/qsa_prompt_attn.cu:1108` |

(`grep -rn "f16.f16.f16.f16"` under `src` and `include` finds nothing, so the F16-accumulator form is absent.) The same
holds for the WMMA wrapper: all eight `wmma::fragment<..., accumulator, 16, 16, 16, float>` declarations name `float`
(`src/kernels/cuda/iq_kernels.cu:6687`, `src/kernels/cuda/iq_kernels.cu:6929`, `src/prefill/gdn_chunk.cu:257`, `src/prefill/gdn_chunk.cu:297`, `src/prefill/gdn_chunk.cu:328`, `src/prefill/gdn_chunk.cu:384`,
`src/dsv41/ds4_iq.cu:848`, `src/dsv41/ds4_gguf.cu:817`). The two places that run MMA on a Turing card are the F16 WMMA
prompt tiles: `set_tc_kernel<T, GU>` (`src/dsv41/ds4_iq.cu:821`) holding
`wm::fragment<wm::accumulator, 16, 16, 16, float>` (`src/dsv41/ds4_iq.cu:848`), and `mx_set_tc_kernel` for MXFP4 packs
(`src/dsv41/ds4_gguf.cu:791`, enabled by `tc_ok()` at `src/dsv41/ds4_gguf.cu:1007`).

**What it is worth.** The F16 tiles are load-bearing on this machine: turning them off with `STRATA_DS4_NO_TC=1`
costs 30-32 % of prompt throughput — 423.9 → 294.6 tok/s at 4K, 671.4 → 458.4 at 16K, 764.9 → 523.4 at 32K, decode
unchanged (28.5 against 29.9), measured here on 4 x 2080 Ti with `tools/strata_bench.py`, 5 runs, medians. Those
figures bound the *stage* the accumulator switch would change — they are not the switch's own value. If it is worth
llama.cpp's 14 % at batch 1, that is the class of win to expect on a shape we already depend on. **Needs
measurement** here, and the honest prior is that a 2 % error budget is too large for a prompt tile that feeds 40
layers; the ExLlamaV3 number is at k=4096 for one projection.

**Risk.** This changes answers by design. `CONTRIBUTING.md` and `strata_control/tune_knobs.py` both classify that as a
`quality` setting: never automatic, always behind a switch, with `tools/needle_bench.py` and
`tools/quant_quality.py` run before it is offered.

### 4. A Turing tile config for the vendored MMQ

**What the docs say.** `sm75optiResearch_2.md` §3 records that llama.cpp's MMQ encodes Turing tuning in exactly one
place (`TURING_MMA_AVAILABLE` raises the X-extent to 128 and pads quantised tiles so `K % 8 == 4`), and doc-1 §2519
warns that reaching an Ampere-named table is not proof of a performance bug.

**Where it lands.** Both halves are visible in the vendored copy and both are worth knowing:

- `third_party/llama.cpp/ggml/src/ggml-cuda/common.cuh:288-289` defines `TURING_MMA_AVAILABLE` for cc 7.5, and
  `third_party/llama.cpp/ggml/src/ggml-cuda/mmq.cuh:165-204` makes a Turing build take the MMA data layout
  (`use_mma_data_layout` via `turing_mma_available`).
- `third_party/llama.cpp/ggml/src/ggml-cuda/mmq.cuh:230-259` then hands a Turing card the **Ampere** tile config:
  `ggml_cuda_mmq_get_config_ampere` whenever `highest_compiled_arch(cc) >= GGML_CUDA_CC_VOLTA`. Nothing in the
  shared-memory arithmetic asks for Turing's 64 KB (`third_party/llama.cpp/ggml/src/ggml-cuda/mmq.cuh:1388-1393` and
  `third_party/llama.cpp/ggml/src/ggml-cuda/mmq.cuh:428-430` are arch-agnostic), so the config is chosen for a card with more shared memory and more resident warps
  than this one.

**What it is worth.** **Needs measurement.** llama.cpp's own Turing-specific work (PR 31533's few-row kernel, 1.34×
step time with 7-token drafts) is **not** in the vendored copy — `grep few_row` finds nothing — so the only harvest
here is a per-card config table, measured the way this engine already measures the interleaved GEMV:
`src/kernels/cuda/native_mmvq.cu:3783` (`kIlArch[]`) filled in from
`mmvq_il_parity --bench --emit-table`, documented in
[MMVQ_IL_TABLE.md](MMVQ_IL_TABLE.md).

**Constraint.** `third_party/llama.cpp` is vendored: `tools/check_cuda_spills.py` deliberately skips its kernels
("a kernel whose demangled name carries no `::` is one of theirs"), and the project's rule is that a vendored tree is
not edited. A measured Turing config therefore has to arrive as a patch carried in this repository's own overlay, or
as an upstream report — not as a local edit to
`third_party/llama.cpp/ggml/src/ggml-cuda/mmq.cuh`.

**Risk.** A tile config caps the shared memory a kernel asks for and the rows a block owns; a Turing config that is
wrong does not fail, it reduces occupancy or spills, and the vendored kernels are excluded from
`tools/check_cuda_spills.py`'s allowance list — so the audit has to be a measurement, not a static check. It is also
the item least likely to matter here, because Strata's own expert path is the i-quant kernel of item 2 and this one is
only reached through the vendored MMQ fallback.

### 5. A narrow-M kernel for the 2-24 column window

**What the docs say.** `sm75optiResearch_2.md` §4 names a dead zone at 2-24 activation columns: a packed-weight GEMV
repeats its multiply per column, and a 128-row tiled kernel has too few rows. llama.cpp's answer (PR 31533) reads each
weight row once, decodes it once into MMA fragments, and multiplies it against every column: measured Q4_K matmul
−14.3 %, spec decode over 7-token drafts 1.34× step time (58.87 → 43.81 ms/step, 57.6 → 77.7 t/s on a 3090 Ti), with
about 44 % of the gain from laying the activation fragment out directly.

**Where it lands.** Turing's equivalent in this tree is `mv_kernel<T, NC>` at `src/dsv41/ds4_iq.cu:911`: one row a
warp, `NC = 1, 4, 16` columns. The 2-24 window is what the verify round of `--dspark` produces
(`src/dsv41/ds4_model.hpp:46-55`), and the Qwen engine's version of this idea is the interleaved GEMV with its
per-card rows-a-warp table.

**What it is worth.** **Needs measurement** on this machine. The Ampere figure is large and the shape is the one
decode verification spends its time in, so this is the most likely kernel win after item 2 — but the same A/B that
made `STRATA_DS4_NO_TC=1` cheap is the way to test it, and no such test exists yet.

**Risk.** Few. A narrow-M kernel that only serves windows of 2-24 columns cannot change a single-query decode, and it
is opt-in by construction (`NC` is a template argument).

### 6. `__restrict__` on the all-reduce kernels

**What the docs say.** `sm75optiResearch_2.md` §8 records a llama.cpp RFC measurement: adding `__restrict__` took a
10 KB all-reduce from 12.49 µs to 6.51 µs. It also records the trap that matters more: "lifting custom-AR gates
naively produced silently wrong output — wire an output-equivalence gate".

**Where it lands.** Three kernels tie their pointers without `__restrict__`: `ar_rs_kernel`
(`src/dsv41/ds4_kernels.cu:1773`, the reduce-scatter taken when `nm > 2`), its F16 twin `ar_rs16_kernel`
(`src/dsv41/ds4_kernels.cu:1834`, `DS4_AR_F16`), and `ar_sum_kernel` (`src/dsv41/ds4_kernels.cu:1883`, the
one-shot form). The choice is at `src/dsv41/ds4_kernels.cu:2658-2681`. This is unconfirmed static analysis: the qualifier is missing, the aliasing is not.

**What it is worth.** **Needs measurement.** The all-reduce is one call a layer per token
(`src/dsv41/ds4_model.cu:10406`, `allreduce(d.moe_out, T * D, ...)`), so a 6 µs saving is 0.24 ms a token over 40
layers — about 0.7 % of a 33 ms token at 30 tok/s. Small, and llama.cpp's 12.49 µs was a 10 KB message, so the size
has to be matched before the number means anything. It is a keyword, not a kernel.

**Risk.** `__restrict__` on pointers that do alias is undefined behaviour. These kernels take `x` (the local buffer)
and `peers`/`rows`/`sums` (the shared buffers) from separate allocations, which the launch site makes explicit, so the
qualifier is sound here — and any change must be checked with the engine's existing output-equivalence runs before it
counts.

### 7. CUDA graphs for the decode step

**What the docs say.** Both research documents name launch overhead as a first-order Turing problem, because it has no
`cp.async`-era compiler scheduling to hide it and no hardware scheduling assumption that the driver helps
(`sm75optiResearch.md` §634; `sm75optiResearch_2.md` §9: "hundreds of small kernels per token").

**Where it lands.** The wrapper exists — `src/core/graph.cpp:49` (`cudaStreamBeginCapture` with
`cudaStreamCaptureModeThreadLocal`) and `src/core/graph.cpp:77` (`cudaGraphLaunch`) — and is used by the Qwen engine. `src/dsv41`
contains no graph capture at all (`grep -rn cudaGraph src/dsv41/` is empty).

**Why it is not a one-liner.** The default DS4.1 decode is asynchronous on purpose: the GPU waits on flags the CPU
lane sets while it runs (`src/dsv41/ds4_model.cu:1129-1136`, `sync_decode`), and a flag wait in the middle of a
capture is exactly what a graph cannot host. So the honest scope is "capture the synchronous shape
(`DS4_SYNC=1`) or the per-layer body", which is a measurement question before it is a code question.

**What it is worth.** **Needs measurement.** Doc-1 §634 is the right warning: do not interpret disabling graphs as a
speed optimisation, and keep a graph-disabled configuration as the diagnostic.

**Risk.** A capture records the allocations and the stream order it saw; the engine's expert cache swaps and its
chunk ring change both between tokens (`src/dsv41/ds4_model.cu:1155`, `src/dsv41/ds4_model.cu:3424`), so a graph captured over a decode
window would have to be invalidated whenever the residency set changes. The failure mode is silent — a stale graph
replays reads of buffers that have since been reused — so any capture has to be checked with an output-equivalence
run, not a timing run.

### 8-9. What the launch path should check on the machine

**What the docs say.** Doc-2 §12 and its §13 action list put the machine first: ECC off, persistence mode, locked
clocks, and the file's §7 requires the real topology before any multi-GPU conclusion. Doc-1 §153-204 lists the exact
commands (`nvidia-smi -L`, `nvidia-smi topo -m`, `nvidia-smi topo -p2p r/w`, `lspci -tv`, `lspci -vv`) and warns that
link state falls at idle, so it must be read under load.

**Where it lands.** Nothing in the product. `strata_control/tune_machine.py` reads only the GPU index, name, memory
and driver (`gpus()`; the `--query-gpu=index,name,memory.total,memory.used,driver_version` call), the CPU, and free
memory. There is no ECC read, no persistence-mode read or set, no clock
lock, and no topology capture anywhere outside `tools/par/e00.py` (a benchmark helper) and two benchmark READMEs.
ECC does not exist on a consumer 2080 Ti, so that item shrinks to persistence mode, locked clocks, and printing the
link width and the P2P matrix at launch.

**What it is worth.** **Needs measurement** on this machine. Doc-2's ECC figure (10-15 % of memory bandwidth on
T4/Quadro) is about hardware we are not running. The value here is diagnostic: the audit found that a four-GPU
DeepSeek benchmark and a two-GPU report disagree about which topology is which, and the engine's own log does not
record the link state at all.

**The measured facts already in the tree** that such a check would have surfaced: the four-2080-Ti benchmark
records `pcie=gen3 x16` and a 250 W power limit, and [DSV41_SM75_ANALYSIS.md](DSV41_SM75_ANALYSIS.md) records the
per-pair asymmetry — GPU0↔1 47.07 GB/s against GPU2↔3 47.03 GB/s on NV2, with the cross-pair link going through the
host bridge (PHB), so two of the three hand-offs a chunk could use move four times the bandwidth of the third. P2P is
off in the engine's deep path: `cudaDeviceEnablePeerAccess` appears only in
`src/core/peer_experts.cpp:149-152`, never in `src/dsv41`, so every cross-GPU byte in DS4.1 goes through driver
staging (`src/dsv41/ds4_model.cu:590-594`).

**Risk.** Persistence mode and clock locking change the machine, not the program: a locked clock hides a thermal or a
power-limit problem instead of fixing it, and it makes a benchmark that a user cannot reproduce. Any such setting
belongs in the report ("measured with persistence on, clocks at 1650 MHz") or behind an explicit flag, never applied
silently by a launcher.

### 10. The Turing interleaved-GEMV table stops at four columns

**What the docs say.** `sm75optiResearch.md` §2237-2250 (reuse quantised KV across GQA heads) and
`sm75optiResearch_2.md` §3 ("the right unit is `__dp4a`") are about the same lever Strata already pulls: read a
packed weight row once and reuse the decode work across several columns. Doc-2 §4 then names the hole this engine
still has — the 2-24 column window where a GEMV repeats work and a tiled kernel has too few rows.

**Where it lands.** This is one of the better-covered items in the tree. `kIlRows_75[]` at
`src/kernels/cuda/native_mmvq.cu:3708` is registered in `kIlArch[]` at `src/kernels/cuda/native_mmvq.cu:3786` and was
"read off `mmvq_il_parity --bench --emit-table` on an RTX 2080 Ti 22 GB (SM 1650 MHz), the same 3 % rule"
(`src/kernels/cuda/native_mmvq.cu:3704-3707`), with the argument for the table being dp4a-bound from two columns on: IQ4_XS 2560x10240 at four
columns 81.1 → 37.4 µs, the Q5_K head 1671 → 1045 µs. Every cell is bitwise `native_mmvq`'s output, so the table is
only speed. The gate is `sm75_il_on()` at `src/kernels/cuda/native_mmvq.cu:3908` (`STRATA_SM75_IL`, which follows
`STRATA_SM75_PERF`), reached through `il_arch_ok()` at `src/kernels/cuda/native_mmvq.cu:3914-3925`.

**The hole.** Turing has a 2-4 column table and **no 5-8 column table**. `kIl58Arch[]` at
`src/kernels/cuda/native_mmvq.cu:3735-3739` lists sm_80, sm_120 and sm_120-with-80-SMs only, and
`native_mmvq_il_rows_for()` returns 0 for `ncols > 4` on any other card
(`src/kernels/cuda/native_mmvq.cu:3829-3840`). Five to eight columns is exactly the verify window that a `--dspark`
draft of 5-7 tokens produces, and exactly doc-2 §4's dead zone. So the decode path has no measured guidance there
and falls back to the plain kernels.

**A documentation defect found by the same reading.** [MMVQ_IL_TABLE.md](MMVQ_IL_TABLE.md)'s card table still says
"sm_75 and older: the path is off; the plain kernels run" (`docs/MMVQ_IL_TABLE.md:21`), and there is no row for
the Turing table. That was true before the sm_75 series existed (commit `004efd46`, "#2069 sm_75: interleaved
small-batch matvec table for Turing (kIlRows_75)"); the code now carries a 2080 Ti's measurements. Unconfirmed static
analysis of a doc-versus-code mismatch — no speed involved, but a reader following that table concludes the wrong
thing, and a maintainer adding a Turing cell does not know the file exists.

**What it is worth.** **Needs measurement** for the 5-8 column cells, and the measurement is already scripted:
`mmvq_il_parity --bench --emit-table` (`docs/MMVQ_IL_TABLE.md:27`) prints a paste-ready block, and the sm_80 row shows
what the result looks like (`kIlRows58_80`, `src/kernels/cuda/native_mmvq.cu:3727`). The 2-4 column gain is already
banked: 37.4 µs against 81.1 µs on one shape is the class of win the 5-8 column cells would carry.

**Risk.** The interleaved path is bitwise equal to the plain kernels by design (`docs/MMVQ_IL_TABLE.md` says so, and
the Turing table's header repeats it), so a wrong cell cannot change an answer — but a wrong cell written from one
shape and applied to another does cost throughput, which is why the table is keyed by type, column count, output size
and card. The doc fix is risk-free; the table cells need the card.

### 11-12. The two structural gaps: no sm_75 row in the tuner, no per-card defaults in DS4.1

**What the docs say.** `sm75optiResearch.md` §539-608 and §680-734 both make the same point from different ends: a
setting is only worth having if the tool that chooses it knows the card, and
`sm75optiResearch_2.md` §13's action list opens with "identify the GPU generation first" for the same reason. Neither
document names a place in this tree; the finding is that both places exist and both are generation-blind.

**Where it lands.** The tuner's knowledge is a set of tables —
`strata_control/tune_knobs.py:137` (`BOUNDS`) and `strata_control/tune_knobs.py:154` (`NEVER`), 17 `Arm(...)` candidates — and none of them has an
sm_75 row. Its answer classes (`exact`, `placement`, `nonrepeatable`, `quality`) are exactly the vocabulary a Turing
default would need. Meanwhile `src/core/nv_defaults.cpp` has per-architecture tables for sm_86 (`src/core/nv_defaults.cpp:26`),
sm_60 (`src/core/nv_defaults.cpp:41`) and sm_121 (`src/core/nv_defaults.cpp:53`) and nothing for 7.5; the pattern a
DS4.1 equivalent would copy is `include/strata/core/defaults_026.hpp:12` with its table test
`src/core/arch_defaults_test.cpp` (no card needed); the Qwen side's equivalent is
`include/strata/core/sm75_series.hpp:25-40`, whose call sites are listed in
[SM75_SERIES_WHY.md](SM75_SERIES_WHY.md).

**What it is worth.** No speed by itself. It is what makes items 1, 3 and 10 a one-line default instead of a new
`getenv` in a file that already has 121 of them (`src/dsv41/ds4_model.cu`). The measured numbers those defaults would
carry are already in [SM75_SERIES_WHY.md](SM75_SERIES_WHY.md), on an RTX 2080 Ti: grouped experts 47.3 → 41.4 µs at
10x1 and 120.2 → 100.2 µs at 30x1 (but 36.5 → 40.8 µs at 12x3, which is why the switch caps groups), the fused
up-projection 86.1 → 70.9 µs at six rows, the interleaved multi-column GEMV 88 → 37 µs at four columns, the
quantised cache-attention chunk 146 → 84.9 µs at four queries, the halved block scores 774 → 255 µs at the largest
shape, and the QSA top-k split 0.21 → 0.033-0.064 ms.

**Risk.** A per-card default is a change to the default path, and `CONTRIBUTING.md` requires the default path to stay
byte-identical to the last release unless the maintainers agree the change. So item 12 is two separate steps: a tuning
table and defaults function that exist and are tested without a card (the shape `src/core/arch_defaults_test.cpp`
already establishes), and — only after a measured win on an actual sm_75 card — an entry that changes behaviour. Doing
the second before the first is what the "opt-in, byte-identical default" rule forbids.

### 13. Warp-specialised expert kernels for Turing

**What the docs say.** Doc-1 §2008-2081 (persistent expert work queues, warp specialisation) and doc-2 §7 (fused MoE
with a 2-stage pipeline) both describe warp-specialised expert kernels as the shape that removes dispatch and pipeline
stalls.

**Where it lands.** Strata has the kernel — `src/prefill/moe_ws.cu` with `src/prefill/moe_ws_down.cu`,
`src/prefill/moe_ws_gu_iq2.cu` and `src/prefill/moe_ws_gu_iq3.cu` — and it is gated to one architecture: `mode()` at
`src/prefill/moe_ws.cu:51-71` requires
`STRATA_MMQ_WS=1` **and** every visible device to be cc 6.0 (GP100), because that is where it was tuned and checked.
`supported()` at `src/prefill/moe_ws.cu:73` then limits it to gate/up types with `Q2_0`/`IQ4_NL` down.

**What it is worth.** **Unconfirmed static analysis**: the Turing machinery the kernel would need (the `__ldg` copy
substitute, the 64 KB budget, the m8n8k16 MMA) is already written elsewhere in `src/prefill`, and the existing kernel
is a separate file behind a switch, so the change is a new gate and a shared-memory audit rather than a port. No
number exists for Turing, and the Qwen-side warp-specialised path (`native_ws_kernel`, guarded at
`src/prefill/moe_fused_iq.cu:1728-1735` with `fa.ptxVersion < 80 → false`) is deliberately closed on sm_75.

**Risk.** This is the largest engineering item on the page and the one with the least evidence: relaxing `mode()` would
hand a Turing card a kernel tuned and checked on one GP100, at a shared-memory budget the GP100 does not share, and
`supported()` would simultaneously widen the type set. It also touches the most fragile part of the codebase — the
shared-memory arithmetic and the static asserts at `src/prefill/moe_fused_iq.cu:569`, `src/prefill/moe_fused_iq.cu:1314`, `src/prefill/moe_fused_iq.cu:1567`, `src/prefill/moe_fused_iq.cu:923`. Treat
it as a last item, not a first.

### 14. `compute-sanitizer` in CI

**What the docs say.** Both kernel documents end on discipline: `racecheck` and `synccheck` find the
independent-thread-scheduling races Turing introduces (`cudaToolkitDoc.md` §1.4.1.2 says this outright and names both
tools), and miscompiled MMA does not crash, it produces fluent nonsense, so needle tests belong in the loop.

**Where it lands.** No script, test or workflow in this repository mentions `compute-sanitizer`, `racecheck`,
`memcheck` or `synccheck`. What the tree does have is `tools/check_cuda_spills.py`, which reads
`cuobjdump --dump-resource-usage` per kernel and per architecture and fails on stack or local-memory use outside a
listed allowance — a different check, and a useful one.

**What it is worth.** **Unconfirmed static analysis.** The sm_75 kernels rely on 85 `__syncwarp` calls and on
volatile-flag protocols that a race checker would review (`src/dsv41/ds4_kernels.cu:1472-1483`, `src/dsv41/ds4_kernels.cu:1704`;
`src/dsv41/ds4_gguf.cu:664`). Race detectors are slow and cannot cover a whole benchmark; the realistic scope is the
handful of kernels that carry a flag protocol.

**Risk.** A race checker reports what it cannot prove, not what is wrong: on a kernel that spins on a peer's flag
through mapped host memory the tool will flag the intended protocol, so a run needs a baseline on an architecture that
is already validated (sm_86) before an sm_75 finding means anything. Adding it to CI also adds minutes; scoping it to
the flag-protocol kernels is what keeps that acceptable.

### 15. sm_75 SASS in the DeepSeek release package

**What the docs say.** `cudaToolkitDoc.md` §1.3 (Application Compatibility) says to check how the application is
compiled *before* any tuning, and §1.4.2 gives the reason: "Any binary compiled for Volta will run on Turing, but
Volta binaries using Tensor Cores will only be able to reach half of Turing's Tensor Core peak performance. Recompiling
the binary specifically for Turing would allow it to reach the peak performance" (`cudaToolkitDoc.md:61`). A binary
built for a later architecture is worse off still: it does not run on an earlier card at all. Both research documents
assume the reader owns the binary they are tuning.

**Where it lands.** `packaging/build_dsv41_cuda.sh:18` builds `ARCH=${ARCH:-86-real;89-real;120}`, and the script's
own header says what that means: "ARCH defaults to the 0.2.7 release shape: SASS for sm_86 (RTX 3090 / 3060), sm_89
(RTX 40, #3935) and sm_120 (RTX Blackwell) + PTX sm_120" (`packaging/build_dsv41_cuda.sh:8`). The DeepSeek release
package therefore contains no sm_75 image and no `compute_75` PTX; `compute_120` PTX is forward-only, so a 2080 Ti
cannot JIT it either. The image simply does not run on that card. The main engine's packaging is fine by comparison:
`packaging/build_cuda_linux.sh:18-19` builds `75;86;89;120` for CUDA 13 and `60;61;70;75;80;86;89` for CUDA 12.

**What it is worth.** Every DS4.1 item in this page ends at "a user with a 2080 Ti cannot run the shipped package".
The audit could only measure because it built the lab by hand (`nvcc -O3 -std=c++17 -arch=sm_75`). Worth stating
plainly: this is the difference between a research item and a shippable one.

**Risk.** Every extra architecture grows the release package and its build time, so this is a packaging decision, not
a free fix: the DeepSeek variant is built per architecture today (`packaging/build_dsv41_cuda.sh:18` takes an `ARCH`
override for exactly that reason). An sm_75 image that nobody has run is worse than no image — the gate for adding it
is a passing run on a 2080 Ti, which is the same gate items 1-7 need.

### 16. The GDN tensor-core scan, gated one card class too high

**What the docs say.** `sm75optiResearch.md` §370-376 (recurrent and hybrid layers) warns that recurrent
state updates can bottleneck outside attention and GEMM, and doc-2 §6 puts the recurrent/hybrid shapes in the same
"rebuild it around the card you have" bracket as attention. This engine's own equivalent is GatedDeltaNet, and it has
a measured prompt win sitting behind an arch gate.

**Where it lands.** `STRATA_GDN_TC=1` runs the chunked recurrence's scan on the tensor cores
(`src/prefill/gdn_tc.cuh`, dispatched from `src/prefill/kernels.cu:2986-2989`). The engine's own measurements, in the
comment block at `serve/server.py:3719-3730`: KL mean 0.0019 / max 0.0125, 61-task suite 55 against 54, prompt
**+4.4 % at 4K, +5.5 % at 16K, +5.7 % at 32K** (Qwen3.8 IQ3_S). The kernel is compiled out below sm_80 —
`#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800` → `STRATA_GDN_TC_SM80 1`, else 0
(`src/prefill/gdn_tc.cuh:32-35`) — and `src/prefill/gdn_tc.cuh:1` says so in its own header: "the chunked DeltaNet recurrence's
scan on the tensor cores (STRATA_GDN_TC=1), sm_80+".

**Why that gate is probably one class too high.** The scan's MMA sites are `m16n8k16` (`src/prefill/gdn_tc.cuh:143`)
and `ldmatrix` (`src/prefill/gdn_tc.cuh:125`, `src/prefill/gdn_tc.cuh:134`) — and Turing has the second and not the first, which is exactly the pair
`src/prefill/moe_fused_iq.cu` and `src/kernels/cuda/qsa_prompt_attn.cu:52-56` already handle by splitting the k=16
step into two `m16n8k8` steps on the same fragments. So the same mechanical port applies here.

**What it is worth.** **Needs measurement.** The +4.4-5.7 % is measured on other hardware (the comment does not name
the card for the Qwen run; the kernel's own speed figure at `src/prefill/kernels.cu:2987-2988` is "3.75x
gdn_rec_kh_kernel on a CMP 170HX"). A Turing port cannot claim that number. What it can claim is that the GDN prompt
stage is a real share of a small-MoE prompt on a 616 GB/s card, and that this is the only *already written and already
quality-gated* prompt win in the tree that an sm_75 build silently skips.

**Risk.** The scan is held to 1e-4 against an FP64 reference by `src/prefill/gdn_rec_parity.cu` (its own comment at
`src/prefill/gdn_rec_parity.cu:11` and the cases at `src/prefill/gdn_rec_parity.cu:962-968`), so the correctness gate exists and a Turing instantiation would be tested by the
same harness. The output-order change is the same class as item 1's.

### 17. The plan and the prompt-chunk rule know nothing about the card

**What the docs say.** `sm75optiResearch.md` §59-90 says to find the bound before optimising and gives
`t ≥ max_i(W_i / B_i)` as the tool; §425 asks for the logical batch and the microbatch to be tuned separately.

**Where it lands.** The scheduler in this tree is not a queue: it is the memory plan plus the prompt-chunk rule, and
neither takes a compute capability as input.

- `strata::plan::make_plan(uint64_t max_context, const Geometry&, const Costs&, uint64_t pool = vram_pool_bytes())`
  (`include/strata/plan/plan.hpp:130-150`) computes `cache_slots = cache_bytes / c.expert_blob` from a **pool constant**
  — `vram_pool_bytes()` returns a hard-coded `5943000000ull` (`include/strata/plan/plan.hpp:126-128`), documented as
  "the one number taken from the architecture notes (§2)". There is no `Geometry` or `Costs` field for the card, and
  `src/plan/plan_main.cpp` (73 lines) exposes only `--max-context`, `--vram`, `--dense`, `--embd`, `--workspace`,
  `--state`, `--pool`.
- The real DS4.1 chunk decision is `prefill_chunk_default(...)` (`src/dsv41/ds4_experts_rule.hpp:64-79`), and every
  branch of it keys on **VRAM size, pack layout and tp** — `gpu_share >= 0.5` → 1024; `min_vram_bytes >= kChunk4096MinVram`
  (10 GiB) → 4096; `min_vram_bytes < kChunkInPlace4096Below` (16 GiB) → 4096; else `big`/2048. A 22 GiB 2080 Ti and a
  22 GiB RTX 4090 take the same branch.

**What it is worth.** **Unconfirmed static analysis**, and the honest framing is that VRAM is a good proxy for most
cards and a bad one for two: a 22 GB Turing card (this pack's whole target) and a 24 GB Ada card are in the same bucket
while differing by 2× in compute and by Tensor Core generation. Item 1 and item 13 both need "is this card sm_75?" at
exactly this layer.

**The shape of the fix** is already in the tree: `prefill_chunk_default` already takes a bag of booleans and sizes, so
one more argument (or a `Costs` field threaded through `make_plan`) is a small change; the hard-coded `5943000000ull`
pool is the part that deserves a comment before it deserves a card branch.

**Risk.** The plan is what decides the expert cache's slots, so a capability-aware branch is a change to residency on
every card, not only on Turing. `vram_pool_bytes()`'s `5943000000ull` is already a proxy for "the pool after everything
else"; replacing a proxy with a card branch without a measurement on at least two cards risks making the common case
worse in order to model the rare one. The `Costs` field is the low-risk half, the default is the other half.

### 18. The serve stack's batch knobs are invisible to a Turing card's best path

**What the docs say.** doc-1 §425-459 (separate the logical batch from the microbatch; speculative decoding first,
model-free n-gram second) and doc-2 §9-§10 ("serving-stack specifics on Turing", "speculative decoding and scheduling")
are about keeping a GPU busy with several requests. Doc-2 §10 also carries the two measured claims that shape this
item: prefix caching gave "up to 31 % TTFT reduction", and dynamic batching gave "~an order of magnitude
(27 → 306 tok/s)", with the advice to prefer a serving engine for more than two users and `--parallel` for two to four.
On a bandwidth-bound card the useful version of this is *not* more concurrency: it is not paying for work the card
cannot use.

**Where it lands.** `serve/server.py` already decides the parallel stack per platform and per engine, and the decision
is explicit about what it has not measured:

- `PARALLEL_STACK_ENV` (`serve/server.py:3675-3679`) is nine switches (`STRATA_CKPT_POOL`,
  `STRATA_CKPT_POOL_READY`, `STRATA_SLOT_CKPT_MOVE`, `STRATA_SLOT_COPY_DIRECT`, `STRATA_SLOT_COPY_BATCHSYNC`,
  `STRATA_BATCH_GDN_GROUPS`, `STRATA_BATCH_ATTN_ROWS`, `STRATA_BATCH_SELF_COMMIT`, `STRATA_QSA_SLOTS`), and
  `parallel_stack_env()` (`serve/server.py:3702-3713`) turns them on only when `parallel_slots(cfg) >= 2`, the engine
  is a CUDA Qwen engine, and the platform is Linux — with the comment "Only where the stack was run alone: Linux CUDA,
  all nine. HIP got no measurement without replicas … Windows none at all". All nine reach real engine code — the
  checkpoint pool and the slot move at `src/program/generate.cpp:301` and `src/program/generate.cpp:340-348`, the rest at
  `src/core/verify.cpp:310-316` — so they are not cosmetic.
- The capability check is absent: `parallel_stack_platform()` (`serve/server.py:3697-3699`) is
  `sys.platform.startswith("linux")` and nothing else. A 4 × 2080 Ti box on Linux takes the same nine-flag stack as
  the Linux CUDA machine it was measured on.
- The GDN prompt speedups next to it (`serve/server.py:3719-3730`) are qualified with their own card gates, so the
  file already knows how to say "measured here, not there".

**What it is worth.** **Needs measurement**, and the risk is the interesting part: nine flags that were measured
together on one machine, silently enabled on a card class nobody measured, is exactly the "enable every experimental
switch" row doc-1 §664 warns against. The audit's concrete recommendation is a capability line in
`parallel_stack_platform()` — or, cheaper and more honest, a note in the serve log naming the card class whenever the
stack is enabled, so a Turing user's report is diagnostic instead of anecdotal.

**Also in this area, and already answered:** the DeepSeek engine answers one request at a time by design
(`strata_cli/launch/runner.py:1503`), so doc-1's microbatch sweep has no dimension to sweep there; and the Qwen
engine's `--parallel` (`strata_cli/launch/cmd.py:100`) plus the tuner's 17 `Arm(...)` candidates are the two places a
batching experiment would land.

**Risk.** These nine flags were measured together on one machine; switching them on by capability alone would repeat
the error the sm_75 series was created to avoid. `serve/server.py:3702-3713` already gates them on a config shape
rather than on a card, which is the conservative default — the item is a log line and a documented gate, not an
automatic enable. The flash-attention and convolution flags in the same stack are also numerics-changing for a
quantised cache, so each one needs its own parity run.

### 19. Fuse the all-reduce with the post-normalisation

**What the docs say.** `sm75optiResearch.md` §2366 proposes a small-message peer-read reduction and §2392 asks for the
all-reduce and the RMSNorm to be fused, because on a Turing card the pair is two launches of a few microseconds each
over the same row.

**Where it lands.** They are two kernels in this engine and they run back to back on the same row:
`if (nm > 1) allreduce(d.moe_out, T * D, ag, mi, ++d.ar_seq, d.xcnt, s);` (`src/dsv41/ds4_model.cu:10406`) is followed
immediately by `hc_post(d.moe_out, d.h, d.f_post, d.f_comb, D, H, T, d.h2, s)` (`src/dsv41/ds4_model.cu:10407`).
`hc_post_kernel` (`src/dsv41/ds4_kernels.cu:409-420`) then reads the row `moe_out` that the all-reduce just wrote, and
combines it with the residual and the hyper-connection mix.

**What the docs do not know, and the reason this is one item and not two:** the collectives they rank third are already
here and already gated correctly. `ar_rs_kernel` (`src/dsv41/ds4_kernels.cu:1773`) is a reduce-scatter and an
all-gather in one launch with `__threadfence_system()` on both sides of the flag (`src/dsv41/ds4_kernels.cu:1760`,
`src/dsv41/ds4_kernels.cu:1765`) — doc-2's two-shot shape — and `allreduce()` (`src/dsv41/ds4_kernels.cu:2656-2680`) takes it whenever
`nm > 2`. Below three members it uses the one-shot form instead, and that is arithmetic rather than an oversight: the
traffic saving of two-shot is `2(W-1)n/W` against `3n` at four ranks, which at two ranks is `n` against `n`. Both forms
also sum in member order per element (`src/dsv41/ds4_kernels.cu:1797-1806`, `src/dsv41/ds4_kernels.cu:1894-1903`), so
they agree bit for bit — the file's comment at `src/dsv41/ds4_kernels.cu:1768` says so and the code bears it out.
`sm75optiResearch_2.md`'s headline `1.4-6x` is at four to eight ranks, which is not a regime this engine gets wrong.

**What it is worth.** **Needs measurement**, and the mechanism is small: one launch less a layer and one read of a
`T * D` row saved, 40 layers a token. At one token that is 20,480 bytes a layer. The Qwen engine already treats this
class of saving as worth a switch (its `STRATA_PF_FUSED` family), so the measurement, not the idea, is the work.

**Risk.** `hc_post_kernel` is where the hyper-connection mix is applied, and it is the same kernel on the
`hc_post_inplace_kernel` path (`src/dsv41/ds4_kernels.cu:1909`) where "every stream's value read before any is
written" is the invariant. A fused form must keep that ordering for the leader's own row and must not run the
normalisation before the last member's contribution has landed — the flag wait is inside `ar_sum_kernel` and
`ar_rs_kernel`, so a fused kernel would have to carry the wait itself. Wrong here is not slow, it is a wrong answer
that only shows as drift.

## Already there: items the documents propose and the tree already does

These are listed so that nobody spends a week on them.

| Proposal | Where it already is |
| --- | --- |
| Replace `cp.async` with a load-then-store pipeline | Written twice: `src/dsv41/ds4_kernels.cu:1001` (packed attention, 16 rows a pass as `uint4`), `src/dsv41/ds4_attn_split.cuh:65` ("all of a warp's row loads are issued before any is converted"), and in `src/prefill/moe_fused_iq.cu:202-229` |
| Fuse the routing weights into the expert epilogue | `combine_kernel` at `src/dsv41/ds4_kernels.cu:1574-1584`: one pass over `w[t][j] * ed[row[t][j] * dim + e]`, then the CPU partial and the shared-expert partial |
| Keep routing metadata on the device | `topk_kernel` at `src/dsv41/ds4_kernels.cu:697` carries `__shared__ unsigned hist[256]` and a prefix/mask radix loop; `hc_mix_split_kernel` (`src/dsv41/ds4_kernels.cu:352-385`) counts on the device |
| Opt in to more than 48 KB of shared memory | `cudaFuncSetAttribute(..., MaxDynamicSharedMemorySize)` at `src/kernels/cuda/iq_kernels.cu:7041-7043`, `src/prefill/moe_fused_iq.cu:1705`, `src/prefill/gdn_chunk.cu:569-572`; carveout 100 at `src/kernels/cuda/fused_gr.cu:1244`, `src/kernels/cuda/qsa_prompt_attn.cu:1414` |
| Use `ldmatrix` to feed MMA fragments directly | `src/prefill/moe_fused_iq.cu:232`, `src/prefill/moe_fused_iq.cu:716`, `src/prefill/moe_fused_iq.cu:764-768`; `src/prefill/gdn_tc.cuh:125-134`; `src/kernels/cuda/native_qsa_score.cu:42-47` |
| Emulate `m16n8k16` with two `m16n8k8` steps | Written out with the register mapping explained at `src/kernels/cuda/qsa_prompt_attn.cu:52-56`, and again at `src/prefill/moe_fused_iq.cu:210-222` |
| Ask the driver for occupancy instead of guessing | `resident_ctas<auto Kernel>()` at `src/dsv41/ds4_iq.cu:1127-1140` |
| Sweep `__launch_bounds__` per kernel | Already measured as a dead end on this engine: `mx_gate_up_kernel<16,*>` sits at 119/94 registers on sm_75 = one block an SM, and a sweep of the Qwen config reduced prefill (recorded in [DSV41_SM75_ANALYSIS.md](DSV41_SM75_ANALYSIS.md)) |
| Use 16-byte units, whole-warp coalescing, no per-thread gather | `dq_rows_kernel2<T>` at `src/dsv41/ds4_iq.cu:1293` writes 32 values through shared memory 32 apart for exactly this reason; the Qwen side is `STRATA_TC_SM75`'s dequant store width (`include/strata/core/tc_sm75.hpp:21`) |
| Adaptive expert-cache admission and profile-based placement | Beyond the documents: `src/core/expert_cache.cpp`, `src/dsv41/ds4_experts_rule.hpp`, `--expert-profile` (`tools/make_profile.py`, `tools/make_profile_mix.py`), `DS4_STORE_POLICY` (`src/dsv41/ds4_model.cu:5438`). Doc-1's §255-291 research questions are answered by code, not pending |

## Considered and rejected, with the reason

| Proposal in the documents | Why it does not transfer |
| --- | --- |
| Native INT4 tensor cores (`8x8x32 s4`, `m8n8k32`) | `sm75optiResearch_2.md` §2 measures it on an RTX 2080 as ~230 cumulative HMMA cycles and calls it an experimental feature. Strata has no `s4.s4` MMA anywhere, and DS4.1's 4-bit experts already reach tensor cores through F16 tiles; the density is obtained by decoding, not by a narrower instruction |
| A W4A16 Marlin-style pipeline | The documents themselves say Marlin's pipeline is sm_80+ and that its numbers are not sm_75's. DS4.1's equivalent already exists in a different shape (`set_tc_kernel`, `src/dsv41/ds4_iq.cu:821`) |
| W8A8 SmoothQuant, QoQ/QServe W4A8KV4, deterministic int8 router GEMM | No such weight formats in this tree; the router is an F32 GEMV (`src/dsv41/ds4_model.cu:8351`, `src/dsv41/ds4_model.cu:10305`, `src/dsv41/ds4_model.cu:10421`). Building a kernel for a format no pack carries is not a harvest |
| A Triton fused-MoE autotune sweep | The engine is CUDA/C++ and the fused expert kernel already exists in four variants; adding a Python kernel stack for one tuning sweep does not pay |
| FlashAttention 2/3/4, FlashInfer, cuDNN SDPA | `cudaToolkitDoc.md` and both research documents agree these need sm_80+. Strata has its own attention kernels and never enters the vendored flash-attention path (`ggml_cuda_flash_attn_ext`, `third_party/llama.cpp/ggml/src/ggml-cuda/fattn.cu:145`), so the `TURING_MMA_AVAILABLE` branches in `third_party/llama.cpp/ggml/src/ggml-cuda/fattn-mma-f16.cuh` are dead code here |
| vLLM / SGLang / LMDeploy / KTransformers / TensorRT-LLM configurations | Those are other engines. The transferable part is the argument, not the flags: pipeline parallelism over tensor parallelism on PCIe (`sm75optiResearch_2.md` §7.2) is the same argument Strata's `--gpu-roles` and layer stages already make |
| Continuous batching, microbatch tuning, `--parallel` sweeps | The DeepSeek engine answers one request at a time by design (`strata_cli/launch/runner.py:1503`, "no `--batch` to pass"); there is no microbatch dimension to tune. The Qwen engine does have `--parallel` (`strata_cli/launch/cmd.py:100`), and doc-1's §425 advice to separate logical batch from microbatch applies there |
| llama.cpp's flag set: `-fa`, `-ctk`/`-ctv`, `-b`/`-ub`, `-ot`, `--cpu-moe`, `--n-cpu-moe`, `-ngl`, `-sm row`/`-sm tensor`, `-mla`, `-fmoe`, `--cache-reuse` | Every one has a Strata equivalent, and the audit maps them rather than repeating the flags: `--kv` for the cache type, `--prefill` for the batch, `--expert-cache`/`--experts` for residency, `--gpu-roles`/`--tp` for the split, `--spec`/`--dspark` for speculation, `--suffix-draft` for the model-free case |
| ECC off for bandwidth | Consumer 20-series cards have no ECC. The item survives only as persistence mode and locked clocks (item 9) |
| BF16 anything | Turing has no BF16 MMA. `cudaToolkitDoc.md` §1.4.2 and doc-1 agree; DS4.1 already asks cuBLAS for F16, not BF16 (`src/dsv41/ds4_model.cu:3458`) |
| Unified memory as a capacity strategy | Doc-1 §525 calls it a failure-recovery experiment, not a default. Strata's own VMM path (`src/core/vmm.cpp`) is the measured alternative |
| The "14× trick" of surgical `-ot` placement | The mechanism is per-layer expert placement, and Strata already does it adaptively (`src/dsv41/ds4_experts_rule.hpp`, `--vram-experts`, `--expert-profile`, `--adapt-every`, and the adapter arm in the tuner). The 14× figure is a llama.cpp study of a 35B-A3B model where the baseline was `--cpu-moe`; it is not a like-for-like comparison to this engine's default |

## Coverage: what each section of each source contributed

So that nothing was silently skipped. Subsection numbers roll up into their parent range: `sm75optiResearch_2.md`
§3.2 is inside §50-77; §6.2 (the "14× trick"), §6.3 (ik_llama.cpp), §6.4 (fewer-experts tricks), §6.6 (RPC) and §6.7
(what to expect on real hardware) are inside §115-173; §7.1 (PCIe physics) is inside §173-199; and §11 (the
don't-waste-your-weekend list) is answered row by row in the rejections table above. `cudaToolkitDoc.md` has no
Markdown headings; its section numbers are cited inline (§1.1-1.3, §1.4.1.1-1.4.1.4, §1.4.2, §1.4.3-1.4.3.1).

| Source section | Outcome |
| --- | --- |
| `sm75optiResearch.md` §15-59 (hardware limits, what Turing does well) | Feeds items 2, 3, 5 and the rejections (INT4, BF16, FP8) |
| §59-90 (find the real bottleneck) | Feeds the "limits" section: the bound `t ≥ max(W/B)` is the tool, but no number transfers |
| §90-210 (multi-GPU, interconnect) | Items 6, 8; the pair-versus-four-GPU experiment is already answered by `bench/results/` |
| §210-286 (MoE residency, admission, profiles) | Already implemented; see the "already there" table |
| §286-317 (quantisation as a system choice, FP16 accumulation) | Item 3 is the one usable piece; the rest is pack-authoring guidance |
| §317-376 (attention and long context) | Item 1, and the GQA/KV-reuse advice behind item 10 |
| §376-423 (GEMM/GEMV/dispatch, fusion, MMQ vs cuBLAS, registers) | Items 4, 5, 6; the fusion and register items are already done |
| §423-473 (batching and speculation) | Rejected for DS4.1 (one request at a time); item 5 is the kernel consequence of speculation |
| §473-531 (CPU/RAM/SSD, NUMA, mmap) | Rejected: the engine measures its own RAM tier (`STRATA_DS4_RAM_TIER`, `src/dsv41/ds4_model.cu:1173-1188`, `DS4_COPY_NUMA`), and doc-1's own advice is not to copy dual-socket findings onto this host |
| §531-539 (WSL) | Rejected: no WSL path in the release flow |
| §539-608 (engines and forks worth testing) | Rejected as engine choice; the Strata-specific candidates it names (stage-weight trimming, split-point tuning, pipeline windows) are the engine's own knobs and already documented |
| §608-680 (control build, graphs, lower-priority tuning, changes to avoid) | Items 7, 9; "changes to avoid blindly" is echoed in the rejections table |
| §680-734 (benchmark discipline, experiment sequence) | Feeds the limits section and the verification commands below |
| §425-459 (separate batch from microbatch, speculation order) | Item 18, and the rejection of a microbatch sweep for DS4.1 |
| §370-376 (recurrent and hybrid layers) | Item 16 — the GDN scan, which the tree has and gates to sm_80+ |
| §1556-2177 (SM75 kernel engineering: constraints, shapes, arithmetic paths, unpacking, load pipeline, MMA, registers, shared layouts, dispatch, fusion, split-K) | Items 2, 3, 5, 13; the shared-memory, `ldmatrix`, two-step-k16 and fusion items are already done |
| §2185-2290 (attention rebuilt around SM75) | Item 1 and its merge-order risk |
| §2290-2462 (multi-GPU inside kernels) | Items 6 and 19. The collectives they rank third are already better placed than the documents assume: the reduce-scatter/all-gather is `ar_rs_kernel` (`src/dsv41/ds4_kernels.cu:1773`) gated at three members or more, which is where two-shot actually beats one-shot; the fusion the section asks for (all-reduce into the following normalisation) is item 19; the "reduce after local expert aggregation" shape is already the engine's (`combine` then one `allreduce`, `src/dsv41/ds4_model.cu:10406`) |
| §2462-2515 (small operators; what existing code proves) | Feeds the rejections: RMSNorm/router/top-k fusion is done or F32 by design |
| §2515-2560 (kernel development programme) | Feeds the verification commands: benchmark the whole routed operator, not the GEMM |
| `sm75optiResearch_2.md` §1-38 (silicon, hardware wins) | Items 8, 9; ECC narrowed to consumer hardware |
| §38-50 (hardware-level wins) | Item 9 |
| §50-77 (the quantisation decision tree) | Rejected: pack authoring, not engine work |
| §77-94 (engine matrix for SM75) | Rejected: engine choice; the usable part is item 4's "what a Turing build compiles" |
| §94-115 (single-GPU llama.cpp deep dive) | Rejected as flags; the `--kv` and `--prefill` equivalents are already the engine's defaults |
| §115-173 (the hybrid-MoE centrepiece) | Already implemented adaptively; the "keep early-layer experts on the GPU" heuristic is a policy the adapter learns |
| §173-199 (multi-GPU scenarios, PCIe physics) | Items 6, 8 |
| §199-232 (reference builds, serving stack) | Rejected as engine choice; the transferable half is item 18 — the batch stack's own gates and what they do not check |
| §232-254 (speculation, scheduling, measurement discipline) | Item 5, item 18 (the scheduling half), and the limits section |
| §254-318 (priority action list) | The priority order of this page follows it where it survives |
| §318-443 (kernel deep dive: ISA, decode GEMV, narrow-M, quantised GEMM architectures, attention, MoE, collectives, graphs, profiling) | Items 2, 3, 4, 5, 6, 7, 14; §2's accumulator-cliff measurement is the source of item 3 |
| `cudaToolkitDoc.md` §1.1-1.3 (architecture, best practices, compatibility) | Confirms item 15: "Volta binaries using Tensor Cores will only be able to reach half of Turing's Tensor Core peak performance", so an arch-targeted build is not optional |
| §1.4.1.1-1.4.1.2 (instruction scheduling, independent thread scheduling) | Item 14; the `__syncwarp` requirement is why `racecheck`/`synccheck` matter |
| §1.4.1.3-1.4.1.4 (occupancy, integer arithmetic) | Already done: 32 warps/SM and 64 KB are respected, and the interleaved GEMV is the integer/FP overlap |
| §1.4.2 (Tensor Core operations) | Items 2, 3 |
| §1.4.3-1.4.3.1 (memory throughput, L1/shared carveout) | Already done; the carveout is set explicitly where it matters |

## Limits of this audit

1. **Nobody on this project has an sm_75 card.** `docs/DETAILS.md:1069-1072` says so, and
   `include/strata/core/tc_sm75.hpp:12-14` repeats it ("0.1.45: the Turing paths are opt-in (nobody here has an sm_75
   card to confirm them): STRATA_SM75_PERF=1"). Every Qwen-engine number in this
   page comes from community reports or the engine's own author, not from a card anyone here can re-measure.
2. **The DeepSeek release package cannot run on a 2080 Ti.** `packaging/build_dsv41_cuda.sh:18` builds
   `86-real;89-real;120` (SASS for those three, PTX for sm_120 only, per the script's header at
   `packaging/build_dsv41_cuda.sh:8`). Items 1-7 were measurable only because the lab was built by hand for sm_75.
3. **No `compute-sanitizer` coverage**, so the flag protocols in `src/dsv41` have been reviewed by eye (item 14).
4. **Cross-GPU experiments are limited on this box**: peer access is enabled only in `src/core/peer_experts.cpp`, the
   two NVLink pairs are not all-to-all, and the cross-pair link goes through the host bridge, so item 6's small
   all-reduce numbers will be dominated by that asymmetry rather than by the qualifier.
5. **The three documents mix sources of very different strength.** Doc-1's second half cites a Turing chat that the
   documents themselves label as community material, and doc-2's §13 cites a T4 benchmark repository for
   "11× from batching". None of that was checked against the underlying sources for this page; the page only used
   claims that either had a measurement or a mechanism that could be checked in the code.

## How to check each item cheaply

| Item | Command | What it settles |
| --: | --- | --- |
| 1 | `strata-dsv41` with `DS4_PROF=1`, then with `DS4_ATTN_SPLIT=1`, 5 runs each, same flags | whether attention holds the critical path at all |
| 2 | `build/ds4_iq_test --bench-tiles <types> 8 1,2,4` (`tests/dsv41/iq_expert_test.cpp:334`) with and without the Turing setup | whether the i-quant tiles beat the F16 tiles on this card before any engine change |
| 3 | the same tile bench with the accumulator fragment changed to `half` and promoted once a k-slice | whether the ~14 % survives a 2 % error budget |
| 4 | the vendored `ggml-cuda` test target for MMQ with a Turing tile config, A/B against the Ampere one | whether an Ampere tile is the right tile on a 64 KB card |
| 5 | `ds4_lab --tokens 2,4,8,16 --only <the verify shapes>` | the narrow-M window's cost before a kernel exists |
| 6 | `ds4_lab --only allreduce`, with and without `__restrict__` on `ar_rs_kernel` | the qualifier's effect, isolated from the collective |
| 7 | `DS4_SYNC=1` with and without a captured layer body, `DS4_PROF_SYNC=1` for the launch gaps | whether launch overhead is measurable at all here |
| 8-9 | `nvidia-smi -q -d CLOCK,PERFORMANCE`, `nvidia-smi topo -m` and `nvidia-smi -q -d PCIE` during a run | the machine facts the log does not record |
| 10 | `build/mmvq_il_parity --bench --emit-table` on a 2080 Ti, then again with 5-8 columns | the missing `kIlRows58_75` cells for the verify window |
| 11-12 | `strata tune` on a 2080 Ti, and a `ds4_arch_defaults()` unit test without a card | whether either layer can be card-specific today |
| 13 | `STRATA_MMQ_WS=1` on an sm_75 card, after relaxing `mode()` | whether the warp-specialised kernel's shared-memory budget fits 64 KB |
| 14 | `compute-sanitizer --tool racecheck` on the flag-protocol kernels only | whether the protocols are sound |
| 15 | `ARCH="75-real;86-real;89-real;120" packaging/build_dsv41_cuda.sh` and run the package on a 2080 Ti | whether the DeepSeek engine can ship to a Turing card at all |
| 16 | `gdn_rec_parity` with a Turing instantiation of `src/prefill/gdn_tc.cuh` (the two-step k=16 split) | whether the scan beats `gdn_rec_kh_kernel` on this card within the 1e-4 gate |
| 17 | `strata-plan --vram <bytes>` on two cards with equal VRAM and different compute | whether the plan can say anything the VRAM size cannot |
| 18 | `strata launch --parallel 2` on a 2080 Ti with the nine `PARALLEL_STACK_ENV` flags on and off | whether a stack measured on one machine helps or hurts here |
| 19 | `ds4_lab --only allreduce` with `DS4_AR_F16=1` on a two-GPU box, then a fused post-normalisation | whether the fused pair is worth a kernel, and whether it keeps the hyper-connection ordering |

## Sources

The three documents are not in this repository. They are at `sm75optiResearch.md`, `sm75optiResearch_2.md` and
`cudaToolkitDoc.md` in the reader's home directory. The repository's own findings are in
[DSV41_SM75_ANALYSIS.md](DSV41_SM75_ANALYSIS.md) (DeepSeek-V4.1 candidates, measured),
[SM75_SERIES_WHY.md](SM75_SERIES_WHY.md) (the Qwen engine's Turing switches, measured),
[DEEPSEEK41_ENGINE_MAP.md](DEEPSEEK41_ENGINE_MAP.md) (the engine's shape), [DETAILS.md](DETAILS.md) (the technical
reference) and [MMVQ_IL_TABLE.md](MMVQ_IL_TABLE.md) (the pattern a per-card table follows).
