# Why the RTX 20 (sm_75) switches exist

This page answers one question: why does the Qwen engine have an opt-in Turing kernel series, and why is it built the
way it is? It is written for a maintainer or a contributor who wants to change a default on a compute capability 7.5
card, or who wants to know why a Turing kernel is off. The DeepSeek V4.1 engine has no such series; see
[DSV41_SM75_ANALYSIS.md](DSV41_SM75_ANALYSIS.md) for what would be entailed there.

The short answer: Turing kernels arrived as community pull requests, each measured on one RTX 2080 Ti. Nobody on the
team owns a Turing card. So instead of a per-architecture defaults table (the shape Ampere, Pascal and GB10 use), the
series became one master switch that a Turing owner turns on deliberately.

## The three mechanisms the engine uses for a per-card default

The engine has three ways to give a card a different kernel. They differ in who takes the risk.

| Mechanism | Shape | Who takes the risk | Used for |
| --- | --- | --- | --- |
| a defaults table | `src/core/nv_defaults.cpp:26`, `:41`, `:53` set environment variables before the run | the team: measured on our own cards, on by default | sm_86 (`src/core/nv_defaults.cpp:26`), sm_60 (`:41`), sm_121 (`:53`) |
| an inline arch predicate | `strata::sm75_default(major, minor)` at the call site | the team, per kernel | the Turing kernels the team has since measured |
| an opt-in series | one master switch, `STRATA_SM75_PERF=1` | the user | every Turing kernel nobody here could confirm |

`nv_defaults.cpp` has a table for sm_86, sm_60 and sm_121 only. There is no sm_75 branch — check the three conditions
at `src/core/nv_defaults.cpp:26`, `:41` and `:53`: Turing cannot match any of them. That absence is the structural
reason the series exists. A Turing card gets no defaults, so any Turing kernel has to be asked for by name.

## The series switch

`include/strata/core/sm75_series.hpp` is the whole mechanism; it is 46 lines.

| Function | Line | What it answers |
| --- | --- | --- |
| `sm75_series()` | `include/strata/core/sm75_series.hpp:25` | `STRATA_SM75_PERF` set and non-zero. Read once, into a `static const bool`. |
| `is_sm75(major, minor)` | `include/strata/core/sm75_series.hpp:34` | the device is 7.5, or emulates it (`STRATA_EMULATE_CC=75`) |
| `sm75_default(major, minor)` | `include/strata/core/sm75_series.hpp:37` | 7.5 **and** the master switch. This is the "follow the series" default. |
| `sm75_switch(name, major, minor)` | `include/strata/core/sm75_series.hpp:40` | 7.5 only; the named variable wins in both directions; unset it follows the master switch |

Two properties matter and both are enforced in `sm75_switch`:

1. A non-Turing card cannot be moved onto these kernels. `sm75_switch` returns false before it reads the variable
   (`include/strata/core/sm75_series.hpp:41`), and `sm75_default` requires `is_sm75` first (`:37`). The header states
   this in words at `include/strata/core/sm75_series.hpp:10-11`.
2. The per-kernel variable wins over the master switch, so a user can turn one kernel off inside an otherwise on
   series, or on inside an otherwise off series.

`STRATA_EMULATE_CC=75` counts as a Turing card (`include/strata/core/emulate.hpp`), which is how the series is
exercised on an sm_86 box. It also sets the 64 KiB shared-memory cap of a real Turing part
(`include/strata/core/emulate.hpp:30`).

The rationale is written into the header at `include/strata/core/sm75_series.hpp:3-5`: the kernels came from PRs
#2069-#2072, #2081, #2091-#2093, #2145-#2146, #2220-#2223 and #1821, measured on one RTX 2080 Ti and a CMP 50HX, and
are bitwise equal to the kernels they replace. The commit that made them opt-in is `7f1964a6` ("No sm_75 card is
available here to confirm the series").

## The four switches

The four the investigation asked about are one master switch and three per-kernel switches, and they are not the same
kind of thing: one is a policy flag, two select kernel variants, one selects a store width.

### `STRATA_SM75_PERF` — the master switch

Read at `include/strata/core/sm75_series.hpp:25`. It selects no kernel itself: every consumer goes through `sm75_default`
or `sm75_switch`, so the master switch only flips defaults. Grep finds 9 `sm75_default` call sites, 3 `sm75_switch`
sites (the two per-kernel variables plus `include/strata/core/tc_sm75.hpp:22`), one direct `sm75_series()` read at
`src/kernels/cuda/qsa_prompt_attn.cu:2450`, and 2 more through `device_is_sm75()` (`src/core/verify.cpp:399`), which is
the same predicate memoised per device:

| Consumer | Line | What flips |
| --- | --- | --- |
| `q2_0_warp_rows_on()` | `src/kernels/cuda/native_mmvq.cu:3192` | one warp per four Q2_0 rows at 1-4 columns |
| `sm75_il_on()` | `src/kernels/cuda/native_mmvq.cu:3909` | interleaved-row multi-column GEMV (its own switch) |
| `exp_mode()` | `src/kernels/cuda/iq_kernels.cu:6178` | grouped native experts, layout mode 4 |
| `up75_on()` | `src/kernels/cuda/fused_gr.cu:1342` | `gr_up75_kernel` for 3+ tokens |
| `down_tg()` | `src/kernels/cuda/fused_gr.cu:1367` | token pairs for the staged down projection |
| `scores_thread_on()` | `src/kernels/cuda/qsa_select.cu:2068` | thread-per-head block scorer |
| `scores_halve_on()` | `src/kernels/cuda/qsa_select.cu:2083` | halving-reduction block scorer |
| `topk_split_device()` | `src/kernels/cuda/qsa_select.cu:2635` | split top-k on the device |
| `topk_sample_on()` | `src/kernels/cuda/qsa_select.cu:2868` | one-read prompt top-k |
| `attn_v3_on()` | `src/kernels/cuda/qsa_decode_attn.cu:449` | QSA decode attention v3 for int8 pools |
| `device_is_sm75()` | `src/core/verify.cpp:411` | the two consumers below |
| `fuse_wait_on()` | `src/core/verify.cpp:438` | the plan's host-flag wait folded into its copy |
| `quant_in` | `src/core/verify.cpp:1498` | q8_1 and its interleaved copy in one launch |
| `launch` selection | `src/kernels/cuda/qsa_prompt_attn.cu:2450` | v1 prompt attention with register prefetch (this one reads `sm75_series()` directly) |

The kernel inventory behind those call sites, with the numbers, is in the commit bodies and in each kernel's own
comment. The measured ones:

**Measured on:** RTX 2080 Ti 22 GB for every number in the table below, summed up by the community pull requests named
in the right column; the tool guard row is unmeasured, as its column shows.

| Kernel | File:line | Result on an RTX 2080 Ti |
| --- | --- | --- |
| mode 4 grouped experts | `src/kernels/cuda/iq_kernels.cu:6178`, `native_gu_amd_kernel` at `:5608`, `native_down_amd_kernel` at `:5665` | 10 groups x 1 entry 47.3 → 41.4 µs, 30 x 1 120.2 → 100.2 µs (commit `26855986`) |
| mode 4 guard | `sm75_down_em` at `src/kernels/cuda/iq_kernels.cu:6189` | mode 4 loses with 3+ entries per group: 12 x 3 36.5 → 40.8 µs, 16 x 4 56.9 → 68.9 µs, so `cap_groups > 30` keeps mode 0 |
| `gr_up75_kernel` | `src/kernels/cuda/fused_gr.cu:556`, launched at `:1502-1508` | T3 53.2 → 50.2 µs, T4 64.1 → 57.6, T5 76.2 → 69.3, T6 86.1 → 70.9 (commit `80872d43`) |
| down token pairs | `down_tg` at `src/kernels/cuda/fused_gr.cu:1349`, used at `:1430` | T4 29.1 → 27.3 µs, T6 38.5 → 32.2 µs |
| `q2_0_warp_rows_kernel` | `src/kernels/cuda/native_mmvq.cu:3141`, used at `:3209` | 1/2/3/4 columns 36.6/60/68/88 → 24.8/31/31/37 µs (commit `bd455008`) |
| `attn_chunk_q8_v3_kernel` | `src/kernels/cuda/qsa_decode_attn.cu:342`, used at `:1018` and `:1210` | 2051 cells, 1-4 queries 51.6/78.5/110/146 → 34.5/41.0/73.0/84.9 µs (commit `996b9f84`) |
| `block_scores_halve_kernel` | `src/kernels/cuda/qsa_select.cu:1968`, used at `:2100-2111` | 420K cells, 1/3/4/6/8 queries 117/265/347/510/774 → 101/118/125/238/255 µs (commit `050229ec`) |
| `qsa_block_topk_split` | `src/kernels/cuda/qsa_select.cu:2640`, used at `:2893` | decode calls at 420K 0.21 → 0.033-0.064 ms |
| `block_topk_sample_kernel` | `src/kernels/cuda/qsa_select.cu:2696`, used at `:2951` | one read of the scores above ~115K cells |
| `block_scores_thread_kernel` | `src/kernels/cuda/qsa_select.cu:1151` and `:1256`, used at `:2143`, `:2157` | unmeasured on Turing in the comment; +1..3% on a P100 at 30-60K tokens |
| folded wait and fused quantize | `src/core/verify.cpp:438`, `:1498` | plan fold +0.6..+1.4% decode, fused quantize +0.5..+1% (108.0 → 109.1 tok/s, commit `88a1c23f`) |
| v1 prompt attention, register prefetch | `src/kernels/cuda/qsa_prompt_attn.cu:2450` | unmeasured for this flag in the comment; #2146's one-stage variant is the measured one (below) |

### `STRATA_SM75_EXP` — grouped expert layout

Gated at `src/kernels/cuda/iq_kernels.cu:6178` by `sm75_switch`, so it is Turing-only and follows the master switch
when unset. It sets `exp_mode()` to 4 (`:6179`: `1 + (... turing ? 4 : kExpModeDefault)`), and `kExpModeDefault` is 0
for CUDA (`:5548`), so the default Turing layout is the plain CUDA layout.

Mode 4 is one of the alternating layouts the file describes at `src/kernels/cuda/iq_kernels.cu:5537-5545`. It is not a
Turing-specific kernel: it is the shared "layout mode" family (`native_gu_amd_kernel`, `native_down_amd_kernel`) whose
other members Volta (`:6175`) and the P40/P100 (`:6169-6173`) use by default at mode 8. The Turing choice of mode 4 is a
measured layout preference, not new arithmetic — bitwise equal to mode 0 in the commit body.

The guard is the interesting part: `sm75_down_em` (`src/kernels/cuda/iq_kernels.cu:6189`) refuses mode 4 for the down
projection when the call has more than 30 groups, and only when nothing forced a mode (`g_exp_mode < 0`,
`STRATA_EXP_MODE` unset). It calls `strata::is_sm75` at `src/kernels/cuda/iq_kernels.cu:6195`, not `sm75_default`, so
with `STRATA_SM75_EXP=1` the guard still applies unless `STRATA_EXP_MODE` is also set. That is deliberate: the guard
encodes a measured crossover, not a preference.

### `STRATA_SM75_IL` — interleaved-row multi-column GEMV

Gated at `src/kernels/cuda/native_mmvq.cu:3909` by `sm75_switch`, then consumed by `il_arch_ok()`
(`src/kernels/cuda/native_mmvq.cu:3914-3927`) and `native_mmvq_il_supported()` (`:3930-3934`).

Three independent conditions must all hold before a Turing card runs it:

1. `ncols` in 1..8 and `g_multi_exact` (`:3931`) — the exact multi-column layout.
2. `il_arch_ok()` — for 7.5 that is `sm75_il_on` (`:3923`).
3. the type is 11, 12, 13, 14 or 23 — Q3_K, Q4_K, Q5_K, Q6_K, IQ4_XS (`:3933`).

The table is `kIlRows_75` (`src/kernels/cuda/native_mmvq.cu:3708-3714`), registered in `kIlArch` at `:3786`. It says how
many matrix
rows one warp takes for each (type, token count, row-count class); 0 means "use the plain kernel". Its comment
(`:3704-3707`) records why it exists: on Turing the plain multi-column kernels are bound by `__dp4a` throughput from 2
columns on (IQ4_XS 2560x10240 at 4 columns 81.1 → 37.4 µs; the Q5_K head 1671 → 1045 µs, both on the 2080 Ti). No
one-column table exists on purpose: the engine's single-column kernels sum in another order and gained 0.5%.

Every table cell is bitwise the plain kernel's output; only the rows-per-warp changes. `docs/MMVQ_IL_TABLE.md:21` says
"sm_75 and older: the path is off", which is the state with the series unset — the table exists for
`STRATA_SM75_PERF=1`.

### `STRATA_TC_SM75` — dequant store width

This is the only one of the four that lives in its own header, `include/strata/core/tc_sm75.hpp:21`, and it is
straightforward: `tc_sm75_enabled(major, minor)` is `sm75_switch("STRATA_TC_SM75", major, minor)` and nothing more.

Its sole consumer is `src/kernels/cuda/dequant_bf16.cu:391`, inside the `vec` predicate at `:387-391`:

```cpp
const bool vec = (reinterpret_cast<uintptr_t>(out) & 15) == 0 && ((uint64_t) ld * sizeof(T)) % 16 == 0 &&
                 cudaGetDevice(&dev) == cudaSuccess && ... && strata::core::tc_sm75_enabled(cmaj, cmin);
if (vec) dequant_kernel<TY, T, true><<<grid, 256, 0, st>>>(...);
else     dequant_kernel<TY, T, false><<<grid, 256, 0, st>>>(...);
```

The `VEC` template parameter is described at `src/kernels/cuda/dequant_bf16.cu:172-176`: the plain form writes each
element where it is computed, so a warp issues 32 store instructions whose lanes are 64 bytes apart — 32 two-byte
transactions each. The `true` form dequantises into a 32-element register buffer and flushes it as `uint4` stores
(`src/kernels/cuda/dequant_bf16.cu:186-193`). The values are identical; only the stores are wider. The comment at `:173`
records the Turing build measuring this kernel at 18% of the row's bandwidth, and commit `d976db34` records the cause:
the per-group kernel writes 2 bytes every 64 bytes, and 0.1.43 had only coalesced the Q8_0 path (#1720).

The name is misleading and worth stating plainly: **it is not an MMA and does not touch any tensor core.** `tc_`
prefixes the header because the file's other sm_80+ paths are tensor-core paths, but the switch selects a store width.

Two neighbouring switches are easy to confuse with it, and neither is part of this one:

- `dequant_wide_on()` (`src/kernels/cuda/dequant_bf16.cu:286`) selects `dequant_wide_kernel` (`:272`) for types
12/13/14/23. It is on
  by default for sm_61, sm_86 and sm_80 (`:300-301`) and has never been on for Turing.
- `STRATA_DEQUANT_WIDE` (`:287`) is its own variable (`docs/DETAILS.md:1067`).

## Why not a defaults table

The three-cards-in-one-table route was taken for sm_86 and sm_60 because the team ran the A/B itself. The numbers in
`src/core/nv_defaults.cpp:29-38` are drop-one measurements on an RTX 3060 and an RTX A4000 with 8 interleaved pairs, and
the identity gate was checked. Turing has no such data, and `nv_defaults.cpp` applies a table only where a measurement
exists (`:100` prints "exact" only for the bit-exact sets).

There is a second reason, specific to Turing: `stderr` transparency. A defaults table prints a line naming the
switches it set (`src/core/nv_defaults.cpp:100-105`) and a real off switch (`STRATA_SM86_PERF`, `:27`). The series has
the same property with a different default direction: `STRATA_SM75_PERF=1` and each kernel's own variable.

## What stays outside the series

Bitwise-equal kernels are safe to flip from a distance; kernels that round differently are not. The header lists the
exceptions at `include/strata/core/sm75_series.hpp:13-15` and `docs/DETAILS.md:1073-1077` repeats them. They keep their
own explicit opt-ins:

| Switch | Where | Why it is separate |
| --- | --- | --- |
| `STRATA_BF16_TC=1` | `src/prefill/gemm.cu:830-841` | BF16 operands run on the FP16 tensor cores; clamped at ±65504, and sums round differently. Turing is the one 7.x card where it is off by default (`src/prefill/gemm.cu:840`: `return cc < 75 ? 1 : 0`) |
| `STRATA_SELECT_SIMT=1\|2` | `src/kernels/cuda/qsa_select.cu` | tiled FP32 block scorer; another summation order |
| `STRATA_PA_V2T=1` | `src/kernels/cuda/qsa_prompt_attn.cu:2443-2445` | v2 int8 prompt attention with register staging in place of `cp.async` |
| `STRATA_PF_FUSED_SM75=1` | `docs/DETAILS.md:1076` | fused int8 prompt experts on a Q2_0 pack, Turing only |
| `STRATA_XN_F16` | `docs/DETAILS.md:1077` | hyper-connection xn16 as FP16 |

`STRATA_BF16_TC` is the clearest example of the policy. It is the biggest Turing number in the repository — 20-34%
prompt on a Quadro RTX 8000 and 15-18% on an RTX 2080 Ti (`docs/OLDER_GPUS.md:336`) — and it is still off by default
on Turing because it changes replies (3 of 9 paired requests on the Quadro).

## The hardware reason the series exists at all

Turing is a real discontinuity, not a slow Ampere. The facts the code relies on are stated in
`include/strata/core/tc_sm75.hpp:3-8`:

- no TF32 MMA and no `cp.async` (both sm_80+), so prompt kernels that use them are compiled out below sm_80;
- fp16 MMA exists, but only `m16n8k8`; `m16n8k16` is sm_80+ (`src/kernels/cuda/qsa_prompt_attn.cu:52-56` shows the
  two-instruction emulation and notes that the two sums round twice, so that path is *not* bitwise the k16 one);
- cuBLAS has no BF16 GEMM below sm_80, so it runs an FFMA one — the reason `STRATA_BF16_TC` exists
  (`src/prefill/gemm.cu:774-776`: "a fifth of a 4K prompt on an RTX 2080 Ti", recorded as 19.5 % in
  `include/strata/core/tc_sm75.hpp:6`);
- 64 KiB of shared memory per block (`include/strata/core/emulate.hpp:30`), half of Ampere's 100 KiB default, which is
  why the prompt attention kernels have separate Turing shared-memory layouts (`src/kernels/cuda/qsa_prompt_attn.cu:105`,
  `Smem2T` at `:510`) and why one kernel went from 45.6 KB to 19.7 KB a block to fit three blocks an SM (commit
  `5d4c7867`).

Four facts explain every kernel in the series: fewer active warps per SM than Ampere, `__dp4a` throughput as the
binding constraint on multi-column matvec, no `cp.async` for prefetch, and half the shared memory.

## How to run it

The series belongs to the Qwen families. Set it before the engine starts, or in the model config's `env` object, then
restart:

```sh
STRATA_SM75_PERF=1 strata launch --family qwen --model q2_0 --yes --no-tui
```

To check one kernel at a time, set the kernel's own variable: `STRATA_SM75_EXP=0` keeps the CUDA expert layout,
`STRATA_SM75_IL=0` keeps the plain matvec, `STRATA_TC_SM75=0` keeps the per-group dequant. To exercise the series on a
newer card, set `STRATA_EMULATE_CC=75`.

`strata launch --family deepseek41` does not read any of these four switches: no source under `src/dsv41` includes
`include/strata/core/sm75_series.hpp` or `include/strata/core/tc_sm75.hpp`. What a Turing card can do there is
[DSV41_SM75_ANALYSIS.md](DSV41_SM75_ANALYSIS.md).

A measurement of the full series on an RTX 20 has not been published; the numbers above are per-kernel
microbenchmarks or per-switch A/Bs. The honest summary is in `docs/DETAILS.md:1071-1072`: nobody on the team has an
sm_75 card to confirm the series.

## History / older measurements

| Date | Change | Note |
| --- | --- | --- |
| 2026-10-12 to 10-14 | PRs #2069-#2072, #2081, #2091-#2093, #2145-#2146, #2220-#2223, #1821 merged | each on by default for 7.5 in its own PR; all measured on one RTX 2080 Ti and a CMP 50HX |
| 2026-10-14 | `7f1964a6` | every bitwise Turing kernel made opt-in behind `STRATA_SM75_PERF=1`, with a per-kernel variable each; the one-read prompt top-k (#2093) was also narrowed from "every CUDA card" to Turing |
| 0.1.45 | the series documented | `docs/DETAILS.md:1069`, `include/strata/core/sm75_series.hpp:13` |
