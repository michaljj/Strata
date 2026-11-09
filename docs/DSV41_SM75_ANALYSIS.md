# sm_75 performance mods for the DeepSeek-V4.1 engine: an analysis

This page answers one question: of the performance work that exists in Strata's Qwen engine for Turing cards (sm_75,
the RTX 20 series and the T4), what could be applied to the DeepSeek-V4.1 engine in `src/dsv41`, and what would it be
worth on an RTX 2080 Ti?

It is an analysis, and nothing here is a claim of a speed-up. Every candidate carries either a measurement I made on
this machine or the label **needs measurement**. No engine code was changed to write it. The engine's shape is in
[DEEPSEEK41_ENGINE_MAP.md](DEEPSEEK41_ENGINE_MAP.md); why the Qwen engine has its Turing switches is in
[SM75_SERIES_WHY.md](SM75_SERIES_WHY.md).

**Measured on:** 4 x RTX 2080 Ti 22 GB (sm_75), Xeon E5-2680 v4, 121.4 GiB RAM, Ubuntu 24.04, driver 595.91.07,
CUDA 13.4, project 0.1.55 (commit c62d5e3a), engine ds4-0.1, pack `deepseek-v4.1-flash-ist-das-2.00bpw`.

## The short answer

Three things are worth naming, and the order matters: the largest measured prompt win on this machine is a bandwidth
item, not an instruction-rate item. The int8 tensor-core expert GEMM is the largest *port* candidate, and it is a port
of something `src/prefill/moe_fused_iq.cu` already contains for Turing and DS4.1 never calls.

| Item | Measured on this machine | Where it would go |
| --- | --- | --- |
| the i-quant expert GEMM on `m8n8k16` int8 tensor cores | the instruction runs at 244-252 TOP/s here against 62-63 TOP/s for the `__dp4a` the decode dots use: **4.0x**; against the F16 tiles that actually run the prompt: **about 2x**. It covers **27 %** of this pack's expert bytes | the prompt tile pair `set_kernel` / `set_tc_kernel` at `src/dsv41/ds4_iq.cu:733`, `:821`, fed from a Turing instantiation of `src/prefill/moe_fused_iq.cu` |
| the split-KV attention kernel, in the tree and off by default | 37.4 µs against 142.1 µs at one query, 3.8x; 187.6 against 422.0 at four, 2.25x | `split_enabled()` at `src/dsv41/ds4_kernels.cu:2027` |
| `cp.async` row prefetch for the packed attention | **no**: sm_80 instruction, and the sm_75 substitute is already there | `src/dsv41/ds4_kernels.cu:1001` |

Read with the bandwidth evidence below, the ranking is: the chunk hand-off and its expert streaming first, then
the int8 expert GEMM, then the attention kernel. An instruction-rate win cannot beat a stage that is waiting on PCIe.

The sections below are numbered in the order they were investigated, not by size. The int8 tensor-core expert GEMM has
no number because it is not one of the candidates that came out of the Qwen engine: it gets its own section, first.

Everything else that the Qwen engine learned about Turing is either already present in DS4.1 (the WMMA expert tiles, the
DP4A integer dots, the 16-byte-store dequantiser) or does not transfer, because DS4.1's decode is not the kind of work
the Qwen engine's Turing kernels were written for. The reason is the next section.

## The largest item: the i-quant expert GEMM on int8 tensor cores

This is what the analysis was worth doing for, and it is not in the Qwen engine's sm_75 series at all. It is in
`src/prefill/moe_fused_iq.cu`.

**What exists.** That file is the fused int8 prompt-expert GEMM for native GGUF packs. It decodes a quantised expert's
blocks into int8 tiles in shared memory, runs `mma.sync` integer products, and turns each int32 dot into a float with one
add (the `0x4B400000` magic-offset trick, `src/prefill/moe_fused_iq.cu:5-14`). It covers gate/up IQ2_XXS, IQ2_XS, IQ2_S,
IQ3_XXS, IQ3_S, IQ4_XS (`src/prefill/moe_fused_iq.cu:1828-1829`) and down Q2_0, IQ4_NL (`:1829-1830`).

**How much of this pack that is, measured from the pack's own tensor list.** That list is
`/home/jurek/Strata-data/packs/deepseek-v4.1-flash-ist-das-2.00bpw/tensors.txt`; the types are per layer, not per pack,
and this pack mixes them:

**Measured on:** `tensors.txt` of pack `deepseek-v4.1-flash-ist-das-2.00bpw` (the 48793-line tensor list the engine
loads), summed from its `bytes` column; nothing run.

| Role | Types in this pack | Experts | GiB | Of which `moe_fused_iq.cu` covers |
| --- | --- | ---: | ---: | ---: |
| gate/up (`w1`,`w3`) | IQ1_M 12288, IQ2_XXS 13056, IQ2_XS 5376 | 30720 | 83.6 | IQ2_XXS + IQ2_XS = 54.1 GiB, 24 of 40 layers |
| down (`w2`) | IQ1_M 10752, IQ2_XXS 3072, IQ2_XS 1536 | 15360 | 39.4 | IQ2_XXS + IQ2_XS = 13.6 GiB, 12 of 40 layers |
| both roles of one layer | — | — | 123.0 | **10 of 40 layers, 33.7 GiB, 27 %** |

The covered layers are 10, 18, 22, 23, 25, 26, 28, 33, 38 and 39. Fourteen more (11, 12, 15, 17, 19, 24, 27, 29, 31,
32, 34, 35, 36, 37) have a covered gate/up and an **IQ1_M down**, so they could take the GEMM for gate/up and would
still have to run the current decode kernel for down.

So this candidate is not "the pack's formats are covered". It is **27 % of the expert bytes in the ideal case, 44 %
(gate/up only, 24 layers) if a layer may mix the two kernels** — and `IQ1_M`, 55.4 of the pack's 123.0 GiB of expert
bytes (45 %), is not
covered by this file at all. That is a real limit on the ceiling below, and it is measured rather than assumed.

**A second finding, and it changes the shape of the port.** `IQ1_M` is not covered by `moe_fused_iq.cu`, but
`src/dsv41/ds4_iq.cu` **does** decode it, on the tensor cores, in `set_tc_kernel` (`src/dsv41/ds4_iq.cu:821`): `deq8<T>`
converts the block to 8 floats and the tile goes into shared memory as F16. So for this pack the int8 GEMM is not
strictly an upgrade over a scalar path — it is a choice between two tensor-core paths, F16 WMMA (what runs) and int8 MMA
(what `moe_fused_iq.cu` has). The 4.0x ratio below is int8 MMA against `__dp4a`; against the F16 tiles the engine
actually uses for prompts, the comparison is 244-252 TOP/s against 124-126 TFLOP/s, i.e. **about 2x**, and any Turing
port has to argue that 2x survives its own load stage. The page says both numbers because both are real.

**What already handles Turing.** The file has a real Turing branch, not a fallback. `src/prefill/moe_fused_iq.cu:201-229`
under
`__CUDA_ARCH__ >= 750` defines `cp16` as a plain `__ldg` copy (`src/prefill/moe_fused_iq.cu:205`), `cp_commit`/`cp_wait`
as no-ops, and
the signed-int8 MMA as `mma8_s8` = `mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32`
(`src/prefill/moe_fused_iq.cu:210-211`) — because
Turing has no `m16n8k32` s8, so the k=32 step is two k=16 steps. `ASTAGES_DEV` drops to 2 for Turing with the reason
written down (`src/prefill/moe_fused_iq.cu:70-73`): "its copies are plain loads, so depth buys nothing, and 4 does not fit
its 64 KB
of shared memory".

**What stops it.** `src/prefill/moe_fused_iq.cu:1817-1826`:

```cpp
static const bool sm75 = [] { const char* v = std::getenv("STRATA_PF_FUSED_SM75"); return v != nullptr && v[0] == '1'; }();
if (major < 8 && !(major == 7 && minor == 5 && sm75)) return d;
...
if (major < 8) {   // Turing: the canonical Q2_0 pack's kernels (the codebook formats' do not fit 64 KB at 4 warps a row)
    d.t75 = true;
    d.ok = setup_one<T_Q2_0, true>(occ) && setup_one<T_Q2_0, false>(occ);
```

The stated reason is shared memory, and that is worth checking rather than believing, because the arithmetic decides
whether this candidate lives. From `smem_bytes()` (`src/prefill/moe_fused_iq.cu:112-115`) with `WLD = 80` (`:66`),
`AB = 80` (`:59`), `ASTAGES_DEV = 2` (`:72`), `NT = 512` (`:65`):

**Measured on:** computed from those constants and the three codebook sizes at `:107-110`; no kernel was run.

| Type | WW=4, 4 stages | WW=4, 2 stages | WW=2, 4 stages | WW=2, 2 stages (the Turing configuration) |
| --- | ---: | ---: | ---: | ---: |
| IQ2_XXS (8 KiB codebook) | 70.25 KiB ✗ | 60.25 KiB ✓ | 66.50 KiB ✗ | **46.50 KiB ✓** |
| IQ2_XS (4 KiB) | 72.25 KiB ✗ | 62.25 KiB ✓ | 68.50 KiB ✗ | **48.50 KiB ✓** |
| IQ3_XXS (2 KiB) | 69.25 KiB ✗ | 59.25 KiB ✓ | 65.50 KiB ✗ | **45.50 KiB ✓** |
| IQ3_S (2 KiB) | 70.25 KiB ✗ | 60.25 KiB ✓ | 66.50 KiB ✗ | **46.50 KiB ✓** |
| IQ2_S (8 KiB) | 76.25 KiB ✗ | 66.25 KiB ✗ | 72.50 KiB ✗ | **52.50 KiB ✓** |

The comment is true for **WW=4 with the sm_80 four-stage pipeline** — every codebook format overflows 64 KiB there. It is
false for the Turing configuration the same file selects 90 lines earlier: `host_astages()` (`:1690-1699`) returns 2
below sm_80, and at **WW=2, two stages** every one of them fits, the largest at 52.5 KiB of 64 KiB. So the Q2_0-only gate
on Turing is a choice whose stated reason does not hold for the configuration it applies to.

The table also says exactly where a Turing port would need care, and it is not the whole list:

- `setup_one<T,GU>` (`:1713`) sets up **both** `WW=4` and `WW=2` (`setup_ww<T,GU,4> && setup_ww<T,GU,2>`), and
  `pick_ww()` (`:1950`) chooses per launch from the average routed rows an expert (128-row items when that average is
  56-112, else 64-row items). Both kernels therefore have to pass, not just the one a launch will use.
- At two stages `WW=4` fits for all but **IQ2_S at 66.25 KiB**, so a naive "add the codebook formats" list would fail
  its own setup on IQ2_S and — because `d.ok` is one flag for the whole device — take every format down with it. The
  safe shape is a Turing setup that instantiates **only** `WW=2` (52.5 KiB worst case), which is also the shape a
  4K-token DS4.1 chunk chooses anyway (about 64 routed rows an expert at six of 384).
- `launch()` (`:1917-1924`) sizes the shared memory with `host_astages()`, so the host and the kernel agree on 2 without
  further plumbing.

That is a small, checkable change inside `src/prefill/moe_fused_iq.cu` — not a new kernel. It still is not proof the
result runs fast. `IQ1_M` is the exception both ways: it is absent from the covered list
(`include/strata/prefill/moe_fused_iq.hpp:11-12` — "but the IQ1_M ones (which MMQ does not cover either)"), and the
table above shows it is 55.4 GiB of the 123.0. The realistic shape is a mixed layer: gate/up through the
fused GEMM where it is IQ2_*, down through the current kernel when it is IQ1_M.

**What the tensor cores are worth here.** The engine's decode dots go through `sub_dot<T>` in `src/dsv41/ds4_iq.cu`
(`:358`), eight `__dp4a` calls a 32-value sub-block, and the prompt tiles use `nvcuda::wmma` F16. Measured, as
register-only instruction streams so the number is the card's and not a kernel's:

**Measured on:** 1 x RTX 2080 Ti 22 GB, `tools/dsv41_lab/mma_peak.cu` (`nvcc -O3 -arch=sm_75`), 68 SMs, 2 to 16 warps a
block, 20,000 iterations; three runs on an idle card (boost clock 1845 MHz) and three more with the other three cards
busy at 1350-1410 MHz. Both sets are below, because the absolute figures moved by a fifth and the ratios did not.

| Instruction | What uses it | Idle card | Loaded card | Ratio to `__dp4a` |
| --- | --- | ---: | ---: | ---: |
| `m8n8k16.row.col.s32.s8.s8.s32` | `moe_fused_iq.cu`'s Turing branch | **244-252 TOP/s** | **204-209 TOP/s** | **3.9-4.0x** |
| `m16n8k8.row.col.f32.f16.f16.f32` | Turing's largest F16 MMA; the WMMA tiles compile to it | 124-126 TFLOP/s | **71-106 TFLOP/s** | 2.0x on the idle card |
| `__dp4a` | `sub_dot<T>`, DS4.1's decode path | **62-63 TOP/s** | **51-54 TOP/s** | 1x |

The integer MMA is **about 4x** the `__dp4a` stream, and that ratio is stable across both clock states while the
absolute numbers are not: the F16 row is the one that swings most (71-106 TFLOP/s loaded), so no argument here should be
built on an F16 figure without its clock. That ratio is this candidate's ceiling, not its result: the same instruction
fed from shared memory through `ldmatrix` in a 16-stage pipeline measured **35.8 TFLOP/s** at two blocks a SM on the idle
card (`tools/dsv41_lab/mma_roof.cu`), and **28.4** in the loaded re-run. A real kernel lives between that and the peak,
and nearer the low end until its staging is as deep as `moe_fused_iq.cu`'s. **Needs measurement**, and it is the most
valuable measurement left on this machine.

One more measured fact that constrains every Turing tile: `mma.sync.aligned.m16n8k16...f16.f16.f32` does not assemble for
sm_75 at all — `ptxas` says "Feature '.m16n8k16' requires .target sm_80 or higher". Turing's F16 shapes are `m16n8k8`
and `m8n8k4`; a 16x16x16 WMMA is lowered to two k8 steps, which is what the Qwen engine's
`src/kernels/cuda/qsa_prompt_attn.cu:52-57` describes from the other direction.

**What a port costs.** Five things, in order of difficulty:

1. **The gate, and it is smaller than it looks.** Three separate places keep a Turing card out, and all three are
   one-line predicates:
   - the device visit at `src/prefill/moe_fused_iq.cu:1817-1819` returns before any setup unless
     `STRATA_PF_FUSED_SM75=1`, and then sets up `T_Q2_0` gate/up and down only (`:1821-1827`);
   - `native_supported()` at `:2013` refuses every pair except Q2_0/Q2_0 when `d.t75` is set, whatever the environment
     says — so even the Q2_0 opt-in cannot mix a codebook gate/up with a Q2_0 down;
   - `gu_covered()`/`d_covered()` (`:1871-1888`) list the codebook formats without a Turing term, which is correct
     today because nothing below the setup can reach them.
   A Turing port therefore adds the codebook formats to the setup at `:1821-1827` and admits them at `:2013`. That is the
   whole gate change: the kernels already exist, already compile for sm_75, and their fragments are already the Turing
   ones. It is roughly the shape of the Qwen engine's `sm75_default()`, and no new kernel.
2. **Geometry.** The kernel hard-codes the Qwen pack's K: `GU_ROWS_K = 2560, D_ROWS_K = 640` (`:78`) — the K of
   gate/up (n_embd) and of down (n_ff) — with `NS = K / 64` (`:573`) and `NFB = (GU ? 1280 : 2560) / WR` (`:574`), where
   the numerator is the output row count: `2 * 640` gate and up rows of a 640-wide feed-forward, and `2560` for down.
   DS4.1 is `n_embd 5120`, `n_ff 2304`, so those become K of 5120 and 2304, `NS` of 80 and 36 (both exact, 2304 is
   `36 * 64`), and tile counts of `4608 / WR` and `5120 / WR`. All four are `constexpr` inside the kernel, so this is a
   template parameter or a second instantiation, not a header change — but every one of them has to be threaded through
   `native_kernel`, `launch`, `smem_bytes` and `pick_ww`, which is the bulk of the work.
3. **Row order.** The file assumes the Qwen pack's gate/up layout — "for gate/up a tile is 8 features, its gate rows
   then its up rows, so a lane holds gate and up of one feature" (`src/prefill/moe_fused_iq.cu:597-598`) — a single
   `geo.up_off` from the start of the blob plus `fb * (WR / 2) + 8 * (ur >> 4) + (ur & 7)` rows
   (`src/prefill/moe_fused_iq.cu:612-616`). DS4.1's engine reaches its three expert matrices through `a.tr[0..2]`, one
   offset a part (`src/dsv41/ds4_iq.cu:640-641`, `:698`), which is the same shape: three offsets, three row strides. A
   port needs the pack's actual per-part offsets and strides, taken from `src/dsv41/ds4_model.cu:1928-1932`'s
   `expert_layout_mx` / `parts[3] = {"w1", "w3", "w2"}` (gate, up, down) and the loader, plus a check that a gate row and
   its up row are laid out the way the tile arithmetic assumes.
4. **The call site.** Nothing under `src/dsv41` calls `strata::prefill::fused`: `strata-dsv41` links `strata_prefill`
   only for `strata_pf_gemm_f16_ld` (`src/dsv41/ds4_model.cu:35`). A port means a new call from `run_set()`
   (`src/dsv41/ds4_model.cu:4181`) beside the existing `x3::set_gate_up` — that is new engine code, outside this page's
   scope, which is why the section stops at a plan.
5. **Build.** `strata_mmq` (which holds `moe_fused_iq.cu`) is built under a CUDA/HIP block that also builds llama.cpp's
   MMQ instances (`CMakeLists.txt:2151-2154`), and `strata-dsv41` does not currently link it.

**The honest verdict.** The highest-value sm_75 mod for DS4.1 is not a Turing trick at all. It is an integer-tensor-core
expert GEMM that already exists in this repository, already has a Turing branch, is gated off by a shared-memory reason
that does not hold for the configuration Turing uses, and would need a new geometry and a call site before it can be
tried. The 4.0x instruction ratio is measured; the kernel-level gain is not, and it applies to at most 27 % of this
pack's expert bytes.

## Why the Qwen engine's sm_75 series does not carry over

The Qwen engine's sm_75 series is about **GEMV of ggml block types at 1-8 columns** (`kIlRows_75` at
`src/kernels/cuda/native_mmvq.cu:3708`), **expert layout** (`src/kernels/cuda/iq_kernels.cu:6178`) and
**16-byte dequant stores** (`src/kernels/cuda/dequant_bf16.cu:391`). Three structural differences stop each of them from
landing in DS4.1:

1. **The types are different.** The Qwen series gates on ggml types 11/12/13/14/23 (Q3_K…IQ4_XS). The 2.00 bpw
   DeepSeek pack's experts are IQ1_M, IQ2_XXS and IQ2_XS only, and the dense weights are BF16/F32/IQ4_NL/MXFP4/Q5_K/Q8_0
   (the pack's own manifest, `pack-info.txt`). The kernels that read those deep types live in `src/dsv41/ds4_iq.cu`,
   are written around a 256-value block with a shared codebook table, and already do the work `kIlRows_75` does:
   decode a sub-block once and dot it with several activations (`gate_up_kernel<T,NE,RW>` at `src/dsv41/ds4_iq.cu:611`,
   `NE` = 1, 4, 8 or 16 entries a pass).
2. **The decode is memory bound, not instruction bound, at batch 1.** A token reads 6 experts a layer x 40 layers x
   8.60 MB = 2.064 GB of expert weights. On this machine one card's share arrives over PCIe gen 3 and from RAM for
   half the experts; the measured decode is 26-41 tok/s over the run, and the whole prompt-versus-decode behaviour says
   the same thing: disabling the MXFP4 tensor-core tiles costs 30-32 % of the prompt's speed and leaves decode
   unchanged (423.9 -> 294.6 tok/s at 4K, 764.9 -> 523.4 at 32K; decode 28.5 -> 29.9 tok/s, i.e. noise).
3. **The dense work already uses cuBLAS.** DS4.1's chunked prompt calls `cublasGemmEx` with F16 inputs and F32
   accumulation (`src/dsv41/ds4_model.cu:3458`). The Qwen engine keeps a Turing switch for exactly the case where that
   call is made with BF16 inputs: `include/strata/core/tc_sm75.hpp:6` records "measured 19.5 % of the prompt's GPU time
   on a 2x2080 Ti", because on Turing `cublasGemmEx` on `CUDA_R_16BF` falls back to a SIMT fp32 kernel (`magma_sgemmEx`,
   `src/prefill/gemm.cu:774-776`). DS4.1 never asks for BF16 in that path, so the trap does not apply.

The exception to all three is the section above: `src/prefill/moe_fused_iq.cu` is the one piece of Qwen-side work that
does fit this engine's formats, and it has a Turing branch — but only for the Q2_0 path, and only opt-in. It is not part
of the sm_75 *series* — it is `#136`'s fused expert GEMM — which is exactly why it was easy to miss.

The lesson from the Qwen series that *does* apply is a method, not a kernel: **find the one stage that is bound by
something a Turing card can fix, measure it alone, and gate the change to the card**. The candidates below are exactly
that.

## Candidate 0: the bandwidth items (largest measured, but not sm_75 work)

This one is out of the Turing scope and belongs first all the same, because it decides whether any of the
instruction-rate candidates below can pay. The prompt path on this card is **waiting, not computing**:

**Measured on:** 4x RTX 2080 Ti 22 GB, 2.00 bpw pack, `--tp 4`; the throughput column from `tools/strata_bench.py` 5 runs a
configuration, the per-chunk figure computed from the pack's own tensor sizes.

| Item | Number | Why it matters |
| --- | --- | --- |
| non-VRAM experts streamed per chunk | ~190 experts a layer x 7.74 MB x 40 layers = **58.8 GB**, about **5.1 s** at the measured 11.61 GB/s pinned H2D | the tile kernels cannot be the limit of a stage that spends 5.1 s a chunk moving bytes |
| measured prompt | 10.9 s at 4K (1 chunk), 24 s at 16K (4), 43.4 s at 32K (8) | linear in chunks, not in FLOPs |
| `DS4_FAT_ROWS=64 DS4_FAT_PIECE=512` | **515.1 / 851.6 / 964.9 tok/s at 4K / 16K / 32K** against 423.9 / 671.4 / 764.9; +21.5 / +26.8 / +26.1 % | the largest measured prompt win here, and it is a *decode once, then GEMM* change, i.e. fewer passes over weights |
| `chunk_hand_bytes()` at `src/dsv41/ds4_model.cu:3258` | 4096 x 92 192 = **377 MB** of F32 `h` per boundary per chunk, three boundaries | about 1.1 GB a chunk that no kernel speed can amortise away |
| peer access | `cudaDeviceEnablePeerAccess` / `cudaMemcpyPeer` appear **zero** times in `src/dsv41/`; the comment at `src/dsv41/ds4_model.cu:591` says peer copies go "through the driver's staging, ~1.7 ms a copy" | measured GPU0<->1 47.07 GB/s, GPU2<->3 47.03 GB/s (NV2), cross-pair PHB; 2 of 3 hand-offs a chunk could use 4x the bandwidth |
| cross-chunk expert cache | the `lc` layer cache at `src/dsv41/ds4_model.cu:727` is single-device only | a per-group version would stream each missing expert once a *prompt* instead of once a *chunk*: 32K goes 470 GB -> 59 GB |

**Consequence for the ranking below.** The int8 expert GEMM is the best *instruction-rate* candidate, and an
instruction-rate win cannot rescue a stage waiting on a bus. Treat candidates 1-4 as worth measuring and this section as
worth more.

**Measured on:** the two `DS4_FAT_ROWS` configurations are `bench/results/2026-11-04-abc-notc-A` and
`bench/results/2026-11-05-t-fat` on the same card, 5 runs each, spread +/-1.8 % at 32K.

## Candidate 1: turn the split-KV attention on for sm_75 (measured, biggest measured win)

**What it is.** `attnsplit::sparse_attn_split<kSplitG, kSplitR>` (`src/dsv41/ds4_attn_split.cuh`, called from
`src/dsv41/ds4_kernels.cu:2445-2452`) splits the 128 window rows plus the 512 selected compressed rows over `R` chunks and
merges them. It is in the tree and off: `split_enabled()` at `src/dsv41/ds4_kernels.cu:2027` reads `DS4_ATTN_SPLIT`, and
the comment beside it says why — "no gain measured on 4x R9700 (21.8 vs 21.9 tok/s)", where the decode is bound
elsewhere.

**Why Turing is different.** The comment on the kernel says 2.9x at one query on an A4000. What the engine runs at
decode on any card is not the split kernel: it unpacks the selected rows into scratch and then runs
`sparse_attn_kernel<true, 8>` over a grid of `dim3(heads, T)` (`src/dsv41/ds4_kernels.cu:2464-2472`) — one block a head,
each block walking all 640 rows (128 window + 512 selected) in order, 8 rows of prefetch. At one query that is 64 blocks
of 256 threads on a card with 68 SMs, and each block's 640 rows are one long chain. The split kernel divides those rows
over 40 chunks and runs them at once. Measured:

**Measured on:** 4 x RTX 2080 Ti, `tools/dsv41_lab/ds4_lab.cu` built for sm_75 with
`nvcc -O3 -std=c++17 -arch=sm_75 -I../../include`, `--ctx 8192 --reps 50`, 64 heads x 512, 128 window + 512 selected
rows, one device, one stream, CUDA graphs so the launch gap is out.

| Kernel | 1 query | 2 queries | 4 queries | max error vs double |
| --- | ---: | ---: | ---: | --- |
| `sparse_attn (fp4 ckv)` — what the engine runs today (unpack, then the staged kernel) | 142.14 µs | 164.47 µs | 421.99 µs | 4.6e-7 |
| `attn_split G8 R40` | **37.36 µs** | **69.63 µs** | **187.59 µs** | 1.0e-6 |
| `attn_split G4 R40` | 44.37 µs | 85.20 µs | 223.64 µs | 1.0e-6 |
| `attn_split G4 R20` | 49.51 µs | 97.81 µs | 245.38 µs | 1.0e-6 |
| `attn_split G2 R32` | 65.66 µs | 124.59 µs | 333.38 µs | 1.0e-6 |
| `sparse_attn (packed)` — the same rows, no scratch, which is the path from 8 queries up | 138.49 µs | 222.24 µs | 553.42 µs | — |

The tolerances in the lab are 1e-4, so all the split variants are exact enough for the engine's own gate. The last two
rows are the same work through the two shipped forms: the scratch form the engine takes at decode (`T <= kAttnScratchT`,
`kAttnScratchT = 7` at `src/dsv41/ds4_kernels.cuh:157`) beats the packed form at every query count measured here, and
the split kernel beats both.

**Why it is not already on for sm_75.** Nobody measured it here. The gate `split_enabled()` is one environment variable
with no per-card rule, and the Qwen engine's answer to exactly this situation was `sm75_default()`.

**The mod, precisely.** Make the default follow the card, as `include/strata/core/sm75_series.hpp` does for the Qwen
engine: on CUDA `major == 7`, `split_enabled()` returns true unless `DS4_ATTN_SPLIT=0`; everywhere else it stays
opt-in. Two lines and a comment; `DS4_ATTN_SPLIT=1|0` keeps working. This is the smallest change in this page and the
only one with a large measured number behind it.

**What it is worth.** The lab says 105 µs saved a layer at one query, 40 layers: 4.2 ms a token. At 30 tok/s a token is
33 ms, so **if attention were on the critical path, this is up to 12 % of decode**. It is probably not all on the
critical path: the engine overlaps the expert reads with the attention, and `DS4_PROF` would show how much of the 33 ms
the sparse-attention stage actually holds. **Needs measurement** end to end, with `DS4_PROF=1` before and after.

One caution about the lab table: neither shipped form scales smoothly (142 → 164 → 422 µs, against 138 → 222 → 553 for
the packed form), so treat the two and four query columns as points rather than a curve. The one-query column is the
shape decode runs and is the one the 3.8x applies to.

**Risk.** The merge sums the chunks in another order, so the last bits differ (4e-7 against a double reference in the
kernel's own comment, 1.0e-6 in my run). That is the same class of change the engine already accepts elsewhere
(`DS4_FAST_ATTN`'s comment: "the owner approved the trade for the prefill"). It must be said plainly: the split path is
not bit-identical to the packed path.

## Candidate 2: `cp.async` in the packed and split attention readers

**What it is.** The Qwen engine's prompt attention uses `cp.async` for its KV tiles
(`src/kernels/cuda/qsa_prompt_attn.cu:45-49`) and records the shared-memory benefit (`Smem2T` 37.9 -> 29.2 KB at
`:105`). `cp.async` is an sm_80 instruction; `__pipeline_memcpy_async` and `cuda::memcpy_async` lower to it only from
sm_80, so on sm_75 they compile to a `ld.global` into registers followed by `st.shared` — the sequence the kernel
already writes by hand.

**The honest answer: this does not port, and DS4.1 already has the sm_75 substitute.** CUDA lowers
`__pipeline_memcpy_async` / `cuda::memcpy_async` to `cp.async` only from sm_80. On sm_75 the compiler emits the same
load-then-store sequence a kernel writes by hand.

That by-hand form is what both DS4.1 readers already do; Turing's own prompt-attention kernel in the Qwen engine
proves the pattern is understood there (`src/kernels/cuda/qsa_prompt_attn.cu:161`, "Turing has no cp.async, and …") —
and it takes the same trade at `:606-607`: "the warp's next pieces wait in registers and are stored into their stage
after this chunk's p.v (store_stage) — the same bytes in the same places as the cp.async pipeline". DS4.1 does the same
thing twice: `sparse_attn_packed_kernel` (`src/dsv41/ds4_kernels.cu:1001`) stages 16 rows a pass through registers as
`uint4` (`kStage = 16` at `:959`), and `attnsplit`'s reader (`src/dsv41/ds4_attn_split.cuh:65-67`) says it outright —
"all of a warp's row loads are issued before any is converted", i.e. the prefetch is already there in the only form
sm_75 has. **Do not do this one.** It is listed because it is the obvious-looking candidate and the answer is no.

The version that *does* apply is the one `SM75_SERIES_WHY.md` records for the Qwen engine: 16-byte units,
whole-warp coalescing, and no per-thread gather. DS4.1's dequantiser already does it —
`dq_rows_kernel2<T>` at `src/dsv41/ds4_iq.cu:1293` writes its 32 values through shared memory 32 values apart, and its
comment records why (a thread writing its own 32 values straight to `dst` wrote 2 bytes every 64, 0.5 G weights/s on
the R9700). Nothing to port.

## Candidate 3: the MXFP4 expert tiles — already there, and already the biggest lever

**Measured on:** 4 x RTX 2080 Ti, `tools/strata_bench.py`, 5 runs, medians, the same flags
(`--tp 4 --experts 5 --dspark-conf 0.5`); A with tensor cores, B with `STRATA_DS4_NO_TC=1`.

| Prompt | With the WMMA tiles | `STRATA_DS4_NO_TC=1` | Cost of turning them off |
| --- | ---: | ---: | ---: |
| 4K | 423.9 tok/s | 294.6 tok/s | -30 % |
| 16K | 671.4 tok/s | 458.4 tok/s | -32 % |
| 32K | 764.9 tok/s | 523.4 tok/s | -32 % |
| decode (short chat) | 28.5 tok/s | 29.9 tok/s | noise |

This is the one place where a Turing card is already doing tensor-core work in DS4.1: `tc_ok()` at
`src/dsv41/ds4_gguf.cu:1007` is `major >= 7`, so a 2080 Ti runs `mx_set_tc_kernel<GATE_UP>` at
`src/dsv41/ds4_gguf.cu:791` (WMMA 16x16x16 F16, BM=64, BN=64, 8 warps, one 32-value MXFP4 block a k-step) and the
i-quant equivalent `set_tc_kernel<T,GATE_UP>` at `src/dsv41/ds4_iq.cu:821`. sm_75 has `ldmatrix` and the `m16n8k8` F16
MMA, and WMMA's 16x16x16 compiles to two k8 steps, so the tile shape is supported. What a Turing card pays for it is the
30-32 % in the table above; nothing in this repository measures that against a newer card's version of the same tiles,
so this page does not claim a ratio for it.

**The mod worth considering here:** none of the Qwen-style ones. The tiles are already the right shape.
`strata::dsv41::wmma_emu` (`tools/dsv41_lab/wmma_emu.hpp`) exists precisely to check what a Turing card can do with
these fragments, and its answer is the one in the bench above.

## Candidate 4: a per-card defaults header for DS4.1 (structure, no speed by itself)

The Qwen engine's real invention on Turing is not one kernel: it is that **a card can change a default, once, in one
place, and every consumer asks that place**. `include/strata/core/sm75_series.hpp:25` is that place, and the
`SM75_SERIES_WHY.md` page lists its 15 consumers (9 `sm75_default`, 3 `sm75_switch`, one direct `sm75_series()` read,
2 `device_is_sm75()`).

DS4.1 has no such place. Its knobs are 121 `std::getenv` sites in `src/dsv41/ds4_model.cu` alone, plus
`src/dsv41/ds4_experts_rule.hpp`'s pure rules, and the only per-card branch on CUDA is `tc_ok()`'s `major >= 7` and
`use_legacy()`'s `major < 8` (`src/dsv41/vendor/exl3.cu:1272`). Candidate 1 needs a card rule; candidate 1's sibling
needs a place to live. A `ds4_arch_defaults()` beside `defaults_026()` — same shape, same `v026` argument, same tests
without a GPU — would cost little and make every later Turing finding a one-line default instead of a new getenv.

**No speed by itself.** It is the plumbing that makes candidate 1 shippable and the next measurement cheap. Worth
saying: `defaults_026()` already proves the pattern in this engine.

## Candidate 5: the trellis (EXL3) decode — the pre-sm_80 kernels are already selected

On an NVIDIA card before sm_80, `use_legacy()` at `src/dsv41/vendor/exl3.cu:1272` sends the trellis decode to
`src/dsv41/vendor/exl3_legacy.cu`, which is v1.1.8's kernels, chosen because on a V100 the modern kernels' CUDA-core
fallback was slower (4.9 -> 4.4 tok/s, the file's own comment). A 2080 Ti therefore runs the legacy trellis path.

The 2.00 bpw pack in use has no trellis experts (its types are IQ1_M/IQ2_XXS/IQ2_XS), so this affects which *other*
packs a 2080 Ti can run, not the throughput of this one. **No measurement possible on this pack.** Leave it.

## Candidate 6: expert-tier policy — the real decode ceiling, and not a kernel question

**Measured on:** this machine, the running server's logs. 7976 of 15360 routed experts (68.2 GB) on the CPU lane,
67.8 GB read in place per layer set, `--experts 5`, 582 MiB free a card.

A token draws 6 of 384 experts a layer. If an expert lives in graphics memory its 8.60 MB come at VRAM speed; if it
lives in RAM it comes over PCIe gen 3. The decode numbers vary 22-41 tok/s run to run on identical flags, which is the
signature of *which* experts a token draws, not of kernel speed. The knobs that move this are `--experts N`,
`DS4_RAM_TIER`, `DS4_STORE_POLICY`, `DS4_DISK_SKIP_W`, `DS4_AHEAD` and `--vram-experts`: all host policy, all already
wired, all outside the sm_75 question. A Turing mod cannot fix them; a **Turing-aware default** could (e.g. the
five-experts default already keys on 16 GiB of VRAM in `src/dsv41/ds4_experts_rule.hpp:39`).

## What is measured to be a dead end

| Idea | Why not |
| --- | --- |
| tensor cores for the **decode** GEMVs | DS4.1's `mv_kernel<T,NC>` (`src/dsv41/ds4_iq.cu:911`) is one row a warp, NC = 1/4/16 columns. The Qwen engine already tried the same shape on Turing: `bench/results/2026-09-30-tc-gemv-sm75` measured the MMA path 1.3-2.2x *slower* (ssm_alpha 8.73 -> 11.32 µs, ple_value 35.08 -> 77.73 µs) because one token fills 1/16 of an m16 tile and the warp kernel already reaches 276-374 GB/s |
| `cp.async` / `ldmatrix` in attention | `cp.async` is sm_80 only; on sm_75 the compiler emits the same code the kernel already has. `ldmatrix` does exist on sm_75 and the WMMA tiles already use it (`src/dsv41/ds4_iq.cu:838-848`) |
| BF16 tensor-core dense GEMM (`STRATA_BF16_TC`'s trick) | DS4.1 asks cuBLAS for F16, not BF16 (`src/dsv41/ds4_model.cu:3458`); Turing's BF16 trap does not apply, and the SM_75 F16 path is already what runs |
| an MMA path for the trellis decode (`mma16816`, `src/dsv41/vendor/exl3.cu:925`) | gated at `__CUDA_ARCH__ >= 800`, and the legacy file exists because the CUDA-core fallback was measured slower on a V100 (4.9 -> 4.4 tok/s). Porting it to sm_75 means rewriting the fragment loaders, not flipping the gate |
| a per-kernel `__launch_bounds__` sweep | `mx_gate_up_kernel<16,*>` already sits at 119/94 registers (measured with `ptxas -v` at sm_75), i.e. one block an SM; that is a shape choice, and `resident_ctas()` (`src/dsv41/ds4_iq.cu:1128`) already asks the driver for the occupancy instead of guessing |
| `DS4_FAT_ROWS` for a Turing card (already measured the other way round) | it turns busy experts into an F16 decode plus a library GEMM. That is a prompt-path win that does not depend on the card's generation, so it is not a Turing candidate at all — but it is the **largest measured prompt win in this corpus**, and it lives in the expert path this page calls the ceiling: `DS4_FAT_ROWS=64 DS4_FAT_PIECE=512` took the same card from 423.9 / 671.4 / 764.9 to **515.1 / 851.6 / 964.9 tok/s** at 4K / 16K / 32K (+21.5 %, +26.8 %, +26.1 %), decode flat. See the bandwidth paragraph below |

## The measurement plan

The engine's own harnesses are enough; nothing new is needed.

| Step | Command | What it gives |
| --- | --- | --- |
| 0. The int8 expert GEMM | a Turing instantiation of `strata::prefill::fused` at DS4.1's geometry, then `ds4_iq_test --bench-tiles` and `DS4_PROF=1` against it | whether the 4.0x instruction ratio survives into a kernel; the one measurement this page could not make without new engine code |
| 1. Find the stage | `DS4_PROF=1 DS4_PROF_SYNC=1` on a 100-token run | the six decode stages and the prompt's eleven, with the expert kernels by type in ms, calls, rows and TFLOPS |
| 2. Kernel A/B, no model | `tools/dsv41_lab/ds4_lab.cu`, `--only sparse_attn --tokens 1`, built for sm_75 | the table above, with parity against a double reference in the same run |
| 3. Expert-kernel A/B | `ds4_iq_test --bench --type 29` (also `--bench-light 29 384 1,2,4,8,16` and `--bench-tiles 29 384 17,32,64`) | i-quant experts in µs and GB/s for each kernel form, against ggml's decode |
| 4. End to end | `python tools/strata_bench.py --long --runs 5` twice, `DS4_ATTN_SPLIT` 0 then 1 | prompt tok/s, decode tok/s and TTFT medians with ranges; the engine's own timings |
| 5. Guard the parity | `tests/dsv41/iq_expert_test.cpp`'s parity checks, and `DS4_DUMP_Y`/`DS4_DUMP_R` against the reference | the same ids, or a stated last-place difference |

A change is a win only when step 4 moves outside the run-to-run band. That band is wide here: the 2026-11-04 and
2026-11-05 runs at identical flags read the same 4K prompts at 423.9 and 515.1 tok/s. Two A/B pairs are the minimum,
and the day-to-day drift must be named in the report.

Two host facts explain most of that band and both must be recorded with any number from this machine: the four cards
clock down together under load (1350 MHz observed against a 1545 MHz boost), and the expert tiers move between runs,
because a token's 22-41 tok/s spread at identical flags is which of the 384 experts a layer happened to draw. The
instruction tables above were taken with the cards idle at their boost clock; a kernel measurement taken while another
bench owns the cards reads lower and the difference is not the kernel's.

## Summary table

| # | Candidate | Mechanism | Expected effect | Effort | Numerics risk | Status |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | the i-quant expert GEMM on `m8n8k16` int8 tensor cores (`src/prefill/moe_fused_iq.cu`) | decode to int8 tiles in shared memory, `mma.sync` integer dots, one add per dot | 4x the instruction rate the decode dots use, ~2x the F16 tiles' rate; covers 27 % of this pack's expert bytes; the kernel-level gain is unmeasured | large: new geometry, row order, call site, link | none measured; int8 products are exact | **largest**, instruction ratio measured, kernel **needs measurement** |
| 1 | split-KV attention on sm_75 | `kSplitG/kSplitR` chunks with a merge | 3.8x the attention stage at one query; up to ~12 % of decode if that stage is on the critical path | 2 lines + a comment | last-place difference (1e-6 vs double) | **measured**, kernel level; end-to-end **needs measurement** |
| 2 | `cp.async` attention prefetch | sm_80 instruction | none on sm_75 | — | — | **no-go**, measured by construction |
| 3 | MXFP4/i-quant WMMA tiles | already on (`tc_ok()` is `major >= 7`) | -30 to -32 % of prompt speed if removed | shipped | none | **already there** |
| 4 | `ds4_arch_defaults()` header | one place a card changes a default | none by itself; makes 1 and later findings shippable | small, with a GPU-free test | none | **proposal** |
| 5 | trellis legacy kernels | already selected before sm_80 | none on this pack | shipped | none | **no-go** for this pack |
| 6 | expert-tier policy | which tier a pick lands in | the decode's real ceiling; 22-41 tok/s run to run | host policy only | none | **outside** the sm_75 question |

## History

| Version | Change |
| --- | --- |
| 0.1.55 | this page. The attention table was measured on the 4 x RTX 2080 Ti box with `tools/dsv41_lab/ds4_lab.cu`; the instruction table came from `tools/dsv41_lab/mma_peak.cu` and `tools/dsv41_lab/mma_roof.cu` on one 2080 Ti, twice: idle at boost and loaded at 1350-1410 MHz |
