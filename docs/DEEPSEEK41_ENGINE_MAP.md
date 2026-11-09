# The DeepSeek-V4.1 engine (strata-dsv41): a map

This page is the map of the DeepSeek-V4.1 engine: what it is built from, what each file does, which kernels run in which
phase, what the model's numbers are, and every place where the code asks the graphics card's generation. It exists so
that a change can be judged before it is written, and so that the sm_75 question in
[DSV41_SM75_ANALYSIS.md](DSV41_SM75_ANALYSIS.md) has somewhere to point. It is a reading of the source, not a
measurement.

The engine is a separate executable, `strata-dsv41`, under `src/dsv41`. The code started as Project Maya's `maya-ds4`
(MIT); the vendored copy and what changed are in `third_party/project-maya/NOTICE.md`. It shares no kernel with the
Qwen engine.

## Build

`option(STRATA_BUILD_DSV41 "Build the DeepSeek-V4.1 engine path (strata-dsv41)" OFF)` at `CMakeLists.txt:909`. The whole
block is guarded by `if(STRATA_BUILD_DSV41 AND NOT STRATA_HIP_GFX906)` at `CMakeLists.txt:910`, so gfx906 builds no
DeepSeek engine.

| Part | Where |
| --- | --- |
| static library `strata_dsv41_kernels` — `src/dsv41/vendor/exl3.cu`, `src/dsv41/vendor/sampler.cu`, `src/dsv41/vendor/moe_win.cu`, `src/dsv41/vendor/exl3_cpu.cpp`; plus `src/dsv41/vendor/exl3_legacy.cu` when not HIP (`CMakeLists.txt:911-915`) | `CMakeLists.txt:911` |
| executable `strata-dsv41` from twelve sources (`CMakeLists.txt:918-920`) | `CMakeLists.txt:918` |
| build id, printed by `--version`: `-DSTRATA_BUILD_ID`, then a `BUILD_ID` file, then `git rev-parse --short=9 HEAD` (`CMakeLists.txt:921-931`) | `CMakeLists.txt:932` |
| links `strata_prefill` (the RDNA3 WMMA dense GEMM), `strata_dsv41_kernels`, the BLAS and the GPU runtime (`CMakeLists.txt:935-937`) | `CMakeLists.txt:937` |
| `STRATA_NATIVE_EXPERTS` adds the CPU trellis lane and defines `STRATA_DSV41_TQ_CPU=1` (`CMakeLists.txt:944-947`) | `CMakeLists.txt:946` |

The CUDA architectures default to `86-real;89-real;120` when `STRATA_BUILD_DSV41` is set, and the comment says why: it is
the release shape of the DeepSeek packages, so a hand build with no architecture runs on an RTX 30, 40 or 50. The default
is set before `enable_language(CUDA)` on purpose (`CMakeLists.txt:218-230`, issue #4023): CMake 3.18 and later fill an
unset `CMAKE_CUDA_ARCHITECTURES` with nvcc's own default, and with CUDA 13 that default is sm_75.

Test targets (all need `STRATA_NATIVE_EXPERTS`): `ds4_gg_bench` (`CMakeLists.txt:950`), `ds4_tq_test` (`:967`),
`ds4_iq_test` (`:989`), `ds4_slot_test` (`:999`), `ds4_experts_rule_test` (`:1002`), `ds4_pick_rule_test` (`:1005`).
None of those six is registered with `add_test`, so `ctest` runs none of them.

## The files

| File | Lines | Role |
| --- | ---: | --- |
| `src/dsv41/ds4_model.cu` | 10799 | the whole forward pass, the expert tiers and their caches, threads and streams, slots, `DS4_PROF` |
| `src/dsv41/ds4_kernels.cu` | 2682 | the non-expert device kernels: attention, hyper-connections, routing, top-k, the indexer, DSpark, the ring |
| `src/dsv41/ds4_iq.cu` | 1736 | the i-quant and K-quant experts (token, window, light and prompt-set kernels, dequant to rows, GEMV jobs) |
| `src/dsv41/ds4_gguf.cu` | 1694 | the MXFP4 experts (token, window, prompt tiles with and without WMMA), the dense BF16/Q8 GEMVs, KV row helpers |
| `src/dsv41/ds4_main.cpp` | 1636 | the CLI, the `--serve` protocol (GEN, GENI, STOP, QUIT), conversation slots, the run summary |
| `src/dsv41/vendor/exl3.cu` | 1755 | the trellis (EXL3) decode and its prompt tiles, the sm_80+ MMA path, the occupancy choice |
| `src/dsv41/vendor/exl3_cpu.cpp` | 1287 | the EXL3 CPU lane (AVX-512 / AVX2 / scalar) |
| `src/dsv41/vendor/exl3_legacy.cu` | 968 | the pre-sm_80 NVIDIA decode kernels `src/dsv41/vendor/exl3.cu` dispatches to |
| `src/dsv41/vendor/sampler.cu` | 774 | the sampler (temperature, top-k, top-p, min-p, greedy) |
| `src/dsv41/ds4_tq.cu` | 540 | the trellis types TQK5/6/7 (`kTQK6=145` … `kTQK4T=227`) on the device |
| `src/dsv41/ds4_vision.cu` | 520 | the vision tower's device kernels (32 layers, hidden 1024, 16 heads) |
| `src/dsv41/ds4_gguf_cpu.cpp` | 418 | the MXFP4 CPU lane (int8 dots) |
| `src/dsv41/ds4_gg_cpu.cpp`, `src/dsv41/ds4_tq_cpu.cpp`, `src/dsv41/ds4_iq_cpu.cpp` | 296, 138, 782 | the other CPU-lane decoders |
| `src/dsv41/ds4_packinfo.cpp` | 273 | reads the pack manifest |

Headers: `src/dsv41/ds4_kernels.cuh` (317) holds the shared constants (`kWin=16`, `kPicks=8`, the KV row formats at
`src/dsv41/ds4_kernels.cuh:23-30`, `kAttnScratchT=7`). `src/dsv41/ds4_iq.cuh` (155) and `src/dsv41/ds4_gguf.cuh` (85) hold
the quant types, and `include/strata/dsv41/exl3.hpp` (196) the trellis layout. `src/dsv41/ds4_attn_split.cuh` (207) is the
split-KV attention. The host rules are `src/dsv41/ds4_experts_rule.hpp` (234) and `src/dsv41/ds4_pick_rule.hpp` (29), both
with a test that needs no GPU. The rest: `src/dsv41/ds4_store.hpp` (592) the RAM expert store,
`src/dsv41/ds4_pool.hpp` (279), `src/dsv41/ds4_os.hpp` (640), `src/dsv41/ds4_engram_io.hpp` (275),
`src/dsv41/ds4_slotmeta.hpp` (98).

## The model's numbers

From the pack's `ds4.txt` — the manifest the engine reads, at the root of the model directory — and `src/dsv41/ds4_model.cu`
where the code reads it.

| Setting | Value |
| --- | --- |
| layers | 40 (two of them Engram: `engram layers 1,14`) |
| hidden size | 5120 |
| routed experts | 384 a layer, 6 a token (`--experts 5` drops one) |
| expert feed-forward size | 2304 |
| attention | 64 heads, head dim 512, MQA (K = V), sliding window 128 |
| q / o low rank | `q_lora_rank` 1280, `o_lora_rank` 1024, `o_groups` 8 |
| hyper-connections | `hc_mult` 4, `hc_sinkhorn_iters` 20, `hc_eps` 1e-6 |
| indexer | 32 heads x 128, `index_topk` 512, `candidate_topk_blocks` 2048 of `candidate_block_size` 8, source layer 20 |
| vocabulary | 129280, `eos_token_id` 1 |
| rope | `original_max_position_embeddings` 65536, factor 16, beta fast 32, beta slow 1, theta 10000; the compressor's `compress_rope_theta` 160000 |
| swiglu | `swiglu_limit` 10.0, `routed_scaling_factor` 1.5, `rms_norm_eps` 1e-20 |
| KV sources | `kv_source_layer_ids` 2, 8, 14, 20; `index_source_layer_ids` 2, 8, 14, 20, 24, 28, 32, 36 |
| compression | `compress_ratios` 0,0 then 2 eighteen times, 1 twenty times, then 0,0,0 — a layer keeps one KV row a 2 or 1 tokens |
| DSpark | block 5, noise token 128799, markov rank 256, 128 routed experts, 3 a token |
| max position | 1048576 |

An expert is 8.60 MB at 2.00 bpw and a token reads 6 of them a layer: 2.064 GB of weights a token before anything else.
That number decides everything in [DSV41_SM75_ANALYSIS.md](DSV41_SM75_ANALYSIS.md).

## The weight formats, and where the dequant runs

| Format | Types | Where |
| --- | --- | --- |
| MXFP4 | `kFmtMXFP4` (32 values in 17 bytes, power-of-two scale) — `src/dsv41/ds4_gguf.cuh:31` | the experts of the source pack; `src/dsv41/ds4_gguf.cu` kernels |
| i-quants | `kQ2_K=10` … `kIQ1_M=29` (256 a block) — `src/dsv41/ds4_iq.cuh:21-26` | the 2.00 bpw pack's experts; `src/dsv41/ds4_iq.cu` |
| trellis (TQ) | `kTQK6=145, kTQK7=146, kTQK5X=224, kTQK4X=225, kTQK5T=226, kTQK4T=227` | `src/dsv41/ds4_tq.cu` on the device, the CPU lane on `src/dsv41/ds4_tq_cpu.cpp` |
| EXL3 | `kTypeEXL3=200` (`include/strata/dsv41/exl3.hpp:30`) | `src/dsv41/vendor/exl3.cu` |
| dense | BF16, F32, F16, Q8_0, MXFP8 (Engram rows only) | `bf16_mv` / `q8_mv`, cuBLAS for the dense GEMM |
| KV rows | `kKvBf16=0`, `kKvFp8=1`, `kKvFp4=2`, `kKvFp4p=3` — `src/dsv41/ds4_kernels.cuh:23-30` | inside attention |

No weight is dequantised ahead of time on the host for the device path. Each kernel decodes the blocks it needs, and the
activations are rounded to int8 per 32 values with an F32 scale (`src/dsv41/ds4_iq.cuh:9-14`), so a sub-block's product is
an exact integer dot on `__dp4a` times its scales in F32. `__dp4a` needs no generation check: it exists from sm_61.

## Kernels by phase

### Decode (one token)

| Phase | Kernel | Where |
| --- | --- | --- |
| hyper-connection mix | `hc_mix_kernel` (24 x 20480 dots) | `src/dsv41/ds4_kernels.cu:254` |
| Sinkhorn split, 20 rounds | `hc_split_kernel` | `src/dsv41/ds4_kernels.cu:343` |
| pre-norm, post | `hc_pre_norm_kernel`, `hc_post_kernel` | `src/dsv41/ds4_kernels.cu:388`, `:409` |
| attention projections | `gemv_f16_kernel`, `gemv_f16_rows_kernel` | `src/dsv41/ds4_kernels.cu:422`, `:448` |
| router | `route_kernel` (`__shared__ float sc[1024]`) | `src/dsv41/ds4_kernels.cu:508` |
| sparse attention | `sparse_attn_kernel<kContig, kPre>`, `sparse_attn_packed_kernel<G>`, `unpack_sel_kernel` | `src/dsv41/ds4_kernels.cu:820`, `:1001` |
| indexer | `index_score_kernel` | `src/dsv41/ds4_kernels.cu:629` |
| top-k over the indexer's scores | `topk_kernel` (`__launch_bounds__(1024)`) | `src/dsv41/ds4_kernels.cu:697` |
| experts, i-quant | `gate_up_kernel<T,NE,RW=1>`, `down_kernel<T,NE,RW=2>` | `src/dsv41/ds4_iq.cu:611`, `:674` |
| experts, MXFP4 | `mx_gate_up_kernel<NE,…>`, `mx_down_kernel<NE,…>` | `src/dsv41/ds4_gguf.cu:523`, `:588` |
| experts, trellis | `gate_up_kernel`, `down_kernel` (and the legacy pair on pre-sm_80) | `src/dsv41/ds4_tq.cu:323`, `src/dsv41/vendor/exl3.cu:607`, `:631` |
| clamp and combine | `swiglu_kernel`, `combine_kernel`, `gather_kernel` | `src/dsv41/ds4_kernels.cu:558`, `:1574`, `:1568` |
| DSpark drafter | `markov_kernel`, `markov_kernel_q8`, `markov_pick_kernel`, `confidence_kernel` | `src/dsv41/ds4_kernels.cu:1361`, `:1389`, `:1468`, `:1497` |
| Engram | `engram_dequant_kernel`, `engram_gate_kernel` | `src/dsv41/ds4_kernels.cu:1521`, `:1543` |
| sampler | `sampler_kernel` (1024 threads), `sampler_greedy_kernel` | `src/dsv41/vendor/sampler.cu:193`, `:102` |

### Prompt (the chunked pass)

| Phase | Kernel | Where |
| --- | --- | --- |
| expert sets, i-quant | `set_kernel<T,GATE_UP,RM>` (F32 tiles, BM=64, BN=64, TM=4, TN=4) or `set_tc_kernel<T,GATE_UP>` (WMMA, `kLdT=40`) | `src/dsv41/ds4_iq.cu:733`, `:821` |
| expert sets, MXFP4 | `mx_set_tc_kernel<GATE_UP>` (WMMA) or `mx_set_kernel<GATE_UP>` (F32) | `src/dsv41/ds4_gguf.cu:791`, `:712` |
| busy experts to F16 rows | `dq_rows_kernel2<T>`, `dq_rows_kernel<T>` | `src/dsv41/ds4_iq.cu:1293`, `:1337` |
| expert grouping | `moe_sort_kernel`, `bounds_scan_kernel` | `src/dsv41/ds4_kernels.cu:1964`, `:1998` |
| dense GEMM | cuBLAS, `cublasGemmEx` with F16 inputs and F32 accumulation | `src/dsv41/ds4_model.cu:3458` |
| attention | the same `sparse_attn` entry point | `src/dsv41/ds4_kernels.cu:2425` |
| vision | `patchify_kernel` … `bias_act_kernel` | `src/dsv41/ds4_vision.cu:76-174` |

The light path is part of the prompt: an expert set of at most 16 rows goes through the token kernels' form instead of
the GEMM tiles (`light_q8_kernel` at `src/dsv41/ds4_iq.cu:953`, `light_kernel<T,NE,GU,F>` at `:1022`, `light_kernel_rows` at
`:1039`; `light_on()` at `:1229`, `STRATA_DS4_NO_LIGHT=1` turns it off). The old dequant kernel `dq_rows_kernel` at
`:1337` is kept for gfx11 only: `dq_use_old_kernel()` at `:1402` is false on every CUDA card.

### The expert tiers

The 384 experts a layer are split three ways and the split is the engine's main design constraint
(`src/dsv41/ds4_model.cu:5525` and the log lines the engine prints): the experts that fit in graphics memory stay there, the
most-routed of the rest are copied into RAM, and the remainder stay in the model's files and are read in place. On this
machine the log says 7976 of the 15360 routed experts (51.9 %, 68.2 GB) on the CPU lane and 67.8 GB read in place. A
decode token draws 6 of 384 a layer, so the tier a pick lands in decides where its 8.60 MB come from: graphics memory, RAM,
or the model's files.

## The rules that are generation-blind

`src/dsv41/ds4_experts_rule.hpp` is host policy — no device code — and each rule is a pure function with a test:

| Rule | What it decides | Where |
| --- | --- | --- |
| `experts_per_word` | the router's top N, renormalised; 5 where the cards hold 16 GiB or less in all, or the GPU shares the machine's RAM; the model's 6 elsewhere (`kFasterModeVramMax`, `kFasterModeExperts`) | `src/dsv41/ds4_experts_rule.hpp:39` |
| `prefill_chunk_default` | the prompt chunk, from the GPU's share of the experts | `src/dsv41/ds4_model.cu:2293` |
| `attn_slab_base_mb`, `attn_slab_rows` | how much attention scratch a chunk gets | `src/dsv41/ds4_model.cu:3331`, `:3354` |
| `staging_ring_mb` | the size of the hand-off ring | `src/dsv41/ds4_model.cu:3294` |
| `store_feature_default`, `settle_store_default` | the RAM store's spare slots, its put and its settle | `src/dsv41/ds4_model.cu:5551`, `:7182` |
| `dense_pfgemm_default` | prefetch the dense GEMM's weights | `src/dsv41/ds4_model.cu:3445` |
| `fast_attn_default` | the matrix-core attention (gfx12 only in practice) | `src/dsv41/ds4_kernels.cu:2084-2089` |

Every one of them takes `defaults_026()` from `src/dsv41/ds4_experts_rule.hpp:25`: `DS4_DEFAULTS=026` restores what
v0.2.6 did, so that the app's "DeepSeek speed-ups" switch can be turned off honestly. None of them asks the card's
compute capability: on CUDA the engine's tuning is the same for a Turing card as for an RTX 40, except where `tc_ok()`,
`use_legacy()` and `mma_supported()` say otherwise.

## Every place the code asks the card's generation

Nine `__CUDA_ARCH__` sites, and none asks for sm_75 by name:

| Site | What | Value |
| --- | --- | --- |
| `src/dsv41/vendor/exl3.cu:270` | `EXL3_TC` — the MMA path | `>= 800` |
| `src/dsv41/vendor/exl3.cu:332`, `:339` | `kMinBlocksX`, `kMinBlocksMv` (launch bounds) | `>= 800` |
| `src/dsv41/vendor/exl3.cu:925` | `mma16816` | `>= 800` |
| `src/dsv41/ds4_gguf.cu:665`, `src/dsv41/ds4_kernels.cu:1473`, `:1705`, `:1746`, `src/dsv41/vendor/exl3.cu:762` | `__nanosleep` | `>= 700` |

Runtime branches, asked once a device:

| Function | Rule | Where |
| --- | --- | --- |
| `tc_ok()` | CUDA `major >= 7` and not `STRATA_DS4_NO_TC=1`; HIP: gfx11xx, gfx12xx | `src/dsv41/ds4_gguf.cu:1007`, `src/dsv41/ds4_iq.cu:1085` |
| `use_legacy()` | NVIDIA `major < 8` decodes with `src/dsv41/vendor/exl3_legacy.cu`; `STRATA_GLM_EXL3_DECODE=legacy\|new` overrides | `src/dsv41/vendor/exl3.cu:1272` |
| `mma_supported()` | `major >= 8` | `src/dsv41/vendor/exl3.cu:1600` |
| `fast_attn_enabled()` | HIP only in effect; `DS4_FAST_ATTN`, default on gfx12 | `src/dsv41/ds4_kernels.cu:2034-2085` |
| `split_enabled()` | `DS4_ATTN_SPLIT`, off by default | `src/dsv41/ds4_kernels.cu:2027` |
| `light_on()`, `dq_use_old_kernel()` | `STRATA_DS4_NO_LIGHT`, `DS4_DQ_OLD`; the old dequant kernel is gfx11-only | `src/dsv41/ds4_iq.cu:1229`, `:1402` |
| `rdna4()`, `device_is_gfx12()` | HIP only | `src/dsv41/ds4_model.cu:936`, `src/dsv41/ds4_kernels.cu:2052` |

What the sm_75 card therefore gets today: the WMMA prompt tiles for MXFP4 and i-quant experts (`tc_ok()` is true at
major 7), the trellis decode in 4-term FMA chains at 2 blocks an SM, the legacy trellis decode kernels, the scalar
sparse attention (no MMA, no split-KV by default), no BF16 tensor-core dense GEMM, and cuBLAS in F16 with
`CUBLAS_GEMM_DEFAULT_TENSOR_OP`.

There is no per-generation defaults table and no series header: `include/strata/core/sm75_series.hpp` and
`include/strata/core/sm120_series.hpp` are Qwen files, and no source under `src/dsv41` includes either. The nearest thing is
`src/dsv41/ds4_experts_rule.hpp`, which is host policy — how many experts, how big a prompt chunk, how much staging — and
asks the card's generation nowhere.

## What the engine does not have

Each of these is a Qwen-engine feature, present there and absent here. They matter for the analysis page.

| Missing here | Present in the Qwen engine |
| --- | --- |
| an integer-tensor-core multi-row expert GEMM (`mmq`) | `src/prefill/moe_mmq.cu:264`, `src/prefill/moe_mmq_tab.cuh:61` |
| weight-stationary expert streams | `src/prefill/moe_ws.cu:122`, `src/prefill/moe_ws_down.cu` |
| one-launch grouped MoE | `src/prefill/moe_fused.cu:674`, gated at cc 75 (`:675`) |
| NVFP4 / W4A4 | `src/prefill/mmq_nvfp4_w4a4.cu:1-10` |
| `cp.async` and `ldmatrix` attention | `src/kernels/cuda/qsa_prompt_attn.cu:45-49`, `:523` |
| CUDA graph capture | `src/qwen35/q36_engine.cu:2373-2379` |
| programmatic dependent launch | `include/strata/kernels/pdl.hpp:68` |
| a per-architecture tuning table | `include/strata/core/sm75_series.hpp:25` |
| kernel A/B harnesses | `tests/cuda/pa_bench.cpp:1-4` |

## Measuring it

| Tool | What it measures | Where |
| --- | --- | --- |
| `DS4_PROF=1` | graphics time a decoded token by stage, and the prompt's by stage, plus each expert kernel's time and TFLOPS | `src/dsv41/ds4_model.cu:1311-1341`, printed at `src/dsv41/ds4_main.cpp:1624-1632` and `src/dsv41/ds4_model.cu:9437-9482` |
| `DS4_PROF_SYNC=1` | the decode's stages timed on the host clock instead of events | `src/dsv41/ds4_model.cu:1332` |
| `DS4_DUMP_Y`, `DS4_DUMP_R`, `DS4_PRED_DUMP`, `DS4_XIN_DUMP`, `DS4_TRACE` | routing, expert sets and inputs for an offline reference | `src/dsv41/ds4_model.cu:1300-1326` |
| `tools/dsv41_lab/ds4_lab.cu` | parity and speed of the non-expert kernels, linking `src/dsv41/ds4_kernels.cu` unmodified | `tools/dsv41_lab/ds4_lab.cu:1-6` |
| `tests/dsv41/iq_expert_test.cpp` (`ds4_iq_test`) | i-quant experts against ggml's own decode and a double reference, `--bench` | `CMakeLists.txt:989` |
| `tests/dsv41/ds4_gguf_test.cu` | the GGUF kernels against a double reference; not wired into CMake | — |
| `tests/dsv41/ds4_mx_bench.cpp` | the CPU lane's MXFP4 kernels in GB/s; not wired into CMake | — |
| `tools/strata_bench.py` | end-to-end prompt and decode throughput over the HTTP API | `bench/results/*/README.md` |

**Measured on:** 4 x RTX 2080 Ti 22 GB, Xeon E5-2680 v4, Ubuntu 24.04, CUDA 13.4; `tools/dsv41_lab/ds4_lab.cu` built
with `nvcc -O3 -std=c++17 -arch=sm_75` and run `--tokens 1 --ctx 8192 --reps 50`. Every non-expert kernel agrees with
its double-precision reference (0 failures). At one query the split-KV attention takes 37.36 µs against the shipped
`sparse_attn (fp4 ckv)` 142.14 µs and `sparse_attn (packed)` 138.49 µs.

## History

| Version | Change |
| --- | --- |
| 0.1.46.1 | the engine arrives as a separate executable (`bbd0a2c6`, 2026-10-20: Project Maya's engine v1.1.17, imported unmodified, then built for HIP gfx1201) |
| 0.1.x | `--experts N` with 5 as the default where the cards hold 16 GiB or less (`3dd11c6d`, 2026-10-24) |
| 0.2.7 | `DS4_DEFAULTS=026` gives every default of v0.2.6 (`a3ab37fc`, 2026-10-26); the rule list is the comment at `src/dsv41/ds4_experts_rule.hpp:16-20` |
| 0.1.55 | the version this page's measurements were taken on (`c62d5e3a`, 2026-10-31) |
