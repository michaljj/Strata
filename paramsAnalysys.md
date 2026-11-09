Read the notes and the engine thoroughly. Big correction first, then new knobs, then structural findings.

## Correction: `STRATA_SM75_PERF=1` did nothing

```
$ strings -a build-dsv41/strata-dsv41 | grep -c STRATA_SM75
0
$ nm -C build-dsv41/strata-dsv41 | grep -ci sm75
0
```
`STRATA_SM75_PERF`, `STRATA_SM75_EXP`, `STRATA_SM75_IL`, `STRATA_TC_SM75` live in `include/strata/core/sm75_series.hpp:25,34,37,40` and are read by `src/kernels/cuda/native_mmvq.cu:3192`, `src/kernels/cuda/iq_kernels.cu:6178` — the **Qwen** engine. `CMakeLists.txt:918-942` links `strata-dsv41` against `strata_prefill`, `strata_dsv41_kernels`, `strata_kernels_cpu` only.

So run4's gain over run3 came from `--dspark auto` + `DS4_ASYNC_T=4`, not SM75. Drop that env var.

## Prefill is PCIe-bound, not compute-bound

Derived per 4096-token chunk: ~190 non-VRAM experts/layer × 7.74 MB (`tensors.txt`: w1+w2+w3 = 3 × 2,580,480 B) × 40 layers ≈ **58.8 GB streamed**, at measured H2D 11.6 GB/s ≈ **5.1 s**.

Cross-check against your log: 4K = 1 chunk → 5.1 s predicted vs 10.7-10.9 s measured; 16K = 4 chunks → 20 s vs 24 s; 32K = 8 chunks → 40 s vs 43 s. Streaming scales exactly with chunk count. That is why 32K crawls.

## New parameters worth testing (all present in this binary, none in notes)

| Knob | Why | Value |
|---|---|---|
| `DS4_COPY_PIN=1` + `vm.nr_hugepages` + `LimitMEMLOCK` | Prompt DMAs experts **in place** instead of RAM→staging→DMA. Auto-pin needs an explicit 2 MB arena (`ds4_model.cu:5140-5145`); your THP arena + `ulimit -l 8192` blocks it. Measured upstream: 2nd 8K prompt 12.0 s pinned vs 12.3-28.8 unpinned (`ds4_model.cu:5136-5138`) | ~68 GB hugepages, `DS4_COPY_HUGETLB=1 DS4_COPY_PIN=1` |
| `DS4_FAT_ROWS=64`, `DS4_FAT_PIECE=512` | Busy experts via FP16 GEMM instead of re-decoding tiles per 64 rows (`ds4_model.cu:3470-3500`). IQ1_M passes `f16_rows_ok` (`ds4_iq.cu:1670`). Measured +6-8% (3×V100 8K/32K), 1400→1625 tok/s on RTX 5090 13K | needs ~53 MB/card; you have 580 MiB |
| `DS4_ATTN_SLAB=512` (or 1024) | CUDA never auto-grows past base 256 rows (`ds4_experts_rule.hpp:169`, `ds4_model.cu:3352`) because the ≥0.5-share branch is RDNA4-only. 256 rows = 16 dense dequant passes per 4K chunk; 1024 = 4 passes. Costs 192 MB (512 rows) / 385 MB (1024) | pair with `--reserve-mib 300` |
| `STRATA_GLM_EXL3_DECODE=new` | You log `exl3: CUDA0..3 decodes with v1.1.8's kernels` — the sm<80 default (`vendor/exl3.cu:1283-1290`). This is a real A/B never run here | decode |
| `STRATA_DS4_DSPARK_CUM=0.5` | Cumulative draft-confidence floor, auto-0.5 only on gfx12 (`ds4_experts_rule.hpp`). +71.7→76.9 tok/s on 4×R9700. Different variable than your `--dspark-conf 0.5` | decode |
| `DS4_DSPARK_MAX=3` | Caps drafts/round (`ds4_model.cu:1059`). Your log: `drafts 18 of 19 kept`, `2 of 5 kept` — acceptance is uneven, wasted verify work is real | decode |
| `DS4_SETTLE_MS=0` | `cache_settle` burns up to 800 ms after **every** chunked prompt (`ds4_model.cu:7164`). That is inside your 13.1 s agent TTFT. Upstream table: prompt+256 at 6000 ms 27.2 s, 800 ms 23.9 s, 0 ms 23.4 s | agent TTFT |
| `--prefill-chunk 2048` | Not faster (2× streaming) — do it only to see the streaming model confirmed, or to free ~470 MB/card. 8192 is impossible: C-scaled buffers ≈ 945 MB at 4096, ~1.9 GB at 8192 | diagnostic |
| `--reserve-mib 300` | 600→300 frees 1.2 GB total ≈ 155 more experts; also lets slab/fat rows fit | +1% |
| `--cpu-threads 20` | Log says `14 threads`; `lane_takes_all` needs 28 (`ds4_model.cu:5644`), so `STRATA_DS4_LANE_FREE` never applies. E5-2680 v4 has HT | CPU lane |
| `--vision off` | 0.97 GB tower for nothing if no images | frees ~250 MiB/card |
| `nvidia-smi -pl 280 -pm 1` | Limit is 250 W, max 280 W, persistence `Disabled` | +12% power headroom |

## Inert here — stop burning runs on them

- `DS4_AR_F16` — all-reduce never runs, `nm == 1` per group (`ds4_model.cu:8335`). Log: `tensor parallel 4 -> 1`.
- `DS4_YOCO*` / `STRATA_DS4_PROMPT_SKIP` — needs `l.nm == 1` for every layer, dead at tp4 (`ds4_model.cu:9280-9284`).
- `DS4_LAYER_MAJOR`, `DS4_LAYER_CACHE_MB` — `devs.size() != 1` returns false (`ds4_model.cu:9192`).
- `DS4_LOOKAHEAD`, `DS4_LA_MAX`, `DS4_SERIAL_PREFETCH` — Metal-gated, absent from `strings`.
- `DS4_FAST_ATTN` — inside `#if defined(STRATA_USE_HIP)` (`ds4_kernels.cu:2480`).
- `DS4_READAHEAD_MB`, `DS4_COPY_GB`, store family — all CPU-lane experts already in RAM ("0 from the files, 0.0 GB through the page cache").
- `DS4_GG_IQ512`, `STRATA_NO_AVXVNNI` — E5-2680 v4 has no AVX-512.
- `DS4_MX_*` — pack is IQ1_M, not MXFP4.
- `--eos`, `--layer-split`, `--dspark-block`, `--dspark-depth` — as notes established.

## Structural findings (upstream `[Engine]` material)

1. **P2P unused, and it exists on half the box.**
   ```
   GPU0->1 peer: 47.07 GB/s      GPU0<->2: no peer
   GPU2->3 peer: 47.03 GB/s      GPU1<->3: no peer
   H2D pinned per GPU: 11.61 GB/s   D2H: 13.18 GB/s
   ```
   No `cudaDeviceEnablePeerAccess` / `cudaMemcpyPeer` in `src/dsv41/` at all. Comment at `ds4_model.cu:591` says peer copies go "through the driver's staging, ~1.7 ms a copy" — that is the PHB-pair behaviour generalized to NVLink pairs. 2 of 3 chunk hand-offs per chunk qualify for 4× the bandwidth.

2. **Chunk hand-off size.** `chunk_hand_bytes()` (`ds4_model.cu:3258`) = 4096 × 92,192 ≈ **377 MB per boundary per chunk** in F32 `h` (81,920 B/token of 92,192). Sending `h` in F16 halves it; 3 boundaries → ~1.1 GB per chunk removed.

3. **No cross-chunk expert cache in multi-GPU mode.** The `lc` layer cache (`ds4_model.cu:727, 9213`) is single-device only. A per-group version would stream each missing expert once per *prompt* instead of once per *chunk*: 32K prompt 470 GB → 59 GB.

4. **PCIe gen is fine.** Idle `gen 1`, ramps to `gen 3` under `nvcc -arch=sm_75` load (verified: 1905 MHz, 90 W, gen 3). No PCIe tuning needed; 11.6 GB/s is correct for 3.0 x16.

5. **Clock/power.** Under synthetic load all 4 hit 1860-1905 MHz of 2100 max, 90 W of 250 W. No thermal or clock throttling. `SW Power Capping` counter is lifetime-cumulative, not current.

## Suggested order

1. `DS4_SETTLE_MS=0` — 1 run, expected −0.5 to −0.8 s agent TTFT.
2. `DS4_FAT_ROWS=64 DS4_FAT_PIECE=512` — prompt only, check log line `busy experts ... as FP16 GEMMs`.
3. `STRATA_GLM_EXL3_DECODE=new` — decode, 1 line in log confirms it took.
4. `STRATA_DS4_DSPARK_CUM=0.5` — decode.
5. Hugepages + `DS4_COPY_PIN=1` — biggest prompt win, requires root, do it last and alone.
6. `--reserve-mib 300` + `DS4_ATTN_SLAB=512` — one combined config.
