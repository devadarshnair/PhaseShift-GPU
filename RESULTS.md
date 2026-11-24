# PhaseShift results summary

All numbers below come from **cycle-level simulation in Vortex SimX** (no silicon, no FPGA). Workload: **Q8_0
(int8 weights and activations, fp32 group scales) LLM decode** on the llama2.c `stories15M` model and on a layer set
with the shapes of an 8B-class model. Each number is a single deterministic simulation run.

## Machines

| Name | What it is |
|---|---|
| **short-latency** | 1 Vortex core (32 threads x 8 warps, issue width 2) with L2 and one DRAM channel, **11-cycle** DRAM latency |
| **long-latency** | the same core and memory system with a **190-cycle** DRAM latency |
| **2 cores / 1 channel** | 2 such cores sharing one DRAM channel (190-cycle latency): the bandwidth-starved case |

The named SimX configurations are in [`experiments/configs.yaml`](experiments/configs.yaml).

## 1. DOT4: an int8 dot-product instruction for decode GEMV

`DOT4` / `DOT4ACC` is a 4-way int8 dot product (`rd = Σ int8·int8`, or `rd += ...`) in the RISC-V **custom-3**
opcode space, added to SimX's integer ALU ([`patches/vortex/dot4/`](patches/vortex/dot4/)) together with a
synthesis-only RTL datapath ([`patches/vortex/rtl/`](patches/vortex/rtl/)). It is paired with a
**warp-cooperative, coalesced Q8_0 GEMV kernel** ([`kernels/ps_gemv_q80dot4w/`](kernels/ps_gemv_q80dot4w/)): several
lanes share one weight row, a host-side weight re-layout makes each warp-wide load read contiguous bytes, and partial
sums are combined with warp shuffles. Results are bit-exact against llama2.c's `runq.c` reference order.

| Comparison (GEMV, geomean over the decode shapes) | short-latency | long-latency | 2 cores / 1 channel |
|---|---|---|---|
| DOT4 kernel vs the **same-layout tuned scalar** kernel | **2.7x** faster | **1.9x** faster | |
| DOT4 kernel vs the **best software GEMV** found |  | **+100 %** | **+92 %** |

## 2. End-to-end token decode (stories15M, Q8_0)

Full-model decode on the device (every op of llama2.c's forward pass runs as a Vortex kernel; host only does argmax),
DOT4 GEMV vs the best software GEMV per matrix:

| Decode position | short-latency | long-latency | 2 cores / 1 channel |
|---|---|---|---|
| 0 | +64 % | +31 % | +29 % |
| 128 | **+28 %** | **+16 %** | **+16 %** |
| 255 | +23 % | +13 % | +14 % |

The gain shrinks as the context grows because attention, which DOT4 does not accelerate here, takes a larger share
of each token.

## 3. DOT4 vs a tuned tensor-core GEMV

Against an equal-effort tuned GEMV on Vortex's int8 tensor core (WMMA-style `wmma_context<..., int8, int32>`):

| Batch | DOT4 vs tensor core |
|---|---|
| 1 | DOT4 **1.32-1.34x faster** |
| 2 | parity |
| 4 | DOT4 1.3-1.6x slower |

The tensor core is about **8-10x the area** of the DOT4 datapath (Yosys + Nangate45 synthesis ratios; not a
sign-off flow). For batch-1 decode, the small instruction wins.

## 4. Thread / warp sweep

DOT4's GEMV gain grows on narrow warps at short latency: **+313 % at 8 threads** per warp vs **+174 % at 32
threads** (8 warps). Narrow warps are where a scalar int8 inner loop is most issue-bound.

## 5. L2 stream prefetcher with a 32-entry MSHR

An L2 tagged next-N-sector stream prefetcher plus a 32-entry L2 MSHR
([`patches/vortex/prefetch/`](patches/vortex/prefetch/)), measured after fixing a bookkeeping bug in the
prefetcher (`0004-pf-late-reset.patch`; numbers below are from the fixed version).

| | 190-cycle DRAM | 2 cores / 1 channel |
|---|---|---|
| GEMV, prefetcher over best software (27 classes, geomean) | +29 % | +24 % |
| Token decode, DOT4 + prefetcher over best software without it (pos 0 / 128 / 255) | +82 / +43 / +29 % | +57 / +32 / +29 % |
| Additivity of DOT4 and prefetcher at token 128 (1.0 = they compose) | 1.02 | 0.98 |
| Extra scheduled DRAM reads vs prefetcher off (token decode) | +18 to +29 % | +16 to +24 % |
| Prefetched sectors evicted unused (lower bound) | 15-40 % | 21-35 % |

The gain comes from more memory-level parallelism (more outstanding misses plus prefetch issue), not from prefetch
accuracy: a large share of prefetched data is never used, and the prefetcher costs extra DRAM traffic. A prefetcher
with the stock 16-entry MSHR gives only about half the GEMV gain (+15 % / +11 %). Re-arming the prefetcher on late
merges was tested and gave no gain.

## 6. What did not work

- **Prefetch re-arm on late merge** gave no held-out gain (-0.25 % to -0.03 %) and slowed some classes by up to 7 %. Not adopted.
- **int8 attention (QK^T with DOT4ACC over an int8 K cache)** did not preserve the greedy token sequence and gave
  only a 3-8 % token-time gain. Not adopted ([`kernels/ps_attn_k8/`](kernels/ps_attn_k8/) is kept for reference).

## Caveats

- **Simulation only.** All timing is SimX cycle-level simulation; no FPGA or silicon measurement.
- **Single deterministic runs.** SimX is deterministic for a given configuration; no run-to-run variance is
  reported.
- **Area numbers are synthesis ratios** (Yosys, Nangate45, placeholder SRAM macros), useful for relative comparison
  only.
- **No fmax, power or energy claims** are made.
- Model quality is not evaluated beyond token-exactness against the llama2.c reference.
