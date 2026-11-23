# PhaseShift methodology

This document defines every equation, modeling choice, measurement convention and kernel/mechanism semantic used in
PhaseShift. Every equation and modeling choice used anywhere in the code is defined here or in a comment next to the
code. It is a working research document: sections were written as the corresponding parts of the project were built,
and later addenda refine earlier sections in place.

Two kinds of pointers refer to material that is not part of this release: `R§n` (a section of an internal literature
synthesis) and `[cand:<ID>]` (an entry of an internal reading list). They are reading pointers, not citations; no
claim in this repository depends on them.

---

## 1. Notation

Symbols from R§1, used unchanged everywhere in PhaseShift (code, docs, results):

| Symbol | Meaning |
|---|---|
| B | batch: number of concurrent sequences in an iteration |
| S | prompt (prefill) length in tokens |
| T | cached context length in tokens (KV entries per sequence) |
| L | number of transformer layers |
| H_q, H_KV | number of query heads, number of KV heads |
| D_h | head dimension |
| d | model width, d = H_q·D_h |
| g | GQA group size, g = H_q / H_KV (g = 1 is MHA) |
| q_w, q_a, q | bytes per weight element, per activation element, per KV element |
| N_tok | tokens processed by one GEMM (prefill: B·S, decode: B) |
| d_in, d_out | input / output width of one linear projection |
| N_params | number of model parameters (weights) |
| M_KV | KV-cache bytes (per token or in total, see E3) |
| BW, FLOPS_peak | peak memory bandwidth (bytes/s) and peak compute (FLOP/s) of the machine |
| AI | arithmetic intensity, FLOPs ÷ bytes moved (FLOP/byte) |
| T(i, c) | cost (cycles or time) of interval i under configuration c (§5) |
| C_sw(c′, c) | cost of switching from configuration c′ to configuration c (§5) |
| c ∈ C | a configuration in the PhaseShift knob space C |

Additional symbols introduced in this document are defined where they first appear. Note the one clash kept from
R§1: inside E8, R is the relative regret; everywhere else in this document R names the synthesis report.

---

## 2. First-principles equations E1–E11

The formulas are copied symbol-for-symbol from R§1 (the team's derivation, methodology notes Q1). "Status" is R§1's
verdict on the proposal's version of each equation.

### E1 — Linear-layer FLOPs

`FLOPs_linear = 2·N_tok·d_in·d_out` (prefill: N_tok = B·S; decode: N_tok = B)

- **Symbols:** N_tok tokens pass through one d_in×d_out projection.
- **Why this shape:** each of the N_tok·d_out output elements is a d_in-long dot product, i.e. d_in multiply-accumulates,
  and one multiply-accumulate counts as 2 FLOPs. The proposal's "∝ 2BSd²" (prefill) and "∝ 2Bd²" (decode) are this
  formula for one square (d×d) projection.
- **Status:** Correct (proposal).

### E2 — Linear-layer arithmetic intensity

`AI_linear ≈ 2·N_tok / (q_w + q_a·N_tok·(1/d_in + 1/d_out)) ≈ 2·N_tok/q_w` when N_tok ≪ d

- **Symbols:** as E1; q_w and q_a bytes per weight and activation element.
- **Why this shape:** AI = FLOPs ÷ bytes. The numerator is E1. The bytes are the weights, read once and reused across
  all N_tok tokens (q_w·d_in·d_out), plus the activations in and out (q_a·N_tok·(d_in + d_out)). Dividing both by
  d_in·d_out gives the expression. When N_tok ≪ d the activation term is negligible and AI ≈ 2·N_tok/q_w: at FP16
  (q_w = 2) decode AI ≈ B FLOP/byte, which is why batching rescues decode linear layers. When N_tok is large, AI
  saturates near d/q_a (for d_in = d_out = d: 2·N_tok / (q_a·N_tok·2/d) = d/q_a), which bounds prefill.
- **Status:** Added (the bytes side was missing in the proposal).

### E3 — KV-cache size

`M_KV/token = 2·L·H_KV·D_h·q`; `M_KV,total ≈ 2·B·L·T·H_KV·D_h·q`

- **Symbols:** per token, every layer stores one K and one V vector of H_KV·D_h elements of q bytes.
- **Why this shape:** the factor 2 counts K and V. The total multiplies by B sequences of T cached tokens.
  Generalizations used by the model code: Σ_i T_i for sequences of unequal length; Σ_i ⌈T_i/b⌉·b with a paging block
  size b; MLA, sliding windows and KV quantization change the per-token term (q, or the per-layer element count).
  Explicit forms used by `model/opgen.py`:
  - **MLA** (DeepSeek-V2-style multi-head latent attention, config field `kv_lora_rank`):
    `M_KV/token = L·(kv_lora_rank + qk_rope_head_dim)·q`. MLA caches one compressed latent c_KV of kv_lora_rank
    elements plus one decoupled RoPE key of qk_rope_head_dim elements shared by all heads, per layer and token;
    per-head K and V are reconstructed from c_KV and never cached, so there is no factor 2 and no H_KV·D_h.
    DeepSeek-V2-Lite: 27·(512 + 64)·2 = 31,104 B; caching per-head K (qk_nope + qk_rope = 192) and V (128) instead
    would take 27·16·(192 + 128)·2 = 276,480 B, which is not what MLA stores.
  - **Sliding window** W_l on layer l (rolling buffer): a layer keeps only the last W_l tokens, so
    `M_KV,total = Σ_i Σ_l ⌈min(T_i, W_l)/b⌉·b · e_l·q`, with e_l = 2·H_KV·D_h (or kv_lora_rank + qk_rope_head_dim for
    MLA) elements per token in layer l; W_l = ∞ for full-attention layers. M_KV/token itself does not change.
  - **KV quantization**: q is the bytes per KV element (e.g. 0.5 for INT4); scale/zero-point metadata is ignored.
- **Status:** Correct (proposal), extended.

### E4 — Attention FLOPs

Per layer per sequence: prefill `4·S²·d` (≈ `2·S²·d` if the kernel skips the masked causal half); decode `4·T·d`.

- **Symbols:** S query and key tokens in prefill; one query token against T keys in decode; d = H_q·D_h.
- **Why this shape:** QKᵀ and A·V each cost 2·(query tokens)·(key tokens)·d FLOPs (a multiply-accumulate per element
  of the d-wide reduction, summed over all heads). Prefill: 2·2·S·S·d; decode: 2·2·1·T·d. GQA does **not** reduce these
  FLOPs (every query head still does its dot products); it only shrinks KV bytes (E3). Cross-check: Kaplan's per-token
  forward cost `2N + 2·n_layer·n_ctx·d_model` [cand:B20] contains the same attention term at the causal-average context.
- **Status:** Missing in the proposal.

### E5 — Decode-attention arithmetic intensity

`AI_attn,decode = 4·T·H_q·D_h / (2·T·H_KV·D_h·q) = 2g/q`

- **Symbols:** numerator = E4 decode FLOPs with d = H_q·D_h; denominator = the KV bytes of one sequence for one layer
  (E3 per token ÷ L × T, since E3's per-token term counts all L layers).
- **Why this shape:** each decode step reads the whole KV cache of every sequence once per layer to perform one query's
  dot products. T cancels, and B never appears because every sequence has its own KV. Examples: MHA FP16 (g = 1,
  q = 2) → 1 FLOP/byte; GQA-8 FP16 → 8; GQA-8 INT4 KV (q = 0.5) → 32.
- **Status:** Missing in the proposal; the key result.

### E6 — Decode step lower bound

`t_step ≥ max{ (q_w·N_params + M_KV,total + act)/BW , (2·B·N_params + 4·B·L·T·d)/FLOPS_peak }`

- **Symbols:** `act` = activation bytes moved in the step; M_KV,total from E3; the FLOP term is 2 FLOPs per parameter
  per token (E1 summed over all projections) plus decode attention (E4 × L layers × B sequences).
- **Why this shape:** a step cannot be faster than streaming all weights plus all KV at peak bandwidth, nor faster than
  its FLOPs at peak compute (the two roofline limits, E7). KV traffic exceeds weight traffic when
  `B·T·2·L·H_KV·D_h·q > q_w·N_params`. R's example: for a 7B-class MHA model this is B·T ≈ 27K tokens (e.g. B = 8 at
  T ≈ 3.4K). Arithmetic check under the assumption of the common 7B MHA shape L = 32, H_KV = 32, D_h = 128, q = q_w = 2
  (shape to be confirmed against the pinned configs): KV/token = 2·32·32·128·2 = 524,288 B; weights = 2·7×10⁹ B;
  B·T = 1.4×10¹⁰ / 524,288 ≈ 26.7K. This crossover defines the decode sub-regions in §4.
- **Status:** Added (estimate).

### E7 — Roofline

`Perf = min(Peak_FLOPs, AI × Peak_BW)`; ridge = `Peak_FLOPs / Peak_BW`

- **Symbols:** Peak_FLOPs = FLOPS_peak, Peak_BW = BW.
- **Why this shape:** attainable throughput is capped by compute or by bandwidth × intensity, whichever is lower; below
  the ridge AI a kernel is bandwidth-bound [cand:B21]. The Vortex ridge must be computed from the configured TCU
  throughput; the TCU is off in every shipped profile (`VX_config.toml:40, 293–311`), so every phase boundary moves
  with configuration. PhaseShift measures the ridge per PS-BASE configuration (§11).
- **Status:** Added.

### E8 — Relative regret

`R = (T_detector − T_oracle) / T_oracle`

- **Symbols:** T_detector = time under the practical detector's schedule; T_oracle = time under the oracle (§5).
- **Why this shape:** the fraction by which the practical design's time exceeds the oracle's. Report it per interval
  and time-weighted; accuracy alone hides the few transitions that dominate latency.
- **Status:** Correct (proposal).

### E9 — Break-even mode length

`N_min = ⌈(C_enter + C_exit + C_warmup) / ΔC_saved⌉`

- **Symbols:** C_enter / C_exit = cost of switching into / out of a mode; C_warmup = cost of re-warming caches and
  prefetchers after the switch; ΔC_saved = per-iteration saving while the mode is active.
- **Why this shape:** a mode pays off only if it lasts long enough for its per-iteration savings to repay the switch
  in, the switch back, and the re-warm; the ceiling gives a whole number of iterations. The proposal's
  `C_switch/ΔC_saved` omits re-warm and the return switch. Computed separately for time and for energy (the two can
  differ in sign). Measured costs come (§13).
- **Status:** Corrected.

### E10 — Capture

`capture = (P_PhaseShift − P_static-best) / (P_oracle − P_static-best)`

- **Symbols:** P = a higher-is-better performance metric (e.g. throughput) under the real PhaseShift design, the best
  static configuration and the oracle.
- **Why this shape:** the fraction of the static-to-oracle gap that the real design recovers (0 = nothing, 1 = all).
  It is meaningful only when the denominator is well above simulation/sampling noise (§5, §6), so the absolute gap
  is always reported too. For latency and energy (lower is better) use reduction ratios instead.
- **Status:** Correct, with caveat.

### E11 — Area, power and energy per token

`ΔA = (A_PS − A_base)/A_base`, `ΔP` likewise; `E_token = total inference energy / generated output tokens`

- **Symbols:** A_PS / A_base = area of the PhaseShift / baseline design in the same flow; ΔP the same for power;
  E_token = energy per generated output token.
- **Why this shape:** relative overheads are what an open, pre-layout flow can defend (absolute watts cannot be);
  SRAM is reported separately; power is anchored at iso-performance or iso-SLO, and EDP is added. Energy model in §17.
- **Status:** Correct, extended.

### Consequences used throughout

1. **Decode is two regimes, not one.** At small B·T decode streams weights; at large B·T it streams KV (E6). Weights
   are reused many times, KV lines are touched once per step, so the two want opposite cache-insertion and prefetch
   policy. This is the first-principles motivation of the KV-aware mechanisms and splits decode into the two regions
   of §4.
2. **Capacity is not the decode lever.** E5 bounds decode attention far below any GPU ridge (hundreds of FLOP/byte).
   The levers are bandwidth efficiency, prefetch timeliness and protection against pollution; only shared KV (prefix
   caching, beam search, tree speculation) or a lower q moves the bound. Published simulations agree
   ([cand:A1], [cand:B1], [cand:B17], summarized in R§1).
3. **Scale artifact.** stories260K's weights (1,056,540 B) are about the size of Vortex's default 1 MiB L2
   (1,048,576 B), so on that model decode would look cache-friendly. This is an artifact of scale, not physics:
   working set relative to L2 is always plotted as a dimensionless axis (§7), and no policy conclusion is drawn from
   the RTL-validation model.
4. **Caches are cold at every launch on Vortex (no cross-launch reuse).** Every SimX launch runs
   `ProcessorImpl::run()` → `reset()` → `SimPlatform::instance().reset()` (`sim/simx/processor.cpp:274-275,332-333`),
   which resets every cache set (`sim/simx/mem/cache.cpp:696-703`); the SimX flush only clears dirty bits, untimed and
   outside any kernel's cycle count; the RTL flush invalidates. Hence: (i) no cross-launch weight reuse appears in any
   Vortex measurement; (ii) the flush switching cost is taken from the RTL formula, analytically and labelled as such
   (§13); (iii) a ps-tree sensitivity knob "persistent L2 across launches" quantifies the avoid-the-flush
   avenue; (iv) the analytical model has `persist_across_launch` (False for Vortex, True for GPU-like machines; §9).
   Kernel-boundary switching is therefore already free on Vortex, which is why §3 adds the per-kernel best static
   baseline. **Measured** (`ps_twolaunch`, the same GEMV launched k = 1, 2, 3 times on the same buffers in one
   app run, REG-CFG): L2 read misses = 1.0000 / 1.0005 / 1.0006 × k · misses(1) for stories15M wq (324 KiB, which
   would fit the 1 MiB L2 if it persisted) and 0.9996 × 2 for a 1 MiB matrix (`results/<run>/coldcache.csv`,
   `docs/vortex_errata.md` item 24). The claim holds; `persist_across_launch` stays False for Vortex.
5. **Launch history: SimX is deterministic per launch sequence, not per launch.** Re-running the same app run
   gives bit-identical instrs and cycles, but launch k ≥ 2 of a run is not a cycle-exact replay of launch 1 even with
   cold caches: later launches start from DRAM-model state left by earlier ones (Ramulator is built once and not reset
   between launches; errata 24, code-supported, not isolated). Measured effect: ≤ 0.7 % of cycles (wq +0.7 % per later
   launch, 512×512 −0.6 %); instruction counts are unaffected. Rules for every SimX measurement step (P3, P4,
   P5, P6): (i) **per-kernel cycles come from the first launch of an app run** (one measured launch per run);
   (ii) a k-launch total divided by k is used only when stated next to the number; (iii) **A/B comparisons** (base vs
   ps no-op regression, mode A vs mode B, policy vs best static) are made only on **identical launch sequences**
   (same kernels, same order, same shapes before the measured launch). Multi-kernel runs (a decode step) are measured
   as the whole sequence; the history effect there is physically realistic (DRAM state does persist on hardware).
6. **Kernel correctness metric.** Kernel apps check device output y against a reference with
   rel_i = |y_i − ref_i| / S_i, where S_i is the natural magnitude of the computation (for a K-long dot product
   S_i = Σ_j |a_ij x_j|), not |ref_i| (unbounded where the dot product cancels). Definition and derivation:
   `kernels/common/ps_common.h`; GEMV use: `kernels/gemv_f32/gemv_host.h`. Two correct float32 implementations with
   different summation orders differ by at most 2γ_K·S_i ≈ 2Ku·S_i (u = 2⁻²⁴), so the default tolerance 1e-4 is
   justified **only for K ≤ 838**; a step testing larger K must state its own tolerance (e.g. 2γ_K) rather than reuse
   1e-4. Softmax and decode attention use their own scales and derived bounds, tol = max(tolerance,
   bound): §14.1 (rel to max(ref, 2⁻¹⁰⁰); 1e-5 holds for the software-exp kernel up to n = 4096 (bound ≤ 8.94·10⁻⁶), the MUFU kernel uses its
   bound ≤ 1.44·10⁻⁵) and §14.2 (S_i = Σ_t p_t|v_t[d]|; data-dependent bound, 1e-4 floor).
7. **Linear layers: tolerances, padding and FLOP accounting.** Kernel apps `ps_linear` (`kernels/linear/`)
   and `ps_gemv_f32`. (a) **fp32 accumulation, any K:** tol = max(1e-4, 2γ_K), γ_K = Ku/(1 − Ku), u = 2⁻²⁴, i.e. item
   6's bound used as the tolerance itself above K = 838 (K = 1792 → 2.14e-4, K = 3584 → 4.27e-4). It is rigorous for
   every summation order used: gemm_small/gemv_rows (1 mul + 3 in-chunk adds + K/4 accumulations ≤ K roundings for
   K ≥ 8), SimX's TCU FEDP (`sim/simx/tcu/tcu_unit.cpp:176-201`: fp16 products exact in fp32, one FMA per packed pair,
   one add per word and per k-step, ≤ K roundings) and BLAS. (b) **fp16-input TCU path:** the strict check compares the
   device against a float64 sum over the *fp16-rounded* inputs (products exact), so input rounding cancels and (a)'s
   2γ_K applies; a second check against the fp32 NumPy result from *unrounded* inputs uses the earlier rel tol 1e-2. Its
   justification: rounding w and x to fp16 (u_h = 2⁻¹¹, round-to-nearest-even) changes each product by ≤ 2u_h + u_h² =
   9.77e-4 relative in the normal fp16 range, plus ≤ 2γ_K from the two accumulations ⇒ rel_i ≤ 1.2e-3 at K = 1792, plus
   an absolute ≤ 2⁻²⁵·|x_j| per fp16-subnormal weight and ≤ 2⁻²⁵·|w_j| per fp16-subnormal activation (|v| < 2⁻¹⁴; measured
  max over the fixtures of the per-tensor fraction of nonzero entries: ≤ 0.48 % of a weight
   tensor, ≤ 0.78 % of an activation tensor); 1e-2 leaves ≈ 8× margin and is a plausibility check against the real
   model, not the kernel check. bf16 inputs (u = 2⁻⁸) would give 7.8e-3 + 2γ_K under the same formula (not run in
  ). (c) **Padding:** the TCU GEMM needs M, N, K multiples of cta_M = ISSUE_WIDTH·xtileM, xtileN, tileK (32, 64, 16
   at NT = 32, ISSUE_WIDTH = 2, WGMMA_NRC = 32, fp16; `sw/common/tensor_cfg.h:336-363`); inputs are zero-padded,
   padded outputs must come back exactly 0, and the padding overhead per shape and M is tabulated in
   `results/<run>/padding.csv`. (d) **FLOP/cycle** is always reported with the *useful* FLOPs 2MNK of the unpadded
   problem (executed FLOPs 2·Mp·Np·Kp listed separately), so padding shows up as lost throughput. (e) **Layout:** the
   TCU path computes C = X·Wᵀ with Wᵀ stored K × N in fp16 — a one-time offline conversion of the llama2.c (N, K)
   fp32 weights, not counted in kernel cycles; the FPU paths read the llama2.c layout directly. (f) **Scaled shapes:**
   "Llama-3.1-8B-shaped" = d/8 with aspect ratios kept (d = 512, head_dim 64, 8 query / 2 KV heads, d_ff = 1792;
   `model/linear_fixtures.py`), the (8, 2, 64) case also uses. The element-wise decode kernels use scale_i = max(|ref_i|, FLT_MIN), and RoPE the norm of the rotated
   input pair; tolerances and their derivations: §14.0.

### Op-trace conventions (`model/opgen.py`)

Modeling choices that turn E1–E5 into a per-op trace. Each is repeated in the docstring of the code that applies it.

1. **Compulsory bytes only.** Every op counts each byte it touches once (weights read, activations read/written,
   KV read/written, tagged by class). Re-reads caused by tiling, cache capacity or cold caches are the memory model's
   job (§9). With this rule a linear op's AI is exactly E2 and decode attention's FLOPs ÷ KV bytes is exactly E5.
2. **Context convention.** A sequence in an iteration has `n_new` new tokens (prefill chunk S, or 1 for decode) after
   `ctx` cached tokens. Query j (0-based) attends causally to `ctx + j + 1` keys, its own K/V included (they are
   written to the cache before attention). For decode at context T, T counts the new token (ctx = T − 1), so the step
   reads exactly E3's T tokens of KV and does E4's 4·T·d FLOPs per layer and sequence.
3. **Causal flag.** `causal=True` (default): the kernel skips the masked half, FLOPs use Σ_j keys_j
   = S·ctx + S(S+1)/2 pairs (≈ E4's 2·S²·d for a full prefill). `causal=False`: the full S × (ctx+S) rectangle
   (E4's 4·S²·d); this is what HF eager attention executes and what the FlopCounterMode cross-check compares.
4. **Fused attention** (default): QKᵀ → softmax → AV is one kernel; scores stay on chip. Unfused: scores and
   probabilities are written/read at q_a bytes per element. QKᵀ reads the K part of the cache, AV the V part.
5. **RoPE fused with the KV write** (as kernel): the K and V projections write activations; the
   `rope_kv_write` op reads q, k, v, rotates q and k (3 FLOPs per rotated element) and writes k, v into the cache.
   cos/sin tables are not counted (computed on the fly or negligible).
6. **Elementwise FLOPs** count one per arithmetic or transcendental operation: RMSNorm 4/element, SiLU-gate
   4/element, residual 1/element, softmax 5/score. They are reported but are < 0.1 % of linear FLOPs at the
   pinned shapes; FlopCounterMode counts them as 0.
7. **Batching and mixed iterations.** Linear layers see the concatenated token rows of all sequences in the
   iteration (one GEMM with N_tok = Σ n_new, weights read once per iteration, as in E6); attention is computed per
   sequence (group of identical sequences). A mixed iteration = prefill-chunk sequences + decode sequences batched this
   way. A prefill chunk that is not the request's last emits no logits.
8. **LM head.** By default only each emitting sequence's last token goes through the LM head (serving engines);
   `lm_head_all_tokens=True` reproduces an HF forward (logits for every token). The final norm is applied to all rows.
9. **MoE.** Router = d → E linear (top-k selection not counted). Routed experts are grouped GEMMs over the N_tok·k
   (token, expert) rows, so FLOPs = 2·N_tok·k·d·d_ff,expert per projection whatever the routing. Expert weight bytes =
   E[distinct experts touched]·d·d_ff,expert·q_w with E[distinct] = Σ_e [1 − (1 − π_e)^N_tok], π_e = probability a
   token selects expert e (Σ_e π_e = k), tokens routed independently. *Uniform* routing: π_e = k/E. *Zipf* routing:
   popularity p_e ∝ (e+1)^(−s) and π_e = min(1, λ·p_e) with λ set so Σ π_e = k (an expert is picked at most once per
   token). Independence overstates distinct experts (real routing is correlated), i.e. it is an upper estimate of
   expert-weight traffic; s is a sensitivity knob. Shared experts (DeepSeek) are a dense MLP of width
   n_shared·moe_intermediate_size; the combine op weights and sums the k outputs (2 FLOPs per element).
10. **MLA** (modeling choice, not a claim about every kernel): by default every phase is modelled in the
    weight-absorbed form: q_nope is multiplied by W_UK per head into the latent space (r = kv_lora_rank), scores have
    per-head length r + ρ (ρ = qk_rope_head_dim) against the cached latent + RoPE key, AV accumulates per-head length
    r over c_KV (read once by the fused kernel), then W_UV (r → v_head_dim) per head and O. W_UK and W_UV together are
    `kv_b_proj`. Absorption is the natural form for decode (it never materialises K/V). For prefill an implementation
    may instead decompress K/V from the latent and run ordinary per-head attention; `gen_ops(...,
    mla_prefill_form="decompressed")` models that (kv_b_proj applied to every key the chunk touches, QKᵀ of length
    qk_nope + ρ, AV of length v_head_dim; decode sequences stay absorbed). Which form is cheaper depends on S, T and
    the machine, so may evaluate both. Attention FLOPs in either form differ from E4's H_q·D_h form (E4 assumes
    per-head K/V of width D_h).
11. **Sliding windows** follow the HF config semantics (Qwen2: only if `use_sliding_window`, only on layers ≥
    `max_window_layers`; Mistral: every layer); a window caps keys per query at W and the KV kept per layer (E3).
12. **Not modelled** (each < 0.002 % of N_params at the pinned configs, measured against the meta-device models in
    `results/<run>/nparams_compare.csv`): Qwen2 q/k/v biases, the per-head q/k RMSNorm of Qwen3 and OLMoE, rotary
    tables, and the top-k/softmax of the router.

---

## 3. Baselines

From R§4 (table copied), plus one baseline the plan adds (deviation 5).

| Baseline | Why it is required | How to realize it on Vortex |
|---|---|---|
| Vortex default (explicit config) | Unmodified reference. "Default" has **no L2**, so state the cache configuration | Profile plus `--l2cache`, PLRU, TCU on |
| Global best static | One chip for the whole suite | Exhaustive sweep over the PhaseShift knob space |
| **Per-workload best static** | Boot-time configuration is cheap; **the core claim must beat this one** | Same sweep, chosen per (model, B, T, trace) |
| **Per-kernel best static** (added by the plan) | With flush/reset per launch, kernel-boundary switching is free on Vortex, so the gap must also be shown against it (G2b) | Same sweep, configuration chosen per kernel launch |
| Prefill-tuned / decode-tuned static | Shows the policy conflict between phases exists | Best config on prefill-only / decode-only intervals |
| Zero-cost oracle and **switch-cost-aware oracle** | Upper bound; the zero-cost version overstates the gain when intervals are short | Per-interval best; dynamic program that pays C_sw (§5) |
| **NVIDIA-practice baseline** | Shows the gain is not already reachable with shipping knobs | Per-kernel LMEM usage via `KMU_LMEM_SIZE`, plus per-launch persisting/streaming windows and eviction hints emulated by the class-bit mechanism |
| Sarathi-style chunked prefill on the best static config | Software already recovers part of the gain [cand:A5] | Chunked prefill tiles in the kernel schedule and the serving simulator |
| Software KV prefetch | Shows hardware policy beats a kernel-level prefetch [cand:B17] | DXA issue in the decode-attention kernel |
| Adaptive baselines that ignore phase: UCP (A), PADC/BLISS + FR-FCFS + static demand-first (B), FDP (C), SHiP-Mem / Hawkeye / Mockingjay (D), CCWS warp throttling | Show that *phase* information, not just adaptivity, provides the gain | SimX C++ policy variants |
| DuetServe-style core partitioning | Closest competitor; show complementarity | Assign Vortex cores or clusters to prefill vs decode (limited by the single-queue CP) |
| SPAD, Splitwise/DistServe, PagedAttention-style KV paging | Specialization vs disaggregation vs allocation | Analytical comparison; paged KV layout in kernels |

Sources of the named baselines: UCP [cand:C40]; PADC [cand:C25]; BLISS [cand:C22]; FR-FCFS [cand:C19]; FDP [cand:C27];
SHiP [cand:C35]; Hawkeye [cand:C36]; Mockingjay [cand:C37]; CCWS [cand:C17]; DuetServe [cand:A2]; SPAD [cand:A1];
Splitwise [cand:A3]; DistServe [cand:A4]; PagedAttention [cand:A6]. The plan implements Hawkeye/Mockingjay and CCWS only
as optional baselines or limitations (item 10).

**Definitions of the static baselines** (let a workload w be a sequence of intervals i = 1…n, §5, and K(i) the kernel
launch that interval i belongs to, numbered in launch order):

- per-workload best static: `T_static(w) = min_c Σ_i T(i, c)` — one configuration for the whole workload;
- per-kernel best static: `T_kernel(w) = Σ_k min_c Σ_{i: K(i)=k} T(i, c)` — one configuration per kernel **launch**,
  switching free at launch boundaries (justified by consequence 4 of §2). Here k ranges over the workload's kernel
  launches and K(i) is the launch that interval i belongs to (§5); two launches of the same kernel function may get
  different configurations, which makes this baseline the strongest static-per-launch choice (and G2b stricter);
- global best static: `c_global = argmin_c geomean_w [ Σ_i T(i, c) / T_static(w) ]` — one configuration for the
  whole suite. Each workload's cost is normalized by its own best static cost so that long workloads do not dominate
  the choice, and the geometric mean is used because the normalized costs are ratios.

Because a per-workload static choice is a valid per-kernel choice, `T_kernel(w) ≤ T_static(w)` always.

The proposal's ablation matrix (A, B, C, D, A+B, all-ideal, full-practical, oracle; absolute and incremental gains) and
its adversarial cases (short prefill, high-batch decode, prefix-cached prefill, long attention-heavy prefill,
speculative or multi-token decode) are kept (R§4).

---

## 4. Regions

Numeric definitions, applied per **iteration** (one serving-scheduler step, §10):

- an iteration is **prefill** if it has prefill tokens and no decode tokens;
- **mixed** if it has both prefill and decode tokens;
- otherwise **decode**;
- a decode iteration is **KV-bound** if `ρ_KV = M_KV,total / (q_w·N_params) ≥ 1`, else **weight-bound**. Here
  M_KV,total is the KV bytes read in that step (E3, summed over the iteration's sequences, Σ_i 2·L·H_KV·D_h·q·T_i) and
  q_w·N_params the weight bytes per step. ρ_KV = 1 is exactly the E6 crossover where KV traffic equals weight traffic,
  so the threshold is physical, not tuned. For MoE models "weight bytes per step" means the bytes actually streamed
  in that step (shared weights plus the experts the step's tokens touch, opgen's per-step `w_rd`), not q_w·N_params,
  which counts every expert; `model/roofline.py: rho_kv(..., weight_bytes=…)` takes that value.

This gives **four regions**: prefill, mixed, decode-weight-bound, decode-KV-bound. A **workload belongs to the region
holding the majority of its baseline time**. Modeling choices made here to make that rule executable:
"baseline time" is time under the explicit Vortex-default configuration of §3 as fixed by PS-BASE, so region
membership does not depend on the configuration being evaluated; when no region holds more than half of the time,
the workload goes to the region with the largest share, and the share is reported so the borderline cases are visible
. A later step that changes this writes a decision record.

---

## 5. Oracle, capture, regret and gates

**Intervals are work units, never cycles.** An interval is a fixed unit of completed work (completed tiles or loop
iterations, marked by `ps_mark`), so interval i covers the same work under every configuration and
T(i, c) is comparable across c. Cycle-based intervals would cover different work under different configurations and
make the DP below meaningless.

**Zero-cost oracle.** `T_zero(w) = Σ_i min_c T(i, c)`: the best configuration in every interval, switching free. It
is an upper bound that overstates the gain when intervals are short.

**Switch-cost-aware oracle (dynamic program).** R§7's formalization of "the Viterbi optimum that pays C_switch" is
`J(i, c) = T(i, c) + min_c′ [ J(i−1, c′) + C_sw(c′, c) ]`, answer `min_c J(n, c)`. PhaseShift uses it with two stated
details:

- `J(1, c) = T(1, c)`: the initial configuration is free, consistent with the static baselines (which also pay no
  cost to start in their configuration);
- the switch cost depends on the interval (modeling choice): `C_sw,i(c′, c) = 0` if interval i is the first
  interval of a new kernel launch (K(i) ≠ K(i−1)), else `C_sw,i(c′, c) = C_sw(c′, c)`, with `C_sw(c, c) = 0` (staying
  costs nothing). Reason: on Vortex every launch resets the caches (§2 consequence 4; errata item 12), so a change of
  configuration at a launch boundary adds no cost beyond what every schedule, static ones included, already pays;
- the recurrence actually computed is therefore
  `J(i, c) = T(i, c) + min_c′ [ J(i−1, c′) + C_sw,i(c′, c) ]` for i = 2…n, and the answer is `T_DP(w) = min_c J(n, c)`;
  J(i, c) is the cheapest way to finish interval i in configuration c.
- Complexity: n intervals × |C| states × |C| predecessors = **O(n·|C|²)** time, O(n·|C|) memory with back-pointers to
  recover the schedule (needed for replay).
- C_sw comes from the measured/analytical flush walk plus writebacks for modes that need a flush (§13) and is near
  zero for policy-only modes (way mask, replacement, prefetch depth). C_sw ≥ 0 always.

Ordering: `T_zero ≤ T_DP ≤ T_kernel ≤ T_static`. Why each step holds:
- `T_zero ≤ T_DP`: every DP schedule costs Σ_i T(i, c_i) plus switch costs ≥ 0, and Σ_i T(i, c_i) ≥ Σ_i min_c T(i, c).
- `T_DP ≤ T_kernel`: the per-launch schedule (configuration c_k on every interval of launch k) changes configuration
  only where K(i) ≠ K(i−1), where `C_sw,i = 0`, so its DP cost is exactly T_kernel; the DP minimizes over all schedules,
  this one included. (With a position-independent C_sw this step would not hold: the per-launch schedule would pay
  C_sw at each launch boundary where its configuration changes.)
- `T_kernel ≤ T_static`: one configuration for every launch is a valid per-launch choice. Speedups are reported as
`T_static / T_DP` (and `T_kernel / T_DP` for G2b); a region's speedup is the geometric mean over its workloads, because
speedups are ratios and the geometric mean does not depend on which configuration is the normalizer.

**Composition assumption.** T(i, c) is measured per interval and composed additively in the DP; this ignores state
carried across an interval boundary (cache contents, in-flight requests). validates composed DP costs by
replaying ≥ 10 DP schedules per region on SimX (`PS_SCHEDULE`); if replay error exceeds the bound, the
gates use replayed costs.

**Capture (E10) and regret (E8).** Capture uses P_static-best = per-workload best static and P_oracle = DP oracle;
regret is reported per interval and time-weighted. Caveat: both are ratios of small differences; they are only
reported where the denominator (the absolute oracle gap) is above the sampling-error bound, and the absolute
gap is always reported alongside.

**Implementation notes (`model/oracle.py`).** Choices the definitions above leave open:
- *Inputs and validation.* Per interval: cost row T(i, ·), workload label, launch id K(i) (required; integers,
  non-decreasing within a workload, i.e. launch order) and an optional kernel-type label (diagnostic only: the
  per-kernel baseline groups by launch, §3). A workload's intervals must be contiguous. T must be finite and ≥ 0;
  C_sw ≥ 0 may contain +∞ (forbidden switch); `C_sw(c, c) ≠ 0` is rejected. The DP runs per workload, so every
  workload's first interval starts free (`J(1, c) = T(1, c)`).
- *Ties.* The DP back-pointer and every argmin take the lowest configuration index, so schedules are deterministic.
- *Prefill-/decode-tuned static.* The global-best-static rule of §3 applied only to the intervals labelled prefill
  (resp. decode), i.e. `argmin_c geomean_w [Σ_{i∈w, phase(i)=p} T(i, c) / min_c′ Σ_{i∈w, phase(i)=p} T(i, c′)]` over
  workloads that have such intervals; that configuration is then used for the whole of every workload.
- *Capture form.* E10 with P = 1/T: each workload performs fixed work, so 1/T is its throughput:
  `capture = (1/T_PS − 1/T_static) / (1/T_DP − 1/T_static)`. The reduction-ratio form for lower-is-better metrics,
  `(T_static − T_PS) / (T_static − T_DP)`, is reported next to it. Both lie in [0, 1] when T_DP ≤ T_PS ≤ T_static
  and are undefined (NaN) when the gap is 0.
- *Regret attribution.* Per-interval cost of a schedule = T(i, c_i) plus the switch cost paid on entering interval i
  (charged to the interval the new configuration serves). Per-interval E8 is `(T_det,i − T_oracle,i)/T_oracle,i`
  (can be negative for single intervals); time-weighted E8 weights by the oracle's interval cost, which equals
  `(Σ T_det − Σ T_oracle)/Σ T_oracle`.
- *Bootstrap.* Percentile bootstrap (default 10,000 resamples, fixed seed) of the statistic (default the geometric
  mean of per-workload speedups) over resampled units: workloads, or whole groups (e.g. all workloads of one
  arrival seed) when a grouping is given. 95 % CI = the 2.5 % and 97.5 % quantiles. Caveat: the percentile
  bootstrap under-covers at small n — the measured 0.905 coverage of the nominal 95 % interval at
  n = 20 workloads — so every CI is reported with its n, and a gate decided by a CI bound with few workloads is
  flagged in its decision record (a better interval, e.g. BCa or a t-interval on log speedups, needs a recorded
  decision before use).

**Gate rules (copied verbatim; thresholds are defaults to agree with the advisor):**

| Gate | Rule | Action |
|---|---|---|
| G2 (R1) | **PASSES iff** ≥ 2 of the 4 regions have geomean (over that region's workloads) DP-oracle speedup over per-workload best static ≥ 10 % **and** the bootstrap 95 % CI lower bound of that speedup > the sampling-error bound. Performance only; DRAM bytes reported as energy proxy. Costs must be schedule-replay validated. | fail → STOP: characterization paper + edge-decode pivot (§4 EP) |
| G2b | DP-oracle speedup over **per-kernel best static** (free kernel-boundary switching) < 5 % in **every** region | STOP: the contribution would reduce to per-kernel configuration (R§2 fact 2, R§Conclusion) |
| G3 (R2) | geomean gain of the **ideal-hint PhaseShift policy** (runtime hints, no detector error; not the oracle) over best static + Sarathi chunking < 5 % in every region where G2 passed | STOP: complementarity claim only if additive, else pivot |
| G6 (R5) | for some region, the 95 % CIs of the oracle gap on S1 and S2 lie on opposite sides of zero | STOP: scope to edge-class GPUs |

(S1/S2 in G6 are the scale tiers of §7 below.)

### 5.1 Schedule replay format — `PS_SCHEDULE`

A DP schedule (above) is a sequence of configurations over work-unit intervals. ps patch `0004-sched-replay` makes
such a schedule executable inside one SimX launch (implementation `vortex-trees/ps/sim/simx/ps_schedule.h`,
generator/validator `scripts/mkschedule.py`, which applies the same rules).

**File format.** One switch per line; `#` starts a comment; blank lines are ignored:

```
[launch=<k>] <work_count> <knob>=<value> [<knob>=<value> ...]
```

* `work_count` — non-negative decimal integer in the **work unit** interval trace (§12.10):
  `PS_SCHEDULE_UNIT=marks` (default; the per-launch `ps_mark` total summed over cores) or `instrs` (instructions retired
  since the launch began, summed over cores). If `PS_SCHEDULE_UNIT` is unset, `PS_TRACE_UNIT` is used, so a schedule
  and a trace of the same run share one unit. Lines must be in non-decreasing `work_count` order (checked).
* `launch=<k>` — optional; the line applies only to launch *k* (0-based launch index since device open). A line
  without it applies to every launch (the work counter restarts at every launch, so the schedule is per launch).
* `value` — decimal or `0x` hex.
* Knobs = the **policy-only** `PhaseConfig` fields of `[dcr_phase]` (mechanisms D, B, C) and the
  classification ranges: `mode`, `insert`, `dxa_depth0`…`dxa_depth7` (8-bit), `dram_sched`, `dram_aging`,
  `kv0_base`, `kv0_limit`, `kv1_base`, `kv1_limit`, `wgt_base`, `wgt_limit`, `range_acct_only` (0/1).
* **Rejected** (SimX aborts at start-up, `mkschedule.py` refuses): `l1_waymask`, `l2_waymask` (A2: a way-mask change
  needs a flush, §13), `lmem` / `kmu_lmem` (A1: needs a kernel boundary), `persist_l2` / `flags` (launch-boundary
  sensitivity knob, §5.2), and any unknown knob.

**Semantics.** At the start of every launch the effective configuration is the host-written (DCR-latched) one; the
launch's lines are then applied in file order, each on top of the previous effective configuration (a line changes
only the knobs it names). Lines with `work_count = 0` apply before the launch's first cycle. A line with
`work_count = W > 0` applies **at the end of the first simulated cycle in which work ≥ W** (several lines crossed in
one cycle apply in order in that cycle); the new configuration is pushed to every consumer through the same path as a
host DCR write (L1/L2/L3 latches, DRAM model, DXA), which costs no simulated cycles (policy-only switch, C_sw = 0 by
§13). The switch check runs after the trace check of the same cycle, so the trace row that closes in the switching cycle
belongs entirely to the old configuration. Consequently, when W is a multiple of the trace step n, rows 1…W/n ran
under the old configuration and row W/n + 1 is the first row under the new one; requests already in flight at the
switch keep the class/policy they were issued with (the class is stamped at the first cache, §12). At the end of the
launch (before its trailing flush) the host configuration is pushed back, so the flush and any host DCR write between
launches see the host latch. Every switch prints
`PSSCHED launch=<k> line=<n> work_count=<W> work=<w> cycle=<c> <PhaseConfig::str()>` (observation only).

**Modeling choice.** Switching on the cycle boundary after the crossing (not mid-cycle) matches how a hardware
controller would act on a counter it samples once per cycle; the alternative (re-classifying in-flight requests) has no
hardware analogue. Policy knobs are latched state read at decision points, so changing them between two cycles is
well defined; way masks and LMEM size change the cache/LMEM geometry and therefore stay at flush/kernel boundaries.

### 5.2 Persistent-L2 sensitivity knob — `persist_l2`

Stock SimX starts every launch with cold caches (§1; `ProcessorImpl::run()` → `reset()`), which hides cross-launch
reuse. The `[dcr_phase]` `flags` bit 0 (`persist_l2`, default 0) is a **labelled sensitivity study** of the
"avoid the flush" avenue (R§2 fact 2), never a default configuration:

* When set (host latch at the launch's reset), every **L2** bank keeps its tag/data/replacement state across the
  per-launch reset and the L2's init walk (`sets_per_bank` cycles, the RTL invalidation walk) is skipped. MSHRs,
  pipelines and every other cache level (L1 icache/dcache, L3) are reset as before. The trailing flush still writes
  back and cleans every dirty line (stock SimX behaviour), so kept lines are clean when the next launch starts.
* **Host writes invalidate.** SimX caches hold data, so a kept line could go stale when the host writes device memory
  between launches (`vx_enqueue_write`, argument/kernel uploads, completion words). In persist mode the runtime's
  device-memory write hook (`sw/runtime/simx/vortex.cpp`, `dram_write`) calls `Processor::ps_host_write`, which drops
  every L2 line overlapping the written range (line granularity, dirty or not — the flush has already cleaned them).
  The run prints `PSPERSIST launches=<n> host_inval_lines=<m>`. `PS_PERSIST_NO_INVAL=1` disables the invalidation;
  it exists only as the negative control of the stale-data test (results of such runs are wrong by construction).
* **Never by default** (two static checks): the `static_assert` that a default `PhaseCfg` has the bit clear, and
  the host helper (`kernels/common/ps_phase.h`): `phase_enqueue` / `phase_write` refuse (return −77, write nothing) a
  config with the bit set unless the host code is compiled with `-DPS_ALLOW_PERSIST_L2=1` (default 0,
  `static_assert` that it is 0 or 1). Apps expose the knob as `-P` (`ps_twolaunch` / `ps_gemv_f32`, `ps_llama`).
* **Scope of the approximation.** Real GPUs keep L2 contents across kernels but still pay coherence actions for
  host-visible writes; here a host write costs no cycles (it happens between launches, outside kernel time) and only
  removes lines. DMA reads by the host see device RAM, which the trailing flush has made current. Results of persist
  runs are reported separately and labelled "persist-L2 (sensitivity)"; G2 is evaluated in the cold mode.

---

## 6. Sampling

LLM structure makes sampled simulation defensible (R§7):

- **Layers are identical.** All L layers have the same shapes, so simulate one or two layers per step and multiply:
  `T_step ≈ T_pre + L·T_layer + T_post`, where T_layer is the measured cost of one layer (the mean of the simulated
  layers) and T_pre / T_post are the embedding and final norm/classifier costs, simulated once. The approximation
  ignores cache state carried between layers; with caches cold per launch (§2) and one launch per kernel this error is
  small, and it is measured (below).
- **(B, T) grid + interpolation.** Step cost is a smooth function of (B, T), so simulate a grid and interpolate (in
  effect stratified sampling). Defaults (modeling choices; may refine them with a recorded reason, and measures
  their error):
  - *Decode (the T / KV dimension):* linear in T, because KV traffic and decode-attention FLOPs are linear in T (E3,
    E4 decode term, E6); linear between the simulated B points.
  - *Prefill (the S dimension):* not linear. Prefill linear layers scale with S (E1, N_tok = B·S) and prefill attention
    with S² (E4). Per (B, configuration), fit `T_prefill(S) = a + b·S + c·S²` by least squares to ≥ 4 simulated S
    points: the constant term covers launch and fixed costs, b·S the linear layers, c·S² attention. Use the fit only
    inside the simulated S range and report its residuals. Mixed iterations are simulated at grid points and
    interpolated the same way: quadratic in the prefill-chunk length, linear in the decode context T.
  - Never extrapolate outside the simulated grid.
- **Early stop inside long kernels.** Stop once IPC converges, as Principal Kernel Analysis does [cand:D1]; the
  convergence rule is set by the step that uses it.
- **Measured vs projected.** The error of layer multiplication and interpolation is measured against a few full runs
that measured error is "the sampling-error bound" used by the gates (§5). Measured results and projected
  (multiplied/interpolated) results are kept in separate columns and never mixed silently.

### 6.1 Sampling-error validation: the error metric and the single number

This subsection and §6.2–§6.4 were written at 07:00–07:04 UTC on 2026-11-07 while the first runs of the batch were going
(corrected; the ledger supports this, not the earlier "before any run had finished"): the five
PS-BASE repro rows, all 11 new decode grid points and 3 prefill grid points had ended by 07:04:16, no (b) data beyond those and no (c) data existed (grid
runs 07:01–07:16, prefix runs 07:16–07:44), and no error had been computed (the first analysis run is the ledger's
`:ana-dev` at 07:04:16; (a1) needs only stored data). The rules below were not tuned on the data; the (b3) bullet
of §6.3 was added later and is marked as such. Results: §6.5. Code:
`results/<run>/make_batch.py` (runs), `results/<run>/analyze.py` (every error), numbers `results/<run>/README.md`.

- **Error of one estimate.** e = (estimate − measured) / measured, where *measured* is a full SimX value of the same
  quantity (a whole decode step, a whole launch) and *estimate* is what the sampling method would have reported without
  simulating it. Statistics use |e|. Measured and estimated cycles are separate columns of `sampling_error.csv`.
- **Sub-tests.** (a1) layer multiplication inside the full sequence, (a2) one layer composed from isolated single-launch
  kernels, (b1) (B,T) interpolation with a 20 % hold-out, (b2) interpolation from the tier-C 3 × 3 grids that
  actually measured, (c) early stop by truncated work (§6.2–§6.4).
- **The bound** (used by G2, replay checks and the E8/E10 "denominator above noise" caveat, §5) =
  **max over the sub-tests of the 95th percentile of |e|** (numpy `percentile(…, 95)`, linear interpolation between order
  statistics). Modeling choice: says "the 95th-percentile relative error over (a)–(c)". Pooling all samples into one
  percentile would let the sub-test with the most samples decide the number ((b) has ~10× more samples than (c)). If every
  sub-test's p95 is ≤ m, at most 5 % of each sub-test's samples exceed m, so at most 5 % of the pooled samples do: the
  pooled p95 is ≤ the max of the per-sub-test p95s (up to the interpolation between order statistics). The max is therefore
  the conservative reading; the pooled p95 is reported next to it.
- **Acceptance.** median |e| ≤ 5 % per sub-test, and in (b) per op family. Where it is exceeded the sampling
  scheme is changed along the failing axis (densify the grid, i.e. simulate that axis at every value used downstream) and
  the change is recorded; the bound is then computed for the scheme that proceeds, and the error of the original scheme is
  kept as a reported column (never silently dropped).

### 6.2 (a) Layer multiplication

- **Full step.** C_step(p) = Σ of the per-launch cycles of the 14 L + 2 launches of position p (`ps_llama`'s `PSL` deltas,
  §14.5; they sum exactly to the run's `PERF:` cycles, asserted per run). The token-embedding row is a device-to-device copy
  outside every kernel's cycles (§14.5), so the "embedding" term T_pre of §6 is 0 cycles in both the full and the sampled
  accounting; T_post = final rmsnorm + classifier GEMV.
- **(a1) in sequence.** For a sampled layer ℓ: Ĉ_ℓ(p) = L · Σ_{op ∈ layer} c(p, ℓ, op) + c(p, rmsnorm_final) + c(p, wcls),
  with every c taken from the same full run. One sample per (run, position p, ℓ ∈ {0 … L − 1}): each ℓ is a possible choice
  of the simulated layer. Tests layer heterogeneity only (data-dependent timing, DRAM-model state left by earlier launches,
  §2 consequence 5); T_post is exact by construction.
- **(a2) isolated composition** (what a tier-C sweep does: P5/P6 cost tables are built from isolated cold launches). One
  single-launch app run per distinct kernel shape of a stories15M layer at REG-CFG: GEMV 288 × 288 (counted 4×: q, k, v, o),
  768 × 288 (2×: w1, w3), 288 × 768 (w2), rmsnorm n = 288 (2× per layer, +1 final), residual n = 288 (2×), silu_gate n = 768,
  and the two position-dependent kernels at position p — `ps_rope` with its row 0 at position p (`-S p+1`) and
  `ps_decode_attn_gqa` with T = p + 1 keys (Tt = 32, as `ps_llama`) — plus the classifier GEMV 32,000 × 288:
  Ĉ_iso(p) = L · Σ (counted isolated launches) + c_iso(rmsnorm) + c_iso(wcls), compared with C_step(p)
  stories15M tf32 run, p ∈ {5, 15, 31}. Besides layer heterogeneity it contains the launch-history effect (§2 consequence 5),
  different buffer addresses (L1/L2 set mapping), synthetic instead of model data (data-dependent cycles of `ps_expf` and
  musl's soft-float libm) and the standalone apps' own launch geometry (checked equal to `ps_llama`'s).
- **Configurations (modeling choice).** (a1) and (a2) at REG-CFG, because stored every stories15M row there (32 + 8
  positions with per-launch counters), so the main test needs no new 2-minute-per-token run; (a1) is repeated on one new
  6-position stories15M run at PS-BASE-S1-4c (the machine P5/P6 use; 6 is the smallest count with a compared, non-prompt
  position, so the app's own TEST applies). Layer multiplication is structural (identical layer shapes, caches cold per
  launch), so both configurations test the same mechanism; PS-BASE differs in warp width and GEMV variants.

### 6.3 (b) (B, T) interpolation

- **Grids** (PS-BASE-S1-4c, base setting, one cold launch per point, the op families and shapes): decode attention
  (B, T) ∈ {1, 2, 3, 4} × {64, 128, 256, 512, 1024}; prefill attention (B, S) ∈ {1, 2, 3, 4} × {16, 24, 32, 48, 64} (s0 = 0);
  mixed attention (chunk c, decode context T) ∈ {16, 24, 32, 48, 64} × {128, 256, 512} (s0 = 0, n = 4 decode sequences;
  the earlier three mixed points are not a grid); TCU GEMM (M, matrix) ∈ {32, 64, 128, 192, 256} × {q, w1, w2} (multiples of
  cta_M = 32; the earlier "M = 16" cell ran as M = 32; the M = 1 GEMV is a different kernel and is not interpolated in M);
  element-wise (kernel, B) ∈ {rmsnorm, residual, silu_gate, rope} × {1, 2, 4, 8, 16}. the earlier tier-C cells are reused (37
  points; their cycles are re-measured on 5 of them, `repro`), 53 points are new. Every prefill row has 5 S points, so §6's
  quadratic still has ≥ 4 points after a hold-out (the earlier carry-over: its 3 S points could not support the fit).
- **Interpolants (modeling choice; model-free).** memhier is not used (§9.8: qualitative only). **Primary** = §6's defaults:
  decode attention piecewise-linear in T within a B row and linear in B between rows (bilinear on the rectilinear grid:
  exact for a + b·B + c·T + d·B·T, the launch constant plus the B·T work of E4's decode term); prefill attention per B row
  the least-squares fit a + b·S + c·S² over the row's training S points when there are ≥ 4 (§6; E1's S-linear plus E4's S²
  terms), otherwise piecewise-linear in S, and linear in B between rows; mixed attention the same in the chunk length c
  (rows = T), linear in T between rows; TCU GEMM piecewise-linear in M per matrix (E1: work ∝ M at fixed N, K);
  element-wise piecewise-linear in B per kernel. **Alternative** (reported, not used for the bound) = the same tensor
  structure with piecewise-linear interpolation of log(cycles) in the log of every axis (exact for power laws such as S²
  or B·T, wrong for a constant plus a power). Never extrapolated: a point outside its line's training range is not
  predicted.
- **(b1) 20 % hold-out.** Per op family, n_hold = round(0.2 · N) points drawn without replacement by
  `numpy.random.default_rng(3303)` (families in the order decode, prefill, mixed, linear, element-wise) from the points that
  are not corners of the family's grid (2-D families: the 4 rectangle corners; families that are a set of 1-D lines — TCU
  GEMM per matrix, element-wise per kernel — the two ends of each line), because predicting a corner needs extrapolation.
  The interpolant is rebuilt from the remaining points and each held-out point is predicted **per axis**: along its line's
  primary axis (T, S, c, M; B for element-wise) from the training points of that line, and, for the 2-D families, also
  along the other axis (B; T for mixed) from the training points of its column, whenever the point is bracketed on that
  axis. Each (point, axis) prediction is one sample: the test thus separates T-, S- and c-interpolation from
  B-interpolation.
- **(b2) tier-C operating error.** Training = only the points measured (decode/prefill 3 × 3 with B ∈ {1, 2, 4};
  GEMM M ∈ {32, 256}; element-wise B ∈ {1, 4, 16}); every new densified point is predicted (2-D where both coordinates are
  new: interpolate along the primary axis in the two bracketing rows, then linearly along B). With 3 S points per row the
  prefill row falls back to piecewise-linear (§6 needs ≥ 4). Mixed has no grid, so no (b2). This is the error P5/P6
  incur if they keep the earlier 3 × 3 grids.
- **(b3) densification round** (added 07:4x UTC, **after** (b1)/(b2) were computed and before its runs; the plan's remedy
  "densify the grid (recorded)"). (b1)/(b2) failed the 5 % median target on decode B, prefill S, prefill B and element-wise
  B (§6.5). Decode and prefill already have every integer B ∈ {1 … 4}, so B cannot be densified: B is simulated at every
  value in the scheme that proceeds. Densified instead: prefill S ∈ {16, 24, 32, **40**, 48, **56**, 64} (step 8) for every
  B, element-wise B ∈ {1, 2, **3**, 4, **6**, 8, **12**, 16}. Test: every point of a densified line except the two line ends
  is predicted leave-one-out from the line's other points by **piecewise-linear interpolation** (local: its two remaining
  neighbours). Recorded refinement of §6's prefill default: the global least-squares quadratic per row failed on 5-point
  rows (it cannot follow the 48 → 64 jump), so the local interpolant is primary for (b3) and the quadratic (over the
  remaining ≥ 4 row points) and log-log are reported as alternatives. Leave-one-out predicts across twice the densified
  spacing, so it over-estimates the error of interpolating inside the densified grid (conservative). Acceptance as §6.1: an
  axis whose (b3) median exceeds 5 % is simulated at every value used downstream (not interpolated) and its samples leave the
  bound; an axis that passes proceeds with the densified grid, and its (b1)/(b2) samples are superseded (kept, in_bound = 0).

### 6.4 (c) Early stop ("PKA-style", by truncated work)

- **Why not an IPC trace.** PKA stops simulating a kernel once its IPC has converged and projects the rest [cand:D1]. That
  needs the IPC as a function of simulated time inside one launch. Release SimX has none: its trace macros are compiled
  out (`sim/simx/debug.h`: `DT` is empty under `NDEBUG`, release builds use `-O2 -DNDEBUG`, `sim/simx/Makefile`), and the
  performance counters are read by the host only after `ProcessorImpl::run()` returns (`sim/simx/processor.cpp`, the
  `run()` loop). A debug build (`-O0`, one trace line per event) was rejected: it changes the build flavour of the shared
  `build/vortex-base`, is much slower and writes huge logs (K9). the earlier work-unit interval trace does not exist yet.
  An offline replay of a full run's IPC is therefore **not feasible** now; this is flagged (re-validate (c) on
  the interval trace).
- **Truncated work (what is implemented).** Causal prefill with s0 = 0 over the first S′ query tokens, with the full run's
  tiles forced (`-Q Qt -t Tk`), launches exactly the first ⌈S′/Qt⌉ query tiles of every KV head of the full problem, each
  with the same work as in the full launch (query tile i sees keys 0 … (i + 1)·Qt − 1 in both). A doubling sequence of
  prefixes S₁ < S₂ < S₃ (largest = S/2; S/8, S/4, S/2) plays the role of PKA's growing simulated window. Work unit
  W = computed (query tile, key tile) pairs (`tiles` of the app's PSINFO line). Marginal rate r_j = (C_j − C_{j−1}) /
  (W_j − W_{j−1}) = cycles per tile pair in the newest window (the steady-state analogue of IPC; cumulative cycles per tile
  would include the launch constant). **Stop rule** (my choice, not PKA's parameters): stop at the first j ≥ 3 with
  |r_j − r_{j−1}| ≤ ε · r_{j−1}, ε = 5 %; projection Ĉ = C_j + r_j · (W_full − W_j). If the rule never fires (it can only
  fire at j = 3 with three prefixes), the kernel would be simulated in full; to stay conservative the bound still uses the
  forced projection from the largest prefix, and the "no stop" outcome is reported. Saving = 1 − (Σ_j C_j) / C_full.
- **Kernels.** The three longest kernels at PS-BASE-S1-4c (its carry-over rows: (8, 2, 64) S = 512 dxa d2, Qt 16,
  Tk 25; (16, 2, 128) S = 256 lmem, Qt 4, Tk 28; stories15M (6, 6, 48) S = 512 dxa d2, Qt 64, Tk 32; 20–30 M cycles) with
  new prefix runs, and the three longest at REG-CFG + DXA (the earlier (16, 2, 128) S = 512 lmem, 672 M cycles;
  the earlier (16, 2, 128) S = 256 dxa d2 and (8, 2, 64) S = 512 lmem) with prefixes from stored bench rows (for the
  S = 256 kernel the stored S = 16, 64, 128, i.e. not an exact doubling at the first step; same tiles by
  the host rule at every S, checked; left the earlier instructions and cycles unchanged) plus one new prefix ((8, 2, 64)
  S = 128) and one re-measured stored row. Supplementary families, from (b)'s grid runs: decode attention B = 4, prefix in
  T (T′ keys = the first T′/Tt key tiles of every CTA's loop; W = T′; 128, 256, 512 → 1024) and the TCU GEMM w2, prefix in M
  (the first M′/32 row tiles; W = M′; 32, 64, 128 → 256).

### 6.5 Sampling-error results and the scheme that proceeds (variants, validity domain and error unit)

**Superseded for gates Revision 2.9** (decision of 2026-11-07): every gate point is simulated in full (no interpolation, no early stop); the binding sampling-error bound for G2/P6 is sub-test (a) (p95 0.18 %) combined with the G2 bootstrap. The 18.41 % bound and its variants below apply only to non-gate (exploratory) use of the cheaper scheme.

Numbers: `results/<run>/README.md`, `bound.json` (second-pass variants under its last key `variants_attempt2`; every earlier
key unchanged), `sampling_error.csv`, `loo.csv`, `variants.csv`, `prefill_tiles.csv` (103 new SimX runs, 0.78 h of job
wall; 6 re-measured stored rows reproduce cycles and instructions exactly after `kernels/common` edits).

- ** bound = 18.41 %** (the pre-registered rule: max of per-sub-test p95 |e|, (c) with the 8 kernels of §6.4 and
  forced projections; set by (b3) element-wise B, with (c) at 17.73 %). Pooled p95 of all counted samples: 2.79 %. Before
  densification: 44.29 % ((b1) on the 5-point lines; m3) and 35.82 % ((b2), i.e. the earlier 3 × 3 scheme, all primary
  samples).
- **Sensitivity of the bound to the (c) convention** (same rule, other components unchanged, their max is (b3)
  18.41 %). With forced projections: the -literal 3 long PS-BASE kernels K1–K3 give (c) p95 23.36 % ⇒ bound **23.36 %**
  (set by (c)); the 6 long kernels K1–K3 + R1–R3 give 19.98 % ⇒ **19.98 %**; the pre-registered 8 kernels (incl. the
  short supplementary S1/S2, whose "early stop" already simulates 88–89 % of the kernel, so they are not early stops of a
  long kernel in the sense) give 17.73 % ⇒ **18.41 %**. Counting refusals as full simulations (error 0, the stop rule's
  operational outcome) gives (c) 2.98 / 2.81 / 2.69 % ⇒ **18.41 %** in every kernel set. S1/S2 were included (§6.4) to cover
  the two other prefix families (decode T, GEMM M); they lower the (c) p95 only because the 8-sample p95 interpolates
  between the largest two order statistics. **Which convention and kernel set bind is an open decision
  (left open by this step; for gates superseded Revision 2.9); this step reports 18.41 % (pre-registered) with 19.98 % and 23.36 % beside it.**
- **(a) layer multiplication is exact for practical purposes:** in-sequence p95 0.18 % (228 samples, REG-CFG and
  PS-BASE-S1-4c; the six layers of a step differ by ≤ 0.8 %), isolated single-launch composition 0.05 % (3 positions). The
  launch-history effect (§2 consequence 5) and buffer-address effects largely cancel over a step (per-op ratios 0.98–1.045).
  T_step = L·T_layer + T_post with T_pre = 0 is adopted for P5/P6. Mixed builds (m6): (a2) compares isolated launches built
  after `kernels/common` edits with the pre-hint `ps_llama` measurement, which was not re-measured at REG-CFG.
- **(b) interpolation works along work axes and fails along batch/tile-quantized axes.** Decode attention in T: p95 0.56 %
  from the earlier {64, 256, 1024}; leave-one-out (LOO) over all 12 interior points 0.62 %. TCU GEMM in M (multiples of 32): hold-out
  ≤ 1.7 %, LOO over 9 points p95 2.14 %. **Mixed attention:** the 20 % hold-out gave only 2 c-axis and 1 T-axis samples (≤ 3.9 %);
  LOO over all 9 interior c points (the (b3) method) gives median 3.88 %, **p95 10.06 %, max 12.03 %** with the
  primary quadratic (piecewise-linear 2.66 / 11.00 / 13.02 %, log-log 2.73 / 12.28 / 14.12 %); the worst point is c = 32 at
  T = 128 (+12.0 %), interpolated across a query-tile step; mixed T LOO (5 points at T = 256) p95 0.62 %. Fails (median > 5 %,
  hence simulated at every value in the proceeding scheme): decode attention in B (the CTA wave count ⌈B·H_kv/cores⌉ makes
  cost a step function of B; B = 3 mis-predicted by ≈ 24 %), prefill attention in B and in S. **Prefill S (m2, verified in
  `prefill_tiles.csv`):** at Qt = 16 the cost steps with the query-tile count ⌈S/16⌉ (and the CTA grid B·H_kv·⌈S/16⌉): S = 40
  costs 0.93–0.95× S = 48 (both 3 tiles), S = 56 costs 0.85–0.88× S = 64 (both 4 tiles), and the jumps are at 32 → 40 and
  48 → 56 (B = 2: 728 k → 1.59 M → 1.80 M at S = 48/56/64), so S interpolation predicts across tile boundaries; densifying to
  step 8 cannot remove this because every LOO prediction spans a boundary — tile/wave quantization is the evident first-order
  cause (L2 set conflicts, errata 32, are not excluded as a second-order one); a further jump at 16 → 24 for B ≥ 2 (×2.0–2.6, 1 → 2 tiles), and B1_S16 runs with Qt = 8. Element-wise B passes on the densified grid
  (median 2.0 %) with a heavy tail (p95 18.4 %: `ps_rope`'s cost depends on the synthetic per-row positions — row 0 at S − 1,
  the others pseudo-random, `kernels/rope/main.cpp` — through musl's soft-float sinf/cosf, not on B alone; the sub-2k-cycle residual
  launches only set the maximum (residual_B2, 20.20 %), while the p95 (18.41 %) is set by rope). The §6 prefill quadratic in S is **withdrawn** for this machine (its (b1) median 20.7 %; on 7-point rows 16.9 %):
  prefill is simulated at the S values used. **The quadratic is kept for mixed c** because the withdrawal rule is per failing
  axis and mixed c passes the 5 % median target under both the hold-out and LOO, and because its LOO tail is the smaller one
  (10.06 % vs 11.00 % piecewise-linear); mixed c does share prefill's query-tile quantization (nqt = 2 for c = 16–32, 3 at 48,
  4 at 64), diluted by the 8 decode CTAs of the same launch, so chunk lengths on tile boundaries (multiples of Qt = 16) should be
  grid points. Log-log interpolation was never better than the primary interpolant by enough to matter except for TCU GEMM M
  (b1 0.41 vs 0.60 % median) and was not adopted. The memhier model played no part (§9.8). Mixed builds (m6): the reused
  rmsnorm/residual/silu_gate base cells have no repro row of their own; only one representative per reused family was re-measured (exactly).
- **(c) early stop by truncated work:** the stop rule fired on 5 of 8 kernels (errors +3.09, −1.95, +0.67, +0.88, +0.16 %)
  and refused on 3; on one of those (stories15M (6, 6, 48) S = 512 dxa, Qt 64) the marginal cost per tile pair rose 24 %
  between the last two windows and the forced projection is +25.6 % off, so the refusal was right. Savings are modest
  (34–45 % of a full simulation is spent on the three prefixes of a causal prefill). **Re-validate on the earlier interval trace**,
  also because the truncated-work stand-in differs from intra-launch early stop in concurrency (m5): a prefix launches fewer
  CTAs than the full kernel (K3: 6 / 12 / 24 CTAs vs 48 on 4 cores), a plausible cause of K3's +24 % rate rise.
- **Validity domain.** The bound and the scheme were validated only on PS-BASE-S1-4c (4 cores), at the base
  setting (no stock-knob or mechanism setting), for the `llama8b_s8` layer shape (attention (8, 2, 64); GEMMs q, w1, w2) plus
  stories15M for (a), with decode attention B ≤ 4 and T 64–1024, prefill attention S 16–64 with s0 = 0, element-wise B ≤ 16,
  mixed attention only with n = 4 decode sequences and s0 = 0 (no prior context), c 16–64 × T 128–512, TCU GEMM M 32–256 ((a) and (c) also used REG-CFG, REG-CFG + DXA for R1–R3), and early stop only for causal prefill with
  s0 = 0 and forced tiles plus the decode-T and GEMM-M prefix families. **Not varied, hence outside the validated domain:** the
  mixed decode count n, mixed prior context s0 > 0, and prefill with prior context (chunked prefill, a workload). Outside
  the domain a step must simulate, or re-validate (the earlier TEST provides 10 held-out direct runs).
- **Error unit (R3).** The bound is an **unweighted per-launch** (in (a): per-step) relative cycle error. Its p95
  (b3, 18.41 %) lies between `ps_rope` B = 4 (16.00 %, 825 k cycles) and B = 6 (18.84 %, 1.49 M cycles): it is **set by rope**
  (synthetic per-row positions), ≈ 26–28 % of an isolated stories15M layer (`iso_ops.csv`); sub-2k-cycle `ps_residual`
  launches supply only the max (20.20 %). G2 compares with a **workload-level** speedup confidence interval; the two are not
  the same quantity, no reweighting was pre-registered and none is applied; the earlier replay validation measures a
  workload-level error.
- **Scheme for P5/P6 (non-gate/exploratory use only; for gates superseded Revision 2.9):** simulate attention at every B and prefill at
  every S used; interpolate decode attention in T, mixed attention in (c, T) with the quadratic in c, the TCU GEMM in M,
  element-wise kernels in B on the densified grid; compose a decode step by layer multiplication; early stop only where its rule
  fires; stay inside the validity domain above. **Post-hoc scenario** (not validated as a bound; chosen after seeing the data;
  not adopted): simulating the no-stop kernels in full and element-wise launches at every B, with the interpolated axes tested by
  LOO (one component per axis), gives a max-of-p95 of **10.06 %**, set by mixed c (reported 3.79 % from the 2 hold-out
  samples of that axis; superseded). Caveat (m7): simulating element-wise launches at every B does not remove rope's
  synthetic-position error, so even this scenario understates the element-wise error of real batches. Adopting it is a recorded
  open decision.

### 6.6 Interval-cost table
Code and data: `results/<run>/` (`workloads.py`, `launches.py`, `make_batch.py`, `inert.py`, `topk.py`, `build_costs.py`,
`check_a.py`; README). Every number that can feed a gate is a **full simulation**: no (B, T), S or c
interpolation and no early stop anywhere in `costs.csv`; the only composition is layer multiplication, re-checked here (6.6.D).

**6.6.A Workloads (modeling choices).**
- *Model:* `llama8b_s8` (d 512, H_q 8, H_kv 2, D 64, d_ff 1792; the shape used by the characterization sweep and the sampling study) with **L = 32** layers (Llama-3.1-8B
  `num_hidden_layers`) and vocabulary **V = 128,256 / 8 = 16,032**: the width scaling applied to the vocabulary keeps the
  classifier's share of a step's MACs equal to the 8B model's (7.5 %). fp32 weights and KV (the P2 kernels): KV 2·L·H_kv·D·4 =
  32,768 B/token; weights q_w·N_params = 4 · 117,260,288 = 469,041,152 B/step (classifier included, embedding gather excluded:
  a device copy outside the cycle count, §14.5).
- *Regions* (§4) per iteration; a decode iteration is KV-bound iff ΣT ≥ 469,041,152 / 32,768 = 14,314 tokens.
- *σ* (§10.4) = (131,072 B / 32 MiB) / (32,768 B / 1 MiB) = **0.125** (reference h100-like Llama-3.1-8B bf16, target
  PS-BASE-S1-4c 1 MiB L2).
- *Synthetic grid* (one iteration each; ≥ 10 per region, because replays ≥ 10 per region): decode-weight (B, T) ∈
  {1,2,4,8}×{128,512} ∪ {(1,2048),(4,2048)}; decode-KV {(8,2048),(4,4096),(16,1024),(2,8192),(8,3072),(16,1536),(6,3072),
  (12,1536),(4,6144),(16,2048)}; prefill (B, S, s0) ∈ {(1,32,0),(1,64,0),(1,128,0),(1,256,0),(2,64,0),(2,128,0),(1,64,64),
  (1,64,256),(1,64,1024),(1,32,512)} (s0 > 0 = chunked prefill with prior context); mixed (c, s0, n, T) ∈ {16,32,64}×{2,8} at
  (s0 0, T 256) ∪ {(64,0,8,1024),(32,256,4,512),(64,0,4,2048),(16,0,8,128)}.
- *Trace iterations:* scheduler, chunked prefill c = 512·σ = 64, B_max = 8 (≤ c, §10.2), first 1,000 arrivals of Azure
  conv / Azure code / Mooncake conversation, lengths × σ with a 2,048-token clip (policy clip; 281 of the Mooncake requests are
  clipped). The scheduler's reference cost is a linear estimate (`workloads.ref_cycles`, 1 GHz) that only sets iteration timing
  (fixed-schedule assumption §10.3); it is never an evaluated cost. Sample: stratified by region, **2 iterations per non-empty
  stratum** per trace (seed 6101), only from iterations the P2 apps express exactly (at most one prefill chunk), each with its
  Horvitz-Thompson weight data (stratum_N, stratum_eligible, stratum_n in `workloads.csv`). Decode-only trace iterations
  use one context T = round(ΣT/B) (ps_decode_attn_gqa has one T; the §9 M14 collapse, exact for quantities linear in T); mixed
  ones keep the ragged contexts. *Reduction (rev 2.9 budget rule):* sampled 100 iterations per stratum and four prefill
  modes; at SimX cost takes 2 per stratum and one chunked mode.
- *Launches of an iteration* (run.c order, `launches.py`): rmsnorm, q, k, v, rope, attention, o, residual, rmsnorm, w1, w3,
  silu-gate, w2, residual (× L via `layer_mult` = 32), then final rmsnorm + classifier over the R rows that emit a token (none for
  a non-final prefill chunk). GEMM rows M = 1 → `ps_gemv_f32 -v lanes`; M ≥ 2 → `ps_linear -v tcu` (WGMMA + DXA; host pads M
  to 32 and N to 64), as. Decode attention `-v dxa -p 2`, prefill attention `-v dxa -p 2` (pilot: 868,342 vs 918,640 cycles
  for lmem at S = 64), mixed attention default; the two marked kernels (decode, mixed attention) are PS_HINTS=1 builds with
  `-R kv` in **every** configuration (one binary per launch; work units = `ps_mark` tiles).

**6.6.B Configuration set and aliasing.** Planned single-mechanism configurations (knobs.yaml values): base; D insert-LRU,
bypass; B strict demand-first (W 0), strict W 256, strict W 1024; C cap 1, 2, 4; A2 ½ and ¼ of the L2 reserved for
non-weights. **A2 was removed** after both settings produced functionally wrong results on launches (A2 ½:
`ps_silu_gate -b 64`; A2 ¼: 7 `ps_rope` launches; deterministic; the same launches PASS unmasked and under every B/C/D
configuration): a failing run cannot supply a cost, and a partial column breaks alignment (README, BUG-QUEUE). **Pruned:** A1
small tile (17–48 % slower on every decode and mixed cell; one CTA per core; `a1_probe.csv`: +11…+36 % on all 8
probe launches); B work-conserving modes (|gain| ≤ 0.45 % at PS-BASE bandwidth). Mechanism A is therefore absent from
`costs.csv`. Cross (tier C): the top-2 settings of each of D, B, C (off a candidate; geometric mean over all workloads of
T_iter(s)/T_iter(base), `topk.py`): D {LRU, bypass}, B {strict W 1024, off}, C {off, cap 4} → 8 combinations, 6 new; 15
configurations in all. **Aliasing:** a mechanism whose probe runs (two or more shapes per kernel type) are bit-identical to base
in instrs, cycles and all 256 class-8 totals, and whose structural precondition is absent (D: no KV_STREAM L2 access; B: no
BULK DRAM read; C: no DXA stream), is inert on that kernel type (`inert.csv`) and dropped from the knob vector before the
launch is simulated, so the cell's run is the same simulation (`provenance.csv`).

**6.6.C Intervals and composition.** Interval = a work unit group: in the marked launches, n = ⌈W/16⌉ `ps_mark` units (W =
the launch's mark count; ≤ 16 intervals per launch; prefill and decode marks counted separately per interval, columns
`marks_prefill`, `marks_decode`); every other launch is one interval. The base run of a marked launch is traced at n = 1 and
regrouped in blocks of n (exact: the n-trace boundary "counter ≥ k·n" is the n = 1 boundary k·n and rows are deltas; checked
against re-traced runs). The post-launch flush rows are added to the launch's last interval, so Σ intervals = PERF cycles.
**Composition assumption:** no state carries across an interval boundary — T(i, c) is measured with the whole iteration in c
and composed additively; across launch boundaries this is exact on Vortex (caches reset per launch, §2), inside a launch it is
the assumption the earlier schedule replay validates. On 4 cores equal mark counts can cover different tiles (§12.10).

**6.6.D Re-check of layer multiplication (rev 2.9).** (i) ps_llama stories15M `-m prefill` (prefill-pass and decode step types)
under base and one configuration per mechanism; (ii) llama8b_s8 layer launches re-run with other synthetic data for one decode,
prefill and mixed workload under every single-mechanism configuration of the attention launch. Results: `check_a_summary.csv`,
README.

### 6.7 Oracle study and schedule replay
Code: `results/<run>/oracle_study.py` (every choice below is also stated in its docstring); results and numbers:
`results/<run>/README.md`.

**6.7.A Cost matrix.** For workload w the oracle runs on `T'(i, c) = layer_mult_i · T(i, c)` from `results/<run>/costs.csv`
(15 configurations; layer launches × 32, final rmsnorm + classifier × 1). Running the DP on the one representative layer and
multiplying is the same as "one schedule in every layer"; since all layers have identical interval costs under layer
multiplication and the DP optimum is per interval when C_sw = 0, a per-layer schedule could not do better.

**6.7.B Switch cost.** Every configuration differs only in policy-only knobs (D insertion, B arbitration, C DXA depth; A2
excluded for wrong results, A1 small pruned), for which §13.1 gives C_sw = 0 device cycles. The C_sw matrix is therefore 0 and
the DP oracle equals the zero-cost oracle. Any real cost of a mid-kernel switch (state left behind by the previous
policy) is not modelled by C_sw; it appears in the replay error (6.7.D).

**6.7.C Baselines.** Vortex default = configuration `base`; global best static (geomean rule of §5/§3); per-workload best
static (G2 reference); per-kernel best static = one configuration per launch (G2b reference); prefill-/decode-tuned static
(phase label of a workload's intervals = its region: prefill → prefill, decode-weight/-KV → decode; mixed iterations carry both
phases inside one launch and join neither tuning set); zero-cost oracle; DP oracle. Region speedups are geometric means over the
region's workloads; CIs are percentile bootstraps over workloads (10,000 resamples) repeated with 5 seeds, and the gate uses the
smallest lower bound (conservative). The percentile bootstrap under-covers at n = 10–16 (§5 caveat); the G2 outcome is decided
by the point estimate, not by a CI bound.

**6.7.D Replay error.** A DP schedule is turned into `PS_SCHEDULE` files (§5.1). For a marked launch (decode / mixed attention)
interval k's configuration becomes a line at `work_count` = the mark count at which interval k starts (the `work_end` of
interval k − 1; unit marks), with all four policy knobs written explicitly; consecutive equal knob vectors are merged. Unmarked
launches are single intervals; for prefill workloads (no marked launch) every launch whose DP configuration is not `base` is
re-simulated with an oracle-generated constant schedule (unit instrs, W = 0, all knobs explicit), and base launches are the
base runs (the same deterministic simulation). Because every launch resets the caches, launches compose exactly, so the
replayed iteration cost is `T_rep(w) = T_DP(w) − Σ_{k replayed} mult_k · (T_DP,k − T_sim,k)`. Replay error =
`|T_DP(w) − T_rep(w)| / T_rep(w)` over the whole iteration; the per-launch error of the replayed launch(es) is reported next to it
(stricter, not diluted by unaffected launches). Acceptance: the 95th percentile of the iteration error over all replayed
workloads ≤ the sampling-error bound (6.7.E); the percentile is the linear-interpolation one (`numpy.percentile` default, `oracle_study.py:396`): on the 58 replayed rows it is 0.77 %, nearest-rank would give 0.87 %, and both lie below 1.04 % and above 0.18 % (`` correction note 3); otherwise the gates are evaluated on replayed costs (`gap_by_region_replayed.csv`),
which exist for every workload.

**6.7.E Sampling-error bound used as "the bound".** 1.04 %, the largest layer-multiplication error measured anywhere
(re-check, stories15M prefill pass), adopted conservatively for every region: the rev 2.9 full-step vs one-layer × 32
comparison on the llama8b_s8 shape was not executable (the only full-model app, ps_llama, runs llama2.c checkpoints with its own
kernel set). The bound enters the G2 CI condition, the replay acceptance and the capture-reporting rule (§5). G2's 10 %
point-estimate condition does not involve the bound.

**6.7.F Energy proxy interface.** DRAM bytes per interval = Σ DRAM read classes (ord, wgt, kvs, dxa) + DRAM writes
(`intervals/*.csv`), × layer_mult; reported for default, per-workload static and DP schedule; not a gate input. For the
serving DES (`serving/des.py` `load_oracle_schedule`) the oracle is delivered per region as `{"ratio": r}`,
`r = geomean_w T_DP(w) / T_base(w)` over the region's workloads (`results/<run>/oracle_schedule.json`; the replayed variant
`oracle_schedule_replayed.json` uses T_rep).

---

## 7. Scaling ladder and dimensionless axes

R§7. The scale tiers (named S0/S1/S2):

- **S0**, tiny, for RTL and SAIF only;
- **S1**, 4–16 cores, for the main SimX studies;
- **S2**, an a100/h100 *slice*: keep per-core resources, L2 per core and bandwidth per core, with fewer cores. This
  preserves the ridge point (E7: per-core FLOPs and per-core bandwidth both scale with the core count, so their ratio
  is unchanged) at about 1/10 of the cost. The slice is the team's recommendation; no cited GPU precedent was found.

Models are shrunk proportionally. The only claim made across tiers is that the **sign and rough size of the oracle
gap** hold across S1 and S2 on **dimensionless axes**: AI relative to the ridge (AI / ridge, E2/E5 vs E7) and KV
working set relative to L2 (M_KV,total / L2 size, E3). Anything beyond that is presented as an analytical projection.
Disclosed distortions of the shipped profiles: the TCU is off; L1 and shared memory are approximated by `DCACHE_SIZE`
alone; L2 is rounded down to 32 MiB; H100 latency is "not calibrated" (R§7). Virgo [cand:B19] is the precedent for a
small Vortex study compared against modeled Ampere- and Hopper-style designs with equal MACs.

---

## 8. Two-level methodology and goodput

R§7. **Level 1 (cycle level):** SimX produces a latency and energy table indexed by (phase, B, T, configuration,
model). **Level 2 (serving):** a small discrete-event serving simulator replays trace arrivals (Azure [cand:D19] or
Mooncake [cand:D20]) against that table. This is the only way to obtain P50/P95/P99 TTFT and TBT, SLO attainment
versus request rate, and goodput on a 7 GB, low-load machine.

**Goodput** is DistServe's metric [cand:A4]: requests or tokens completed within phase-specific SLOs (in PhaseShift: a
TTFT SLO for the prefill phase and a TBT SLO for decode). Several arrival seeds are run and bootstrap 95 % confidence intervals are
reported. The cycle simulator is deterministic for a fixed launch sequence (launch-history rule: §2, consequence 5),
so all variance comes from workload sampling (arrival seeds, request sampling). Serving metrics and statistics are specified in §15.

---

## 9. Analytical memory-hierarchy model (memhier)

Code: `model/memhier.py` (class `MemHier`, parameters `Hier`, knob vector `Knobs`); tests `tests/test_memhier.py`;
example outputs `results/<run>/`. The model costs one serving iteration (the opgen op trace of §2) in cycles on a
roofline machine with an L1 / LMEM / L2 / DRAM hierarchy, under a PhaseShift configuration c = (A1, A2, B, C,
D). It was **uncalibrated** and exists to rank configurations and size the oracle gap before SimX (G1 is
informational for that reason, item 4). Every modeling choice is numbered M1–M14 below with its expected
failure mode; the code comments refer to these numbers.

> ** outcome (§9.8): the model is QUALITATIVE ONLY.** Calibrated against 306 single-launch SimX rows on
> PS-BASE-S1-4c, its held-out MAPE is **70.6 %** (> 25 %), and it does not reproduce the earlier stock-knob rankings. 's
> numbers (G1) are therefore **not used for any gate**; they remain an informational record of what the uncalibrated
> model says.

### 9.1 Knobs (the configuration c)

| Knob | Field | Meaning in the model |
|---|---|---|
| A1 LMEM usage | `lmem_use` (bytes per CTA, 0 = no LMEM tiles) | tile size β (M10) → re-read factors; CTAs per core n_cta (occupancy). On GPU-like machines LMEM is carved out of the combined L1/shared array, but because L1 is not modelled as a general cache (M11) the L1 side of that split has no effect here |
| A2 way mask | `a2_weight_ways` (None = shared) | weights fill only a/ways of the L2; every other class fills the rest (M7) |
| B two-class DRAM | `b_two_class` | decode-al requests get non-preemptive priority over bulk requests (M5) |
| C DXA depth | `dxa_depth` k (0 = LSU loads) | tile streams are fetched asynchronously with k tiles in flight per CTA (M11) and overlap compute (M3); needs `lmem_use` > 0 (DXA lands in LMEM) |
| D KV insertion | `kv_insert` ∈ {mru, lru, bypass}, `kv_insert_scope` ∈ {decode, all} | fills of the demoted KV class are inserted at MRU (normal), at LRU, or not allocated (M8); the phase-aware default demotes only KV read by decode, scope "all" is the phase-ignorant variant |

**"Stock" knobs** (`Knobs()`: A1 = 0, no A2, B off, C = 0, D = mru) mean, for linear and attention launches, a kernel
**without shared-memory (LMEM) tiling**: its tiles live implicitly in the L1. On Vortex that matches the plain-load
kernels. On GPU-like machines no vendor library ships such a kernel, so a GPU-like A1 gain (for example, h100-like
Llama-8B prefill: stock 2.26× its roofline bound, best knob vector 1.24×) is a gain **relative to that kernel**, not
relative to a tuned vendor GEMM. must read GPU-like A1 numbers that way.

### 9.2 `persist_across_launch`

When **False** (default for every `vortex-S1-*` machine), a reference can hit in L2 only if the previous reference to
the same byte was **in the same kernel launch**: tiling re-reads (M10) can hit, the first touch of every byte in a launch
is a miss, whatever the capacity. This is what SimX measures: every launch resets every cache (§2 consequence 4,
errata 12; confirmed, `results/<run>/coldcache.csv`). When **True** (default for `a100-like`/`h100-like`, and
persistent-L2 sensitivity runs on Vortex), first touches may also hit if the previous reference in an
earlier launch or step is within the capacity (M4). Test: with an infinite L2 and persistence off, DRAM reads equal
opgen's compulsory reads exactly and the cross-launch hit counter is 0.

### 9.3 Model structure and equations

Per launch (M1), opgen's compulsory read bytes are split into **streams**, each with a class (M9), an issue path
(LSU or DXA), a priority (al or bulk), compulsory bytes B, a tiling re-read factor r ≥ 1 (M10) and two reuse
windows: the cross-launch window X (distinct bytes, per class, inserted between this launch's first touch and the
previous reference in an earlier launch; M4) and the within-launch window Y of the re-reads (M10).

- **Hits** (M2): a reference hits with probability h given by its window F (summed over the classes that can
  displace it) and the capacity its class gets under the knobs (M7, M8). Default `hit_model = "cliff"`:
  `h = clamp((cap + w/2 − F)/w, 0, 1)`, a ramp centred on the capacity with width w = `cliff_ways` · C/ways (one L2
  way); `cliff_ways = 0` is the ideal step 1[F ≤ cap]. `hit_model = "proportional"`: h = min(1, cap/F).
- **DRAM read bytes of a stream:** `D = B·(1 − h_X) + (r − 1)·B·(1 − h_Y)`, with h_X = 0 unless persistence is on.
  L2 requests are B·r; the difference is L2 hits.
- **Bandwidth bounds:** `t_bw = (Σ D + W_timed) / (η·BW)` (BW = DRAM bytes/cycle of the machine, η = `eta_dram`; W_timed
  = timed write-back bytes, M6) and `t_l2 = (Σ B·r + writes) / BW_L2`.
- **Latency bound per issue path p (Little's law, M11):** a path with N_p outstanding request slots needs at least
  `t_p = [n_hit,p·L_hit + Σ_class n_miss,p,class·(L0 + W_class)] / N_p` cycles, where n are request counts (bytes/line),
  L_hit = L1 + L2 latency, L0 = L1 + L2 + DRAM latency and W_class the queueing wait (M5). Little's law says the
  number of requests in flight equals throughput × time each spends in the system; with N_p slots always full, each
  request occupies a slot for its latency, so the path finishes no earlier than total slot-time / N_p.
- **Queueing wait (M5),** per DRAM channel (n_ch channels, requests spread evenly): deterministic service
  `s = line·n_ch / (η·BW)` cycles; utilizations at memory time t: `ρ_crit = D_crit/(η·BW·t)`,
  `ρ_bulk = (D_bulk + W_timed)/(η·BW·t)`, ρ = ρ_crit + ρ_bulk. Mean residual work seen by an arrival (deterministic
  service: E[S²] = s²) `W0 = ρ·s/2`. B off (FIFO): `W = W0/(1 − ρ)` (Pollaczek–Khinchine for M/D/1). B on
  (non-preemptive priority): `W_crit = W0/(1 − ρ_crit)`, `W_bulk = W0/((1 − ρ_crit)(1 − ρ))` — a al request
  waits for the residual service plus the al requests already queued; a bulk one also for every al
  request that arrives while it waits. Both satisfy the work-conservation identity ρ_crit·W_crit + ρ_bulk·W_bulk =
  ρ·W_FIFO (tested), and the formulas match a 10⁵-request discrete simulation within 5 % at ρ ≤ 0.6 (tested; at
  ρ = 0.8 a 10⁵-request run has ~5 % sampling noise, 10⁶ requests agree within 1.5 %). Because the real system is
  closed (at most N outstanding requests), waits are capped: FIFO and bulk at `(N_tot/n_ch)·s` (a request cannot queue
  behind more than all other outstanding requests), al at `(N_crit/n_ch + 1)·s` (behind the al ones plus
  one bulk request in service).
- **Memory time:** `t_mem = min{ t ≥ max(t_bw, t_l2) : t ≥ t_p(t) for every path }`. W decreases with t (lower
  utilization), so t − max_p t_p(t) is increasing and the root is unique; it is found by bisection.
- **Launch time (M3):** `cycles = body + L0 + launch_overhead`, `body = max(t_comp, t_mem)`, except for LMEM-tiled
  launches without DXA (single-buffered: load tile, barrier, compute): `body = max(t_comp, t_mem) + min(t_comp, t_mem)
  / n_cta`. `t_comp = Σ_ops FLOPs / (peak FLOP/cycle of the op's unit · η_comp)` with the unit from
  `roofline.op_unit`. One unloaded miss latency L0 is added because the first request of a launch cannot be overlapped
  with anything (L_hit if the launch never reaches DRAM, 0 if it reads nothing). Reporting label
  (`LaunchResult.bound`): "compute" if t_comp ≥ t_mem; else "bandwidth"/"l2" if t_mem is within 5 % (`BOUND_TOL`) of
  max(t_bw, t_l2) (the fixed point always lands slightly above it once queueing is counted); else "latency".
- **Iteration cost** = Σ over launches (× the layer multiplicity, M13). `MemHier.__call__` returns seconds = cycles /
  clock for the scheduler (§10, whose time base is seconds).

### 9.4 Modeling choices and their expected failure modes

- **M1 Launch = cost unit.** Every op is one kernel launch, except the attention core of a layer (`attn_qk`,
  `attn_softmax`, `attn_av` of every sequence group), which is one launch (the fused mixed-batch kernel;
  `fused_mixed=False` gives one launch per group). *Fails* when the real runtime fuses more (e.g. norm into GEMM, QKV
  as one GEMM: fewer cold first touches on Vortex) or less.
- **M2 Steady-state LRU cliff (one-way ramp).** For LRU, a reference hits iff the distinct bytes touched since the
  previous reference to the same byte (its stack distance, including the reused data itself) fit the capacity; for a
  repeated stream of footprint F this is the cliff 1[F ≤ C]. The model uses that cliff as a ramp one L2 way wide
  **centred on the capacity**. A set-associative cache with pseudo-LRU replacement neither switches at exactly C nor
  holds a footprint precisely up to C (set conflicts, PLRU approximating LRU), so hits fall from 1 to 0 as F goes from
  C − ½·C/ways to C + ½·C/ways. Centring makes the ramp unbiased: its mean effective capacity is C (tested). The
  second-pass ramp ended at C and so lowered the capacity by about ½ way.
  **Power-of-two ties:** caches, model dimensions and tiles are all powers of two, so windows often equal a capacity
  exactly (the stock Llama-8B activation-panel window on Vortex was exactly 1,048,576 B = the L2, and
  one 64 B line less cost +53 % of prefill time). The continuous ramp removes every such knife edge; `tests/
  test_memhier.py::test_one_line_l2_change_robustness` checks that ±1 line changes no example cost by more than 1 %.
  *Fails* near F ≈ C, where the real replacement policy, associativity and address mapping decide the hit rate; the
  "proportional" option, `cliff_ways` and the earlier fit bound this. `results/<run>/sensitivity_*.csv` show how much the
  knob gains depend on it (D in mixed iterations most of all).
- **M3 Overlap.** Memory and compute overlap fully (warps interleave) except single-buffered LMEM tiles; there, a
  CTA alternates load and compute and only the other n_cta − 1 co-resident CTAs hide the shorter phase, hence the
  min/n_cta term. The memory time of a launch is one aggregate quantity (bytes, request counts, mean waits): there is
  no al path inside a launch and no load imbalance between its classes, so reordering requests between classes
  cannot shorten it (see M5 for B). *Fails* for small launches that cannot fill all cores (perfect load balance is
  assumed), for tail effects (partial waves), for intra-launch phase changes, and wherever one class's requests gate
  the launch's progress (decode attention waiting on its KV while bulk prefill traffic is in flight), which is the
  effect B targets. **Overlap factor ω (`Hier.overlap`):** the hidden share of the shorter phase is
  scaled by ω ∈ [0, 1]: `body = max(t_comp, t_mem) + (1 − ω)·min(t_comp, t_mem)` (fully overlapped launches) and
  `body = max + (1 − ω·(1 − 1/n_cta))·min` (single-buffered LMEM tiles). ω = 1 (the default) is exactly the rule above
  (the code takes the original expressions at ω = 1, so outputs are bit-identical); ω = 0 serializes compute
  and memory. ω is a calibration knob for kernels whose warps cannot hide one phase behind the other (too few warps,
  barriers between load and compute).
- **M4 Reuse windows.** (a) *Steady state across steps:* the iteration is assumed to follow an identical one, so a
  weight byte (or an old KV byte, or a norm vector) is re-referenced after one step; its window is the step's distinct
  bytes: all weights + all old KV read + all KV written + the activation buffers. Activation buffers are reused across
  layers (as in llama2.c's RunState runtime), so they count once: the largest layer's written activations
  plus the pre/post ops'. (b) *KV written this step* (the chunk's own keys, the decode token's own K/V) is re-read by
  attention; its window = everything inserted from the KV-write launch of that layer (all of that launch's bytes,
  not the ½ tail used for activations in (c): a conservative choice, since the KV write is spread over that launch)
  up to the reading launch.
  (c) *Activations* are produced by the previous launch; a given output byte is followed, on average, by the share
  `act_tail_frac` = ½ of the producer's other traffic (outputs are written uniformly while the producer streams), so
  the window is ½ × the previous launch's bytes + the activation read itself. (d) Embedding rows of new tokens have no
  earlier reference. *Fails* for the first iteration of a workload (cold, not steady), for consecutive iterations of
  different shape, for activations read long after they were produced (the residual stream: the window is
  underestimated) or by an op other than the next (gate/up share the norm output: overestimated), and when buffers are
  not reused across layers.
- **M5 DRAM queue = M/D/1 per channel, Poisson arrivals.** *Fails* because DRAM service is not deterministic (row
  hits vs conflicts, bank parallelism, read/write turnaround), arrivals are bursty and Ramulator's FR-FCFS reorders
  row hits. DRAM-model state carried between launches (§2 consequence 5, ≤ 0.7 % of cycles) is not modelled.
  **Structural limit for B:** when al and bulk misses share an issue path, its time is ∝ Σ_c n_c·(L0 + W_c), and
  by the work-conservation identity (9.3) priority leaves Σ_c ρ_c·W_c, hence that sum, unchanged. B therefore only
  moves waiting between classes. Its only model gains come from (a) reads served ahead of posted writes (which are in
  the bulk class), and (b) the outstanding-request caps and asymmetry between the LSU and DXA paths. With
  `dram_lat` scaled on stories15M mixed 256 + 15×256 (vortex-S1-8c), the gain is not monotone in DRAM latency: 1.010×
  at 10 cycles, 1.012× at 20, 1.018× at 40, 1.024× at 80, 1.006× at 160, 1.000× at 300 and 1000 (third-pass model). It peaks where launches sit at the
  latency/bandwidth crossover, and it vanishes when latency dominates the queueing wait. Tests:
  `test_b_structural_limit_conservation` and `test_b_gain_does_not_grow_with_latency`. **'s B on/off rows are
  therefore not evidence about B.** B is evaluated on SimX, which has per-request al paths.
- **M6 Writes.** Writes are posted (no latency bound) and bulk priority. On machines with a timed write-back path
  every written byte eventually costs DRAM write bandwidth (upper estimate: a buffer overwritten in place while dirty
  is never written back, so activation writes are overstated on GPU-like machines). With an untimed end-of-launch flush
  (`flush_untimed`, SimX: errata 12) only dirty lines evicted during the launch are timed. A line written at a uniform
  random point of the launch is evicted iff more than C bytes follow it, so the timed fraction is `(1 − C/F)⁺`, with
  F_launch = the launch's reads including re-reads, plus its writes. This holds whatever `hit_model` is. *Fails* for
  write-no-allocate L1 effects, partial-line (dirty-byte) writes and non-uniform write timing (e.g. all outputs at the
  end).
- **M7 A2 partitions.** With a way mask, weights live in C·a/ways and all other classes in C·(1 − a/ways); the
  partitions do not displace each other. *Fails* for low associativity (a 4-way cache cannot be split finely and a small
  partition degrades its replacement: measured) and because L1 masks are not modelled.
- **M8 D semantics.** A demoted line inserted at **LRU** is the next victim of its set: it survives only while fewer
  than one way's worth of other fills reach its set, so it hits iff its window fits C/ways; demoted fills hold about
  one way per set, so the other classes keep C − C/ways and do not count demoted bytes in their windows. **Bypass**:
  demoted fills never allocate (hit 0, displace nothing). A hit on a line does not depend on how it was filled, so the
  decode token's own K/V (written this step with normal insertion) is class `kv`, not demoted. *Fails* where SimX's
  MRU fill and the RTL's PLRU update differ (errata; `rtl_like_fill`) and where per-set imbalance matters.
- **M9 Classes and priorities.** `weight`; `act`; `kv` = KV read by prefill chunks (re-used across query tiles) and
  every KV write; `kv_stream` = old KV read by decode (touched once per step). Priority *al* = everything except
  prefill-chunk traffic inside a launch that also serves decode sequences (bulk) and writes (bulk). With B off,
  priorities only label traffic. *Fails* if a real kernel issues decode KV through DXA streams tagged bulk (tags
  the stream class; the model assumes the phase-aware tag).
- **M10 Tiling.** Tile buffer per CTA: `β = lmem_use/(k + 1)` (k = DXA depth: k tiles in flight + 1 being consumed)
  or, without LMEM, `β = l1_tile_frac · L1 / n_cta`. The second form means the n_cta co-resident CTAs' implicit tiles
  together use the share l1_tile_frac = ½ of the L1, sharing it with everything else (gave every CTA ½ of the
  L1, which overflowed it 2× on Vortex and 8× on h100-like and biased A1 down). GEMM: square output tiles of T×T with
  K-depth Tk = `gemm_tk` = 32 (assumption; the kernel fixes the real value):
  `T = min(2^⌊log2(β/(Tk·(q_a + q_w)))⌋, 2^⌊log2(√(acc_per_thread·warps_per_cta·threads_per_warp))⌋)`. The second term
  is the **accumulator cap**: the T² fp32 outputs must fit the CTA's registers. `acc_per_thread` is an assumption: 16
  on Vortex, half of the 32 RV32F floating-point registers, giving T ≤ 32; 128 on GPU-like machines, a 128×128 tile on
  a 4-warp CTA. Prefill attention tiles are capped the same way: Tq·g·d_v accumulators.
  A row-block-major loop re-reads every weight panel once per row block and every activation panel once per column
  block: `r_w = ⌈M_e/T⌉` (M_e = rows per matrix: N_tok, or N_tok·k/E[active experts] for MoE), `r_a = ⌈N/T⌉`.
  **Concurrency:** P = cores·n_cta output tiles are in flight at once, in row-block-major order, so they span
  min(P, cb) column blocks of ⌈P/cb⌉ row blocks (cb = ⌈N/T⌉, row blocks capped at r_w). With M_t = min(T, M_e) and
  N_t = min(T, N):
  - an activation panel's window is those row blocks' activation panels (M_t·K·q_a each), the in-flight weight panels
    (K·N_t·q_w each) and the output tiles written meanwhile (M_t·N_t·q_a each);
  - a weight panel is re-referenced only at the next row block, so its window is the whole matrix plus the in-flight
    row blocks' activation panels and outputs (M_t·N·q_a each).
  - **Lockstep and wave boundaries.** If the earlier reference is by a tile still in flight (activation panels when
    P > 1; weight panels when P > cb), the re-read happens while the in-flight tiles step through K together. Its
    window is then their current K-chunks, `n_fly·Tk·(M_t·q_a + N_t·q_w)` with n_fly = min(P, tiles). Tiles are
    issued in waves of P, so a share of these partners falls in the *next* wave and is not co-resident: 1/P of the
    activation re-reads and cb/P of the weight re-reads. Those use the serial windows above (tested). Lockstep K
    progress is an assumption; if CTAs drift apart, the co-resident windows grow toward the panel sizes.

  Prefill attention: query tiles of `Tq = 2^⌊log2(β/(g·d_k·q_a + e_KV))⌋` rows (Q of the g query heads sharing one KV
  head + that head's K and V per key, e_KV bytes); each tile streams the keys it attends, so `r_KV = Σ_tiles
  keys(tile) / keys(chunk)` (exact causal sum with opgen's key counting; non-causal ⌈n/Tq⌉). Work items are
  (sequence, KV head) pairs, each walking its query tiles; at most cores·n_cta items run at once. The window is that
  many items' KV and Q tiles, plus the decode KV streamed during one tile pass (decode KV of the launch / r_KV).
  Decode attention reads its KV once (GQA heads share the KV read; r = 1). *Fails* for rectangular or split-K tiles,
  swizzled tile orders, other register budgets (the cap changes A1's ceiling), MoE experts processed concurrently
  (not counted in the windows), and kernels that do not use LMEM at all (the A1 knob applies to every linear and
  attention launch of the iteration).
- **M11 Outstanding requests.** CTAs per core `n_cta = min(⌊warps/warps_per_cta⌋, ⌊LMEM/lmem_use⌋)` (warps_per_cta = 4,
  assumption); active warps `min(warps, n_cta·warps_per_cta)` with LMEM tiles, else all warps. LSU slots `N_lsu =
  cores·min(l1_mshr, active_warps·mlp_per_warp)` (mlp_per_warp = 2 lines, from the `VX_config.toml:116` comment "~2
  in-flight loads per warp"); DXA slots `N_dxa = cores·n_cta·k·⌈β/line⌉` (k tiles of β bytes in flight per CTA). Both
  paths draw on the same L2 MSHRs: if the active paths together exceed l2_banks·l2_mshr, both are scaled down in
  proportion (tested). *Fails* where MSHR merging, LSU queue limits, DXA serial multicast (R§5 risk) or L1 hits on
  fine-grained re-reads matter. L1 is modelled only as the capacity of the implicit tile, not as a cache of general
  reuse.
- **M12 Machine parameters.** Vortex (`vortex_hier`): sizes, MSHRs and latency formulas from `VX_config.toml`
  (line pointers in `Hier.sources`, literals re-read by the tests); L2 banks derived from the L2 request count (formula-verified for `VX_CFG_VM_ENABLE` =
  false and SOCKET_SIZE 1; with VM on, the page walker adds an L2 request slot; not measured on a build); DRAM latency = 50 ns (midpoint of the default DRAM's 14-cycle open-row and 26-cycle
  row-conflict idle reads at 400 MHz, `docs/simulation.md:203-205`; SimX times the arrays in ns) × clock + the fixed
  `PLATFORM_MEMORY_LATENCY` (0); L2 bandwidth = one 64 B sector per bank per cycle (assumption). GPU-like
  (`gpu_hier`): sizes from the profiles, the same latency formulas, the a100 profile's fixed DRAM latency (190 cycles;
  also used for h100, whose profile says latency is not calibrated; the resulting h100 unloaded miss latency
  of 289 cycles ≈ 146 ns looks low and is unverified, and it matters only for latency-bound launches), the whole combined L1/shared array usable as LMEM
  (assumption: real maximum carveouts are smaller), and MSHRs by a bandwidth-delay rule: outstanding lines = 2 × BW ×
  L0 / line, split over SMs and banks (GPUs are provisioned to saturate DRAM; the factor 2 is an assumption).
  **GPU-like effective L2 bandwidth: anchored to the roofline, an assumption, not a measurement.** The
  `l2_bytes_per_cycle` value is `GPU_L2_BW_ANCHOR` = 4 × (one 64 B sector per bank per cycle). The unanchored value
  (×1, with the profiles' 32 banks, a power-of-two DRAM-interleave choice copied from a100, not the L2 slice count)
  left Llama-3.1-8B prefill at S = 512 L2-bound at ≥ 4.35× its roofline bound for every knob vector on h100-like,
  while vendor GEMMs of that size run near compute-bound. The anchor is the smallest integer for which the best
  `knob_grid` vector is within 1.5× of max(FLOP bound, DRAM bound) on both GPU-like machines: 1.24× on h100-like and
  1.10× on a100-like (×3 gives 1.56× on h100). It is an *effective* bandwidth: it also stands for the tile-reuse
  mechanisms the model lacks (larger rectangular warpgroup tiles, cluster multicast, L1 hits on tile loads).
  - **Why this parameter and not the tile shape** (the alternative, 8-warp CTAs with larger tiles): the L2 bandwidth
    has no source at all. CTA and tile shape are what A1 and C vary, so bending them to meet the anchor would bias
    the knobs under study. 8-warp CTAs also do not meet the anchor alone (best 4.34× on h100), nor with 2× L2
    bandwidth (2.25×).
  - `test_gpu_prefill_roofline_anchor` enforces the anchor on both machines. The unanchored value is a §9.6
    sensitivity row. Vortex is not anchored: it has no ground truth.

  η_dram =
  η_comp = 1 and launch_overhead = 0 until measured/fitted. *Fails* to describe either machine
  quantitatively until then: every number marked unmeasured or assumption is a placeholder.
- **M13 Representative layers.** Layers of one opgen layer class have identical ops, so one layer per class is costed
  and multiplied (`use_layer_classes`); exact (tested against all layers, persistence on and off).
- **M14 Scheduler glue.** `iteration_seqs` maps a scheduler iteration to opgen groups: a prefill chunk [s0, s0+n) →
  `Seq("prefill", n, ctx=s0, emits=last)` (non-final chunks emit no logits, which opgen's `mixed()` cannot express);
  a decode at context T → `Seq("decode", 1, ctx=T−1)`. B decodes with Σ T are collapsed into at most two groups at
  ⌊ΣT/B⌋ and ⌊ΣT/B⌋+1 with the same B and ΣT: exact for every quantity linear in T (KV bytes, attention FLOPs), i.e.
  for all of them unless a layer has a sliding window, in which case decodes are grouped exactly. uses
  `B_max ≤ c` in chunked mode (§10.2). *Fails* only in what the scheduler itself does not model
  (§10.2: prefix caching, scheduling overheads); the mapping is exact.

**Not modelled:** L1 as a general cache (bank conflicts, hit-under-miss), LMEM bank conflicts, TLBs, the instruction
cache, NVIDIA-practice persisting windows, the L2-prefetch-only DXA mode (iii), MSHR class merging (§12),
switching costs (§13), DRAM state across launches, energy.

### 9.5 Calibration targets

The calibration fits, on 70 % of `table_v0` (single-launch cold-cache rows, so `persist_across_launch=False`): η_dram
(`eta_dram`), η_comp (`eta_comp`), the latencies (`l1_lat`, `l2_lat`, `dram_lat`), `launch_overhead`, `mlp_per_warp` and
`act_tail_frac`, and chooses `hit_model` (cliff vs proportional), reporting MAPE on the remaining 30 %. The structural
parameters (`l1_mshr`, `l2_mshr`, `l2_banks`, `warps_per_cta`, `gemm_tk`, `l1_tile_frac`, `flush_untimed`) are set from
the configuration and the P2/P3 kernels rather than fitted, as are `threads_per_warp` and `acc_per_thread` (register
budget GEMM). `cliff_ways` (cliff ramp width) is fitted together with the hit-model choice.
What actually fitted, and why it differs from this list (η_comp per unit, latencies as two scales, ω added,
`act_tail_frac` not identifiable from cold single launches): §9.8.

### 9.6 Sensitivity list for the analytical oracle sweep

The model is uncalibrated, so G1's gaps must be shown against the parameters that move the knob gains most
(`results/<run>/sensitivity_params.csv` measures each on the example scenarios). Run G1 at the default and at each of:
- `hit_model` cliff vs proportional, and `cliff_ways` 0 / 1 / 2;
- L2 size ±1 line and ×½ / ×2 (exact power-of-two ties, M2);
- the stock implicit tile `l1_tile_frac` ¼ / ½ / 1 (M10, it sets the baseline A1 is compared with);
- the accumulator cap `acc_per_thread` ×½ / ×2 (M10, it caps A1's gain);
- `warps_per_cta` 2 / 4 / 8 (M11, A1 occupancy);
- `dram_lat` ×½ / ×4, including the h100-like latency, which is unverified (M12);
- `l2_bytes_per_cycle` ×½ / ×2 (M12 assumption), and on GPU-like machines the unanchored L2 bandwidth
  (÷ `GPU_L2_BW_ANCHOR`): stock/best for h100-like Llama-8B prefill S = 512 is 1.82× anchored and 1.97× unanchored,
  but the best vector is 1.24× vs 4.35× its roofline bound.

**Mandatory baseline rules for the analytical oracle sweep.** The sweep must
also repeat both rules in `results/<run>/README.md` and ``.
1. **Stock is not a baseline.** On GPU-like machines "stock" is an untiled kernel (§9.1). One tiled configuration is
   best on every scenario (A1 = 64 KiB with C = 1 on h100-like, A1 = 32 KiB with C = 1 on a100-like), so the
   stock/best gaps (prefill 1.82×, mixed 1.27–1.59×) measure untiled vs tiled. They are **not** phase-adaptation
   evidence. G1 compares adaptive and oracle costs only against the per-workload best static and the per-kernel best
   static (§5), never against stock, and no stock/best ratio may be reported as a phase-adaptation gain.
2. **Near-tie breaking.** Decode cost is nearly flat in the knobs (≤ 0.2 %). So the decode-tuned static's argmin can
   land on stock and show a fake regret: 1.815× on prefill and 1.27× on mixed on h100-like. In every phase-tuned
   static baseline and every argmin over configurations, configurations within 0.5 % of the minimum are tied, and ties
   go to the **global best static**, never toward stock.

Effective-rate note (M12): the anchored GPU-like L2 rate is an effective ≈16.2 TB/s on h100-like and ≈11.6 TB/s on
a100-like, applied to all L2 traffic; it stands in for tile reuse the model lacks, not for a measured L2 bandwidth.

B on/off rows are not evidence about B (M5). Per-launch costs for the per-kernel best static (T_kernel) need
`cost_seqs(..., detail=True)`, which bypasses the memo. Measured machine parameters (§11) replace
the `vortex_hier` placeholders first. If MAPE > 25 %, this section marks the model "qualitative only" and the earlier numbers
are not used for any gate. **Outcome (§9.8): MAPE 70.6 % — the model is marked "qualitative only";
the earlier numbers are not used for any gate.**

### 9.7 Oracle-sweep choices

Code `model/oracle_sweep.py`, configuration `experiments/<run>.yaml`, results `results/<run>/`, decision
``. Choices the plan leaves open, each with its reason:

- **Setups.** vortex-S1-8c runs stories15M in fp32 (the model the Vortex kernels run); h100-like runs Llama-3.1-8B in
  bf16. h100-like is the σ reference: σ = (M_ref/L2_ref)/(M_tgt/L2_tgt) = (131,072/32 MiB)/(13,824/1 MiB) = 0.296
  (§10.4). stories15M's 256-token context caps the scaled requests: many are clipped (all of the Mooncake ones), so
  the requested median KV/L2 ratio is not reached on Vortex. Every workload records its clipped share and both ratios.
- **Configuration space.** `knob_grid` with A1 ∈ {0, ¼, ½, 1} × LMEM, C ∈ {0, 1, 2, 4} (C > 0 only with A1 > 0), D ∈
  {mru, lru, bypass}, B ∈ {off, on}: 78 vectors. A2 is off (it is not one knobs).
- **Intervals = launches, so DP = per-kernel static = zero-cost.** The model has no structure below a launch (M3: a
  launch's memory time is one aggregate). Every sub-launch work unit therefore prefers the same configuration as its
  launch, and an intra-launch switch can never pay. Interval i is one launch, K(i) = i, and C_sw,i = 0 everywhere, so
  `T_zero = T_DP = T_kernel` exactly, for any C_sw (§5; invariance property). The analytical DP gap over the
  per-kernel best static is 0 **by construction**: cannot inform G2b, which needs SimX's intra-launch
  intervals. G1 uses the gap over the **per-workload** best static. Two coarser oracles are also reported, as
  diagnostics only. The *iteration-level* oracle is one configuration per serving iteration, which is phase adaptation
  proper. The *per-op-type static* is one configuration per op name for the whole workload, which is what a compiler
  could set per kernel. **Both diagnostics are degenerate by construction on a single-iteration workload.** There the
  iteration-level oracle equals the per-workload static (one iteration, one choice). The per-op-type static equals
  the DP, because the layers of a class are identical, so every launch of an op name has the same cost row. Only
  multi-iteration (trace) workloads can separate iteration-level adaptation from per-launch choices. therefore
  reports both diagnostics on the traces separately (`results/<run>/tables.md`, "Trace workloads only"), and those
  traces do not rule out iteration-level adaptation: it captures 32–46 % of the h100 Mooncake gaps.
- **Composition.** A launch's cost under c is computed with the whole iteration in c. With persistence on (h100-like),
  a launch's cross-launch hits depend on what earlier launches inserted, so mixing configurations within an iteration
  is approximate (§5 composition assumption). With persistence off (Vortex), launches are independent and composition
  is exact.
- **Compression.** Each (iteration, op, layer class) is one row: its weight (the number of iterations it stands for)
  × its layer multiplicity × its cost. Every static baseline and oracle is a sum, over intervals, of T(i, c) for one c
  or of min_c T(i, c), so this is exact (tested against `model/oracle.py` on expanded interval lists).
- **Stratified sampling of trace iterations.** Each trace workload is evaluated on a stratified random sample: strata
  are the four regions, with at most 100 **exact** iterations per stratum (`iteration_seqs`, M14), each weighted
  N_h/n_h. That weighting (Horvitz-Thompson) gives unbiased estimates of every sum over iterations. A percentile
  bootstrap that resamples within strata gives each trace workload's 95 % sampling CI of its gap. Two properties of
  these estimators:
  - **Bias toward STOP.** The sampled per-workload static T̂_static = min_c Σ̂_i T(i, c) is a minimum of noisy
    unbiased sums, so it is biased low (E[min] ≤ min E). The oracle sum Σ̂_i min_c T(i, c) is unbiased. Sampled gaps
    are therefore biased slightly low, i.e. toward STOP.
  - **Conservative CIs.** `bootstrap_gap` also resamples strata that were taken exhaustively (N_h ≤ 100, e.g. a
    prefill stratum of 4 iterations). Those strata have no sampling error, so the CIs are slightly wider than
    necessary.
  - Bucketing iterations was tried first and **rejected**: rounding batch size, mean context, chunk length and offset
    to 8 points per octave. Near the L2 cliff (h100-like mixed iterations with long prefixes), ±4 % rounding changed
    single-iteration costs by up to 79 %. It moved a validation sample's gap from 1.5 % to 2.6 %
    (`results/<run>/bucketing_rejected_validation.csv`).
  - Buckets are used only for the scheduler's reference cost, which sets iteration timing. They are never used for an
    evaluated cost.
- **Workloads.** One workload is one single-iteration point of the synthetic grids, costed exactly: prefill B×S,
  decode B×T, and mixed (chunk c − B prefill tokens + B decodes). Trace workloads are the first 1,000 arrivals of
  each trace, unscaled in rate. They run in full-prefill mode and in chunked mode with c ∈ {256, 512, 2048}
  (× σ on Vortex), B_max ≤ c (§10.2). Each schedule is generated once with a reference configuration: the
  synthetic-grid global best static (fixed-schedule assumption, §10.3). The schedules stay fixed across the §9.6
  variations, so a variation changes only the costs.
- **Region membership** (§4, majority of the time). There is no PS-BASE configuration, and stock is not a
  meaningful configuration on GPU-like machines (§9.6 rule 1), so "baseline time" is the time under the machine's
  global best static. Shares are recorded per workload.
- **Region gap** = geomean over the region's workloads of T_static/T_DP, minus 1 (§5). A region-restricted
  diagnostic is also reported (each workload's iterations of one region, treated as a sub-workload).
- **§9.6 rule 2 (tie rule)** applies to every argmin: global best static (never toward stock), per-workload static,
  per-launch and per-iteration choices, phase-tuned statics. Each choice moves a cost by ≤ 0.5 %. G1 is evaluated with
  the rule (primary) and with plain argmins; both outcomes are reported.
- **B.** B on/off rows are not evidence about B (M5). G1 is also evaluated with B fixed off, and an outcome that
  depends on B is flagged.

### 9.8 Calibration against SimX

Code `model/calibrate.py` (CLI writes `results/<run>/`), tests `tests/test_calibrate.py`, numbers and file list
`results/<run>/README.md`. Model-only: no SimX run was made; the data are 's.

**Data and rows (modeling choice).** `results/<run>/table_v0.csv`, machine PS-BASE-S1-4c, one cold-cache launch per row, so
memhier runs with `persist_across_launch=False` and every row is exactly one memhier launch (`MemHier.cost_launch`, added in
the `cost_seqs` path with empty cross-launch windows; tested equal to `cost_seqs` summed over an iteration's
launches). Used: all **306 `how = run` rows** of the five op families (decode attention 90, prefill attention 90, mixed
attention 21, linear 45 = GEMV 15 + TCU GEMM 30, element-wise 60), i.e. the base cell and every stock-knob setting that changes
the binary or its arguments. Excluded: the 132 `same_as_base` rows (their value *is* the base row's; counting them would
weight a cell by how many knobs leave it untouched) and the 9 `naive` decode rows (a kernel that re-reads K/V once per
query head; opgen's GQA op has no such structure). Element-wise rows are **kept** (fixed in the code before the first fit: the first fit, `build/tmp/p34_quick`,
05:26 UTC, already used all 306 rows; there is no earlier written record): they are real Vortex launch costs that
memhier must predict when it costs an iteration (§9.3), even though their musl soft-float code (§14.0, errata 32) is outside
the model.

**Row → launch mapping (modeling choices).** opgen ops of the `llama8b_s8` layer (d = 512, H_q = 8, H_kv = 2, D = 64,
d_ff = 1792): decode/prefill/mixed attention = the fused `attn_qk/softmax/av` launch of `decode(B, T)`, `prefill(B, S)`,
`mixed([(c, s0)], [(n, T)])`; GEMV / TCU GEMM = `opgen.linear` of the run's (M, N, K) (the M = 16 TCU cell ran as M = 32
and is costed so); element-wise = opgen's `attn_norm`, `attn_residual`, `silu_mul`, `rope_kv_write`. Dtypes as the kernels:
fp32 everywhere except the TCU GEMM's fp16 inputs. **Compute unit as the kernel**: the rule `op_unit` puts every
linear/attention op on the TCU; the SimX attention and GEMV kernels are scalar FPU code, so they are costed on a machine
without a TCU entry. Knobs: `lmem_use` = the launch's measured `lmem_bytes`, `dxa_depth` = the setting's depth; the TCU GEMM
is a single-buffered 3,072 B LMEM tile (stock app, DXA_DOUBLE_BUFFER off; its log: CTA tile 32 × 64 × K 16, 2 warps per
CTA), so `gemm_tk` = 16 and `warps_per_cta` = 2 there; every FPU kernel launches CTAs of all NT·NW threads of a core
(`warps_per_cta` = 8; §14.6 for attention, an assumption for GEMV/element-wise, where it only sizes the implicit tile). L2
knobs set `l2_bytes` (½, ¼) and `l2_ways` (16). **FIFO has no model counterpart** (memhier has no replacement policy) and
is predicted equal to base; 16 ways only widens the cliff ramp.

**Fitted parameters θ (physical meaning, bounds).** Positive scale parameters are fitted in log space.
| parameter | meaning | bounds and why |
|---|---|---|
| `eta_fpu` | achieved / measured FPU peak (56.1 FLOP/cycle, 8-chain FMA loop, §11.3) | [10⁻³, 4.56]: 4.56 = structural 256 / 56.1 |
| `eta_tcu` | achieved / measured TCU peak (155.9 FLOP/cycle) | [10⁻³, 26.3]: structural 4,096 / 155.9 |
| `eta_dram` | achievable / measured LSU-stream bandwidth (40.7 B/cycle) | [0.02, 1.57]: 1.57 = channel peak 64 / 40.7 |
| `overlap` ω | hidden share of the shorter phase (M3) | [0, 1] by definition |
| `lat_miss_scale` | × the DRAM part of L0 (L0 − L2 hit = 51.7 cycles, measured unloaded) | [1, 4]: an effective latency cannot be below the measured unloaded one; the earlier loaded stream latency is ≤ 2.87 × L0 (§11.4) |
| `lat_hit_scale` | × the L1 + L2 hit latencies (13.5 + 14.7 cycles) | [1, 4], as above |
| `mlp_per_warp` | effective outstanding lines per warp on the LSU path (M11) | [0.05, 2]: above 2 the 16 L1 MSHRs per core bind at 8 warps, so larger values change nothing |
| `launch_overhead` | cycles added to every launch | [0, 836]: 836 cycles is the shortest measured launch (residual B = 1 at L2 ½/¼), which no per-launch overhead can exceed |
| `hit_model` | cliff (ramp 0 / 1 / 2 ways) or proportional | discrete outer loop; the default (cliff, 1 way) is kept unless another choice lowers the train loss by > 0.1 % |

Why this set differs from §9.5's list: η_comp is split **per unit** (FPU, TCU) because the measured peaks are per unit and the
two kinds of kernels share nothing (a scalar η_comp would be pinned by whichever unit dominates the loss); this is expressed
by scaling the roofline machine's per-unit FLOP rates, memhier's own `eta_comp` stays 1. The three latencies enter memhier
only through L_hit = l1 + l2 and L0 = l1 + l2 + dram (§9.3), so only two latency combinations are identifiable: they are
fitted as two scales of the measured machine parameters. ω is new (M3). `act_tail_frac` (M4c) only shapes cross-launch windows,
which do not exist with persistence off: cold single launches carry **no information about it**, so it stays 0.5. 's
§11.4 warning is respected: `mlp_per_warp` is capped at the MSHR limit and nothing assumes extra LSU depth raises bandwidth.
The bounds were first set wider (latency scales [0.25, 8], overhead [0, 20,000]); that development fit put the latency
scales at 0.25 (effective latencies below the measured unloaded ones) and the overhead at 11,508 cycles (13 × the shortest
launch), both unphysical, so the bounds were tightened to the physical limits above before the final fit. The
development numbers are kept in `results/<run>/dev_wide_bounds/` (primary test MAPE 62.8 %, same verdict).

**Loss, split, optimizer.** Loss = mean over train rows of r_i², r_i = ln(pred_i / meas_i): symmetric in over- and
under-prediction, scale-free (a 900-cycle element-wise launch and a 3.7 M-cycle prefill weigh the same), the standard
choice for multiplicative errors. Rows are unweighted. **Split (rule: fixed by seed, recorded):** by grid *cell*,
not by row — all settings of a cell are on one side — stratified per family: within each family the sorted cell list is
permuted with `numpy.random.default_rng(3404)` and the first round(0.7·n) cells are train (`SPLIT_SEED`, `TRAIN_FRAC`;
the assignment is written to `results/<run>/split.csv` and a test asserts it). 204 train / 102 test rows (decode 60/30,
prefill 60/30, mixed 14/7, GEMV 10/5, TCU 20/10, element-wise 40/20). A row split would let the fit see every shape and test
only on knob values of seen shapes; the cell split tests generalisation to unseen shapes. Optimizer:
`scipy.optimize.least_squares` (trust-region reflective, bounds, finite-difference Jacobian with step 10⁻⁴ in the
transformed space) from 16 starts — the uncalibrated defaults (clipped into the box) and 15 points drawn uniformly in the
transformed box with `default_rng(0)` — per hit-model choice; the best end point wins. The model is piecewise smooth
(max/min, ramps), hence the multi-start. **Reported metric:** MAPE = mean |pred/meas − 1| on the 30 % test rows, with
median, 90th percentile and maximum APE, per family, phase and setting (`mape.csv`). **Identifiability** (`identifiability.csv`):
parameters on a bound are reported as such; for the interior ones the Gauss-Newton covariance s²(JᵀJ)⁻¹ from the
residual Jacobian gives relative standard errors and correlations ("identified" = interior and relative s.e. < 25 %). For parameters on a bound, scipy's returned Jacobian column can be
degenerate: the `overlap` column norm is listed as 0, while a one-sided difference into the box gives 2.78 (gradient of
½Σr² +9.8, pushing ω to 0); the parameter is correctly flagged on-bound and no conclusion depends on the value.
**TEST** (`tests/test_calibrate.py`): on synthetic rows generated by the model at a known interior θ* (the real 306
cases, their real split), the fit recovers every parameter within 2 % and predicts the held-out rows within 0.001 %; with
5 % log-normal noise the hit-model loop picks the generating hit model and every parameter the identifiability report calls
well determined is recovered.

**Results (primary = global θ, all families; `results/<run>/fit.csv`, `mape.csv`).** Fitted: `eta_fpu` 0.0717 (4.0
FLOP/cycle effective), `eta_tcu` 26.3 (**upper bound**: TCU compute never binds), `eta_dram` 0.0760 (3.1 B/cycle
effective), ω = 0 (**bound**: fully serial), `lat_miss_scale` 1.99, `lat_hit_scale` 1.00 (**bound**), `mlp_per_warp` 2
(**bound**), `launch_overhead` 836 (**bound**), hit model proportional (train loss 1.366 vs 1.386 for the cliffs). Five of
eight continuous parameters sit on bounds (`lat_hit_scale` 0.03 % above its bound): the structure cannot reach the data
inside the physical box. Of the interior parameters only `eta_fpu` (relative s.e. 16 %) and `eta_dram` (19 %) are
identified; the two latency scales trade off against each other (correlation −0.998) and are not. Held-out **MAPE
70.6 %** (102 rows; median 32 %, p90 108 %, max 541 %; train 82.2 %); uncalibrated (values, η = ω = 1, MLP 2, no
overhead) 84.2 %. Per family (test): decode attention 24.7 %, prefill attention 27.6 %, mixed attention 36.0 %, GEMV
**535 %**, TCU GEMM 74.3 %, element-wise 98.0 % (the GEMV and mixed test sets are **one cell each** — M1_w2, 5 rows;
c32_s32_n8_T512, 7 rows — so those two per-family numbers come from a single shape); per phase: prefill 30.1 %, decode 98.2 %, mixed 36.0 %. The global fit
buys element-wise and TCU-GEMM time with an effective DRAM bandwidth of 3.1 B/cycle, which contradicts the measured
throughput of the GEMV itself (13–14 B/cycle), so the GEMV is predicted 6× too slow: **`eta_dram` is a compensating
artefact, not a bandwidth.** Diagnostic variants (not gate-eligible; they add structure did not have or drop data):
- **no element-wise rows** (global θ fitted and tested without them): test MAPE **28.9 %** (decode attention 26.9, prefill
  29.2, mixed 20.3, GEMV 94.4, TCU 7.2 %) — still above 25 %;
- **one compute efficiency per op family** (6 η's: decode attention 0.042, prefill 0.072, mixed 0.074, GEMV 0.118, TCU 1.21,
  element-wise 0.0037; other parameters global): test MAPE 153 % overall because one η cannot cover rope (soft-float sin/cos,
  0.3 % of predicted speed) and residual alike (element-wise 713 %), but **16.7 %** on the non-element-wise rows (decode
  attention 26.3, prefill 16.1, mixed 8.9, GEMV 5.8, TCU 1.0 %). **This variant does not exercise memhier's memory
  structure** (re-verified): every memory-side parameter sits at its fastest bound or is
  unidentified (`eta_dram` 1.572 = upper bound with relative s.e. ≈ 31,300 %, both latency scales 1 = lower bound,
  `launch_overhead` ≈ 0, ω = 1), and **all 246 non-element-wise rows come out compute-bound**, with t_comp / cycles
  0.97–1.00 for decode, prefill and GEMV, 0.96–0.98 for mixed and 0.95–0.97 for the TCU GEMM. The 16.7 % is therefore a
  **FLOPs ÷ (achieved rate per kernel family)** model: one achieved FLOP rate per family (4–12 % of the measured FPU
  peak for the scalar attention and GEMV kernels) explains those launch times to 16.7 %, and the byte and latency
  accounting contributes ≤ 5 % of any prediction and is not tested by it. That is also why this variant predicts every
  cache knob flat. No variant validated memhier's byte/latency structure; the earlier measured attention traffic is 1.8–5.6×
  the compulsory bytes the structure assumes (errata 32).
**Split robustness** (`results/<run>/seed_robustness.csv`): refitting the primary structure
(proportional hit, 4 starts) on split seeds 1, 2, 3 gives test MAPE 144 / 102 / 102 % (non-element-wise 106 / 83 / 83 %)
vs 70.6 % for seed 3404, and moves both latency scales to their upper bound 4 with `eta_dram` 0.040–0.053. The fitted
values are therefore split-specific (`calibrated_psbase()` is not a stable parameter set), and seed 3404 is the most
favourable of the four splits tried: the verdict is robust, the primary seed is unchanged.
Failure modes visible in the residuals: decode attention at B = 1 is under-predicted (×0.62–0.66 at the base setting; 0.44–0.70 over all settings) and at B ≥ 2 over-predicted
(×1.15–1.29), because B·H_kv = 2 CTAs leave 2 of 4 cores idle at B = 1 while the model assumes perfect load balance (M3); the
soft-float element-wise launches are slower than predicted by 9.5–11.5× (silu_gate) and 99–275× (rope), while residual is over-predicted
2.7–4.8× (2.73–4.77×) by the fitted DRAM rate (§14.0; errata 32's instruction-fetch traffic is not in opgen's bytes); the attention kernels' 1.8–5.6× conflict-miss traffic (errata 32) is outside the
compulsory-traffic model.

**Knob rankings (`results/<run>/knob_ranking.csv`; the earlier tension metric, §11.5, recomputed on predicted cycles; the same
code reproduces the earlier measured R to 10⁻⁵, tested).** The calibrated model does **not** reproduce the earlier stock-knob
effects: it predicts no effect of 16 ways, FIFO or the LMEM budget (measured: decode −4.4 %, ±0.5 %, prefill +22 % at
16 KiB); the wrong sign for L2 size in prefill (+14.9 % / +26.2 % at ½ / ¼ vs measured −0.5 / −0.8 %) and for DXA in
prefill (+1.5 … +4.1 % vs measured −3.3 % at d2). Consequently the only measured per-type bests it reproduces are decode's DXA d2 and decode's L2 size 1× (where the
model predicts R = 1.000 / 1.002 / 1.006 for 1× / ½ / ¼); elsewhere
it picks another value (LMEM 64 KiB instead of 32 KiB, L2 1× instead of ¼ for prefill, d0 instead of d2 for prefill) or
sees no effect (ways, replacement), and it predicts a **spurious material DXA
tension** (prefill best d0, decode best d2, cross loss 3.2 %) where measured none. Spearman correlation of predicted
vs measured phase-level R: 0.217 over the 22 informative (phase, setting) pairs (0.20 over all 30, which include 8 trivial
pairs where both sides are 1 by construction: the three base rows, decode LMEM 32/16 KiB and mixed DXA d1/d2/d4); of per-cell knob effects (264 cells × settings): 0.17;
sign agreement of per-cell effects 30 %. The per-family-η variant predicts no material tension, but only because it
predicts every cache knob as nearly flat (|R − 1| ≤ 0.35 %), not because it ranks the knobs correctly. **'s null
phase-tension result is a SimX measurement; the analytical model neither confirms nor refutes it.**

**Verdict and consequences.** Held-out MAPE 70.6 % > 25 % ⇒ **the analytical model is "qualitative only"**.
the earlier numbers (G1) are not used for any gate; G2/G2b rest on SimX costs (P6). Downstream use of memhier is limited to
(i) byte-class accounting as a qualitative structural aid (which byte classes a mechanism moves), itself **not validated**
by any variant here (measured attention traffic is 1.8–5.6× compulsory); (ii) **compute-only** order-of-magnitude
estimates (FLOPs ÷ the per-family achieved rates of the diagnostic variant), labelled as such, which say nothing about
memory mechanisms or knobs; and (iii) nothing that needs knob rankings: on this machine the model ranks stock knobs no
better than chance. A calibrated parameter set is available
(split-specific, see above) as `calibrate.calibrated_psbase()` (separate from memhier's and roofline's defaults, which the validated results
use and which are unchanged; `tests/test_calibrate.py` asserts both).

## 10. Iteration scheduler, fixed-schedule assumption and trace length scaling σ

Code: `serving/traces.py` (loaders, statistics, σ), `serving/scheduler.py` (scheduler); tests
`tests/test_scheduler.py`; trace statistics `results/<run>/trace_stats.csv`. This section is the specification the
code implements; item 9 names the fixed-schedule assumption and σ as explicit modeling choices.

### 10.1 Traces

- **Azure LLM inference 2023** [cand:D19] (`code`, `conv`): one row per request, `TIMESTAMP,ContextTokens,
  GeneratedTokens`. S = ContextTokens, O = GeneratedTokens. TIMESTAMP has 7 fractional digits; it is parsed to integer
  100 ns ticks (read as UTC, so no local-time or DST effect) and the first arrival is subtracted in integers before
  converting to seconds, so relative arrivals are exact. The files have 8,819 (`code`) and 19,366 (`conv`) data rows
  (no trailing newline, so `wc -l` shows the same numbers including the header line).
- **Mooncake FAST'25** [cand:D20] (`conversation`, `toolagent`, `synthetic`): JSONL with `timestamp` = relative
  arrival in **integer milliseconds** and `hash_ids` = prefix-block hashes of **512-token** blocks (both from the
  release's `FAST25-release/README.md` at the pinned commit, read 2026-10-31; the block size also holds on the data:
  `len(hash_ids) = ⌈input_length/512⌉` on every row). S = input_length, O = output_length.
- Requests are sorted by arrival (stable, so equal timestamps keep file order) and numbered 0…n−1. All five files
  have S ≥ 1 and O ≥ 1 (loaders reject anything else).
- Statistics: percentiles use linear interpolation between order statistics (NumPy's default, identical to pandas'
  `quantile`); `mean_rate_rps = (n − 1) / (last arrival − first arrival)`.

### 10.2 Iteration scheduler (continuous batching)

An **iteration** is one forward step of the model over a batch; it is the unit the cost function and the oracle's
work intervals (§5) are built from. The scheduler follows vLLM's iteration-level scheduling with paged KV blocks
[cand:A6]; the chunked mode follows Sarathi-Serve's stall-free batching [cand:A5].

- **State.** Requests that arrived by the start time t of an iteration join a FCFS *waiting* queue. *Running* =
  admitted and not finished, ordered by admission. At most **B_max** requests are running.
- **Token accounting.** A prefill chunk covers prompt positions [s0, s0+n): those queries attend to [0, s0+n) and
  their K/V are stored. The iteration that completes a prompt also emits the request's first output token (the
  logits of the last prompt position). A decode processes the most recent output token: it attends over
  **T = T_stored + 1** entries (cache plus itself), stores that token's K/V and emits the next token. Hence a request
  with S prompt and O output tokens takes one (possibly chunked) prefill and exactly O − 1 decode steps, emits O
  tokens, and stores at most S + O − 1 KV entries (the last token's K/V is never computed).
- **KV capacity (E3 with paging).** A request storing T_i entries holds ⌈T_i/b⌉ blocks of b tokens. The cache has N_blk
  blocks; the invariant `Σ_running ⌈T_i/b⌉ ≤ N_blk`, i.e. `Σ_i ⌈T_i/b⌉·b ≤ N_blk·b` token slots, is asserted after
  every allocation and re-checked by an independent replay in the tests. Rounding each request up to whole blocks is
  what paging costs in capacity; b and N_blk are scheduler parameters (`kv_blocks_for_bytes` converts a KV byte budget
  with E3's per-token size).
- **Full-prefill mode** (`chunk_size=None`, prefill-prioritising like vLLM's original scheduler): if any waiting
  request can be admitted (B_max, KV blocks for its whole prompt, and Σ admitted prompt tokens ≤ the token budget),
  the iteration prefills the admitted prompts whole and runs no decodes; otherwise it decodes one token for every
  running request. Iterations are pure prefill or pure decode. Modeling choice: a prompt longer than the token budget
  is admitted alone rather than rejected (vLLM rejects such prompts; we keep every trace request).
- **Chunked mode** (`chunk_size=c`): each iteration has a token budget c (prefill tokens + decode tokens ≤ c). Running
  requests are served first, oldest admission first: a decode costs 1 token, a partially prefilled prompt gets its
  next chunk of min(remaining prompt, remaining budget). Then the remaining budget admits waiting requests FCFS with
  prompt chunks (stopping at the first request that does not fit: head-of-line FCFS). Modeling choice: if the budget is
  exhausted before every running request is served (only possible when more than c requests run), the rest skip this
  iteration; experiments therefore use B_max ≤ c.
- **Reservation and preemption.** `ondemand` (vLLM default): blocks are allocated when tokens are stored; admission
  needs the blocks of the scheduled chunk; when a running request needs a block and none is free, the most recently
  admitted running request is preempted by *recompute*: its blocks are freed, it returns to the front of the waiting
  queue, and when re-admitted it prefills prompt + generated tokens (S + g), which emits its next token, so its output
  count is unchanged. No waiting request is admitted in an iteration that preempted (as in vLLM; otherwise the victim
  would be re-admitted at once). `full`: admission requires, and reserves, ⌈(S+O−1)/b⌉ blocks for the whole footprint,
  so preemption never happens; it uses the trace's output length as an oracle and serves as a bound, not a real
  policy. No watermark of spare blocks is kept (vLLM keeps ~1 %); the effect is earlier preemption under pressure.
- **Time.** The iteration's duration is `cost(iteration, c)` from a pluggable cost function for hardware
  configuration c (≥ 0, finite); t advances by it. With nothing running or waiting, t jumps to the next arrival.
- **Region of an iteration** (§4): prefill / mixed / decode by token types; a decode iteration is KV-bound iff
  ρ_KV = Σ_i M_KV/token · T_i / (q_w·N_params) ≥ 1 with T_i the attended contexts (`iteration_region`).
- **Not modelled (expected failure modes).** Prefix caching: Mooncake's `hash_ids` are carried but not used, so
  prefill work on Mooncake traces is the no-reuse upper bound (Mooncake's own system reuses prefix KV). Speculative
  decoding, beam search and early stops: every request emits exactly O tokens. Scheduling overheads (CPU time,
  sampling) are not part of the iteration cost unless the cost function adds them.
- **Storage.** The schedule is kept as flat arrays (per iteration, per prefill chunk, per decode item) with offsets,
  ~8 bytes per field and item; Azure `conv` at σ = 1 is ~10⁶ iterations and ~4·10⁶ decode items.

### 10.3 Fixed-schedule assumption (modeling choice (a))

For the oracle study the iteration sequence of a workload is generated **once** with a reference cost (a reference
configuration c_ref) and then **every** configuration c is costed on that same sequence: `T(i, c) = cost(iteration_i,
c)` (`run_schedule` once, `cost_schedule` per c). Why: the oracle DP (§5) compares configurations interval by
interval, which requires that interval i is the same work under every c (the same reason intervals are work units,
not cycles). If each configuration produced its own schedule, batches and iteration boundaries would differ and the
DP would compare different work.

What it ignores: closed-loop feedback. Under a faster configuration requests finish sooner, fewer queue up, and later
iterations would have different batch sizes and prefill/decode mixes (and, at high load, different preemptions).
The open-loop costing is exact at low load (every request is served on arrival, so the sequence does not depend on
speed) and can be off near saturation. P7 (discrete-event serving simulation) re-schedules per configuration
(closed loop) and reports the difference to the fixed-schedule numbers; the gates of §5 use the fixed schedule only
as the oracle's interval definition.

### 10.4 Trace length scaling σ (modeling choice (b))

The stories models have short contexts (seq_len 256 for stories15M, 512 for stories260K) while Mooncake prompts
have medians of 6.3–11.6 K tokens and Azure `code` 1.5 K (`results/<run>/trace_stats.csv`). What matters for an
on-chip memory policy is the KV working set **relative to the L2** (§7's dimensionless axis), so traces are scaled
to preserve that ratio instead of the absolute lengths.

- **Scaling.** `S' = max(1, ⌊σ·S + ½⌋)`, `O' = max(1, ⌊σ·O + ½⌋)` (round half up; at least one prompt and one output
  token). Arrivals are not scaled. Mooncake `hash_ids` are kept and then describe σ·512-token blocks (unused, 10.2).
- **Matching rule.** With M_ref, L2_ref the KV bytes per token (E3) and L2 size of the reference GPU setup, and
  M_tgt, L2_tgt those of the target Vortex setup, σ is chosen so that the median per-request KV working set at
  completion over L2 is equal:
  `median_i [M_tgt · (S'_i + O'_i) / L2_tgt] = median_i [M_ref · (S_i + O_i) / L2_ref]`.
  The per-request working set at completion is E3 with B = 1 and T = S + O (the largest KV footprint of the request).
  Ignoring rounding, S'_i + O'_i = σ·(S_i + O_i), and a median commutes with multiplication by a positive constant,
  so `σ = (M_ref / L2_ref) / (M_tgt / L2_tgt)` (`sigma_for_kv_l2`): **the trace cancels** and σ depends only on the two
  (model, machine) pairs. Rounding moves the achieved median by at most one token per length (tested);
  `median_kv_l2_ratio` reports the achieved value, which every experiment records next to σ.
- **Why per request, not per iteration.** The KV bytes an iteration reads, Σ_i T_i, also scale with the number of
  concurrent requests, which depends on B_max, the arrival rate and (through queueing) on the cost model. Matching
  that quantity would make σ depend on the scheduler and the hardware being evaluated. The per-request rule keeps σ a
  property of (model, machine); the batch dimension is set by B_max, and each experiment reports the per-iteration
  ratio from the schedule's `kv_tokens` column so the batch-level working set is visible.
- **Context limit.** Optional `max_context` (e.g. the stories model's seq_len): a request with S' + O' above it is
  clipped (O' ← min(O', max_context − 1), then the prompt is left-truncated to S' = max_context − O', the usual server
  behaviour for over-long prompts) or dropped (policy `drop`). Clipping affects only requests above the limit, so the
  median is unchanged while the scaled median footprint is ≤ max_context; if it is not, the requested σ cannot be
  realised on that model and the experiment must say so (record σ, the achieved median ratio and the clipped
  fraction).
- **What σ distorts.** Arrival times are kept while each request's work shrinks (σ < 1) or grows, so offered load
  changes; serving experiments (P7) set the arrival-rate scale separately and record it. Output lengths are scaled
  too, so decode lengths shrink with σ.
- **G4.** The gate compares N_min (E9, §13) with the **median decode length of Azure `conv` after the σ scaling used
  in the experiment**, i.e. `median_i O'_i` of the scaled trace; the unscaled value (median O = 129 tokens for Azure
  `conv`) is reported alongside.

## 11. Machine parameters (ridge, bandwidth, latencies)

Decision record: `` (configurations and the reason for every choice). Configurations:
`experiments/configs.yaml` `PS-BASE-*`. Numbers: `results/<run>/machine_params.csv`, generated from `results/<run>/rows.jsonl`
by `results/<run>/summarize.py` (`--check` verifies the CSV and that every repeated run reproduced exactly). Measured
machines for the analytical model: `model/roofline.py` `vortex_psbase()` → machines `psbase-S1-{4,8,16}c`
(`measured=True`); the unmeasured `vortex-S1-*` placeholders remain only so that stay reproducible and
must not be used.

**The Vortex default configuration has no L2** (`VX_CFG_L2_ENABLE = false`, `VX_config.toml:16`): every PS-BASE enables it.

### 11.1 Microbenchmarks (what each number means)

All runs are single launches with cold caches (§2 consequence 4), `--perf=1` (core counters) unless stated, on
`vortex-trees/base`. New app `kernels/ps_memprobe` (microbenchmark code, not an validated P2 kernel; its header comments
give the details):
- **Load latencies (pointer chase, `-m chase`).** One thread follows a chain of 64 B nodes (one line each); every load's
  address depends on the previous one. Reported per level: the core's own average load latency counter (`load_lat`: cycles
  from a load's issue to its response, `sw/runtime/common/perf.cpp`) over a chase whose loads all live in that level, and
  the cycles per dependent hop of the chase (`hop`: load-to-use plus one address add and the loop; single-warp ALU
  dependencies add ≈ 25–36 cycles in SimX, so `hop` − `load_lat` is not a memory quantity). Footprints: L1 = 4 KiB (warm
  pass then timed pass); L2 = 256 KiB (128 KiB on S0), > 16 KiB L1 and < L2: the warm pass misses to DRAM and the timed pass
  hits in L2, so `L_L2 ≈ 2·avg − L0` (modeling choice: the warm pass's DRAM latency is approximated by L0 of the 64 MiB
  chase; labelled an estimate); DRAM = 64 MiB, 4,096 hops, no warm pass: every hop is the launch's first touch of its line
  (cold caches), so `load_lat` = **L0, the unloaded DRAM-miss load latency**. Two of these are **biased estimates**
 : the L1 figure averages in the 64 cold misses of its warm pass (+≈ 64·(L0 − L1)/8,197 ≈ +0.5
  cycle at +11, +1.9 at +190, relative to an unbiased L1 latency of ≈ 13.0), and the L2 estimate includes the timed-pass loads that still hit in L1; both feed
  `psbase_hier`. `rand` = lines in one random cycle (row
  conflicts), `seq` = consecutive lines (open-row hits; Vortex has no prefetcher). `dram_side_lat` = perf class 7
  `memory: lat` (memory request to response at the DRAM model).
- **Latency-hiding ratio ρ = L0 · ISSUE_WIDTH / NUM_WARPS** (cycles): the unloaded miss latency per resident warp per issue
  slot. Why this ratio (modeling choice): an issue slot with W_s = NUM_WARPS / ISSUE_WIDTH warps, each with one dependent
  miss outstanding, stays busy only if each warp supplies ≈ L0 / W_s cycles of independent work per miss (Little's law per
  issue slot: work in flight = issue rate × latency). Equal ρ means a kernel with the same instruction mix exposes the same
  share of its DRAM latency, which is what mechanisms B and C act on. It is the criterion for PS-BASE's DRAM latency.
  **Assumptions (/m11):** (i) NUM_WARPS is the *maximum* resident warps of each machine and ρ ∝
  1/occupancy; the P2 kernels fill all 8 PS-BASE warps, while the occupancy of LLM kernels on an A100 is not sourced here —
  below 64 resident warps the a100's ρ is higher (§11.2). (ii) "Keep the issue slot busy" is approximate: on PS-BASE the
  2 issue slots share one ALU, one LSU and one FPU block, and even the memory-free FMA loop reaches only 0.28 IPC per core;
  ρ is a first-order ratio, not an exact equivalence.
- **DRAM bandwidth.** `ps_memprobe -m stream` (all NT·NW·cores threads read a 32 MiB uint32 array with a grid stride, 4
  independent loads per thread-iteration, exact host-checked sums): bytes / global cycles = the achievable LSU-path
  bandwidth; the stock `dxa_copy -d1` (32 MiB in 1 KiB tiles, one tile per block; a block may not exceed NT·NW = 256
  threads) gives the DXA path. Model peak = channels × 16 B × data rate / clock (`model/roofline.py` `dram_peak_bw`).
- **FPU peak.** `ps_memprobe -m fma`: every thread runs 8 independent FMA chains (16 FLOP per thread-iteration, 1,000
  iterations); FLOP / cycles. Structural = cores × 32 lanes × 2 FLOP × NUM_FPU_BLOCKS (1).
- **TCU peak.** Stock `sgemm_tcu_wg_dxa` (fp16 → fp32) at K = 64 with M = N ∈ {512, 1024}; 2MNK / cycles, the
  best shape is reported. K = 128 was tried once on PS-BASE-S1-4c and **holds** (PASSED; errata 28 already showed the stock
  ULP gate scales with K): 170.3 FLOP/cycle at 512² × 128 vs 149.9 at 512² × 64, **+13.6 % at the same M, N**. Structural = cores × ISSUE_WIDTH TCU blocks × 512 FLOP/cycle (`model/roofline.py` `vortex_s1`).
- **Ridge (E7)** = measured peak FLOP/cycle ÷ measured stream bytes/cycle (FLOP/byte), per unit. These ridges, not the
  structural ones, define the AI/ridge axis of §7 for Vortex from now on. The denominator is the **LSU-path** stream;
  the TCU GEMM loads through DXA, and at 16 cores and at +190 the LSU stream is not DRAM-bound (on-chip queueing at
  16 cores, the per-core in-flight cap at +190; §11.4), so those ridges use a lower bound on DRAM bandwidth (an upper bound on the ridge).

### 11.2 PS-BASE configurations (summary)

Common core = the G-SPEED TIER-CFG core (NT = 32, NW = 8, ISSUE_WIDTH = 2, TCU + WGMMA, DXA, one ALU/LSU/FPU block) + MUFU,
PLRU at L1 and L2, L2 on; per core 256 KiB of L2 and 16 B/cycle of peak DRAM bandwidth (0.5 HBM2 channel; S0: one channel
at half the data rate); fixed DRAM read latency +11 core cycles. S0 = 1 core, S1 = 4 (primary) / 8 / 16 cores;
sensitivity variants of the primary: `-lat0` (Vortex default DRAM latency) and `-lat190` (a100's fixed latency in absolute
core cycles).

**DRAM latency (modeling choice, the one that decides how much memory-side mechanisms can matter).** PS-BASE matches the
a100-like **latency-hiding ratio ρ** (§11.1), not the a100's latency in core cycles. a100: ρ = L0 · 4 / 64 with L0 measured
on SimX at the a100 profile's memory keys (`third_party/vortex/profiles/a100.json` [cand:D35], `P3.1-a100lat-ref`) = 19.84
(random) / 17.45 (sequential); PS-BASE core at the default model: 17.22 / 14.41; with +11 cycles: 19.97 / 17.16 (+0.7 % /
−1.7 %). What this rests on:
- **Occupancy.** ρ uses 64 resident warps per a100 SM, the maximum. If LLM kernels on an A100 keep fewer warps resident
  (not sourced here), the a100's ρ is higher, so **+11 is the low-latency end** of the plausible range. The `-lat190`
  bracket (ρ = 64.7 / 61.9) equals an a100 at 317.4 · 4 / 64.7 ≈ 19.6 (random; 18.0 sequential) resident warps per SM,
  ≈ 28–31 % occupancy; `-lat0` lies 13–17 % below a100's full-occupancy ρ.
- **The a100 L0 is a SimX value**, not an A100 measurement: GPGPU-Sim's fixed 190 cycles (per the profile's `sources`
  field [cand:D35]) on Vortex's cache-latency formulas and DRAM model, run with 4 cores and 2 channels; the profile's
  32 L2 banks, queue size 257, 32 channels and 108 cores were not applied (PS-BASE's L0 grows 79.9 → 83.1 cycles from 4
  to 16 cores, so the 4-core reference probably understates a 108-core one a little).
- **h100 is not a latency target**: its profile has no fixed-latency key and its `sources` field says latency is "not
  calibrated" [cand:D35].
- **Absolute time is not an independent check.** At Vortex's default 400 MHz clock the chosen L0 is 200 ns vs 225 ns for
  the a100-like reference (default model: 172 ns), but that agreement depends entirely on the 400 MHz default, which is
  arbitrary for a simulated baseline: at an A100-like 1,410 MHz, 200 ns is ≈ 282 cycles, at or slightly beyond the `-lat190`
  bracket (whose L0 is 258.8 cycles = 184 ns at 1.41 GHz).
Matching the a100's absolute cycles (+190) gives ρ ≈ 65 (3.3× a100's at full occupancy). The DRAM-vs-L2-hit latency ratio
was not used: the a100 profile reuses Vortex's own cache latency formulas, and no ledger source gives an A100 L2 hit
latency.

### 11.3 Measured machine parameters (`machine_params.csv`; cycles, unloaded unless stated)

| config | cores | +lat | L1 | L2 (est.) | L0 rand | L0 seq | DRAM-side rand | ρ rand | ρ seq | peak B/cyc | stream B/cyc | of peak | dxa_copy B/cyc | FPU FLOP/cyc | (struct.) | TCU FLOP/cyc | (struct.) | ridge FPU | ridge TCU |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| PS-BASE-S0 | 1 | 11 | 13.5 | 27.02 | 82.92 | 71.69 | 45.9 | 20.73 | 17.922 | 16.0 | 9.987 | 0.6242 | 0.98 | 14.055 | 64 | 35.767 | 1024 | 1.407 | 3.581 |
| PS-BASE-S1-4c | 4 | 11 | 13.48 | 28.19 | 79.87 | 68.64 | 40.9 | 19.968 | 17.16 | 64.0 | 40.707 | 0.636 | 4.13 | 56.104 | 256 | 155.854 | 4096 | 1.378 | 3.829 |
| PS-BASE-S1-8c | 8 | 11 | 13.49 | 25.54 | 81.26 | 70.42 | 41.8 | 20.315 | 17.605 | 128.0 | 91.391 | 0.714 | 8.22 | 112.429 | 512 | 304.047 | 8192 | 1.23 | 3.327 |
| PS-BASE-S1-16c | 16 | 11 | 13.51 | 23.85 | 83.07 | 72.61 | 41.6 | 20.767 | 18.152 | 256.0 | 143.563 | 0.5608 | 16.413 | 224.282 | 1024 | 515.92 | 16384 | 1.562 | 3.594 |
| PS-BASE-S1-4c-lat0 | 4 | 0 | 13.39 | 28.19 | 68.87 | 57.64 | 29.9 | 17.218 | 14.41 | 64.0 | 41.615 | 0.6502 | 4.214 | 56.115 | 256 | 158.916 | 4096 | 1.348 | 3.819 |
| PS-BASE-S1-4c-lat190 | 4 | 190 | 14.96 | 28.31 | 258.83 | 247.6 | 219.9 | 64.707 | 61.9 | 64.0 | 27.036 | 0.4224 | 3.036 | 55.875 | 256 | 134.303 | 4096 | 2.067 | 4.968 |
| P3.1-a100lat-ref | 4 | 190 | 18.25 | 41.78 | 317.4 | 279.18 | 262.5 | 19.837 | 17.449 | 68.948 | — | — | — | — | — | — | — | — | — |

L0 = DRAM-miss load latency; ρ = L0 · ISSUE_WIDTH / NUM_WARPS (for the a100 reference with the a100 profile's 4 / 64);
TCU at K = 64, best passing shape (`tcu_best_args` in the CSV; 16c: 512² only, errata 31); K = 128 on PS-BASE-S1-4c: 170.3
FLOP/cycle at 512² (+13.6 % vs 512² × 64, holds). **Stream B/cyc at S1-16c and -lat190 are lower bounds on DRAM
bandwidth, not DRAM-bound** (§11.4: on-chip queueing between LSU and DRAM at 16c, the per-core in-flight cap at +190); S1-4c,
S1-8c and -lat0 are DRAM-limited, S0 is mixed. The a100 reference row is a latency probe only (4 cores of the PS-BASE core with the a100 profile's
memory keys), not a machine. WGMMA vs non-WG (NT = 8, `P3.1-tcu-nt8-ref`): 155.9 vs 105.5 FLOP/cycle at 1024² × 64.

### 11.4 Reading the numbers (implications)

- **Ridges are low.** Measured ridges are ≈ 1.2–1.6 FLOP/byte (FPU) and ≈ 3.3–3.8 FLOP/byte (TCU, K = 64) on every S1/S0
  config, vs 4 / 32 structural placeholders at 4 cores: the FPU reaches 22 % of its structural peak (56 of 256
  FLOP/cycle at 4 cores; an 8-chain FMA loop with no memory access issues ≈ 0.22 FMA warp-instructions per cycle per core,
  0.28 IPC per core in total; the cause of this limit is **unexplained** — it is not the 8-cycle FMA latency, which 8 chains
  per warp cover, and doubling the chains did not change the rate in an **unrecorded development run** — 16 chains × 500 iterations
  291,737 cycles vs 8 × 1,000 291,974; that run's host check failed by construction (`bad=1024`, its host reference
  still modelled 8 chains) and it lives in the private dir `build/tmp/p31/res`, so treat it as unverified) and the TCU 3–4 % (WGMMA
  + DXA single-buffered). So on PS-BASE an fp32 decode GEMV (AI ≈ 0.5 FLOP/B) is below the FPU ridge, while fp16 TCU
  GEMMs with AI > ≈ 4 are compute-bound. AI / ridge (§7) must use these measured ridges; against GPU ridges (a100 tensor
  ≈ 200 FLOP/B with the roofline.py assumptions) the Vortex TCU is two orders of magnitude weaker relative to bandwidth.
  This is a disclosed distortion of the dimensionless AI axis, not something a configuration knob can fix.
- **Achievable LSU-stream bandwidth and where its latency builds up (rewritten; the second-pass "LSU pending
  queue" attribution for 16 cores was wrong).** The LSU path keeps at most `VX_CFG_LSU_PENDING_SIZE` = 16 warp-loads in
  flight per LSU block (`VX_config.toml:118`, enforced at `sim/simx/lsu_unit.cpp:339`), each 32 lanes × 4 B = 128 B, i.e.
  2 KiB per core with PS-BASE's single LSU block. (The L1 MSHRs, 16 per bank × 4 banks at NT = 32, `sim/simx/mem/cache.cpp:653`,
  do not bind.) In this closed loop the queue is **nearly full in every configuration**: the stream runs at 83–94 % of
  cores · 2,048 B / (stream load latency). So the queue does not tell the regimes apart; **where the extra latency
  accumulates** does. Decomposition (perf 1 `load_lat` of the stream and of the unloaded sequential chase, perf 7 DRAM-side
  `memory: lat` of the same two runs; columns `stream_latency_inflation`, `dram_share_of_inflation`, `stream_limit` of
  `machine_params.csv`, rule in `results/<run>/summarize.py`):

  | config | measured B/cyc | stream load_lat / L0 seq | inflation (cycles) | of which DRAM-side | non-DRAM | 2 KiB/core cap at unloaded L0 | where the latency builds up |
  |---|---|---|---|---|---|---|---|
  | S0 | 10.0 | 187.1 / 71.7 = 2.61 | 115.4 | 53.9 (47 %) | 61.5 | 29 | **mixed** DRAM and on-chip |
  | S1-4c | 40.7 | 167.9 / 68.6 = 2.45 | 99.3 | 90.5 (91 %) | 8.8 | 119 | **DRAM** |
  | S1-8c | 91.4 | 157.3 / 70.4 = 2.23 | 86.9 | 72.4 (83 %) | 14.5 | 233 | **DRAM** |
  | S1-16c | 143.6 | 207.3 / 72.6 = 2.86 | 134.7 | 13.7 (10 %) | 121.0 | 451 | **on-chip, LSU→DRAM path (unidentified)** |
  | -lat0 | 41.6 | 165.4 / 57.6 = 2.87 | 107.7 | 98.0 (91 %) | 9.7 | 142 | **DRAM** |
  | -lat190 | 27.0 | 286.1 / 247.6 = 1.16 | 38.5 | 17.4 (45 %) | 21.1 | 33 | **LSU in-flight cap** |

  (With the random chase as reference the shares are 41 / 90 / 81 / 3 / 90 / 23 % and the classes are the same.) Readings:
  - **-lat190:** the stream's load latency is only 1.16× unloaded, and 2 KiB per core at that latency allows ≈ 33 B/cycle
    vs 27.0 measured: the per-core in-flight cap really is the limit.
  - **16 cores:** 121 of the 135 extra cycles accumulate **between the LSU and the DRAM model's input** (L1, L2,
    interconnect, the memory input or the response path); the perf-7 interval starts when a request enters the DRAM model
    (`sim/simx/processor.cpp:165-176`) and the DRAM queues are unbounded (`PLATFORM_MEMORY_QUEUE_SIZE = 0`), so this is not
    DRAM time. The congested resource is **not identified**: the memory crossbar collision count (perf 7 `bank_st`,
    `sim/simx/mem/memory.cpp:77`) sits inside the memory model, downstream of where the DRAM-side latency measurement starts,
    so its time is part of the DRAM-side share, not of the ≈121 non-DRAM cycles; it is 55,415 at 16c vs
    37,317 / 75,575 at 4c / 8c and in any case does not single 16c out; L1/L2 bank-stall counters (perf 4/5) were not recorded for the stream runs.
    At unloaded latency 2 KiB per core would allow ≈ 451 B/cycle, above the 256 peak, so the LSU cap is not what holds 16c
    at 143.6. ** (memhier MLP calibration) and P5 (B, C) must not assume that more LSU depth or MLP would raise the
    16-core bandwidth toward the DRAM peak**; the on-chip queueing would have to be located first.
  - **4c, 8c, -lat0:** 83–91 % of the inflation is DRAM-side (DRAM queueing): DRAM-limited. **S0** (1 L2 bank, one L1
    memory port, one channel at half rate) has about half of its inflation on-chip: mixed.
  So 56–71 % of the channel peak mixes regimes. **The 16c and -lat190 stream figures are lower bounds on DRAM bandwidth
  (not DRAM-bound)**; they are the machine's achievable LSU-path bandwidth and are used as such (roofline `psbase-S1-16c`
  says so). Per core: 9.0–11.4 B/cycle for S0/S1 at +11 (6.8 at -lat190) (a100's peak is 10.2 B/cycle per SM). The stock `dxa_copy` with 1 KiB tiles reaches
  only 1–16 B/cycle (per-block overhead dominates: 13.8 M instructions for 32 MiB), so it is not a bandwidth ceiling either.
- **L0 grows slightly with cores** (79.9 / 81.3 / 83.1 cycles at 4 / 8 / 16 cores, 82.9 on S0 with its half-rate channel),
  so ρ is 20.0–20.8 on S1 (within 5 % of a100's 19.84).
- **Reproducibility:** three rows re-run (rep 1: DRAM chase, stream, 1024² TCU GEMM) are identical in instructions and
  cycles (`summarize.py --check`).
- **16-core TCU (errata 31):** the stock WGMMA GEMM fails its own check at 16 cores for M = N ≥ 768; the 16c TCU peak is from
  512 × 512 × 64 (passing) and under-states the unit at larger shapes. The failure is timing-dependent, so any WGMMA/DXA run
  whose memory timing differs from the configurations tested here (P5 mechanisms included) must check correctness per run.
- **memhier:** `model/memhier.py` `psbase_hier(cores)` (machines `psbase-S1-*`) uses these sizes and splits the measured
  latencies so that the model's L2-hit latency and L0 equal the measured ones (l1 = L1, l2 = L2 − L1, dram = L0 − L2; both
  inputs are the biased estimates of §11.1); its L2 bank count (8 / 16 / 16 at 4 / 8 / 16 cores) equals SimX's
  (`` (d), "L2 banks"); every other memhier parameter stays uncalibrated.

### 11.5 Baseline characterization sweep

Code: `results/<run>/make_batch.py` (grid, settings, batch files; its docstring is the design record),
`results/<run>/summarize.py` (table and analysis), `results/<run>/figures.py`; numbers in `results/<run>/README.md`.
Machine: **PS-BASE-S1-4c only** (simulation-speed tier C ⇒ 4 cores). Every point is one
cold-cache launch (§1), on the fixed build (`-fno-math-errno`), the P2 kernels' device code unchanged.

**Grid (modeling choices).** One layer of the scaled Llama-3.1-8B (`llama8b_s8`: d = 512, H_q = 8, H_kv = 2,
D = 64, d_ff = 1792), so the five op families describe the same layer: decode attention (B, T) ∈ {1, 2, 4} × {64, 256,
1024} (`gqa_shared_kv` as base; `naive_per_qhead` measured as a variant at base); prefill attention (B, S) ∈ {1, 2, 4}
× {16, 32, 64}, s0 = 0 (`lmem_tiled`); linear (M, matrix) ∈ {1, 16, 256} × {q 512×512, w1 1792×512, w2 512×1792}:
M = 1 the fp32 GEMV (`ps_gemv_f32`, variant by the §14.5 rule at NT = 32: `lanes` for K = 512 and 1792), M ≥ 16 the stock
WGMMA + DXA TCU GEMM (`sgemm_tcu_wg_dxa`, fp16 in; M = 16 is run as an M = 32 GEMM — cta_M = 32 — on the stock app's own
inputs, and FLOPs are counted at M = 16; cycles are data-independent);
element-wise (kernel, B) ∈ {rmsnorm 512, residual 512, silu_gate 1792, rope 512/128/64} × {1, 4, 16} (musl soft-float libm,
§14.0); mixed attention (`ps_mixed_attn`, one binary) at three Sarathi-style points (c, s0, n, T) = (16, 48, 4, 256),
(64, 0, 4, 256), (32, 32, 8, 512). Grids are tier C's 3 × 3 (G-SPEED), sized so the whole step fits its 6 h budget
(`results/<run>/budget.csv`: 4.4 h estimated a priori). **Phases:** prefill = prefill attention + linear M = 256; decode =
decode attention + linear M ∈ {1, 16} + element-wise; mixed = `ps_mixed_attn`.

**Stock-knob settings, one at a time around the PS-BASE baseline** (G-SPEED's "baseline + one-at-a-time"): L2 size ½× / ¼×
(`VX_CFG_L2_SIZE`; the L2 hit latency formula `4 + max(0, log2(size) − 20)` keeps it unchanged), L2 16 ways (the a100
profile's associativity) instead of 8, FIFO instead of PLRU at L1 and L2 (FIFO is the Vortex default), software DXA depth
1 / 2 / 4 (attention kernels; depth 0 = no DXA = base), and the LMEM budget the kernel requests per launch (the launch's
`lmem_size` = KMU_LMEM_SIZE): the kernel's own host tile rule evaluated as if the core LMEM were 32 / 16 KiB instead of the
PS-BASE 64 KiB, the tiles passed explicitly (`-Q/-t`). The LMEM budget does not change `gqa_shared_kv` (its 2.6 KiB
footprint fits every budget), so for decode it is measured on the DXA variant (`dxa2` at 32 / 16 KiB, two-knob settings,
labelled). A setting that changes neither an op's binary nor its arguments (DXA/LMEM on linear and element-wise, DXA on
`ps_mixed_attn`, which compiles no DXA path) is `same_as_base`: identical by construction, its value is the base value, and
the phase is "n/a" for that knob in the tension analysis. Combinations of knobs are **not** measured (one-at-a-time only).

**Perf classes.** One class per run (audit Q3): class 7 (DRAM requests and average DRAM latency, LMEM) on every run cell,
classes 1 (scheduler/stalls) and 5 (L2) on every base cell, class 16 (DXA) on the `dxa2` base cells. Cycles are identical
across classes (checked on all 60 multi-class groups), which is what makes this reduction lossless for cycles.
**Correctness evidence per app family** (`summarize.py evidence()`, required by `--check`, a missing line fails): attention
apps PASSED + PSCHECK errors 0 + PSINFO `within_plan_tol` = 1; GEMV and element-wise apps PASSED + PSCHECK errors 0 (they
print no PSINFO line); the stock TCU app its own PASSED.

**Derived quantities** (summarize.py docstring): useful FLOPs as (attention 4·D per (query head, visible key),
linear 2·M·N·K unpadded); DRAM bytes = (reads + writes) × 64 B; AI = FLOPs / DRAM bytes; DRAM B/cycle = bytes / cycles (an
occupancy, a lower bound on memory time); working set = compulsory bytes touched (attention q, K, V, out; linear weights at
the path's precision + activations; element-wise inputs + outputs), and "working set / L2" uses the setting's L2 size.

**Policy-tension metric (modeling choice).** Per phase P and setting s, R_P(s) = Σ_{c∈P} cycles(c, s) / Σ_{c∈P}
cycles(c, base): one launch of every grid cell is the phase's workload, time-weighted because a static configuration pays
the phase's total time (the per-cell geometric mean is reported beside it). Per knob dimension (L2 size, L2 ways,
replacement, DXA depth, LMEM budget) the best value per phase is argmin R_P over the dimension's values; the **cross loss**
L(P ← Q) = R_P(best_Q) / R_P(best_P) − 1 is what phase P loses at phase Q's best value. A **material** tension needs
different best values and a cross loss ≥ 1 % (`MATERIAL`; sub-percent differences in this deterministic simulator come from
set-mapping and scheduling details that another shape or patch can flip). Also reported, **within each knob type**: the type's global best
value vs the per-phase best (geomean over the applicable phases of R_P(global)/R_P(best_P)), and the per-cell best within the
type (Σ_c min_{v ∈ type} cycles(c, v) relative to the phase's best value; the per-launch analogue of §3's per-kernel best
static, restricted to one knob type), plus the same for the three hardware (cache) knobs taken together. Comparing single
settings **across** types (e.g. DXA d2, a kernel variant, against 16 ways, a cache knob) compares alternatives whose
combination was not measured; it is reported only as a labelled side line, never as tension evidence. Op-family results
are reported next to the phase totals because the phase totals weight ops by grid cell, not by a layer's op mix. In the mixed
kernel the same question is asked **on one binary**: per setting, the prefill-region and decode-region busy
core-cycles of the per-CTA timestamps (§14.6) relative to base; fused vs separate launches are never compared. The
**launch-level** quantity is the mixed launch's cycles relative to base, and the launch-level size of a region tension is
cycles(prefill-region-best setting) / cycles(decode-region-best setting) − 1 (a static configuration applies to the whole
launch).

**DRAM-latency share (carry-over 1).** The method (§14.3): three fixed-latency points per bracket (the bracket's
own `VX_CFG_PLATFORM_MEMORY_LATENCY` and +50, +100), the measured average DRAM latency L (perf 7) of each, and share =
s·L₀/cycles₀ with s the slope of the lowest segment (first order). Two robustness read-outs: the least-squares slope through
the three points (`share_ls3`) and the earlier quadratic extrapolation to L = 0. For the attention kernels the two segment slopes
differ by up to 2× in both directions (scheduling noise of a few % of cycles per 45-cycle latency step), so the quadratic
read-out is unstable for them (it even goes negative) and is not used. The two linear read-outs are reported side by side;
they do not bound the share: for a concave kernel both are lower estimates (the earlier argument; GEMV `rows` K = 528 at +11:
quadratic 19.2 % vs 18.1 / 17.7 %). As, a zero-latency DRAM is not physical: the shares are sensitivities.

## 12. Request classes, hints and the MSHR class-merge rule

Patch `patches/vortex/ps/0002-mem-class.patch` (ps tree), host/device helpers `kernels/common/ps_hint.h` and
`kernels/common/ps_phase.h`, report parser `scripts/psclass.py`, tests `tests/test_mem_class.py`, evidence `results/<run>/`.
**In this patch a class is carried and counted, never acted on**: no replacement, arbitration or DXA decision reads it,
so simulated timing is unchanged (measured: §12.8). The mechanisms consume it.

### 12.1 The class field
Every SimX memory request (`MemReq`, `LsuReq`; `sim/simx/types.h`) carries `MemFlags.mem_class`, 3 bits at **bits 21–23
whether or not DXA is built**. Without DXA a 17-bit pad occupies the DXA completion sideband's bits 4–20, and bits 24–31
are an explicit reserved field, so the bit-field list covers the whole word. The list is written once and instantiated
twice (inside the `MemFlags` union and as a plain `MemFlagsLayout` struct); `static_assert`s on `MemFlagsLayout` check the
exact positions with GCC's constant-evaluated `__builtin_bit_cast` (the compiler that builds SimX; clang cannot yet
evaluate bit-casts of bit-fields, so there only the width checks apply), and the processor checks the layout once more at
run time. Fixed bits mean a later RTL/SimX parity check and every mechanism patch see one encoding.

### 12.2 Classes (VX_types.toml `[ps_mem_class]`)
| id | class | meaning | set by |
|---|---|---|---|
| 0 | ORDINARY | unclassified (reset value; instruction fetch, scalars, activations, outputs) | default |
| 1 | WEIGHT | parameters | `mem_hint` or the weight range |
| 2 | KV_PERSIST | KV rows re-read within the launch (a prefill chunk's K/V: every query tile of the chunk re-reads them) | `mem_hint` or range `kv[1]` |
| 3 | KV_STREAM | KV rows read once per step (decode) | `mem_hint` or range `kv[0]` |
| 4 | BULK_DXA | DXA engine GMEM reads (a copy engine; no warp waits on the individual request) | DXA, always |
| 5–7 | reserved | | |

### 12.3 Where a request gets its class (precedence)
1. **DXA** tags every GMEM read BULK_DXA where it builds the request (`sim/simx/dxa/dxa_core.cpp:579-586`, errata 13);
   the LMEM write is untouched. Per-stream DXA classes are mechanism C.
2. **`mem_hint`** (CSR 0x8C0, per warp, §12.6): the LSU stamps the issuing warp's hint on every lane of a load/store
   (`LsuUnit`, next to `amo_unsigned`; warp-uniform like the other `LsuReq` flags). Hint 0 means *unset*, so a hint cannot
   force ORDINARY inside a range (not needed by any kernel; a kernel that wants ORDINARY there simply does not program
   the range).
   *Caveat (measured):* the hint labels **every** load/store the warp issues while it is set — including the
   compiler's stack spills, fills and stack-resident locals (the per-thread stack is in global memory). In
   `ps_decode_attn_gqa` PS_HINTS=1 the reads and stack write-backs **outside** the KV range therefore carry KV_STREAM:
   9,344 of 31,488 non-KV DRAM bytes at REG-CFG (30 %), and **34,176 of 49,024 (70 %) on the 1-core smoke-dxa config
   without L2** (last report, after the flush; an independent end-of-launch count is 32,640 of 46,912;
   `results/<run>/README.md`). Hint quality is judged on the bytes inside the KV range, but a
   mechanism keyed on KV_* (B, D) would stream or bypass stack lines too. **Suggestion for P5:** exclude the stack region
   (`VX_MEM_STACK_*` / the per-thread stack window) from hint stamping in the LSU, or classify KV by range only; range
   classification does not have this problem.
   *Timing of a hint:* `csrw` is an SFU instruction that does not stall the warp (`sim/simx/decode.cpp:700-716`,
   no `wstall`), and the LSU reads the warp's hint when it builds the request (`LsuUnit`), not when the load issues. A
   load issued right after a hint write can therefore still be stamped with the previous hint if the SFU queue is
   behind, and a load queued before a reset can get the new value. The asm `"memory"` clobber orders only the compiler.
   A hint is reliable only across a **barrier or a data dependency**: write it, then a CTA barrier (or loads that depend
   on a value produced after it), and reset it only after a barrier or after stores that depend on the hinted loads. The
   two kernels satisfy this on the **reset** side (reset after the end-of-tile barrier / after the LMEM stores that depend
   on the loads). On the **set** side they do not strictly: both write the hint right after a barrier and immediately
   before the hinted loads (`decode_attn_gqa/kernel.cpp:83→99`, `mixed_attn/kernel.cpp:98→108`, `:237→240`), with no
   barrier or dependency between the `csrw` and the loads. This is safe in practice (measured labelling 100 %) but not
   guaranteed by the rule; move the hint before the barrier the next time the hint binaries change.
3. **Address ranges** of `[dcr_phase]` (the earlier KV0, KV1, WGT pairs): a request that is still ORDINARY when it enters the
   first non-bypass cache is classified by the first range containing its address, in the fixed order kv[0] → KV_STREAM,
   kv[1] → KV_PERSIST, wgt → WEIGHT. With `VX_PHASE_FLAG_RANGE_ACCT_ONLY` (flags bit 1) the ranges only label DRAM bytes
   for the accounting of §12.7 and classify nothing — that is how a PS_HINTS=1 build is measured on its hints alone. Empty
   ranges (the reset value) match nothing.
4. Otherwise ORDINARY.

*Design choice — classify at the cache input, not in `lsu_mem_adapter` (the earlier pointer).* The range check sits in
`Cache::Impl::tick` where a core request enters the bank crossbar (`sim/simx/mem/cache.cpp`, the non-I/O branch). It is
the first place that already holds the latched config (put a copy in every cache), it covers both LSU→dcache paths
(the memory coalescer and the per-lane adapter both copy the LSU flags), the DXA path into L2 and every other requester,
and it is idempotent at later levels (a classified request is never reclassified). The hint, which needs the warp, is
applied in the LSU. Limitation: if every cache level were bypassed, ranges would never apply (no PhaseShift
configuration does that; PS-BASE has L1 + L2).

### 12.4 Propagation below L1 (stock SimX dropped the flags)
Stock SimX builds the requests a cache sends downstream from scratch, so the requester's flags were lost at the first
miss (fills `cache.cpp:1414-1419,1630-1636`, write-through stores `:1206,1246,1556`, write-backs `:1056,1325,1679`;
pristine line numbers). The patch copies **only the class** (never `io`/`local`/`strsp`/DXA bits, which would change
behaviour):
- a **fill** carries the class of the request that allocated the MSHR chain (§12.5);
- a **write-through store** carries the store's class;
- a **write-back** (victim, AMO probe, flush walk) carries the class of its sector: each sector keeps the 3-bit class of
  its most recent fill or write (`sector_t::ps_class`). So the bytes that leave a cache are labelled by whoever last put
  them there — a stored KV row written back is KV_*, not the class of the unrelated miss that evicted it.
- the non-LLC AMO pass-through already forwarded the whole flag word (stock).

### 12.5 MSHR merge rule
A request that misses on a (set, tag, sector) whose fill is already pending joins that MSHR chain (stock SimX coalescing,
at admission or at pipe exit). Rule:
- **class = class of the first requester.** The fill was sent, with that class, when the chain was allocated, and a
  request in flight in SimX is immutable; re-tagging it would need a search of every downstream queue. It also keeps the
  byte accounting conservative: each DRAM transfer is attributed exactly once, to the miss that caused it.
- **priority = max over the chain** of `ps_class_prio(class)`, with prio(BULK_DXA) = 0 and prio(every other class) = 1 in
  this patch. A demand load that finds a bulk DXA copy of its line already in flight must not inherit "background"
  status: it now waits on that transfer. The chain priority is kept equal in every entry of the chain
  (`mshr_entry_t::ps_chain_prio`) so the fill response can read it; this patch only records it and counts the events
  (`mixed_merges` = joins whose class differs from the first requester's, `prio_upgrades` = joins that raised the
  priority; PS_DEBUG_CLASS `PSCLASS-CACHE` lines). Whether an upgrade must also reach the already-queued DRAM request is
  mechanism B's decision, and a finer priority table is the mechanism patches' (new patch, §2 discipline).
  *Coverage:* in every run both counters are 0 — none of the measured kernels has two classes touching one line
  within a launch (DXA fills LMEM, so no warp load shares a DXA line; prefill and decode caches are disjoint). The rule
  is implemented and reviewed but not yet exercised; the first mechanism that relies on it needs a
  microbenchmark that makes a demand load meet an in-flight DXA line.

### 12.6 The two CSRs and `ps_hint.h`
Custom user read/write CSRs (RISC-V: 0x800–0x8FF has bits[11:10] = 10 (read/write) and bits[9:8] = 00 (user); 0xC00–0xFFF
is read-only by convention; no other `VX_CSR_*` uses 0x8xx; errata 16), block 0x8C0–0x8C7 reserved for PhaseShift:
- `VX_CSR_PS_MEM_HINT` = 0x8C0: per-warp state (`warp_t::ps_mem_hint`, cleared at reset, at every CTA dispatch and at
  wspawn), low 3 bits = class; read returns it.
- `VX_CSR_PS_MARK` = 0x8C1: a write counts one work unit of type `value & 7` (`[ps_mark_type]`: 0 OTHER,
  1 PREFILL_TILE, 2 DECODE_TILE) on the writing core **per active SFU-packet trace** of the CSR instruction: with
  `NUM_SFU_LANES < NUM_THREADS` the SFU dispatcher splits a warp instruction into one trace per packet with an active
  lane, and each trace executes the write (errata 33b, corrected; the `ps_mark_written` guard in
  `csr_unit.cpp` cannot de-duplicate across traces and its comment is wrong). **Contract: write `PS_MARK` from exactly
  one lane** (`if (tid == 0)`), which makes the count one per call on every configuration; both kernels do, and
  `test_marks_single_lane` checks it. On all configurations SFU lanes = NUM_THREADS, so even an unguarded mark would
  count once per warp, but kernels must not rely on that. Counters are per core and per launch (cleared by the core's
  reset); read returns the core's total. builds its work-unit intervals on them (carry-over: count only the first
  packet trace, `trace->sop`).
- `kernels/common/ps_hint.h`: `PS_MEM_HINT(cls)` / `PS_MARK(type)` are `csrw` (rd = x0) with a memory clobber, and
  expand to `((void)0)` unless the kernel is compiled with `-DPS_HINTS=1` (passed through CONFIGS, so it is part of every
  row's canonical CONFIGS and run key). Every warp writes its own hint; thread 0 of a CTA marks each finished key tile.
  `ps_decode_attn_gqa`: KV_STREAM around the key-tile loop (its only global loads are K/V rows), one DECODE_TILE per
  (CTA, tile) → host expectation grid × ntiles. `ps_mixed_attn`: prefill CTAs KV_PERSIST around the K/V tile copy, decode
  CTAs KV_STREAM around the tile loop; marks PREFILL_TILE per (prefill CTA, key tile) = the host's `tiles_p` and
  DECODE_TILE per (decode CTA, key tile) = Σ_b H_kv ⌈ctx_b / Tt⌉.
- Host side (`-R none|kv|kvacct` in both apps): one range must cover exactly a part's K and V caches. `ps_decode_attn_gqa`
  uses the span from the lower to the upper end of its K and V buffers; the host checks only that **q and out** lie
  outside it. Allocations made later (the kernel argument block, the module) could in principle land in the
  allocator gap between K and V (2,560 bytes at the REG shape); in every measured `-R` run the kv0 bytes equal exactly the
  K + V bytes read, i.e. no foreign traffic fell inside.
  `ps_mixed_attn` cannot: the runtime allocator places small buffers in the holes between page-rounded allocations
  (measured), so with `-R` it allocates each part's K and V as ONE buffer (V at the next 4 KiB boundary). That moves
  K/V, so `-R` rows of the mixed app are compared only with each other; `-R none` keeps the validated layout and code
  path unchanged.

### 12.7 Hint builds are distinct binaries
A PS_HINTS=1 binary contains extra instructions (two `csrw` per warp around the loop, one guarded `csrw` per tile) and
therefore different timing from the PS_HINTS=0 binary, and it aborts on the base tree (unknown CSR). Rules:
- **every later comparison of mechanisms is made between configurations running the same kernel binary** — never a
  PS_HINTS=1 run against a PS_HINTS=0 run, never fused against separate launches. The hint flag is in the canonical
  CONFIGS of every vxrun row, so same-binary filtering is mechanical;
- the no-op regression uses PS_HINTS=0 builds only; PS_HINTS=0 ELFs of the validated kernels are byte-identical
  to their pre-hint builds (`results/<run>/elf_sha_{pre,post}.txt`);
- the hint overhead reported in `results/<run>/README.md` is a property of the binaries, not a mechanism effect.

### 12.8 Accounting (`PS_DEBUG_CLASS=1`) and the no-timing-effect check
SimX prints after every launch and after every cache flush (format: `scripts/psclass.py`): DRAM bytes by class (counted by
`DramSim::send_request`, which now receives the class through a defaulted sixth parameter so the RTL simulators' calls
are unchanged), the independent total (MPM `mem_reads + mem_writes`, counted where requests enter the memory model) ×
block, DRAM bytes by (range region kv0/kv1/wgt/other, class), core-side requests by class at L1i/L1d/L2/L3 with the
merge counters, DXA GMEM requests by class, and per-core ps_mark counts. Counters are cumulative since device open (the
marks per launch). Counting and printing do not touch simulated state; within one binary PS_DEBUG_CLASS on/off and
`-R none/kv/kvacct` give identical instrs and cycles (tested).

### 12.9 MPM class 8 `PHASESHIFT`: the PhaseShift counters (ps patch `ps-counters`)
Patch `patches/vortex/ps/0003-ps-counters.patch`, parser `scripts/pstrace.py`, tests `tests/test_pstrace.py`, evidence
`results/<run>/`. Stock MPM class 8 was `RESERVED2`; it is renamed `VX_DCR_MPM_CLASS_PHASESHIFT` (no stock code used the
name). Like the classes of §12.1–§12.8, every counter is **observation only**: nothing the timing model reads is
changed (the no-op regression, the trace-on/off runs and the existing-class comparison in `results/<run>/` check this).

*Layout and paging.* An MPM class exposes one window of 29 user counters (CSR ids 3..31 = `VX_CSR_MPM_USER`..0xB1F);
class 8 holds `VX_PS_CNT_NUM` = 256, so it is **paged**: counter k is read as MPM class `8 | (page << 5)`, page = k / 29,
CSR id 3 + k mod 29. The host's DCR tag carries 10 class bits (`vx_device_mpm_query`, tag bits 22–31), the stock classes
use values 0–16 (5 bits), so pages 0–8 never collide with another class. The group bases are defined once in
`VX_types.toml [ps_counters]`; SimX (`sim/simx/ps_counters.h`, values and the trace header) and
`sw/runtime/common/perf.cpp` (names, printing) both use them, and `test_counter_names_agree` checks that the two name
tables agree. All class-8 counters are **device-wide** (every core returns the same value; the mechanisms of P5 act on
the shared L2 and DRAM, and per-core marks remain in the `PS_DEBUG_CLASS` report) and **cumulative since device
open** (like every MPM counter in SimX, errata 24), except `dxa_max_inflight_*`, which are maxima. `--perf=8` prints
them as `PERF: ps: name=value, ...` (one line per group) plus a `PERF: ps-derived:` line (mean DRAM read residency and
L2 hit rate per class with traffic). Classes are indexed by `VX_PS_CLASS_*` (`ord wgt kvp kvs dxa c5 c6 c7`), mark types
by `VX_PS_MARK_*` (`other prefill decode m3..m7`).

| group (base) | per | definition (where counted) |
|---|---|---|
| `l1l2_rd_bytes` / `l1l2_wr_bytes` (0 / 8) | class | requests delivered into an L2's core-side input ports × the L1 line (one block per request; 64 B in every config), reads / writes, class as it arrives on the link. Counted by a delivery callback on the L2's input channels (`Cache::ps_count_link_inputs`), so the L1/DXA→L2 link is counted also when the L2 is bypassed (then it is the L1→memory link). A write-through or write-back request carries one block with byte enables; it counts as one block (link occupancy, the same "one request = one block" rule as the DRAM bytes of §12.8). |
| `l2_acc` / `l2_miss` (16 / 24) | class | L2 bank accesses / misses, incremented at exactly the places the stock `reads`/`writes` and `read_misses`/`write_misses` are (MSHR chain joiners count as misses, the stock convention), so Σ_class equals the stock class-5 totals (tested); hits = acc − miss. 0 for a bypassed L2. |
| `l2_fill_demoted` / `l2_fill_bypassed` (32 / 40) | class | L2 fills inserted at low priority / not allocated: the hooks for mechanism D. **Always 0 in this patch** (no mechanism yet). |
| `dram_rd_bytes` / `dram_wr_bytes` (48 / 56) | class | the earlier DRAM byte counters (`DramSim::send_request`, one block per request). |
| `dram_occ` (64) | class | DRAM **occupancy integral**: at the start of every DRAM-model tick, the number of class-c requests *in the model* is added. A request is in the model from `send_request` (accepted into its channel queue) until its response callback returns true (a read: its data was handed to the memory crossbar; a write: all its transactions were accepted by Ramulator) — exactly the interval `DramSim::can_accept`/`queue_size` accounts for. It therefore spans (i) the wait in the wrapper's per-channel transaction queue `pending_txs` in front of Ramulator (unbounded at the configs, see `dram_bp`), (ii) Ramulator's controller queue, scheduling and DRAM timing, (iii) the fixed `VX_CFG_PLATFORM_MEMORY_LATENCY`, and (iv) for reads, any stall while the memory crossbar refuses the response (callback returns false). It is **DRAM-model residency, not device load latency**: the memory crossbar and the cache hierarchy in front of `Memory`, and the response path back to the core, are outside it. Mean occupancy of an interval = Δocc / Δcycles. |
| `dram_bp` (72) | class | channel-cycles in which a head request of class c could not enter the DRAM model because `DramSim::can_accept` refused it (`Memory` input). **Identically 0 in every / PS-BASE configuration**: `VX_CFG_PLATFORM_MEMORY_QUEUE_SIZE = 0` (`VX_config.toml:96`, no configs.yaml override) makes `can_accept` always true (`sim/common/dram_sim.cpp:335-336`), and every log has `dram_bp_* = 0`. Queueing in front of Ramulator then happens in the unbounded `pending_txs` queue, i.e. INSIDE `dram_occ` / `dram_rd_wait` / the histogram, and queueing in the memory crossbar before `Memory` (`bank_st` of MPM class 7) is in neither. So `dram_bp = 0` does **not** mean "no queueing before the controller"; the counter only becomes informative with a non-zero queue size (must separate the wrapper-queue and Ramulator parts itself if it needs the queue location). |
| `dram_rd_wait`, `dram_rd_done` / `dram_wr_wait`, `dram_wr_done` (80, 88 / 96, 104) | class | summed residencies (DRAM-model ticks) and number of completed reads / writes; mean wait = wait / done. Because occupancy is sampled at the start of each tick and a request resident for k ticks is counted in k samples, **Σ occ = Σ rd_wait + Σ wr_wait** whenever nothing is in flight (Little's law, L = λW, holds exactly in this discrete form; tested at the end of a run for every class with no request left in the model). **Write completions are incomplete (stock SimX artefact, errata 24 addendum).** The write-backs of a cache flush are sent to the DRAM model during the flush, but the flush loop stops when `SimPlatform` is idle, which does not cover DramSim's internal queues; the writes still waiting in `pending_txs` / `write_rsps` are then DISCARDED by `DramSim::Impl::reset()` at the start of the next launch (`ProcessorImpl::run()` → `reset()` → `SimPlatform::reset()` → `Memory::on_reset()` → `DramSim::reset()`), and after the last launch they are still queued when the host reads the counters. They count in `dram_wr_bytes` and (up to the reset / the read) in `dram_occ`, but never in `dram_wr_wait` / `dram_wr_done`. This happens in EVERY launch, not only the last: measured `ps_phase_rw -L 5` 16 write-backs per flush, 11 completed, in each of the 5 launches (80 sent, 55 done); `ps_twolaunch -L 2` 18 / 11 per launch (36 / 22); decode gqa REG-CFG (1 launch) 208 / 105; sgemm 256 / 153. Consequences: (1) `dram_wr_done` and `dram_wr_wait` undercount, and the shortfall grows with the number of launches (ps_llama: 14L + 2 launches per token); (2) Σ occ = Σ waits holds only for classes that do not write (and only over whole runs: a request in flight across a row boundary is split between rows by `dram_occ` but counted in one row by its wait), so **Little's law must not be used per row or per launch for any class that writes**; (3) **use `dram_wr_bytes` (issued) for write traffic, never `dram_wr_done`**, and do not use write residency in P5/P6. Functional results are unaffected (`Memory` writes the data into RAM when the request arrives, `sim/simx/mem/memory.cpp:105-113` pristine). Reads complete before a launch ends (the kernel waits for them), so the read counters are complete. A write's "residency" is its acceptance time (Ramulator reports no write completion, `dram_sim.cpp`), so it is not a write latency. |
| `marks` (112) | mark type | `ps_mark` work units, cumulative over launches, summed over cores (the per-launch per-core counters of §12.6 stay). |
| `dxa_bytes_s<k>` (120) | DXA stream | DXA GMEM read bytes issued (one L1 line each) by worker slot k = min(worker id, 3) of every DxaCore. A "stream" is a DXA worker (one transfer at a time); per-stream *classes* are mechanism C. Workers ≥ 3 share slot 3 (≤ 2 workers per DxaCore configs, `VX_CFG_NUM_DXA_CORES = max(1, cores/8)`). |
| `dxa_max_inflight_s<k>` (124) | DXA stream | **maximum** number of GMEM reads in flight (issued, response not yet arrived) of the slot's workers, max over sockets. MPM: since device open; trace rows: within the row (a window restarted at every row close at the *current* level, so a level held across a boundary appears in both rows). |
| `dram_rd_hist_<cls>_b<j>` (128 + 16·cls + j) | class × bucket | log2 histogram of read residencies: bucket 0 = [0, 2), bucket j = [2^j, 2^(j+1)) cycles, bucket 15 open-ended (≥ 32,768). |

*Why a log2 histogram for P99.* The P99 DRAM read wait per class is what mechanism B (two-class arbitration) must keep
bounded for demand classes. (What it measures: DRAM-model residency as defined in the `dram_occ` row — wrapper queue +
Ramulator + fixed latency + response hand-back stall — not the device's load-to-use latency.) An exact P99 needs every sample; a log2 histogram needs 16 counters per class, covers 1 to
32k cycles (the measured mean residencies are tens of cycles), merges across intervals by addition (so it obeys the
Σ-intervals rule like every other counter) and gives the P99 to within a factor of 2: `pstrace.hist_quantile` returns
the bounds [lo, hi) of the bucket that contains the 0.99-quantile. That resolution is enough to see a tail move between
configurations by ≥ 2×; finer claims need a finer patch.

*Not counted:* L3 (disabled in every configuration), per-class L1 traffic (the `PS_DEBUG_CLASS` report of §12.8
has L1 requests by class), per-core class traffic. The counters are plain integers updated from the single simulation
thread (SimX is built with `SIMX_MT` = 0, `sim/simx/types.h`); a multi-threaded SimX build would need atomics — never
used in PhaseShift.

### 12.10 Work-unit interval trace (`PS_TRACE`)
Env `PS_TRACE=<file>` (path of a CSV written by SimX), `PS_TRACE_UNIT=marks|instrs` (default marks),
`PS_TRACE_EVERY=<n>` (default 1). Read once when the device is opened; without `PS_TRACE` nothing is traced.

*Work counter.* `marks`: the per-launch `ps_mark` total summed over cores (all mark types). `instrs`: instructions
retired in the launch summed over cores (for kernels without marks).

*Row rule.* For each launch, row k (k = 1, 2, …) closes at the end of the first cycle at which the work counter is
≥ k·n; **the last row of the launch closes at the end of the launch instead** and so also holds the tail after the last
unit (epilogue, drain). A launch with W > 0 units therefore has exactly **⌈W / n⌉ work rows** (W = 0: one row for the
whole launch); if several boundaries are crossed in one cycle, each gets its own row at that cycle (the first holds the
deltas; the others have zero deltas, except that `dxa_max_inflight_*` repeat the current window maximum, because a max
window is the level at the close). Implementation: the row of the most recent boundary is held back; at the next boundary it
is written, at the end of the launch the tail is folded into it when no unit followed it (W a multiple of n), otherwise
the tail becomes the final partial row. After every cache flush (the runtime flushes after each launch) a `kind=flush`
row records the flush's write-backs and flush cycles (work columns repeat the last launch's W). Rows never overlap and
leave no gap: each holds the **deltas** since the previous row of the file (maxima: the maximum within the row), so
**Σ rows = the end-of-run class-8 totals exactly** (a telescoping sum; tested against the `--perf=8` report), and Σ of
the `cycles` / `instrs` columns = the global `PERF: instrs, cycles` line. **Kernel cycles of a launch = Σ `cycles` of
its `kind=work` rows**; the global PERF `cycles` also include the cycles of every flush walk (`kind=flush` rows; e.g.
238 + 3 cycles after the decode kernel at PS-BASE-S1-4c). Stock SimX always counted the flush ticks in the core cycle
counters (base tree too: the flush loops tick the whole platform), so every earlier PERF `cycles` number in this
project is consistent with this definition.

*Columns.* `launch, row, kind (work|flush), interval (k; 0 for flush), unit, every (n), work_end (k·n, or W for the last
row), work_at (counter value at the close), cycle (launch-local cycle at the close), cycles (Δ cycles), instrs
(Δ instructions)`, then all 256 class-8 counters by name. `scripts/pstrace.py check <trace> <log>` verifies the totals
and the row counts.

*What "the same work" means (alignment).* With unit = marks, interval k of a launch covers the k-th group of n
completed work units (tiles) under every configuration — the property (C_warmup), (schedule replay at work
counts) (interval costs) need. The row boundaries are therefore defined in work, not cycles: the same kernel
under REG-CFG and PS-BASE-S1-4c has identical `work_end` sequences (tested). Caveat: with several cores, *which* tiles
complete first can differ between configurations (completion order), so "interval k" is the same amount of work, not
necessarily the identical set of tiles; this is the multi-core alignment assumption that states and the earlier replay
validates. With unit = instrs, alignment across configurations holds only when they execute the same
instruction stream (same binary and the same NT/NW/cores); it is meant for unmarked kernels within one configuration.

*No timing effect.* The trace only reads state after a simulated cycle: a per-cycle comparison of the work counter
with the next boundary and, at a boundary, a snapshot of the counters. Traced and untraced runs of the same binary give
identical instrs, cycles and class-8 totals (`test_trace_no_timing_effect`). Wall-time overhead (measured, not a
pass/fail): `results/<run>/README.md` "Overhead".

### 12.11 `ps_mark` counts one unit per warp instruction (fix of errata 33b)
Patch 0002 counted a `ps_mark` write once per SFU-packet trace: with `NUM_SFU_LANES < NUM_THREADS` the SFU dispatcher
splits a warp instruction into one trace per packet that has an active lane (`dispatcher.cpp`, `sop` set only on the
first emitted trace) and `CsrUnit::process` runs once per trace. Patch 0003 performs the `ps_mark` write only on the
instruction's first trace (`trace->sop`), so **every warp CSR-write instruction counts exactly one unit**, whatever the
SFU width and the active lanes (the first emitted trace always contains the first active lane, and other CSRs are
unchanged). Checked with the test app `kernels/ps_marklanes` (unguarded marks from every lane, from the last lane only,
from the upper half of the lanes) on NT = 4 / SFU = 2 and NT = 8 / SFU = 2 (2 and 4 packets) and the control SFU = NT:
the count equals (#warps × marks per warp) in every case (`test_marks_one_per_warp_instruction`). The validated kernels
keep marking from one lane (`if (tid == 0)`, `test_marks_single_lane`), which is what makes one unit mean one
(CTA, tile). Limits of that static test: it checks uses of the `PS_MARK` macro only — a raw `csrw 0x8c1`
or `VX_CSR_PS_MARK` access outside `ps_hint.h` is not detected — and `threadIdx.x == 0` means one lane per CTA only for
1-D CTAs (a 2-D/3-D CTA has one such thread per row); neither occurs in the current kernels. `kernels/common/ps_hint.h`'s header comment still describes patch 0002 ("per active SFU-packet trace"); it
was left unedited because any edit in `kernels/common` changes every `ps_*` app-source hash (vxrun run keys) — this
section and errata 33(b) are the correction of record until `kernels/common` next changes for a functional reason.

## 13. Switching cost C_sw and break-even N_min
Code `scripts/csw.py` (pure functions + the CLI that writes the tables), tests `tests/test_csw.py`, traces
`experiments/<run>.yaml`, evidence `results/<run>/` (`csw.csv`, `warmup.csv`, `nmin.csv`, `medians.csv`). Two kinds of
number appear and every output row says which: **"analytical (RTL formula)"** (the flush, which SimX cannot time) and
**"measured (SimX trace)"** (C_warmup). Policy-only and LMEM rows are 0 **by construction** (no device cycles exist to
measure) and say so.

### 13.1 C_sw per change type
| change | C_sw (C_enter = C_exit) | host DCR writes | why |
|---|---|---|---|
| D insertion, B DRAM arbitration, C DXA depth (policy-only) | 0 device cycles | 1 (`INSERT_POLICY`) / 2 (`DRAM_SCHED`, `DRAM_AGING`) / 2 (`DXA_DEPTH0/1`) | between launches the host writes `[dcr_phase]` DCRs; SimX's `dcr_write` is host-side and untimed, and these knobs change no cache content. A mid-kernel switch by schedule replay latches a new `PhaseConfig` at a work count, also with no state to drain; that value is a design value measures it. |
| A1 LMEM size | 0 extra | 0 | a kernel-boundary (launch/build) change; every launch on this substrate already starts with cold, reset caches (§1), so the change adds nothing that every schedule, static ones included, does not already pay (the same reason as §5's `C_sw,i = 0` at launch boundaries). |
| A2 way mask (L1 or L2) | analytical flush (below) | 1 (`L1_WAYMASK` or `L2_WAYMASK`) | a way that leaves the allowed set may hold dirty lines, so the mask change goes through a flush; Vortex's flush cascades through the whole hierarchy, so an L1-only mask pays the same walk. |

**Flush formula (analytical (RTL formula)).** From `third_party/vortex/docs/designs/cache_subsystem.md` §6.5 and the
§6.7 table (lines 671–692):

`C_flush = Σ_levels walk(level) + N_dirty · t_wr`,  `walk = LINES_PER_BANK` (write-through) or
`LINES_PER_BANK · NUM_WAYS` (write-back),  `LINES_PER_BANK = size / (banks · ways · line)`.

- *Symbols.* LINES_PER_BANK = sets per bank (the walk counter's line range, `CS_LINE_SEL_BITS`); N_dirty = dirty
  64-B sectors written back; t_wr = cycles per written-back sector.
- *Why this shape.* Each bank's FSM walks its counter one entry per cycle (write-through: one entry per set, all ways
  cleared together; write-back: one per (way, set), because each dirty line emits its own memory request); all banks
  of a level walk concurrently, so a level costs one bank's walk. The levels add because a level's bank 0 forwards
  the flush downstream only after its own walk (§6.5). Dirty write-backs are "serialized through `mreq_queue`" (§6.7)
  and the walk pauses when that queue is almost full, so they add N_dirty·t_wr. Adding them to the walk (no overlap)
  is an **upper bound**; the 1-cycle FSM states are ignored.
- *Geometry* (resolved from the ps tree's generated `VX_config.h` with the configuration's flags by
  `results/<run>/l2_geometry.cpp`, output `l2_geometry.out`): PS-BASE-S1-4c L1D 16 KiB, 4 ways, 4 banks, 64-B lines,
  write-through → 16 sets/bank → **16 cycles**; L2 1 MiB, 8 ways, **8 banks** (L2_NUM_REQS = 4 sockets × 2 L1 memory
  ports), **128-B lines** (2 × MEM_BLOCK, sectored), write-back → 128 sets/bank → **1,024 cycles**; walk total
  **1,040 cycles**. REG-CFG: L1D has 2 banks (NT = 4) → 32; L2 identical → 1,056. The worked example of the TEST
  (1 MiB, 8-way, 4-bank, 128-B line → 256 sets × 8 = 2,048; a 64-B line would wrongly give 512 sets) is the
  document's own example, reproduced exactly by `test_worked_example_l2_1mb_8way_4bank_128B`.
- *t_wr.* No write-bandwidth microbenchmark exists. Primary: the measured stream bandwidth at PS-BASE-S1-4c,
  40.707 B/cycle (a **read** stream, used as the service-rate proxy) → 64 / 40.707 = **1.572 cycles/sector**; bound:
  the peak 64 B/cycle (2 channels) → 1.0. The SimX flush row's cycles are not used: SimX's flush is untimed per line and
  its write-backs are partly discarded at the next reset (§12.9).
- *N_dirty.* Two bounds and the measured end-of-kernel values: 0; all of L2 dirty (1 MiB / 64 B = 16,384 sectors); and
  the write-backs **issued** in the `kind=flush` rows of each traced launch (`Σ dram_wr_bytes / 64`, issued not
  completed, §12.9), which is the dirty content at the end of that kernel (164–1,438 sectors). The dirty content at a
  mid-kernel point is not observable in SimX and is bounded by the all-dirty case.

Results (PS-BASE-S1-4c, t_wr = 1.572): **1,040 cycles clean, 1,298–3,301 cycles at the measured end-of-kernel dirty
counts, 26,799 cycles all-dirty** (peak t_wr: 1,040 / 1,204–2,478 / 17,424). Against a stories15M decode step at
PS-BASE-S1-4c (7.38 M cycles, `tf6` run: 44,301,707 cycles / 6 positions) the worst case is 0.36 %.

### 13.2 C_warmup (intra-kernel, measured)
`C_warmup(k) = cum(k) − k · s`,  `s = (cum(b) − cum(a)) / (b − a)`,  `a = P·⌊R/2⌋`, `b = P·(R − 1)`, `R = ⌊W/P⌋`.

- *Symbols.* From the `PS_TRACE` of **one launch of one binary** (unit = marks, n = 1;: compare within the
  same binary only): cum(k) = launch-local cycle at which the k-th work unit completed (the close of the row with
  `work_end` = k); W = units of the launch; P = units completed per "round" = resident CTAs (4 at 4 cores, one CTA per
  core); s = steady-state cycles per unit; k = P by default (every resident CTA's first tile), 2P reported too.
- *Why this shape.* It is the definition (cycles of the first k units minus k × the steady-state per-unit cost of
  the same launch). Units finish in bursts of P (one per core within a few hundred cycles), so per-unit deltas are
  bimodal (one ~40 k-cycle gap, then three short ones); only round-aligned differences are stable, hence whole rounds.
  The steady window is the second half of the launch in whole rounds, **without the last round**, whose row also holds
  the epilogue and drain (§12.10). Requires R ≥ 3.
- *What it contains.* Everything the first round pays and later rounds do not: cold caches (instructions, L2 tags,
  DRAM rows) **and the per-CTA prologue** (loading q, initialising running max/sum, the first LMEM setup), which a
  CTA pays once. With one CTA per core (one wave), the two cannot be separated; launches with **two CTA waves**
  separate them: the second wave's first round has the same prologue but warm instruction caches, so
  `round1 − round(wave-2 first)` is the cold-launch part alone (`cold_minus_wave2_round` in `warmup.csv`).

Results (PS-BASE-S1-4c; `warmup.csv`):

| kernel (PS_HINTS=1 binary) | W | s (cyc/unit) | C_warmup(k=P) | C_warmup(k=2P) | share of kernel |
|---|---|---|---|---|---|
| decode gqa B2 H8 K2 D64 T128 | 16 | 10,804 | 10,264 | 10,229 | 5.6 % |
| same, T512 | 64 | 10,646 | 10,395 | 10,676 | 1.5 % |
| same, T1024 | 128 | 10,644 | 11,196 | 12,081 | 0.81 % |
| mixed c64 s0 n4 T256 (H8 K2 D64) | 76 | 15,713 | 187,941 | 248,746 | 11.7 % |
| decode B2 H8 K4 D32 T128 (trace, 2 waves) | 32 | 6,212 | 6,064 | 3,047 | 3.1 % |
| decode B4 H8 K2 D64 T1024 (trace, 2 waves) | 256 | 10,711 | 10,085 | 10,454 | 0.37 % |
| REG-CFG: decode B2 H8 K4 D32 T128 (trace, 2 waves) | 32 | 7,250 | 2,813 | 1,658 | 1.2 % |

Reading:
- Decode attention has a **T-independent first-round excess of ≈ 10.3–11.2 k cycles** (≈ one extra unit's cost), all
  in round 1 (k = 2P adds ≤ 0.9 k). Later rounds drift slowly downward (T1024: 43.5 k → 36–37 k cycles per round
  late in the launch), which the second-half window absorbs into s; that is why C_warmup grows slightly with T.
- The two-wave traces split it: B4 K2 D64 T1024, wave-1 round 52,930 vs wave-2 first round 48,407 → **≈ 4.5 k cycles
  cold-launch, ≈ 5.6 k per-CTA prologue**; B2 K4 D32 T128 (smaller tiles) −323 (cold part ≈ 0 within noise), REG-CFG
  +265. So most of the decode C_warmup is prologue, which a mid-kernel switch would **not** re-pay; using the whole
  C_warmup for a switch is conservative.
- **mixed_attn's -literal number is not a warm-up.** Its first rounds are prefill tiles (the first round of 4 prefill tiles takes 250,793 cycles
  at c = 64, vs s = 15,713 per unit) and its second half is mostly decode tiles, so the estimator compares two different kinds of work.
  It is reported as an upper bound only, never used as the C_warmup of a switch. A per-type estimator would need
  per-type rounds, which the co-running prefill and decode CTAs of the fused launch do not provide (phases are
  mostly serial in time, the round structure is irregular).
- A warm-up after a switch only exists where state is lost: after an A2 flush (caches cold) the decode/prefill units
  re-pay the cold-launch part; policy-only switches keep cache contents, so their C_warmup is 0.

### 13.3 N_min (E9)
`N_min = ⌈(C_enter + C_exit + C_warmup) / ΔC_saved⌉` (E9), in decode iterations, **time only** (energy:).

- C_enter = C_exit = C_sw of the change type (§13.1); C_warmup = 0 for policy-only and LMEM (no state lost / launch
  already cold), and for A2 the **largest measured decode C_warmup(k = P)** = 11,196 cycles (all of it charged, prologue
  included — conservative), with the mixed estimator (187,941) as a labelled upper-bound row.
- ΔC_saved is a mechanism result (P5/P6) and is not known yet, so `nmin.csv` gives N_min for savings
  f ∈ {0.1, 0.5, 1, 5, 10} % of a stories15M decode step at PS-BASE-S1-4c (T_step = 7,383,618 cycles) and the
  **break-even saving** `f* = (C_enter + C_exit + C_warmup) / (m · T_step)` at which N_min equals a median decode length
  m (N_min ≤ m ⇔ C/(f·T_step) ≤ m ⇔ f ≥ C/(m·T_step)).
- **Median decode length of Azure `conv`** (`medians.csv`, trace, `serving.traces.scale_lengths`): unscaled
  **129**; σ = (M_ref/L2_ref)/(M_tgt/L2_tgt) with the reference (h100-like, Llama-3.1-8B bf16, 131,072 B/token,
  32 MiB) and stories15M fp32 (13,824 B/token): PS-BASE-S1-4c (1 MiB L2) σ = 0.2963 → **38** (also 38 with the
  256-token context clip; 59.6 % of requests clipped, almost all by prompt truncation (O' changes only where σ·O > 255) — the median O' is unchanged);
  PS-BASE-S1-8c (2 MiB) σ = 0.5926 → 76.

| change type | C_enter + C_exit + C_warmup (cycles) | N_min at f = 0.1 % / 1 % / 10 % | f* for N_min ≤ 38 (σ, S1-4c) | f* for ≤ 129 (unscaled) |
|---|---|---|---|---|
| policy-only D/B/C | 0 | 0 / 0 / 0 | any f > 0 | any f > 0 |
| LMEM A1 | 0 | 0 / 0 / 0 | any f > 0 | any f > 0 |
| A2, measured end-of-kernel dirty | 2·3,301 + 11,196 = 17,798 | 3 / 1 / 1 | 0.0063 % | 0.0019 % |
| A2, all L2 dirty | 2·26,799 + 11,196 = 64,794 | 9 / 1 / 1 | 0.023 % | 0.0068 % |
| A2, all dirty + mixed estimator (upper bound) | 241,540 | 33 / 4 / 1 | 0.086 % | 0.025 % |

N_min = 0 means the mode pays from its first iteration. **G4 input:** with these costs, N_min exceeds the σ-scaled median
(38) only if a mode saves less than 0.023 % of a decode step (A2 worst case; 0.086 % with the mixed upper bound). G4
itself is decided at P7 with the measured ΔC_saved; this step supplies the costs, the medians and f*.

*Limits.* (1) The flush is analytical; SimX gives no check of it. (2) t_wr uses a read-stream rate. (3) C_warmup was
measured at PS-BASE-S1-4c (4 cores, one decode shape at three T, one mixed point) and on one REG-CFG trace; other
core counts change P and the wave structure. (4) The intra-kernel A2 switch it prices is not executable in this
project (rejects A2 in a schedule); A2 switches happen at launch boundaries, where on SimX they cost nothing beyond
the reset every launch pays — the analytical C_sw is the cost on hardware whose caches persist across launches
(RTL `persist_l2` sensitivity mode). (5) N_min uses whole decode steps as iterations and charges one
enter + one exit per mode instance.

## 14. Mechanism semantics (A, B, C, D)

### 14.0 Prelude: device KV-cache layout and the element-wise decode kernels

**KV-cache layout.** The device keeps two arrays, K and V, each laid out `[layer][b][pos][kv_head][head_dim]` (float32,
row-major). This is run.c's `key_cache[layer][pos][kv_dim]` (`run.c:254-257`, `loff = l·seq_len·kv_dim`) with a batch
axis of capacity `b_cap` inserted between layer and position, so `kv_dim = n_kv_heads·head_size` is the innermost
contiguous run:

`off(l, b, p) = ((l·b_cap + b)·seq_len + p)·kv_dim` (elements; K and V use the same offset in their own array).

Why this order: (i) with `b_cap = 1` it is byte-identical to run.c's cache, so the driver is checked against run.c
without a layout transform; (ii) one sequence's cache for one layer is contiguous (`seq_len·kv_dim` elements), so
decode attention for (b, l) streams a single region — the E3 per-sequence traffic is read as `T·kv_dim·4` contiguous
bytes, with no gaps between positions; (iii) the batch axis is a capacity, not the live batch size: a launch with B ≤
b_cap rows writes rows 0..B−1 (Tests use b_cap = B). Cost of the choice: a (b, kv_head) attention CTA reads
`head_dim` floats per position at stride `kv_dim`; when `head_dim·4` is smaller than a 64-byte line (stories260K:
head_dim 8 → 32 B), one line holds two heads and a per-head CTA uses half of each line it fetches. reports this.
The fixtures carry the offsets (`params.kv_cache_offset_elems`, `model/make_fixtures.py`), and `ps_rope` checks that it
writes exactly these slots and nothing else (`kv_untouched`).

**Element-wise kernels** (`kernels/{rmsnorm,rope,silu_gate,residual}`; shared host code `kernels/elt_common/elt_host.h`).
Every kernel takes B rows (batched decode: one row per sequence) and follows run.c's float32 formula and order of
operations (cited next to each kernel). Design choices: RMSNorm uses one warp per row with lane-strided partial sums
and a butterfly shuffle reduction (every lane ends with the bit-identical total, so every element is scaled by the
same factor); RoPE uses one thread per (row, pair) and evaluates `freq = 1/powf(10000, head_dim/head_size)`, `cosf`,
`sinf` on the device as run.c does (no host tables), then writes the rotated k and the v pair into the row's cache
slot (the fusion of `opgen` convention 5, §2); SiLU-gate and residual use one thread per element over the B·n
elements. All use grid-stride loops (tested with a reduced grid).

**Correctness metrics and tolerances** (each derived in the app's `main.cpp`; u = 2⁻²⁴; the derivations take the device libm's
`powf`/`sinf`/`cosf`/`expf` as ≤ 1 ulp, and each row states how much libm error the tolerance absorbs. *Measured* (SimX probe, ``; text updated): the device libm — musl from the toolchain's `libc32`,
evaluating in soft-float double precision (cost note below) — is **correctly rounded, ≤ 0.500 ulp**, on every input the fixtures
use and on probe grids (`powf(10000, e)` for all 24 exponents the kernels use: max 0.481 ulp; `sinf`/`cosf` on every fixture
angle and 2,048 points of [0, 600): 0.500 / 0.500 ulp; `expf` on 4,096 points of [−20, 20] and 2,048 random points of
[−87, 87): 0.500 ulp); NumPy float32 reached 2.25 ulp on the same inputs. So the ≤ 1-ulp inputs of the derivations hold with
margin; the measured errors remain the empirical check. Every app checks the device
against a float64 evaluation of the same formula from the same float32 inputs **and** against the NumPy float32
fixture):

| kernel | scale_i in rel_i = \|y_i − ref_i\| / scale_i | tol | worst-case bound (device vs float64) | why this scale |
|---|---|---|---|---|
| RMSNorm | max(\|ref_i\|, FLT_MIN) | max(1e-5, bound) per check | (m/2 + 5)u, m = ⌈n/NT⌉ + log₂NT: 2.5e-6 (n = 288), 8.3e-7 (n = 64); vs the NumPy fixture add its (n/2 + 5)u (BLAS depth ≤ n): 1.14e-5 at n = 288 (> 1e-5, so dev_vs_fixture uses the bound), 3.0e-6 at n = 64 | sum of squares has no cancellation; every later op is a product |
| RoPE (+KV) | max(hypot(v₀, v₁), FLT_MIN) of the input pair | 1e-4 | angle error ≤ pos·freq·(4.6e + 1.5)u + ulp(val)/2 (+ ~4u): 2.7e-5 (stories15M), 7e-6 (stories260K), with libm powf/cosf/sinf ≤ 1 ulp (measured ≤ 0.5 ulp; the bound holds up to ~8 ulp of powf error) | a rotation moves both components by ≤ \|Δangle\|·r; a single component can cancel to ~0, so \|ref_i\| is not a valid scale (measured: the plain element-wise error of correct runs reaches 5.6e-4 > 1e-4, while the pair-norm metric stays ≤ 7.7e-6) |
| SiLU-gate | max(\|ref_i\|, FLT_MIN) | 1e-5 (chosen) | ≈ 3 ulp = 1.8e-7 (libm expf ≤ 1 ulp, measured ≤ 0.5 ulp, + 4 roundings; holds up to ~160 ulp); vs NumPy ≈ 10 ulp | a product of factors; an error e in exp(−v) changes 1 + exp(−v) by ≤ e |
| residual | max(\|ref_i\|, FLT_MIN) | 1e-5 | u/2 = 3e-8; vs the fixture bit-identical (checked separately) | one IEEE add; exact under cancellation (Sterbenz) |

The KV write is checked bit for bit for V (a copy, tol 0) and with the RoPE metric for K; every other cache element
must keep a sentinel bit pattern. The FLT_MIN floor makes exact zeros (and subnormal results) require practically
exact agreement. A step that changes shapes beyond the stated bounds (e.g. ⌈n/NT⌉ > ~320 for RMSNorm, pos·freq ≳ 1000
for RoPE) must restate its tolerance.

**Measured cost note.** On the rv32imaf Vortex target, the device libm (musl: the toolchain's `libc32` ships
`musl-clang`; linked by `tests/regression/common.mk:95` as `-lm -lc`) computes `expf`, `sinf`, `cosf` and `powf` in
double precision, which has no hardware support there (no D extension), so the kernels link soft-float routines
(`__adddf3`, `__muldf3`, `__extendsfdf2`, … — `llvm-nm` kernel ELFs); `sqrtf` is the hardware `fsqrt.s`.
The result is ~760 instructions per element for SiLU-gate (one `expf`) and ~3,800–6,400 per RoPE pair (`powf` +
`cosf` + `sinf`), against ~10 per element for the residual add and 5–8 for RMSNorm (REG-CFG, `results/<run>/bench.csv`;
e.g. stories15M B = 4: SiLU-gate 1.59 M cycles and RoPE 1.66 M cycles vs RMSNorm 17 k and residual 8.7 k). This is a property of the toolchain's libm,
not of the algorithm; the MUFU extension (`vx_mufu.h`: ex2, sin(πx), cos(πx), off by default) or a float-only
exp/sincos would remove it. Which variant the earlier full-model runs use is that step's decision; it changes the
element-wise share of a decode step and must be stated with any per-op breakdown. ** uses this musl libm unchanged**
(the validated kernels, device code untouched; RoPE and SiLU-gate pay the soft-float cost in every per-op number of
§14.5).

**RMSNorm tolerance.** used tol = 1e-5 for every check, but its own worst-case
device-vs-fixture bound at n = 288 is (42 + 149)u = 1.14·10⁻⁵ > 1e-5 (the NumPy fixture's summation depth is only bounded
by m ≤ n, since BLAS `sdot`'s partial-sum order is not fixed); observed ≤ 3.2·10⁻⁷, so this was a paper gap. Since
`kernels/rmsnorm/main.cpp` uses tol = max(1e-5, that check's derived bound): dev_vs_f64 max(1e-5, (m/2 + 5)u),
fixture_vs_f64 max(1e-5, (n/2 + 5)u), dev_vs_fixture max(1e-5, sum of both) — 1e-5 wherever it is provably sufficient
(every check at n = 64, two of three at n = 288), the bound only at n = 288 device vs fixture (1.14·10⁻⁵). `-t` still forces
one tolerance (the liveness tests).

**RMSNorm cycles at the inline-FMA build.** The rmsnorm cycle counts quoted in the earlier results are old-build
numbers: `-fno-math-errno` (errata 30) removes its never-taken `call sqrtf` fallback. Fixed build, REG-CFG, same fixtures:
stories15M (n = 288) B = 1 16,784 → 16,593, B = 4 17,421 → 17,100; stories260K (n = 64) B = 1 4,989 → 4,804, B = 4 5,228 → 4,996
(1.1–4.4 % fewer; `results/<run>/README.md` note, `results/<run>/carry.csv`). The other element-wise kernels are unchanged.

**RoPE capacity stride b_cap > B.** `ps_rope -c b_cap` allocates the cache with a batch
capacity larger than the launch's B; the kernel must write row b at ((l·b_cap + b)·seq_len + p)·kv_dim and leave rows
B..b_cap − 1 of every layer untouched (the `kv_untouched` sentinel check covers them). Tested (`results/<run>/carry.csv`)
and end to end by `ps_llama -c 2` (§14.5).

** addendum (full text §18.7).** The soft-float cost above is repaired by opt-in variants that leave this section's kernels
untouched: `rope_table` + `rope_kv_tab` (angles computed once with the same musl expressions, so q, K and V are bit-identical to `rope_kv`; table
entries checked one by one against a correctly rounded host recomputation, against `rope_kv`'s own per-launch angles and across three
machines) and `silu_gate_expf_poly` / `silu_gate_expf_mufu` (numerics-changing; derived bound `e(x)·E/(1+E) + 2u` per element, accepted in P32 only under
the near-tie token rule). Default `ps_llama` behaviour (musl, per-launch angles) is unchanged; the table is reached with `-R table`, the exp
variants with `-E poly|mufu`. The soft-float figures quoted in this section remain the baseline "as shipped".

### 14.1 Softmax kernel `softmax_fixed`

**Semantics (max subtraction).** For a row x of length n: m = max_j x_j (the exact row maximum; the running maximum
starts at −∞), e_j = exp(x_j − m), S = Σ_j e_j, p_j = e_j / S — run.c's `softmax` and `model/ref_llama2c.py`
`Llama2cRef.softmax`. Subtracting the maximum is an overflow guard, not an approximation: softmax is shift-invariant,
every exponent is ≤ 0, so e_j ≤ 1 and S ≥ 1 (the maximum contributes exactly 1). The stock regression kernel starts
its running maximum at 0.0 (`tests/regression/softmax/kernel.cpp:16`, errata 7), so for an all-negative row it
subtracts 0 instead of the maximum; it also replaces exp by a 4-term polynomial (`tests/regression/softmax/common.h`)
evaluated in double precision. The decode-attention kernel (§14.2) uses the same semantics per key tile with the
online rescaling; its first-tile rescale factor exp(−∞ − m) = 0 relies on the same −∞ initialisation.

**Design** (`kernels/softmax_fixed/kernel.cpp`). One warp per row, lane l takes columns l, l + NT, … (a warp load
touches consecutive floats); three passes (max; exp, store, sum; scale by 1/S); warp reductions with the butterfly
shuffle (`warp_ops.h`, the `lanes` pattern). p_j = e_j · (1/S): one division per row (one extra rounding,
counted below). Each lane sums in blocks of 32 elements before adding a block to its total, so every term passes
through at most h = 31 + ⌈⌈n/NT⌉/32⌉ + log₂NT additions (65 at n = 4096, NT = 4) instead of ⌈n/NT⌉ − 1 + log₂NT (1025).

**Exp** (modeling choice; `warp_ops.h`). The device libm is musl, whose `expf` evaluates in double precision; rv32imaf
has no D extension, so a call runs soft-float double routines: **≈ 752 instructions and ≈ 510 cycles per element** (exp probe at REG-CFG) (exp probe;
found the same for expf/sinf/cosf/powf, §14.0). A GPU's non-SFU exp is a single-precision polynomial, so the
default kernel uses `ps_expf`: Cody-Waite reduction k = round(x·log₂e), r = x − k·ln2_hi − k·ln2_lo (fdlibm's split),
e^r by the degree-7 Taylor polynomial (truncation ≤ 0.09u on |r| ≤ ln2/2), scaling by 2^k in two steps so subnormal
results are rounded rather than flushed; inputs clamped to [−110, 88.8]. Measured on the device over 65,541 points in
[−104, 89]: **max 0.864 ulp on that probe grid** (an independent 8 M-point random sample reached
0.93 ulp; asserted ≤ 2 ulp by the TEST), 36 instructions (25 cycles at REG-CFG) per element including
the load/store. The MUFU variant uses `vx_expf_fast(x) = ex2(x·fl(log₂e))`: 2 ulp from ex2 plus the prescale, whose
rounding and the constant's own representation error (fl(log₂e) is 0.224u below log₂e) give a relative error up to
(1.224|x| + 4)u — measured max 62 ulp at x = −81.8 (relative 3.8·10⁻⁶), within that allowance at every point; subnormal results are
flushed to zero.

**Correctness metric and tolerance.** rel_j = |p_j − ref_j| / max(ref_j, τ), τ = 2⁻¹⁰⁰, against a float64 softmax of the
same float32 inputs. Elements with ref_j ≥ τ are checked relatively; smaller ones absolutely (|p − ref| ≤ tol·τ), which
covers float32's subnormal range and MUFU's flush to zero. Worst-case bound (first order, u = 2⁻²⁴; added
the third term below, which had left out): with w_k = e_k/S and the computed e_k(1 + ε_k),
p̂_j/p_j − 1 = ε_j − Σ_k w_k·ε_k − ε_sum + ε_rcp + ε_mul, where |ε_k| ≤ (1 + c)|d_k|u + E_exp (rounding of d_k = x_k − m
shifts the exponent by |d_k|u; E_exp = 4u for ps_expf (≤ 2 ulp), 2u for musl; MUFU: E_exp = 4u and c = c_L = 1.224 for the
prescale, otherwise c = 0), |ε_sum| ≤ h·u and |ε_rcp| + |ε_mul| ≤ 2u. The other elements' errors reach p_j through S:
|Σ_k w_k·ε_k| ≤ (1 + c)·W(n)·u + E_exp with W(n) = max Σ_k w_k·|d_k| over rows of length n. ∂/∂t_k = 0 forces every
non-maximum element to the same depth t = 1 + W, so W(n) = max_t (n − 1)·t·e^{−t} / (1 + (n − 1)·e^{−t}) = 2.31, 3.34, 4.44,
5.60 at n = 64, 256, 1024, 4096 (0 at n = 1; checked against a random search). Over the checked elements p_j ≥ τ and S ≥ 1
imply |d_j| ≤ 100 ln 2 = 69.3, so bound = γ((1 + c)·69.3 + E_exp + (1 + c)·W(n) + E_exp + h + 2): for ps_expf
6.8·10⁻⁶ (n = 1) to 8.94·10⁻⁶ (n = 4096, NT = 4) — **the rel 1e-5 is justified and used** for the default kernel; for
MUFU 1.18·10⁻⁵ (n = 1) to 1.44·10⁻⁵ (n = 4096) > 1e-5, so its tolerance is its own bound (deviation).
Comparisons against the NumPy float32 `probs` fixtures add NumPy's bound γ(69.3 + 8 + W(n) + 8 + n + 1). tol = max(1e-5, bound) is printed in every PSCHECK line. Observed
maxima are far below: ≤ 1.2·10⁻⁶ for the software exp and ≤ 3.5·10⁻⁶ for MUFU over every TEST case (`results/<run>/summary.csv`).

**Stock kernel on the bug input** (report-only runs, `results/<run>/summary.csv` kind `stock`, 8 rows × 64):

| input (8 × 64) | stock, verbatim (max = 0, polynomial exp) | stock with accurate exp (max = 0) | fixed (`warp`) |
|---|---|---|---|
| U[0, 1) (the stock app's own inputs) | wrong: rel 1.6·10⁻² | correct: 2.5·10⁻⁷ | correct: 1.4·10⁻⁷ |
| U[−30, −1) (all negative) | wrong: rel 5.6·10¹¹ (p = 0.073 where 1.3·10⁻¹³ is correct) | correct: 2.5·10⁻⁷ | correct: 1.1·10⁻⁶ |
| U[−120, −100) | wrong: rel 3.3·10⁷ | wrong: rel 1.05 (subnormal exps) | correct: 2.0·10⁻⁷ |
| U[−200, −150) | wrong: rel 1.2·10²⁰ | wrong: 512/512 NaN (0/0) | correct: 1.4·10⁻⁷ |

So the stock kernel as shipped is wrong on all-negative rows — and on any row whose spread exceeds ~1, including its own
U[0, 1) inputs at 1.6·10⁻² relative error (its CI test passes only because the host reference uses the same polynomial
and the same max = 0). With an accurate exp the max = 0 initialisation alone is harmless for moderate negative logits
(shift invariance) but fails once every exp(x_j − 0) is subnormal (x_j < −87.3: precision loss) or zero (x_j < −103.9:
0/0 = NaN). The −∞ initialisation removes both.

**Cost** (`results/<run>/bench.csv`, REG-CFG, 64 rows; cycles of the single launch ÷ elements):

| kernel | n = 64 | 256 | 1024 | 4096 | marginal (1024 → 4096) |
|---|---|---|---|---|---|
| `warp` (ps_expf), REG-CFG | 20.5 | 19.0 | 18.7 | 18.7 | 18.65 cycles, 26.7 instrs / element |
| `mufu`, REG-CFG + MUFU | 6.5 | 5.5 | 5.2 | 5.2 | 5.15 cycles, 7.5 instrs / element |
| `libm` (musl expf), REG-CFG | | 507.9 | | | (748.8 instrs / element at n = 256: 27× the software exp) |

The MUFU extension cuts softmax from 18.7 to 5.2 cycles per element (3.6×); the exp alone (probe: one exp plus a load
and a store per element) costs 25.3 cycles with ps_expf, 7.8 with MUFU and 510 with musl's expf. The MUFU build
does not change the software kernel (identical cycles on both builds). At the stock app's shape (64 × 64, U[0, 1)):
stock app 1,218,463 cycles, the verbatim reproduction 1,218,045 (instructions within 0.01 %: the reproduction is
faithful), the fixed kernel 83,892 (14.5× fewer), the stock kernel with ps_expf 78,398 (one thread per row is
competitive at 64 rows, where it fills the 64 lanes of REG-CFG; the warp-per-row kernel is the one that scales to long
rows and few rows).

**Addendum (2026-11-06): costs at the inline-FMA build (errata 30).** The text and table above were measured on a
build in which every `__builtin_fmaf` of `ps_expf` was a call to musl's `fmaf` (30 call sites). adds
`-fno-math-errno` to every generated app Makefile, so the 8 FMAs of `ps_expf` are inline `fmadd.s`. Outputs are
bit-identical (all 47 batch cases: equal output digests on the old and the fixed build, and error statistics equal to
the validated logs, `results/<run>/ab.csv`, `vs_old.csv`); the error analysis above is unchanged. New costs
(`results/<run>/bench_p24.csv`, REG-CFG, 64 rows; old values in parentheses):

| kernel | n = 64 | 256 | 1024 | 4096 | per element at n = 4096 |
|---|---|---|---|---|---|
| `warp` (ps_expf), REG-CFG | 12.7 (20.5) | 11.5 (19.0) | 11.1 (18.7) | 11.1 (18.7) | 11.09 cycles, 17.2 instrs (18.66, 26.8) |
| `mufu`, REG-CFG + MUFU | 6.5 | 5.5 | 5.2 | 5.2 | unchanged (no FMA calls in the MUFU path) |
| `libm` (musl expf) | | 507.8 | | | unchanged (soft-float double; the call is to expf) |

The exp probe: ps_expf **16.4 cycles, 22.8 instructions per element** including the load and store (was 25.3 and 36),
MUFU 7.8 and musl 510 unchanged. So the MUFU extension now saves 11.1 → 5.2 cycles per element, **2.1×** (was 3.6×):
part of the old MUFU advantage was the fmaf call overhead of the software exp. At the stock app's shape (64 × 64,
U[0, 1)) the fixed kernel takes 51,819 cycles (was 83,892), **23.5× fewer** than the stock app's 1,218,463 (was 14.5×);
the stock kernel with ps_expf (one thread per row) takes 42,131 (was 78,398) and is still the faster one at 64 short rows.

### 14.2 Fused GQA decode attention `decode_attn_gqa`

**Operation and layout.** One decode step of one layer: for sequence b and query head h,
o[b][h] = softmax_t(q[b][h]·K[b][t][h / kv_mul]·scale)·V[b][t][h / kv_mul] over t < T, scale = 1/sqrtf(D) (run.c divides by
`sqrtf(head_size)`). q and o are [B][H_q][D]; K and V are [B][Tmax][H_kv][D], i.e. the §14.0 cache of one layer (passes the
layer base `K + l·b_cap·seq_len·kv_dim` and Tmax = seq_len). QKᵀ, softmax and PV run in one launch; scores never leave the CTA
(opgen convention 4, §2).

**Work decomposition** (`kernels/decode_attn_gqa/kernel.cpp`). A CTA is NT·NW threads = all warps of one core, so one CTA is
resident per core and owns that core's LMEM. Three variants share one body:
- `gqa_shared_kv` (`-v gqa`): one CTA per (b, KV head), serving its G = kv_mul query heads; each K and V element is loaded once
  per step and used G times from registers — the access pattern E3/E5 assume (KV bytes per step = 2·B·T·H_KV·D·q).
- `naive_per_qhead` (`-v naive`): the same code with G = 1, one CTA per (b, query head); each KV head is streamed by kv_mul
  CTAs (kv_mul× the E3 KV loads; whether they reach DRAM depends on L2 reuse — measured below).
- `gqa_dxa_dbuf` (`-v dxa -p depth`): `gqa_shared_kv` with each key tile's K and V rows copied into LMEM by the DXA engine
  (one 4-D descriptor each over [B][Tmax][H_kv][D]; tile = D × Tt rows of one head; rows ≥ T are out of range, zero-filled and
  never used), `depth` stages: tile j + depth − 1 is issued while tile j is computed, completion through `expect_tx` on the
  stage barrier (`vx_barrier.h`), one issuer warp (the `sgemm2_dxa` pattern). depth 1 = copy, then compute. This is the
  software KV-prefetch baseline (R§4) and mechanism C's zero-hardware layer.

**Online softmax** (single pass, so T is not bounded by LMEM). Per query head a running max m (from −∞, §14.1), sum l and
accumulator acc[D]; per key tile of Tt positions: scores s_t, m' = max(m, maxₜ s_t), c = exp(m − m'), p_t = exp(s_t − m'),
l' = l·c + Σp_t, acc' = acc·c + Σp_t·v_t; o = acc/l after the last tile. This is the same function as run.c's three-pass
softmax (shift invariance: the rescale by c makes every term relative to the final max), with a different rounding sequence.
Per tile four phases separated by CTA barriers: S (thread per key position: its G scores; q from LMEM), M1 (thread per head:
tile max, m', c), M2 (thread per (head, position): p), V (thread per output column d: G accumulators in registers, each V
element loaded once; threads h < G update l). `ps_expf` (§14.1) for every exp. LMEM per CTA (`common.h`): q[G][D], acc[G][D],
p[G][Tt], 3G statistics, plus 2·depth·Tt·D for the DXA stages. Tt = 32 for gqa/naive; for dxa the host picks the largest
Tt ≤ 32 that fits the 16 KiB LMEM: (16, 2, 128): Tt = 7 / 3 / 1 at depth 1 / 2 / 4; (8, 2, 64): 27 / 13 / 6.

**Correctness metric and tolerance.** rel_i = |o_i − ref_i| / S_i, S_i = Σ_t p_t·|v_t[d]| (§2 consequence 6 generalised: the
natural magnitude of o[h][d] = Σ_t p_t·v_t[d]), against a float64 attention of the same float32 inputs (and the NumPy float32
`out` fixtures). Bound per head (u = 2⁻²⁴, first order; corrected):
rel ≤ 2E_s + 2(s_max − s_min)·u + γ(n_acc + n_l + 10), n_acc = T + n_tiles, n_l = Tt + n_tiles, with
E_s = maxₜ[γ_D·scale·Σ_d|q_d·k_td| + 3u·|s_t|] the absolute error of a logit (a D-long dot product, plus fl(sqrtf(D)), its
reciprocal and the multiply). An error ε_t in a weight p_t moves o_i by at most Σ_t p_t|ε_t||v_t| + |o_i|·|Σ_t w_t ε_t| ≤
2·max|ε|·S_i (numerator and denominator, |o_i| ≤ S_i); this doubles E_s, the rounding of s_t − m (|s_t − m|u ≤ (s_max − s_min)u)
and ps_expf's 4u. The summation chains: acc is **one** sequential FMA chain across all tiles (phase V: a = acc·c, then nt FMAs per
tile), so the first key's term passes through T + n_tiles roundings; l sums each tile from 0 and folds it in (l·c + sum),
Tt + n_tiles. The rescale factor c multiplies acc and l alike, so its own error cancels in acc/l; +10 covers the exp terms and
the final division. [Carry-over (2026-11-06), text only, code unchanged: the cancellation holds only
for a single tile. c_j rescales only the contributions of tiles < j, so an error in c_j perturbs a subset of the weights and
adds up to 2·4u per max-raising tile, ≤ 8u·(n_tiles − 1); the argument roundings of c telescope into the existing
2(s_max − s_min)u term, and ps_expf(0) = 1 exactly. The corrected bound is γ(n_acc + n_l + 8(n_tiles − 1) + 10). The published
bound was therefore at most stricter than intended in the runs where it exceeded 1e-4, never looser, and every run met
1e-4 directly; the `main.cpp` comment and the host bound are to be fixed at the next code touch.] (Attempt 1 wrote γ(2(Tt + 2·n_tiles) + 10) and 1×(s_max − s_min)u, which undercounts the acc chain: e.g.
(16, 2, 128), B = 4, T = 1024 gave 7.0·10⁻⁵ where the corrected bound is ≈ 1.25·10⁻⁴;.) The bound grows with
T and with the logits' magnitude, as for any float32 implementation, so the host computes it per case and uses
tol = max(1e-4, bound); in addition every test asserts the criterion itself (rel ≤ 1e-4, PSINFO `within_plan_tol`).
Every first-pass run met 1e-4 outright (128/128 TEST logs, 102/102 batch logs); the largest observed error was 9.4·10⁻⁷ vs float64
and 1.15·10⁻⁶ vs the NumPy fixtures, ~100× below the bound (roundings are not aligned in practice).

**Results** (`results/<run>/bench.csv`; REG-CFG = 4 cores, L2 on; cycles of the single launch; DRAM bytes = memory reads × 64 B,
perf class 7; E3 = 2·B·T·H_kv·D·4):

| (H_q, H_kv, D) | B | T | gqa cycles | naive / gqa | gqa DRAM / E3 | naive DRAM / E3 |
|---|---|---|---|---|---|---|
| (16, 2, 128) | 1 | 17 / 256 / 1024 | 0.38 M / 3.89 M / 15.5 M | 0.73 / 0.75 / 0.77 | 1.85 / 1.057 / 1.015 | 1.86 / 1.057 / **2.016** |
| (16, 2, 128) | 4 | 17 / 256 / 1024 | 0.77 M / 7.80 M / 31.0 M | 1.46 / 1.49 / 1.53 | 1.63 / 1.045 / 1.011 | 1.57 / 1.039 / **2.012** |
| (8, 2, 64) | 1 | 17 / 256 / 1024 | 0.12 M / 1.16 M / 4.61 M | 0.70 / 0.70 / 0.70 | 1.89 / 1.059 / 1.016 | 2.02 / 1.068 / 1.019 |
| (8, 2, 64) | 4 | 17 / 256 / 1024 | 0.23 M / 2.33 M / 9.27 M | 1.40 / 1.39 / 1.38 | 1.51 / 1.034 / 1.010 | 1.43 / 1.029 / 1.009 |
| (8, 4, 8) | 4 | 512 | 1.72 M | 1.38 | 1.033 | 1.030 |

- **KV traffic:** gqa reads E3 + 1–6 % at T ≥ 256 (the rest is q, out, code and partial lines; at T = 17 the fixed q/out/code
  bytes dominate). naive reads 2.0× E3 at (16, 2, 128), T = 1024: the kv_mul = 8 CTAs of a KV head run in 8/4 = 2 *waves* on the 4 cores;
  CTAs of the same wave share lines through the L2, but one KV head's K + V (1 MiB per sequence) does not survive in the
  1 MiB L2 until the next wave, so each wave re-fetches it. The factor is set by CTA waves per KV head (kv_mul / cores here),
  not by kv_mul, and changes with the core count (PS-BASE 8 and 16 cores). At T ≤ 256, or for (8, 2, 64) (512 KiB per
  head), the L2 absorbs the re-reads (≤ 1.07× E3). The stories260K shape (head_dim·4 = 32 B < 64 B line, §14.0) shows no extra DRAM traffic (1.03×): the
  other KV head's half-lines are fetched by the CTA next to it and hit in L2.
- **Cycles: not memory-bound; in-core latency/issue-bound.** Scope: REG-CFG = 4 cores × 4 lanes against 64 B/cycle DRAM, an
  FPU ridge of 4 × 4 × 2 FLOP per FMA = 32 FLOP/cycle over 64 B/cycle = 0.5 FLOP/B (far more bandwidth per FLOP than a GPU;
  unmeasured, measures it); this scalar fp32 kernel with 16 threads per CTA; T ≤ 1024, B ≤ 4, the shapes above. It is
  not a statement about decode attention in general. Evidence: (i) **the FPU is not the limit**: 33.5 MFLOP in 31.0 M cycles
  is ~3 % of the 32 FLOP/cycle peak, so "compute-bound" would be wrong. The the earlier counters for gqa (16, 2, 128),
  B = 4, T = 256 (identical cycles; `build/tmp/p25_check/`) show the scheduler idle 65 % of cycles, scoreboard stalls 28 %,
  operand stalls 14 %, load latency 12.6 cycles and FPU 48 % of instructions (perf class 1), an L1 read hit rate of 79 %
  (class 4) and LMEM bank stalls at 58 % utility (class 7): in-core latency and issue limits (dependent FMA chains, LMEM
  operands, barriers) with only 4 warps per core to hide them. These counters alone do not split the idle time between
  memory, FPU latency and barriers. (ii) **The sound memory evidence is the equal-tile DXA result** below: with 64 KiB LMEM
  and Tt = 16 the dxa variant executes the same instructions as gqa (44.71 M vs 44.72 M) with K/V served from LMEM and
  prefetched ahead, and is only 2.3 % faster, which bounds the exposed global-memory latency of this kernel at a few percent.
  (iii) The DRAM *bandwidth occupancy* is 8.48 MB / 64 B per cycle ≈ 0.13 M of 31.0 M cycles (0.4 %); this is only a lower
  bound on memory time (it cannot see latency stalls) and is quoted as occupancy, not as stall time. Hence naive is *faster*
  than gqa at B = 1 (0.70–0.77×): gqa has only B·H_kv = 2 CTAs for 4 cores, naive has 16; at B = 4 (8 CTAs) gqa is faster
  (naive 1.38–1.53×). The KV-sharing benefit shows in traffic, not in cycles, at this scale.
- **DXA prefetch does not help, because exposed memory latency is small for this kernel**. At equal tile
  size the DXA variant is 0.4–2.3 % faster than gqa (overlap study: (8, 2, 64) Tt = 6: 0.983–0.988; (16, 2, 128) Tt = 1: 0.996;
  with 64 KiB LMEM: Tt = 16 at depth 1 / 2: 0.988 / 0.977, and depth 4 at Tt = 12: 0.982; Tt = 13 would also fit, 62,016 B,
  while 14 needs 66,112 B), and depth 1, 2, 4 give the same cycles (± 0.5 %). With the 16 KiB default LMEM the staging buffers
  force small tiles, and the per-tile barrier cost dominates: dxa/gqa = 1.05 / 1.13 / 1.80 at depth 1 / 2 / 4 for (8, 2, 64)
  and 1.62 / 3.08 / 8.2 for (16, 2, 128) (T ≥ 256). Consequences: (i) software KV prefetch is LMEM-capacity-bound at D = 128
  with 16 KiB (one K+V position = 1 KiB in fp32), a direct link to mechanism A1 (LMEM size); (ii) within the scope above, the
  KV-bound decode region of §4 is KV-bound in *bytes* but not in *cycles*; mechanisms that act on memory (B, C, D) can only
  show cycle gains where memory stalls are exposed (larger L2 pressure, more cores per DRAM channel, lower-precision KV, a
  kernel with less in-core overhead, or faster compute such as the TCU), which must establish before P5 relies on them.
- **Group sizes.** The kernel supports G ∈ {1, 2, 4, 8} (template dispatch). That covers every shape, but it refuses,
  before launch, models with g = 7 (Qwen2.5-7B), 5 (Qwen2.5-32B) and 3 (SmolLM2-135M). must add those
  instantiations (or a generic-G path) before simulating real-g shapes. The "(16, 2, 128)" test shape has g = 8, which matches
  neither Llama-3.1-8B (4) nor Qwen2.5-7B (7).

**Addendum (2026-11-06): results at the inline-FMA build (errata 30).** Every number above (cycles, instruction
counts, ratios, counters) was measured on a build in which each `__builtin_fmaf` (251 call sites: the QKᵀ and PV FMAs, the
l update and ps_expf) was a call to musl's `fmaf`. adds `-fno-math-errno` to every generated app Makefile (one
shared line, `scripts/install_kernel_apps.sh`); the FMAs are inline `fmadd.s` and the outputs are **bit-identical**
(19 runs on both builds with equal output digests, 102 fixed-build batch runs and the TEST grid with error statistics
equal to the validated logs; `results/<run>/`). The text above is kept as the record of the old build; where it differs,
these numbers supersede it (`results/<run>/bench_p25.csv`, REG-CFG, perf class 7):
- **Cycles.** gqa: (16, 2, 128) **2.33–2.40× fewer** (B = 4, T = 1024: 31.0 M → 13.08 M cycles, 43.87 M → 18.19 M
  instructions); (8, 2, 64) 1.96–2.00× (9.27 M → 4.63 M); (8, 4, 8) 1.55×. naive gains only 1.39× (one FMA per loaded K/V
  element, G = 1, so the call overhead was a smaller share). DRAM reads change by ≤ 1.6 % at T ≥ 256 (DRAM/E3 as in the
  table above to ±0.02) and drop 4–13 % at T = 17, where the smaller kernel code (fewer instruction-line fetches) is a
  visible part of the traffic.
- **naive vs gqa changes sign at B = 1.** naive/gqa is now 1.21–1.32 at (16, 2, 128), B = 1 (was 0.73–0.77: naive was
  *faster*), ≈ 1.0 (0.98–1.005) at (8, 2, 64), B = 1 (was 0.70), and 1.6–2.6 at B = 4 (was 1.38–1.53). The old "naive is
  faster at B = 1 because gqa has only B·H_kv = 2 CTAs for 4 cores" was largely the call overhead: with inline FMAs, gqa's
  G-fold register reuse outweighs its idle cores at (16, 2, 128); the CTA-count effect remains visible as parity at (8, 2, 64).
- **Counters** (gqa (16, 2, 128), B = 4, T = 256, perf classes 1 and 4; `results/<run>/rows.jsonl` kind `counters`; old
  values from `build/tmp/p25_check/` in parentheses): instructions 4.59 M (11.01 M), cycles 3.31 M (7.80 M), IPC 1.39 (1.41),
  scheduler idle **65 % (65 %)**, scoreboard stalls **67 % (28 %)**, operand stalls 4 % (14 %), FPU 25 % of instructions
  (48 %), branches 0.16 M (2.34 M: the call/return pairs), load latency 14.9 (12.6) cycles, L1 read hit 76 % (79 %). The
  removed instructions were call overhead, so the remaining ones wait more on dependencies: the kernel is still **in-core
  latency/issue-bound**, now dominated by scoreboard (dependency) waits; FPU use 33.5 MFLOP / 13.08 M cycles = 2.6 FLOP/cycle,
  8 % of the 32 FLOP/cycle peak (was 3 %).
- **Equal-tile DXA (the memory evidence).** With 64 KiB LMEM, Tt = 16, dxa executes the same instructions as gqa
  (18.92 M vs 18.86 M, +0.3 %) and is **4.2 % (depth 1) / 6.2 % (depth 2) faster** (was 1.2 % / 2.3 %); depth 4 at Tt = 12:
  3.4 % (was 1.8 %). At the 16 KiB default LMEM's tiles the DXA staging now *costs* cycles even at equal tile size:
  (8, 2, 64) Tt = 6 dxa/gqa = 1.029–1.037 (was 0.983–0.988), (16, 2, 128) Tt = 1: 1.007 (was 0.996); and with LMEM-sized
  tiles dxa/gqa = 1.05 / 1.14 / 1.71 at depth 1 / 2 / 4 for (8, 2, 64) and 1.45–1.48 / 2.73–2.78 / 7.2–7.4 for (16, 2, 128), T ≥ 256
  (was 1.05 / 1.13 / 1.80 and 1.62 / 3.08 / 8.2).
- **DRAM-latency sensitivity (new, labelled; the method, §14.3).** `VX_CFG_PLATFORM_MEMORY_LATENCY` = 0 / +50 / +100
  cycles on gqa (16, 2, 128), B = 4 at REG-CFG (`results/<run>/dramlat.csv`): T = 1024: 13.08 M / 14.05 M / 14.98 M cycles
  (1.075× / 1.146×) at a measured average DRAM latency L = 37.3 / 87.4 / 136.8 cycles; T = 256: 1.074× / 1.143×. Instructions
  and DRAM reads are identical across the three. Read-out (modeling choice): the slopes are 19,531 then 18,784
  cycles per cycle of L (mildly **concave**), so the first-order share s·L₀/cycles₀ = **5.6 %** is a lower estimate and the
  quadratic through the 3 points gives **5.8 %** (both T). At +100 fixed cycles the same accounting gives ≈ **17 %**.
  Extrapolating to L = 0 is an assumption (a zero-latency DRAM is not physical); the shares are sensitivities.
- **Re-stated memory-share conclusion (scope as above: REG-CFG, this scalar fp32 kernel, T ≤ 1024, B ≤ 4).** The errata-30
  overhead did bias the memory share low, by about 2×: exposed memory latency is now ≈ 4–6 % of cycles by two independent
  measures (equal-tile DXA gain at 64 KiB, and the DRAM-latency share ≈ 5.6–5.8 % at the default DRAM model), up from
  "a few percent" (≤ 2.3 %). DRAM *bandwidth* occupancy is 8.48 MB / 64 B per cycle ≈ 0.13 M of 13.08 M cycles = **1.0 %**
  (was 0.4 %; a lower bound on memory time, not stall time). The kernel is still **not memory-bound**; the headline
  consequence for mechanisms B/C/D stands: they can show cycle gains here only where memory stalls are exposed, which at the
  default DRAM model is ≈ 6 % for this kernel (≈ 17 % at +100 cycles; the share depends on the DRAM latency picks for
  PS-BASE, as for the earlier GEMV, ≈ 13–21 %).
- **Error bound (done).** The host now includes the rescale term:
  bound = 2E_s + 2(s_max − s_min)u + γ(n_acc + n_l + 8(n_tiles − 1) + 10) (`kernels/decode_attn_gqa/main.cpp`, header comment
  rewritten; the "cancels in acc/l" sentence is replaced by the argument in the bracket above). The bound only grew, so tol
  can only be looser where it exceeds 1e-4; every run also met the 1e-4 itself (`within_plan_tol=1`).

### 14.3 Linear layers `ps_linear` / `ps_gemv_f32`

Tolerances, padding, FLOP accounting, weight layout and the scaled Llama shapes: §2 consequence 7. Kernel paths:
decode B = 1 → `ps_gemv_f32` (`rows` = thread per row, `lanes` = warp per row); small M = B ∈ 2…16 →
`ps_linear -v small` (thread per output feature, ≤ 8 fp32 accumulators per pass over its weight row, so weights are
streamed once per launch for M ≤ 8); prefill → TCU WGMMA + DXA GEMM (the stock `sgemm_tcu_wg_dxa` device code; timings
from the stock app, real-weight correctness from `ps_linear -v tcu`, same instruction count). Measured at TIER-CFG
(4 cores, NT = 32; `results/<run>/layer.csv`, `bound.csv`), against the earlier *unmeasured* vortex-S1-4c placeholders
(FPU 256, TCU 2,048 FLOP/cycle, DRAM 64 B/cycle; ridges 4 and 32 FLOP/B):
- **Regimes.** Whole layer, best path: M = 1 → 7.0 FLOP/cycle (llama8b_s8, GEMV `lanes`), M = 32 → 168, M = 256 → 187
  (TCU). The FPU small-M kernel saturates at ≈ 7.8 FLOP/cycle (3 % of the FPU peak); the TCU with M zero-padded to 32
  is faster from M = 2 (10.5 vs 3.1). **Caveat (datatypes differ):** the TCU path reads fp16 weights (half the bytes
  of the fp32 FPU paths) and computes on fp16-rounded inputs (errors vs the fp32 model up to 2.3e-4, vs ≤ 2.3e-7 for
  the fp32 kernels), so it is not a drop-in replacement: must stay token-exact vs run.c and may use
  the TCU for M ≥ 2 only if it accepts fp16 numerics or shows token-exactness with it. Among the fp32 paths, GEMV
  `lanes` is best at M = 1 and the small-M kernel at M ≥ 2. TCU peak fraction: 189.5 FLOP/cycle at M = 256 is 9 % of
  the earlier 2,048 placeholder (which assumes ISSUE_WIDTH = 1) and **4.6 %** of TIER-CFG's structural 4,096 (ISSUE_WIDTH = 2
  ⇒ 2 TCU blocks per core, `VX_config.toml:295`).
- **The decode GEMV (second-pass wording).** Within TIER-CFG/REG-CFG, these shapes and these scalar fp32 GEMV kernels,
  the decode GEMV is **not DRAM-bandwidth-bound** (≤ 22 % of 64 B/cycle: llama8b_s8 w1 `lanes` 14.1 B/cycle, REG-CFG
  `rows` 3.8 B/cycle, although the roofline puts AI ≈ 0.5 FLOP/B ≪ ridge 4, i.e. memory-bound *in bytes*; achieved
  is 22 % / 6 % of the roofline bound ≈ 32 FLOP/cycle) and **not throughput-bound** (IPC ≤ 0.42 per core, 0 % FPU/LSU/ALU
  structural stalls). It is **latency-bound**: warps are idle on scoreboard dependencies 41–71 % of cycles (scheduler
  idle 58–97 %), with 8 warps/core too few to hide them. Average load latency is 19–167 cycles and loads are the likely
  main source, but the counters do not separate load from FPU/shuffle latency. **How much of it is memory-side was
  measured in the second and third passes** (`results/<run>/diag.csv`, `diag_latency.csv`): a fixed extra latency of 0 / 50 / 100
  cycles on every DRAM read response (`VX_CFG_PLATFORM_MEMORY_LATENCY`, `sim/common/dram_sim.cpp:264-267`, SimX-only
  knob) gives, for llama8b_s8 w1 at TIER-CFG, `lanes` K = 512: 261,884 / 324,782 / 389,169 cycles at a *measured*
  average DRAM latency L of 28.3 / 78.5 / 127.9 cycles, and padded-stride `rows` K = 528: 247,430 / 271,351 / 292,526
  cycles at L = 82.0 / 127.3 / 173.6 (Ramulator's own part falls 82.0 → 77.3 → 73.6 as the knob spreads requests, so the
  real increase at +100 is 91.6 cycles, not 100); the TCU GEMM (M = 64) is only 1.04× slower at +100. Read-out
  (modeling choice; slopes against the measured L): `lanes` is mildly **convex** (1,253 then 1,303 cycles per cycle of
  L; ≈ 45 reads overlapping in both segments), so its first-order share s·L₀/cycles₀ = **13.5 %** (13.1 % from a
  quadratic through the 3 points) is an upper estimate; `rows` K = 528 is **concave** (528 then 457 cycles per cycle;
  the overlap *grows* from 124 to 140 reads as latency rises), so its linear figure is a lower estimate: **≈ 17.5 %**
  (lowest segment) to **≈ 21 %** (quadratic). So at the default DRAM model the base DRAM latency accounts for **about a
  fifth or less** of the GEMV's cycles, and **the GEMV is latency-bound mostly by latency that is *not* DRAM latency**
  (L1/L2 hit latency, the FMA/shuffle dependency chains; not separated further). Extrapolating to L = 0 is an assumption
  under any curvature (a zero-latency DRAM is not physical), so the shares are sensitivities, not removable components.
  This is consistent with for decode attention (exposed memory latency a few percent there, §14.2), but the two
  kernels and experiments differ. **Codegen check (the earlier errata 30):** the dot products are inline `fmadd.s`, not calls
  to musl `fmaf`. The built TIER-CFG kernels contain 0 `fmaf` symbols and 0 calls; `gemv_rows`, `gemv_lanes` and
  `gemm_small` contain 4 / 4 / 144 `fmadd.s` (the `gemv_rows` hot loop is 8 `flw`, 1 `fmul.s`, 3 `fmadd.s`, 1 `fadd.s`
  and the back-branch), so the in-core time, and hence these DRAM shares, are not inflated by soft fma calls.
  **Scope of the mechanism consequence.** (a) The share is a property of the DRAM model: the default adds **0** fixed
  latency (an idle open-row read is 14 cycles), and with +100 fixed cycles the same accounting gives ≈ **43 %**
  (`lanes`) and ≈ **27 %** (`rows`). memhier's GPU-like profiles assume 190 fixed cycles (§9), and PS-BASE's DRAM
  latency is not fixed, so any "a memory-side mechanism can move these kernels by ≲ x %" depends on the
  latency chooses. (b) The bound applies only to mechanisms that shorten DRAM-side read latency: B (arbitration)
  and C's L2-only prefetch mode. It does **not** apply to C's software DXA → LMEM path, which also hides L1/L2 latency.
  D (insertion) has no intra-launch reuse to exploit in a streaming GEMV. (c) Two kernels, isolated launches on
  4 cores with no contention; more cores per DRAM channel, L2 pressure or slower DRAM raise the DRAM share.
- **Row stride matters more than the variant:** at row strides whose 64 B-line count has a large power-of-two factor (K = 512, 768, 1792 fp32) the thread-per-row layout hits only 2–4 of
  the 64 L1 sets per warp load (errata 29); padding K = 512 → 528 makes `rows` 3.67× faster and faster than `lanes`.
  should store weights with a row stride of an odd number of 64 B lines (e.g. K + 16 floats), or choose the variant per (shape, NT).
- **The prefill TCU GEMM is not compute-saturated either:** 189 FLOP/cycle at M = 256, DRAM occupancy 12 % (18.6 MB read
  for 3.9 MB compulsory: per-CTA re-reads of the A/B strips not absorbed by the L2), scheduler idle 98 %, and only
  1.04× slower with +100 cycles of DRAM latency: the single-buffer DXA stage → barrier → WGMMA schedule of the stock
  kernel serialises copy and compute. should measure the real TCU and DRAM ceilings before the ridge numbers are
  used (the values are structural upper bounds).

### 14.4 Tiled prefill attention `prefill_attn`

**Operation and layout.** Causal attention of one layer for S query tokens at absolute positions s0 … s0+S−1 (s0 = 0:
whole-prompt prefill; s0 > 0: one chunk of a chunked prefill, the shape needs): for sequence b, token i (position
p = s0 + i) and query head h, o[b][i][h] = softmax_{t ≤ p}(q[b][i][h]·K[b][t][h/G]·scale)·V[b][t][h/G], scale = 1/sqrtf(D)
(run.c). q and o are [B][S][H_q][D] (token-major, like run.c's per-token q/xb); K and V are one layer of the §14.0 cache,
[B][Tmax][H_kv][D], with keys 0 … s0+S−1 present. The S×S score matrix never exists: QKᵀ, the online softmax of §14.2 and
PV run fused per key tile. This is the tiled forward pass of FlashAttention [cand:A7], written from the algorithm.

**Work decomposition** (`kernels/prefill_attn/kernel.cpp`). One CTA per (b, KV head, query tile of Qt tokens). A CTA is all
NT·NW threads of a core, so one CTA runs per core and owns the whole core LMEM. The CTA serves the G = H_q/H_kv query heads of
its KV head. Its R = G·Qt rows (row r = token i, head h, r = i·G + h) all multiply the **same** key and value tiles, so each K/V
element brought into LMEM is used R times (decode, reuses it G times). Per key tile of Tk positions:
1. **load:** the K and V tile into LMEM;
2. **S:** item = (block of 4 rows, key t): 4 dot products over D, with q rows and the k row from LMEM. Causal mask: a key whose
   position exceeds the row position gets −∞;
3. **M1:** one thread per row: running max m' and rescale c = exp(m − m');
4. **M2:** one thread per (row, key): p = exp(s − m'); a masked entry becomes exactly 0;
5. **V:** one thread per row: l = l·c + Σp. Then item = (4 rows, 4 columns): acc = acc·c + P·V as a 4×4 register block
   (4 p loads and 4 v loads per 16 FMAs).

acc[R][D] and q[R][D] stay in LMEM for the CTA's lifetime; out = acc/l after the last tile. Rows are padded to a multiple of 4;
padding rows compute with q = 0 and are never stored. Every exp uses `ps_expf` (§14.1), and m starts at −∞ (errata 7).
- `lmem_tiled` (`-v lmem`): every thread copies K/V elements (coalesced global loads, then LMEM stores), then a barrier. The
  copy is exposed (no overlap).
- `dxa_dbuf` (`-v dxa -p depth`): the DXA engine copies tile j + depth − 1 into a ring of `depth` LMEM stages while tile j is
  computed. One issuer warp issues the copies; completion is signalled by `expect_tx` on the stage barrier. This is the
  pattern and descriptor, with position extent s0 + S. depth 1 = engine copy, then compute.

**Causal tile skipping.** A CTA visits only key tiles 0 … ⌈kend/Tk⌉ − 1, where kend = 1 + its last query position. Tiles
entirely above the diagonal are never loaded or computed: with s0 = 0, that is about half of the S×S tiles, E4's ≈ 2S²d.
Tiles crossing the diagonal are masked per element; that work is executed but useless and is reported as `exec_over_useful`.
Work is uneven across query tiles: tile q visits ≈ q + 1 key tiles. The KMU hands CTAs to cores in strict round-robin
(`sim/simx/kmu/kmu.cpp:122-153`: one CTA per cycle, destination fixed in advance). The query tile is the innermost CTA index, so
each core receives every 4th query tile of every head. Per-core work therefore differs by at most a few query tiles (not
measured separately).

**Tiles (modeling choice).** LMEM per CTA (`common.h`), with Rp = G·Qt rounded up to 4, is
2·Rp·D (q, acc) + Rp·Tk (scores) + 3Rp (m, l, c) + 2·nbuf·Tk·D (K, V) floats, each region rounded to 16 floats; nbuf = 1 for
lmem and depth for dxa. Reuse per K/V element is R, and the fixed per-tile cost (5 barriers, the M1/M2 passes) is amortised
over Tk keys. Default rule (host side; -Q/-t override it):
1. Take the largest Qt ∈ {64, …, 1} with G·Qt ≤ 128 and Qt ≤ S whose largest fitting Tk (≤ 32) is ≥ 8, with that Tk.
2. If no Qt reaches Tk ≥ 8, take Qt = 1 with its largest Tk.
3. Halve Qt while the grid B·H_kv·⌈S/Qt⌉ has fewer CTAs than there are cores.

| (H_q, H_kv, D) | lmem (Qt, Tk), 16 KiB | dxa depth 1 / 2 / 4, 16 KiB | 64 KiB: lmem / dxa-2 |
|---|---|---|---|
| (16, 2, 128) | (1, 7): R = 8 | (1, 7) / (1, 3) / (1, 1) | (4, 28) / (4, 14) |
| (8, 2, 64) | (4, 13): R = 16 | (4, 13) / (2, 11) / (1, 6) | (16, 32) / (16, 25) |
| (6, 6, 48) | (16, 22): R = 16 | | |
| (8, 4, 8) | (64, 11): R = 128 (S ≥ 64) | | |

At D = 128 the q and acc tiles of a single token (G = 8 heads) already take 8 KiB. The 16 KiB LMEM therefore allows only R = 8
with Tk = 7, and with DXA double buffering only Tk = 3 (depth 4: Tk = 1). Prefill K/V reuse is LMEM-capacity-bound: the prefill
analogue finding, and a link to mechanism A1.

**FMA code generation (measured; affects).** The regression build flags (`tests/regression/common.mk:86-90`) leave
clang's default `-fmath-errno` on for riscv32-unknown-elf. As a result `__builtin_fmaf` compiles to a **call** to musl's `fmaf`
(`fmadd.s; ret`, plus argument moves). `__builtin_elementwise_fma` (llvm.fma, which has no errno) is emitted inline as `fmadd.s`.
Both are the same IEEE-754 fused multiply-add with one rounding.
- `prefill_attn` maps every `__builtin_fmaf` to the inline form with a macro, including the 10 in the shared `ps_expf`, whose
  file is unchanged.
- Measured on smoke-dxa with the same tiles: (16, 2, 128), S = 64, lmem went from 31.05 M to 16.06 M instructions and from
  90.1 M to 49.1 M cycles (1.83×). The D = 8 deep-negative case went from 8.42 M to 4.84 M instructions and from 28.4 M to
  15.6 M cycles. The error statistics are identical to the last digit (same max_abs_err and worst element), as expected for the
  same operation (`build/tmp/p27_kernel_{fmaf_libcall,inline_fma}.dis`).
- `llvm-objdump` of the current builds counts 251 `jal <fmaf>` in `ps_decode_attn_gqa`, 30 in `ps_softmax_fixed` and 0 in
  `ps_linear`. Part of the in-core instruction overhead that measured for decode attention, and of ps_expf
  instruction count, is therefore this call. Those numbers stand as measured for those builds. A rebuild with inline FMAs
  (e.g. `-fno-math-errno` in the shared kernel flags, a change outside) would lower them. Recorded as errata 30.
  the earlier instruction, cycle, issue-bound and DXA/memory-share figures must be re-measured after that fix
  before P3.x/P5 rely on them. the earlier rows all come from the inline build (0 `jal <fmaf>`), so they are unaffected.

**Correctness metric and tolerance.** As in §14.2, per output row (token at position p, head h): rel_i = |o_i − ref_i|/S_i with
S_i = Σ_{t≤p} p_t·|v_t[d]|. The reference is a float64 causal attention of the same float32 inputs, plus the NumPy float32 out
of the prefill fixtures. The bound is the §14.2 bound with the prefill chain lengths:

rel ≤ 2E_s + 2(s_max − s_min)u + γ(n_acc + n_l + 8(n_tiles − 1) + 10), with n_acc = (p + 1) + n_tiles and n_l = Tk + n_tiles.

- The rescale term 8(n_tiles − 1) was added. §14.2's claim that the error of c "cancels in acc/l"
  holds only for one tile. c_j rescales only the contributions of tiles < j, in both acc and l, so an error in c_j (ps_expf,
  ≤ 4u) perturbs a subset of the weights; by the weight argument of §14.2 that costs ≤ 2·4u per max-raising tile. The rounding
  of c's argument m_j − m_{j+1} telescopes with the s_t − m roundings into the existing 2(s_max − s_min)u term, and
  ps_expf(0) = 1 exactly when the max does not change. The extra term is ≤ 8.1·10⁻⁵ in the TEST grid's worst case (D = 128,
  Tk = 3, S = 512).
- **Implemented formula (not yet corrected):** the host (`main.cpp`) still computes the bound without the 8(n_tiles − 1) term,
  and its header comment repeats the cancellation sentence. Changing either alters the app-source hash and every run key, so
  both are to be fixed at the earlier next code touch. Direction of the omission: the published bound is smaller
  than the true worst case, so tol = max(1e-4, bound) was at most stricter than intended in the 20 runs where the bound
  exceeds 1e-4, never looser. Every run met 1e-4 directly (largest 3.3·10⁻⁵), so no verdict changes.
- **Addendum:** the host formula and the `main.cpp` header comment now include the 8(n_tiles − 1) term
  (`kernels/prefill_attn/main.cpp`; NumPy's chain is single-pass and unchanged). The rows above predate the change;
  the re-runs points are bit-identical (equal output digests and statistics, unchanged cycles), and 's
  `-fno-math-errno` leaves this kernel's assembly identical (its local macro already inlined the FMAs).

- n_acc: acc is one FMA chain across the visited tiles, with p + 1 visible keys and one rescale per tile. A masked key has p = 0
  exactly, and fma(0, v, a) = a adds no rounding.
- n_tiles is the number of key tiles the row's CTA visits.
- tol = max(1e-4, max bound). Every test also asserts the 1e-4 itself (`within_plan_tol`).
- At S ≤ 512 the chain term is ≤ γ(1100) ≈ 6.6·10⁻⁵. Large logits raise the bound through E_s and (s_max − s_min)u.

**Prefill fixtures** (`model/make_prefill_fixtures.py`). The NumPy reference processes the sequence token by token, and token
p's attention reads only cache rows 0 … p. Its per-token attention outputs are therefore exactly causal prefill attention over
the same K/V. A cheap test checks this against a float64 causal attention, and checks token 17 bit for bit against the
decode fixture. Any prefix and any chunk of a fixture is a valid case.

**Results** (`results/<run>/bench.csv`). Machine: REG-CFG + the DXA extension = 4 cores × 4 lanes, L2 on, B = 1. Cycles are of the
single launch. Useful FLOPs = 4·D per visible (query, key, head); FPU peak = 32 FLOP/cycle. DRAM bytes = perf-class-7 requests
× 64 B.

| (H_q, H_kv, D) | S | lmem cycles | lmem FLOP/cycle (FPU util) | dxa-p2 / lmem cycles | DRAM reads / (K+V+q) | DRAM bw occupancy |
|---|---|---|---|---|---|---|
| (16, 2, 128) | 16 / 64 / 128 / 256 | 1.28 M / 12.7 M / 45.7 M / 172.8 M | 0.87 / 1.35 / 1.48 / 1.56 (2.7–4.9 %) | 1.16 / 1.24 / 1.26 / 1.27 (Tk 3 vs 7) | 1.91 / 1.83 / 1.81 / 1.82 | 0.56 % → 0.06 % |
| (8, 2, 64) | 16 / 64 / 256 / 512 | 0.40 M / 3.30 M / 41.2 M / 157.1 M | 0.70 / 1.29 / 1.63 / 1.71 (2.2–5.4 %) | 0.78 / 0.91 / 0.95 / 0.96 (2/11 vs 4/13) | 2.02 / 1.75 / 1.69 / 1.69 | 0.57 % → 0.04 % |

- **Cycles vs S** grow ≈ S² (causal work). FLOP/cycle rises with S as fixed per-launch costs amortise. Causal tile skipping
  visits 51–61 % of the S×S tiles at S ≥ 64. Executed/useful FLOPs are ≤ 1.012 at S ≥ 256 (1.18 at S = 16 with Qt = 4).
- **DRAM traffic ≈ compulsory.** At S ≥ 256, reads are within 2 % of K + V + q + out bytes (5 % at S = 64). This is consistent
  with the output lines being fetched on write-allocate (code-unverified). The 1 MiB L2/L1 serves every K/V re-read caused by
  the R-limited reuse: K/V bytes staged into LMEM are 8.5–131 × the compulsory K/V bytes at S ≥ 64 (2.5–9.4 × at S = 16;
  `kv_tile_bytes`).
- **DXA vs LMEM-only, at equal tiles** (kind `equal`, same Qt and Tk): dxa takes 0.85× of lmem's cycles for (16, 2, 128) at
  Qt 1, Tk 3, and 0.79× for (8, 2, 64) at Qt 2, Tk 11. With 64 KiB LMEM and larger tiles the ratios are 0.92× and 0.96×.
  - The gain is the removed copy phase: its instructions and their load latency. The lmem thread copy is 14.6 % / 20.9 % of
    cycles at these equal 16 KiB tiles; at 16 KiB lmem executes 70.6 M vs 56.2 M and 58.6 M vs 43.7 M instructions, i.e.
    0.68–0.81 cycles saved per removed warp-instruction (kernel average 0.84–0.95). Its loads are L1 misses served by the L2
    (L1 read hit 74–80 %, DRAM traffic compulsory); the counters cannot separate their issue cost from that L2-hit latency.
  - **Depth 1 (copy, then compute) and depth 2 (double buffering) give the same cycles within 0.05 %** in all four equal-tile
    pairs, so overlapping the copy buys nothing. The engine's copy of a 3–14 KiB tile is negligible next to the 4·10⁴–8·10⁵
    cycles of compute per tile per core.
- **DXA vs LMEM-only, at the default LMEM-sized tiles:**
  - D = 128: double buffering halves Tk (7 → 3), so dxa-p2 is 1.16–1.27× slower. dxa-p1 keeps Tk = 7 and runs at 0.78× of lmem.
  - D = 64: dxa-p1 runs at 0.87× of lmem.
  - Depth 4 forces Tk = 1 at D = 128 (3.13×) and Tk = 6 at D = 64 (1.50×).
  - As a prefetch mechanism, software double buffering does not pay on this machine; what DXA saves is copy *instructions*.
- **64 KiB LMEM** (labelled sensitivity): the default rule picks Qt = 4 (R = 32) for D = 128 and Qt = 16 (R = 64) for D = 64.
  lmem cycles drop to 0.76× (34.8 M vs 45.7 M) and 0.92× (38.1 M vs 41.2 M), with 0.70× and 0.86× of the instructions (fewer
  tiles, barriers and M1/M2 passes). DRAM bytes are unchanged (≤ 0.2 %). Larger LMEM helps prefill by cutting per-tile
  overheads, not memory traffic.
- **Cycles: not memory-bound; in-core latency/issue-bound.** Scope: REG-CFG with the **default DRAM timing model** (0 fixed
  extra latency, `VX_CFG_PLATFORM_MEMORY_LATENCY = 0`, `VX_config.toml:95`; measured average DRAM read latency 21–25 cycles here);
  this scalar fp32 kernel with 16 threads per CTA; B = 1; S ≤ 512 (≤ 256 for D = 128 in the benchmark; the S = 512, D = 128
  point exists only on the 1-core TEST config — should add it on REG-CFG if it needs those cycles); the two shapes. It is
  not a statement about prefill attention in general, nor about a slower DRAM model.
  Evidence:
  - (a) the FPU runs at 1.5–7 % of peak, so the kernel is not compute-bound either;
  - (b) class-1 counters at S = 128/256: scheduler idle 68–76 %, scoreboard stalls 46–52 %, average load latency 10–14
    cycles, FPU instructions 18–26 % of all instructions. Class 4: L1 read hit 74–80 %. Class 7: LMEM bank utility 59–66 %;
  - (c) DRAM bandwidth occupancy is ≤ 0.3 % at S ≥ 64. This is a lower bound on memory time, not stall time;
  - (d) a serialised upper bound on DRAM-latency time is ≤ 3.5 % of cycles at S ≥ 128, ≤ 5.8 % at S = 64, and 11–18 % only
    for the S = 16 launches, which cold misses dominate. The bound counts every DRAM request (reads + writes) at the measured
    average read latency (21–25 cycles) as if it stalled the whole machine — a conservative choice: SimX's latency counter is
    Σ cycles of pending reads / reads, so reads × lat is the integral of in-flight reads, and the read-only integral gives
    ≤ 2.4 % at S ≥ 128 (2.8–4.1 % at S = 64, 7.8–12.3 % at S = 16). This holds at the default DRAM model only; a bound of this
    kind scales roughly with DRAM latency, so with +100 cycles of fixed latency it could reach roughly 15–20 % at S ≥ 128
    (estimate, not measured);
  - (e) depth 1 = depth 2 at equal tiles (above): the K/V copy latency is not exposed;
  - (f) the plain-REG-CFG control matches the DXA build within 0.16 %, and perf classes 1, 4 and 7 give identical cycles.

  The counters do not split the idle cycles between LMEM/L1 latency, L2-hit latency (the lmem copy's L1 misses), FPU latency
  (8-cycle FMA, dependent chains), barriers and integer overhead. `llvm-objdump` of the kernel shows 17 instructions per 4 FMAs in the inner S loop and 37 per 16 in the V
  loop. This is the same in-core picture as (§14.2; whose build pays the errata-30 fmaf calls) (§14.3). 's
  latency experiment found the DRAM-side share of GEMV cycles to be about a fifth or less (≈ 13–21 %) at the default (0 fixed
  latency) DRAM model and 27–43 % at +100 cycles, and any such bound applies only to B and C's L2-only mode. As a comparison at
  the default model only: prefill, with R-fold K/V reuse, sits further from DRAM limits than the GEMV.
  Consequence for P5: on this configuration **and at the default DRAM timing model**, memory mechanisms (B, C, D) are not
  expected to change prefill-attention cycles; with a slower DRAM model the DRAM share grows (estimate above) and must be
  re-measured before P5 relies on it.
  A1 (LMEM size) acts through the tile geometry (reuse R and per-tile overhead), as the 64 KiB rows show.
- **Correctness.** All 64 checked TEST runs (smoke-dxa) met the 1e-4 against float64.
  - Largest observed error: 4.0·10⁻⁶ for ordinary logits; 3.3·10⁻⁵ for the deep-negative case (logits −332 … −156, bound
    4.6·10⁻⁴).
  - The derived bound exceeds 1e-4 in 20 runs: the real-model fixtures, whose logits are larger (up to 3.1·10⁻⁴); D = 128 at
    S = 512 (1.0–1.1·10⁻⁴); and the deep-negative case.
  - Device vs the NumPy fixtures: ≤ 5.7·10⁻⁶. NumPy vs float64: ≤ 4.4·10⁻⁶. Every benchmark run also met 1e-4 (max 3.3·10⁻⁷).

To be written (and their baseline steps): the exact semantics of each mechanism and baseline policy.

### 14.5 Full-model driver `ps_llama` (`runtime/llama_vx`)

**What runs.** run.c's `forward` (llama2.c [cand:D15]) with every op on the device, built only from the kernels validated in
(device code unchanged). Per decode token and layer, 14 launches in run.c's order: `rmsnorm` → q, k, v GEMV →
`rope_kv` (RoPE + KV-cache write, §14.0) → `attn_gqa` (§14.2; G = kv_mul, Tt = 32) → o GEMV → `residual` → `rmsnorm` →
w1, w3 GEMV → `silu_gate` → w2 GEMV → `residual`; then the final `rmsnorm` and the classifier GEMV, i.e. **14 L + 2
launches per token** (72 for stories260K, 86 for stories15M). The logits are read back and the argmax is taken on the host
(run.c's `sample` at temperature 0, first index of the maximum). Weights (one device buffer laid out as the checkpoint) and the
KV cache stay in device memory for the whole run; the token-embedding row is gathered by a device-to-device DMA copy
(`vx_enqueue_copy`, no kernel; run.c's `memcpy`), outside every kernel's cycle count, like the host uploads. The device
module is one multi-entry `.vxbin` (the `tests/regression/multikernel` pattern) linked from 8 translation units, each
`#include`-ing one validated `kernel.cpp` through per-file symlinks, so the kernels' file-local helpers cannot collide.
**Libm variant:** the musl libm of the toolchain's `libc32`, unchanged (soft-float double `powf`/`sinf`/`cosf`/`expf`,
correctly rounded on SimX, §14.0); RoPE and SiLU-gate pay that cost in every number below.

**Per-launch flushes.** The runtime appends one `CMD_CACHE_FLUSH` after every launch (`sw/runtime/common/device.cpp:877`,
`cp_submit_launch_qmd` — the path SimX takes since it advertises QMD support, `cmd_processor.cpp:126-128`; the
plain-launch fallback `cp_submit_launch` at `:816` adds one flush the same way); copies, reads, writes and DCR reads add none. So **flushes = launches = 14 L + 2 per token** (and the
caches are cold at every launch, §1 / errata 24): a stories15M token is 86 cold-start launches. `ps_llama` counts launches
and prints `flushes=` in its summary line (equal by construction of the runtime, not separately observable from the host).

**Cycle attribution (modeling choice).** MPM counters run from device reset (errata 24), so the host reads them before and
after each launch (DCR reads through the CP, outside any kernel's cycles) and attributes the difference to that launch:
cycles = max over cores of `MCYCLE` (every core ticks every platform cycle of a launch), instrs = Σ cores `MINSTRET`; at
perf class 1 also Σ cores scheduler-idle and scoreboard-stall cycles (reported as fractions of cycles × cores), at class 7 the
DRAM read/write requests (64 B each) and summed read latency. The launch deltas sum exactly to the global `PERF:` cycles
line (asserted per run). A launch's cycles include its own start-up (KMU dispatch, cold I-cache and caches) — a real cost of
this launch structure. Launch k ≥ 2 of a run is not a cycle-exact replay of an isolated launch (errata 24: DRAM-model state
survives), so the per-op numbers are those of the sequence as executed, not of isolated kernels.

**GEMV variant per matrix (errata 29).** `gemv_rows` (thread per output row) is the faster kernel
unless the NT rows a warp loads at once collide in the L1: lane j reads row r0 + j at the same column offset, so with row
stride s bytes its lines are (a + (r0 + j)·s) / 64. Rule (`rows_set_pressure` in `main.cpp`, Python port
`model/llama_vx_eval.py`): over every warp-aligned r0 and every 16-byte offset a within a line, count the NT rows per L1 set
(sets = DCACHE_SIZE / (64 · ways)); choose **rows iff the maximum ≤ DCACHE_NUM_WAYS**, else `gemv_lanes` (a warp reads NT
consecutive float4 of one row, no conflict). A stride of an odd number of 64-byte lines always passes (NT ≤ 64 rows get
distinct sets); the rule reproduces errata 29 at TIER-CFG (NT = 32: lanes at K = 512, 768, 1792; rows at K = 288, 528). At
REG-CFG (NT = 4, 4 ways) the maximum is 1 for every shape used here, so **every matrix uses rows**; per shape (K = row length,
stride in 64 B lines): stories260K wq/wo 64×64, wk/wv 32×64, w1/w3 172×64, wcls 512×64 (K = 64: 4 lines), w2 64×172
(10.75 lines); stories15M wq/wk/wv/wo 288×288, w1/w3 768×288, wcls 32000×288 (K = 288: 18 lines), w2 288×768 (48 lines).
No weight padding is needed at this config. The forced-lanes runs measure the alternative (results below). G ∈ {1, 2, 4, 8}
(the earlier decode-attention limit): stories260K has kv_mul = 2, stories15M 1; the host refuses other group sizes.

**KV cache.** `[layer][b][pos][kv_head][head_dim]` with capacity `b_cap` (§14.0; default 1 = run.c's layout), poisoned with
NaN at upload, so a read of a slot no `rope_kv` wrote shows up as NaN logits. `-c 2` runs the live sequence b = 0 inside a
b_cap = 2 cache (the capacity-stride path of `rope_kv` and of the attention's layer base `K + l·b_cap·seq_len·kv_dim`).

**Batched-prefill mode** (`-m prefill`;: validated by logits tolerance, not token exactness). The P prompt tokens
(P = 5 for "Once upon a time" with BOS) are one pass of P rows: every linear layer is `gemm_small` with M = P (fp32, the
same per-row summation order as `gemv_rows`), attention is `attn_prefill_lmem` (causal, s0 = 0, its host tile rule:
Qt = 4, Tk = 32 for stories260K), the element-wise kernels take B = P rows. `rope_kv` writes every row into sequence 0's
slots by receiving the layer's sequence-0 cache base with `seq_len = 0` and `layer = 0`, so its slot index
((layer·b_cap + b)·seq_len + pos)·kv_dim reduces to pos·kv_dim for every row b — a host-side use of the unchanged kernel. The
final rmsnorm and the classifier run on the last row only (run.c computes logits for the current token). Positions P..n − 1
then continue in teacher-forced decode mode from the cache the batched pass wrote. The TCU GEMM is not used: it computes on
fp16-rounded inputs (§14.3, errors up to 2.3·10⁻⁴ per layer), whereas this mode is checked at 1e-3 on the logits after L
layers; gemm_small keeps the prefill pass fp32.

**TEST rules.** *Token rule* (the carry-over): at every compared position (not forced by the
prompt) the device argmax a equals the golden next token g, or the disagreement is an fp32 near-tie in **both** senses —
|l₁ − l₂| / |l₁| ≤ 1e-4 (top-1 vs top-2, the literal rule) **and** |l₁ − l_g| / |l₁| ≤ 1e-4 (l_g = the device logit
of g). The second condition closes the gap the found: with the top-2 rule alone, a bug that pushes g far down
passes whenever the two leading wrong logits happen to be close. Teacher forcing feeds the golden token at every position;
free running feeds the device's own argmax after the prompt and compares until the first disagreement (which must be a
near-tie; afterwards the sequence is no longer the golden one). *Logits metric* (modeling choice): rel = max_i |l_i − r_i| / max_j |r_j|
(norm-wise, L∞) against the NumPy float32 reference of the same golden history (`model/llama_vx_ref.py`, the reference):
the logits are one vector whose ranking matters; an element-wise relative error is meaningless for the many logits near 0, and
the largest logit sets the scale at which a ranking change can occur. Tolerance 1e-3 (batched prefill), applied to every
position whose history is golden as a sanity bound. Why it is comfortable: device and NumPy differ only in rounding order
(dot products: γ_K-scaled, §2 consequence 6), libm (≤ 0.5 vs ≤ 2.25 ulp, §14.0) and `ps_expf` vs `np.exp` in attention
(≤ 2 ulp, §14.1) — per-op relative differences ~10⁻⁶ that a 5–6-layer residual network does not amplify by 10³ (the
float32-vs-float64 logits differ by ≤ 3·10⁻⁶ absolute).

**Results** (`results/<run>/`; REG-CFG = 4 cores × 4 warps × 4 threads, 1 MiB L2, PLRU; fixed build; musl libm).
- *TEST.* stories260K teacher-forced 32 positions: 32/32 device argmaxes equal the golden tokens (28 compared, 0 near-ties,
  smallest top-2 margin 8.1·10⁻³ ≫ 1e-4); free-running 32 tokens: the generated sequence equals the golden one at every step
  (rendered text = the first 32 tokens of `results/S8/golden_260K.txt`); stories15M teacher-forced 8 positions (tier C)
  and 32 positions (tier A/B extension): 4/4 and 28/28 compared positions agree, 0 near-ties. No run needed the near-tie
  rule. Logits vs NumPy (norm-wise): ≤ 9.1·10⁻⁷ (260K), ≤ 1.84·10⁻⁶ (15M) at every position. Batched prefill (P = 5):
  final-position logits 3.8·10⁻⁷ / 1.04·10⁻⁶ from NumPy (tol 1e-3; norm-wise metric — element-wise the 15M final logits reach
  1.012·10⁻³ at one near-zero logit, absolute error 1.9·10⁻⁵, so the pass relies on the documented norm-wise metric) — and bit-identical to the token-by-token device logits
  at that position (observed, not derived: gemm_small uses gemv_rows' per-row summation order; that the two attention kernels
  also round identically at T ≤ 5, one key tile each, is an observation) — and decoding continues token-exactly from the cache the batched pass wrote. b_cap = 2 and perf class 7 give
  bit-identical logits; forcing `lanes` changes logits by ≤ 8.9·10⁻⁷ (another summation order), tokens unchanged.
- *Cycles per decode token per op* (mean over the run's positions; launch deltas sum exactly to the `PERF:` line):

  | op (per token, all layers) | stories260K (2.25 M cycles/token, 32 pos.) | stories15M (19.9 M, 8 pos.; 20.5 M over 32 pos.) |
  |---|---|---|
  | logits GEMV (wcls) | 1.5 % | **42.4 %** (41.1 % over 32 positions) |
  | 7 layer GEMVs (q, k, v, o, w1, w3, w2) | 12.9 % | 29.6 % |
  | RoPE + KV write (`rope_kv`, musl powf/cosf/sinf) | **56.7 %** | 13.8 % (16.0 % over 32: its cycles grow with the position, i.e. with the angle; the cause inside musl's soft-float sinf/cosf was not isolated) |
  | SiLU-gate (musl expf) | **22.8 %** | 12.1 % |
  | attention (`attn_gqa`) | 3.4 % | 0.8 % (1.3 % over 32) |
  | rmsnorm ×(2L + 1) | 2.4 % | 1.1 % |
  | residual ×2L | 0.4 % | 0.2 % |

  The logits GEMV dominates stories15M, as R§6 expects (32,000 × 288 fp32 = 36.9 MB of the 60.8 MB of weights). With the
  toolchain's soft-float libm, RoPE + SiLU-gate cost 26 % (15M) and 80 % (260K) of a token: a faster float-only/MUFU sincos
  and exp would change the element-wise share completely (§14.0 cost note), so any P3 per-op breakdown must state the libm.
- *DRAM traffic* (perf class 7): stories15M 61.6 MB read per token = 1.013 × the checkpoint's weight bytes (every GEMV reads
  its matrix once: wcls 1.0035×, w1 1.006×; caches are cold at each of the 86 launches), i.e. 61.6 MB / 64 B per cycle ≈ 0.96 M
  of 19.9 M cycles = **4.8 % bandwidth occupancy**; stories260K 1.38 MB per token (1.3× its 1.04 MB of weights: activations,
  KV, code), 1.0 % occupancy. Average DRAM read latency 23–40 cycles (default DRAM model). Scheduler idle 70–75 % in the
  GEMVs (scoreboard stalls 50–56 % of core-cycles), 63–82 % in RoPE/SiLU, 97 % in rmsnorm (one warp at B = 1). Scope as
 : REG-CFG, these scalar fp32 kernels; not a memory-boundedness statement beyond them.
- *GEMV variant.* At REG-CFG the rule picks `rows` everywhere; forcing `lanes` is 1.15–2.17× slower per launch for stories260K
  and 1.28–1.55× for stories15M (`gemv_variants.csv`): the rule's choice is the faster one at every shape here.
- *Batched prefill* (one pass of P = 5 rows): stories260K 5.86 M cycles vs 8.92 M for the same 5 positions token by token
  (1.52×); stories15M 49.4 M vs 98.0 M (1.98×). In the pass, RoPE + SiLU are 78 % (260K) / 43 % (15M) and the classifier,
  run once, 17 % (15M): the GEMM amortises weight streaming over 5 rows, the element-wise soft-float work does not shrink.
- *Launch structure.* 72 / 86 launches = cache flushes per token; SimX speed 0.27–0.30 M cycles/s (260K) and 0.15–0.17 M
  cycles/s (15M), so a stories15M token costs ≈ 2 min of wall time (≈ 72 min for 32 positions), peak RSS 0.2 GB.

### 14.6 Fused mixed-batch attention `mixed_attn`

**Operation.** One launch serves one Sarathi-style mixed iteration [cand:A5] of one layer:
- a **prefill chunk** of one sequence: c query tokens at positions s0 … s0+c−1 attend causally to keys 0 … s0+c−1 (the §14.4
  operation; q_p, o_p [c][H_q][D], K_p, V_p [Tmax_p][H_kv][D]);
- **n decode sequences**: sequence b has one query that attends to its keys 0 … ctx[b]−1 (the §14.2 operation; q_d, o_d
  [n][H_q][D], K_d, V_d [n][Tmax_d][H_kv][D]). ctx[b] differs per sequence (a continuous batch), so the host passes a
  `ctx` array in device memory and each decode CTA reads its own length.

**CTA split.** CTAs [0, n_p) are prefill CTAs, one per (KV head, query tile of Qt tokens), numbered as numbers
them (query tile innermost), n_p = H_kv·⌈c/Qt⌉. CTAs [n_p, n_p+n_d) are decode CTAs, one per (sequence, KV head) serving its
G = H_q/H_kv query heads, numbered as the earlier `gqa_shared_kv`, n_d = n·H_kv. This is a static CTA-level split in the spirit of
POD-Attention [cand:B10]; POD-Attention additionally binds CTAs to SMs at run time, which this kernel does not do (below).
The branch on the CTA index is uniform per CTA, so no warp diverges on it. `-r` (order = 1) puts the decode range first; it is
a measured sensitivity, not the layout.

**Reuse, not redesign (modeling choice: equal work).** The two CTA bodies are the validated kernels' bodies:
`lmem_tiled` (`prefill_cta<Mode::Lmem>`) `gqa_shared_kv` (`attn_cta<G, Mode::Plain>`). Only the CTA index (now a
parameter), the argument struct, B = 1 for the chunk, and the per-sequence ctx changed; the DXA paths are not compiled
(measured that DXA does not pay at the default 16 KiB tiles). Every floating-point phase (S, M1, M2, V and the final
division) is text-identical to the validated code (cheap test `test_bodies_match_approved_kernels`). The host keeps the
tile rule for the chunk unchanged, including its last clause (halve Qt while H_kv·⌈c/Qt⌉ < cores), evaluated on the prefill
CTAs alone, and Tt = 32 for decode. The synthetic generator draws the inputs exactly as `ps_prefill_attn` (B = 1, S = c,
-o s0, seed -s) and `ps_decode_attn_gqa` (B = n, T, seed -u) draw theirs. So, for equal ctx, the fused launch and the two
separate launches do exactly the same arithmetic on the same inputs. This is **tested** (for equal contexts): the fused prefill and decode outputs
are bit-identical (PSDIGEST, FNV-1a 64 over the output bits) to the separate launches' outputs (`test_mixed_equals_separate`,
three cases incl. G = 1 and an all-negative case) and in every benchmark point (`compare.csv` `digest_eq`). A cycle difference
between fused and separate launches is therefore code generation (the dominant term in the results below), scheduling,
launch overhead and interference, not different work.

**LMEM (modeling choice).** A launch has one LMEM size for all CTAs; the host passes max(prefill footprint ps_pattn_lmem(G, D, Qt,
Tk, 1), decode footprint ps_attn_lmem(G, D, Tt, 1, 0)), and each body lays out its own regions from offset 0. A CTA is all NT·NW
threads of a core, so only one CTA is resident per core in either case and the larger LMEM request costs no occupancy here. On a
machine with several CTAs per core it would cut decode occupancy to the prefill CTA's LMEM budget; this is one concrete way in
which a single static configuration of a mixed launch cannot fit both parts (mechanism A1's question).

**CTA placement on Vortex (measured consequence, not a design choice).** The KMU hands CTAs to cores in strict round-robin, one
per cycle, with the destination fixed before the CTA is sent and at most LAUNCH_CREDITS = 4 admissions outstanding per core
(`sim/simx/kmu/kmu.cpp:122-153`, `kmu.h:59`); a busy core stalls the launch bus rather than letting a neighbour take its turn.
CTA i therefore runs on core i mod 4, after that core's earlier CTAs, so decode CTA j runs on core (n_p + j) mod 4 behind that
core's share of the prefill CTAs. Nothing in the kernel balances prefill against decode work across cores at run time.

**Per-CTA timestamps (attribution).** Thread 0 of each CTA reads the core's cycle counter (`vx_rdcycle`, mcycle) at CTA entry
and, after a CTA barrier, at exit, and writes one 64 B record per CTA (start, end, core, region, region-local index). Every SimX
core increments its counter once per machine cycle (`sim/simx/core.cpp:343-352`; mcycle reads that counter, `csr_unit.cpp:125`), and the counters are free-running from device
reset (`core.cpp:336-340`), so the records of different cores are on one clock (consistent with the class-1 PERF output: every
core reports the same `cycles` as the whole launch). The host validates every record (magic,
region, index, end ≥ start) and prints per-region and per-core sums. Derived quantities (`results/<run>/summarize.py`):
- busy_p, busy_d = Σ over the region's CTAs of (end − start), in core-cycles (one CTA per core at a time, so this is core
  occupancy by that region); idle = 4·F − busy_p − busy_d (cores with no CTA: launch start/drain, dispatch gaps, the tail);
- interference_x = busy_x in the mixed launch / busy_x when the same part is launched alone through the same app (kinds
  fused-p-only / fused-d-only): > 1 means the part's CTAs ran slower next to the other part's CTAs (shared L2 and DRAM);
- lb_fused = max((busy_p,alone + busy_d,alone)/4, longest CTA): the makespan of a perfect 4-core packing of the measured CTA
  times without interference — a lower bound for any CTA order only under the assumption of no interference. cycles/lb_fused measures what the static round-robin placement
  loses.
Cost of the timestamps: two counter reads, one barrier and eight stores per CTA (kind ts-off measures it).

**Correctness.** Per part, the error metric and bound are §14.4's (prefill rows: n_acc = (p+1) + n_tiles, n_l = Tk + n_tiles)
and §14.2's (decode rows), with the decode chain now per sequence (n_acc = ctx[b] + n_tiles, n_l = Tt + n_tiles, n_tiles =
⌈ctx[b]/Tt⌉), each with the rescale term 8(n_tiles − 1); tol_part = max(1e-4, max bound of the part) and both parts must meet
the 1e-4 itself (`within_plan_tol`). The bounds carry over unchanged because each row executes the same floating-point
operations in the same order as in the separate kernels. "vs NumPy": `model/make_mixed_fixtures.py` writes
`attention_mixed` fixtures whose outputs are `Llama2cRef.attention` (the earlier NumPy attention) of each query over the keys it may
see; for real-model sources (the prefill fixtures of stories260K L4 and stories15M L0) the generator also checks each
output row bit for bit against the prefill fixture's out row. The app compares device vs float64, NumPy vs float64 (NumPy's
bound: chains 2T) and device vs NumPy (sum of both bounds), per part. Cache rows a part must not read are NaN.

**Results** (`results/<run>/compare.csv`, `regions.csv`; REG-CFG = 4 cores × 4 warps × 4 lanes, L2 on, fixed build, perf class
7; every launch the single launch of its app run). Points: chunk c at s0 with every chunk ending at position 256, n decode
sequences of context T. F = fused launch; P = separate `ps_prefill_attn` (lmem); Dc = separate `ps_decode_attn_gqa`
(gqa); F_p / F_d = the mixed app with only one part (same binary as F).

| (H_q, H_kv, D) | c, s0 | n, T | F | P + Dc | F / (P + Dc) | F_p / P | F_d / Dc | F − F_p − F_d | idle share | F / lb |
|---|---|---|---|---|---|---|---|---|---|---|
| (8, 2, 64) | 16, 240 | 4, 256 | 6.07 M | 5.99 M | 1.013 | 1.017 | 1.000 | −6.2 k | 1.7 % | 1.017 |
| (8, 2, 64) | 64, 192 | 8, 256 | 20.05 M | 19.69 M | 1.018 | 1.021 | 1.000 | −10.9 k | 2.2 % | 1.023 |
| (8, 2, 64) | 64, 192 | 8, 1024 | 26.94 M | 26.61 M | 1.012 | 1.021 | 1.000 | −43.2 k | 1.6 % | 1.016 |
| (8, 2, 64) | 256, 0 | 16, 256 | 46.80 M | 45.88 M | 1.020 | 1.023 | 1.001 | −20.1 k | 3.9 % | 1.041 |
| (16, 2, 128) | 16, 240 | 4, 256 | 24.61 M | 23.75 M | 1.036 | 1.044 | 0.999 | −33.9 k | 0.3 % | 1.002 |
| (16, 2, 128) | 64, 192 | 8, 256 | 83.79 M | 80.33 M | 1.043 | 1.048 | 1.000 | −83.5 k | 0.6 % | 1.005 |
| (16, 2, 128) | 256, 0 | 16, 256 | 193.87 M | 185.75 M | 1.044 | (not run) | 1.000 | — | 1.0 % | — |

- **Equal work holds exactly.** In all 7 points the fused outputs are bit-identical to the separate launches' (digest_eq = 1);
  DRAM reads of F are 0.995–1.003 × those of P + Dc; the instructions of F equal those of F_p + F_d to within 64 warp
  instructions (one launch prologue). The separate decode launches reproduce the fixed-build rows to the cycle where
  has the same case (3,310,558 and 1,171,059 cycles), and the (8, 2, 64) S = 256 prefill launch reproduces 's
  REG-CFG control row to the cycle (41,208,587).
- **The fused launch is 1.2–4.4 % slower than the two separate launches here.** Decomposition
  F − (P + Dc) = (F_p − P) + (F_d − Dc) + (F − F_p − F_d):
  - (F_p − P): the prefill body inside the fused kernel executes 2.6–4.1 % more instructions and takes 1.7–4.8 % more cycles
    than in `ps_prefill_attn`, with identical outputs. It is code generation, not work or scheduling: the S-phase inner loop is
    instruction-identical (17 instructions per 4 FMAs; corrected from 13), but the M2 exp loop body grows from 118 to 125 instructions and the
    tile loop from 423 to 430 (`llvm-objdump` of both kernel.elf; the fused function also holds the four decode
    instantiations, and the compiler reloads more loop invariants from the stack). The decode body does not suffer (F_d / Dc =
    0.999–1.001; −0.3 … +0.1 % instructions).
  - (F − F_p − F_d): what one launch instead of two buys on SimX: 6–84 k cycles, i.e. 0.04–0.16 % of F. SimX's fixed cost per
    launch is small (the first CTA starts at cycle ≈ 685; caches are cold either way, §2 consequence 4), and the overlap
    of the two regions is short (below).
  - So on this machine the measured benefit of fusing is ≤ 0.16 % and the cost of the fused binary 1.7–4.8 %. A fused kernel
    without the code-generation penalty (e.g. a non-inlined body per part) would be within ≈ 0.2 % of the separate launches.
- **Per-region attribution (timestamps).** Prefill CTAs occupy 64–92 % of the core-cycles, decode CTAs 6.8–34 %, no CTA 0.3–3.9 %.
  The two regions run almost one after the other: with strict round-robin placement and one CTA per core, a decode CTA starts
  only when its core has drained its share of the prefill CTAs, so decode starts at 0.62–0.92 × F and both regions are active
  together for only 0.6–6.5 % of F (`overlap`). Decode-first order (-r) changes F by −0.01 … +0.09 %: the same near-serial
  pattern in the other order. F is 1.002–1.041 × the packing lower bound lb (largest at c = 256 on (8, 2, 64), where the
  diagonal makes later query tiles heavier and core 3 executes the most instructions; per-core instructions 5.88 / 5.97 / 6.06 / 6.15 M in the class-1
  run at c = 64, the same skew as the separate prefill launch).
- **No slowdown from interference** (at (16,2,128) the decode CTAs are 0.7–1.0 % *faster* inside the mixed
  launch, so "no measurable interference" overstates it). Each region's CTA time inside the mixed launch equals its time alone: busy_p ratio
  0.999–1.000, busy_d ratio 0.990–1.001. With the regions barely overlapping, and both bodies far from DRAM limits (DRAM read
  latency 30–36 cycles in the fused launches vs 22–24 for prefill and 37–39 for decode alone; class-1 counters at (8, 2, 64),
  c = 64: scheduler idle 70 / 70 / 68 %, scoreboard stalls 51 / 48 / 52 % for fused / prefill / decode), there is nothing for
  the two phases to contend for on this configuration.
- **Timestamps cost nothing measurable:** −0.003 % and +0.009 % cycles (kind ts-off), ≈ 90 warp instructions per CTA, same
  outputs.

**Consequences.** (i) The fused mixed-batch launch works and is exact; adds its mem-hint/ps_mark CSR writes per region,
can run phase-specific policies inside one launch. (ii) On REG-CFG at the default DRAM model the two phases share the
machine in *time* (one after the other on each core), not in *space*: a mixed iteration is two nearly serial phases with a short
mixed tail, so a sub-kernel policy switch at the prefill→decode boundary of each core is what the mixed region offers here;
genuine concurrent prefill/decode co-residence needs several CTAs per core (smaller CTAs, or more LMEM) or a different CTA order
(interleaving prefill and decode CTA indices so that every core alternates), which the characterization sweep can measure with this kernel.
(iii) Kernel-level fusion gains nothing measurable on SimX by itself (≤ 0.16 %): SimX launch cost is tiny and caches are cold
at every launch anyway; on GPUs the launch saving is the usual argument for fusion. (iv) The per-kernel code-generation
penalty of a fused binary (here 1.7–4.8 % for the prefill part) is a real cost on this toolchain and must be kept out of any
mechanism comparison by comparing fused vs fused (same binary), never fused vs separate.
Scope: REG-CFG, the default DRAM timing model, two shapes, T = 256 (one point T = 1024), one prefill chunk per launch,
lmem/gqa bodies (no DXA), static CTA order; not a statement about mixed attention on GPUs.

### 14.A Mechanism A — A1 per-launch LMEM size, A2 cache way masks (ps patch `0011-waymask`)

Patch `patches/vortex/ps/0011-waymask.patch` (8 SimX files), host helper `kernels/common/ps_waymask.h`, host option `-W` in
`ps_wkv_micro`, `ps_decode_attn_gqa`, `ps_mixed_attn`, tests `tests/test_waymask.py` + `tests/fixtures/ps_waymask_unit.cpp`,
evidence `results/<run>/` (README there has every number).

**A1 — per-launch LMEM usage (software knob).** LMEM *capacity* stays a compile-time constant: `Core` creates its `LocalMem`
with `1 << VX_CFG_LMEM_LOG_SIZE` bytes (`sim/simx/core.cpp:131-138`; the `128-138` includes the comment block above the
call, line 127). What varies per launch is the LMEM *usage*: the runtime writes `vx_launch_info_t::lmem_size` to
`VX_DCR_KMU_LMEM_SIZE` (0x01C, `VX_types.toml:109`) for every launch (`sw/runtime/common/queue.cpp:469`, refused above the
device capacity at `queue.cpp:364`), and the CTA dispatcher strides CTA slots by it (`cta_dispatcher.cpp:110`). The tile
variants are the apps' existing tile options (decode `-t Tt`, mixed `-t Tt` = the prefill key tile; mixed `-w` is the decode tile and leaves LMEM constant, README correction notes, BUG-QUEUE line 356; footprints `ps_attn_lmem` / `ps_pattn_lmem`,
§14.2/§14.6): a smaller key tile needs less LMEM per CTA, at the price of more tiles (more online-softmax rescale steps and
barriers). No kernel instruction stream changes: the tile is a launch argument. The patch adds only observation: with
`PS_WAYMASK_REPORT=1`, SimX prints `PSLMEM launch=<k> lmem_size=<B> lmem_capacity=<B>` at each launch start (the latched KMU
value, `Kmu::lmem_size()`), and the test checks it equals the host's `lmem_bytes`. In the PS-BASE / P2 kernels one CTA uses all
threads of a core (§14.2), so one CTA is resident per core whatever its LMEM size: in this setup A1 changes the tile shape,
not occupancy. Occupancy effects would need kernels with several CTAs per core (none of the P2 kernels).

**A2 — way masks: encoding.** `[dcr_phase] l1_waymask` (0x02A, obeyed by the L1 dcaches only; icaches, L3 and the other caches
ignore it) and `l2_waymask` (0x02B, the L2 only), one 32-bit word each:
- bits 15:0  = allowed-victim mask for every fill whose request class is not WEIGHT (0 = all ways);
- bits 31:16 = allowed-victim mask for WEIGHT-class fills (0 = the same mask as bits 15:0).
WEIGHT is the class of §12 (from the `wgt` range or a `mem_hint`), i.e. the class of the fill's MSHR chain (§12.5). A word of
0 is stock. With PS-BASE-S1-4c geometry (L1D 16 KiB, 4 ways, 4 banks, 64 B lines; L2 1 MiB, 8 ways, 8 banks, 128 B lines)
the fractions used are L1 {0xF all, 0x3 ½, 0x1 ¼} and L2 {0xFF all, 0x0F ½, 0x03 ¼}; a weight partition sets both halves,
e.g. `0x00F0000F` = weights in L2 ways 4-7, everything else in ways 0-3.

**A2 — what a mask does (modeling choices).**
1. *Allocation only, not lookup.* A mask restricts which way a new line may occupy (the victim); every lookup still searches
   all ways, so a line resident in a now-disallowed way still hits. This is the semantics of way-based allocation masks in
   Intel CAT [C42] and of way partitioning in UCP [C40], chosen because it needs no change to the hit path.
2. *Victim rule inside the mask.* `set_t::select_victim(policy, rand_idx, &free, &repl, way_mask)`: the lowest allowed free
   way if any; else PLRU = the allowed way with the largest age (ties → the higher way, as stock); FIFO / RANDOM = the
   (fifo_ptr mod k)-th / (rand_idx mod k)-th allowed way, k = number of allowed ways (fifo_ptr and the random counter advance
   per fill as in stock). With every way allowed each rule is exactly the stock rule (k = ways; checked: the explicit
   all-ways word gives cycles identical to word 0 on both test apps). The PLRU age update is unchanged (all valid lines of
   the set age, masked or not), i.e. the mask partitions *placement*, not the replacement state — the simplest choice;
   per-partition ages would be a different design.
3. *Refills.* A fill that adds a sector to a resident line keeps the line's way (no eviction), so it is not a victim
   selection and is not counted.
4. *Legality (launch boundary + flush).* A mask is a launch-boundary knob. Every launch is followed by the runtime's cache
   flush and preceded by SimX's per-launch reset (cold caches, §1), so when a new mask comes into force every way is empty and
   no line can be stranded in a disallowed way. The cost of that flush is the earlier A2 cost (analytical RTL walk ≈ 1,040 cycles
   at PS-BASE-S1-4c, + dirty write-backs; `results/<run>/csw.csv`); SimX's own flush is untimed (§13), so a mask change is
   charged that analytical cost in any schedule comparison (the host helper prints `changes × 1,040` as
   `flush_walk_cycles_analytical`). Enforcement: (a) `PS_SCHEDULE` rejects `l1_waymask` / `l2_waymask` (unchanged) and
   `scripts/mkschedule.py` does the same; (b) SimX aborts at a launch start or a schedule switch whose config has a mask bit
   ≥ the cache's associativity, a mask on a disabled cache, an L2 mask together with a non-zero L2 insert policy (D, the
   window baseline, SRRIP, SHiP pick or age victims themselves; combining them is not defined here) [ update: patch 0013-compose defines and allows an L2 mask with D's own modes 1/2, §14.E; the other insert policies stay exclusive] or a mask with
   `persist_l2` (lines would outlive the mask); checked on the whole config at launch start, not per DCR write, because the
   host writes the block register by register; (c) the host helper `ps::WayMaskCtl` refuses (`kWaymaskRefused` = −78, config
   untouched) any mask change between `begin_schedule()` and `end_schedule()`, i.e. inside a declared run of launches that
   must share one partition state.
5. *Counters.* Per cache (summed over banks and caches of a level, cumulative since device open): `wm_fills` (victim
   selections under a non-zero word), `wm_fills_wgt` (of those, WEIGHT fills), `wm_violations` (victims outside the fill's
   allowed mask, an independent re-check after `select_victim`; must be 0), `wm_ways_used` (OR of victim ways). Printed with
   `PS_WAYMASK_REPORT=1` as `PSWAYMASK launch=<k> level=l1d|l2 word= ways_used= fills= fills_wgt= violations= reqs= misses=`
   (`reqs`/`misses` per class from §12.9, so a weight-region hit rate is `1 − misses[1]/reqs[1]`). Counting is off for word 0.

**Vantage warning (C41).** Way partitioning gives a partition granularity of one way and reduces the associativity each
partition sees; Vantage [C41] motivates fine-grain partitioning by exactly this cost at low associativity. The 4-way L1 has
only ¼-steps and a ¼ partition is direct-mapped. This step therefore measures the L1 case (ps_wkv_micro `-w 16/32 -c 64`,
weights in the upper ½ or ¼ of the L1 ways) rather than assuming the L2 result carries over (numbers: `results/<run>/README.md`).

**Limits.** (1) Only the L1 dcache and the L2 obey masks. (2) Classes come from ranges/hints; unclassified weights are
ORDINARY and follow bits 15:0. (3) No timing cost is modeled for the masked victim search (RTL would add a mask AND in the
victim-select path). (4) A1 in this kernel set does not change occupancy (one CTA per core).

#### 14.A.b Phase-ignorant baseline UCP-lite (ps patch `0012-a-baselines`)

Patch `patches/vortex/ps/0012-a-baselines.patch` (5 SimX files: `sim/common/phase_config.h`, `sim/simx/mem/cache.{h,cpp}`,
`sim/simx/processor.cpp`, `sim/simx/ps_schedule.h`), `scripts/mkschedule.py` (knob `l2_ucp`), tests `tests/test_ucp.py`,
evidence `results/<run>/` (README there has every number). No host app, kernel or `kernels/common` file changed: UCP-lite
is switched on by a `PS_SCHEDULE` line at work 0, so every app's default instruction stream and run key are unchanged.
Purpose: A2 (above) partitions the L2 by the request class with a *static* mask chosen per launch; UCP [C40] chooses the
partition *at run time from measured utility*, with no phase information. "A2 vs UCP-lite" separates what knowing the
right partition per phase buys from what an online utility monitor finds by itself. CCWS-lite (C17, warp throttling) is
**not implemented** (optional; effort): a limitation — this comparison has no scheduling-side baseline for A.

**What the paper does (checked against the text, read-only: the authors' technical report TR-HPS-2006-009, the long form of
the MICRO 2006 paper C40).** Per core a UMON: an auxiliary tag directory (ATD) with the cache's associativity and true LRU,
and one hit counter per LRU stack position, so the counters give each application's hits as a function of the number of
ways it would get. UMON-DSS samples 32 sets ("set 0 and every 33rd set"). The partitioning algorithm runs every 5 M
cycles; afterwards every hit counter is halved. Each application gets at least one way. For two applications the paper
evaluates every split exhaustively (its Eq. 6); for many it proposes the Lookahead algorithm (its Algorithm 2: repeatedly
give the application with the highest maximum marginal utility, `max_mu = max_k (U(a+k) − U(a))/k`, the smallest k that
reaches it; ties "broken arbitrarily"). Enforcement (§3.6): each line records the core that installed it; on a miss, if
the missing application holds fewer lines in the set than its allocation, the LRU line among the *other* applications'
lines is evicted, else the LRU line among its own; added ways are therefore taken over gradually, on misses only.

**What "applications" are here (our choice).** A Vortex launch is one kernel on every core, so the paper's per-core
applications would partition a cache among identical copies of one program. The meaningful competitors in this
project's workloads are the **request classes** of §12 — exactly the split that A2's two-mask encoding uses. UCP-lite
therefore has two applications: **A = one request class** (`ucp_class()`: WEIGHT by default; KV_STREAM for the
decode / mixed runs, whose class mix is KV_STREAM vs ORDINARY and contains no WEIGHT) and **B = every other class**.
The class of an access is the class of the request (range or hint, §12), never a phase. A line's owner is the class of
the fill that installed it (`line_t::ps_fill_cls`, written on every new-line install, read only by UCP-lite).

**Knob.** `[dcr_phase] flags` bits 15:3 (`PhaseConfig::FLAG_L2_UCP` … ): bit 3 = on; bits 11:4 = epoch N in units of 1,024
L2 cycles (0 → 16, i.e. 16,384 cycles); bits 15:12 = class of A (0 → WEIGHT, else that class id; ≥ 8 rejected). Fields set
without bit 3 are rejected. The `PS_SCHEDULE` knob `l2_ucp=<bits 15:3>` sets / clears exactly these bits ('s
`dxa_fdp` keeps bits 15:3, so the two compose). Exclusive with a non-zero `l2_waymask`, a non-zero insert policy and
`persist_l2`; needs a PLRU L2 (abort at the launch start / schedule switch, `ProcessorImpl::ps_waymask_check`). Bit 3
clear (the reset value) = every path exactly as patch `waymask` (no-op 8/8 identical; the earlier 83 RESULTS cells (79 new runs + 4 sharing run keys with TEST rows; the earlier "79" counted only the new runs, BUG-QUEUE line 367) re-run
identical on the new binary). L2 only (`Cache::ps_ucp_capable()`, the waymask level 2 cache): the 4-way L1 is not
partitioned (Vantage warning above: ¼-way steps).

**UCP-lite as implemented (`struct PsUcp` in `cache.cpp`, one per L2, shared by its 8 banks).** Choices marked *(ours)*
deviate from or are unspecified by the paper.
1. *UMON-DSS.* 32 sampled sets by the paper's rule on the global set index g = bank × sets_per_bank + set (1,024 sets at
   PS-BASE-S1-4c: g = 0, 33, …, 1023). One ATD per application per sampled set: a true-LRU stack of up to 8 line tags.
   Each committed core access of the L2 (the commit point where SimX counts hits/misses and updates PLRU, plus the hit
   deferred behind an ordered MSHR chain, which is committed there) to a sampled set increments `hits[app][p]` if the
   tag is at stack position p, then moves / inserts it at MRU. Line-granular *(ours)*: a sector miss on a resident 128-B
   line (the second 64-B sector) is a hit at the line level, as the ATD has no sectors. Replays of a filling chain are the
   same access and are not counted again.
2. *Epochs.* N cycles (default 16,384; swept 4,096 / 16,384 / 65,536). The paper's 5 M cycles is longer than every launch
   here (0.34–5.2 M cycles), so N is scaled *(ours)* to give 5–780 repartitions per launch (measured). The epoch clock starts at the first
   L2 access with UCP-lite on (the launch start, or a schedule switch) *(ours)*; boundaries are start + k·N. Evaluation
   is lazy (at the first L2 access or fill at or after a boundary, before that access is counted); since nothing happens
   between the boundary and that access, the partition in force for any fill at cycle c is the one decided at the last
   boundary ≤ c, exactly as if evaluated at the boundary.
3. *Lookahead.* At each boundary: each application starts with 1 way (the paper's at-least-one-way rule), then the
   Lookahead with balance = 8 − 2. `U(n)` = sum of the first n hit counters; marginal utilities are compared exactly as
   fractions; for each application the smallest k reaching its maximum mu; the higher mu wins, **ties → A** *(ours: the
   paper breaks ties arbitrarily)*. If neither application has a positive marginal utility for any k, the remaining ways
   are split evenly with the odd way to A *(ours: Algorithm 2 leaves `blocks_req` undefined there)*. The paper
   says Lookahead behaves as the (optimal) greedy algorithm for what it calls convex utility functions; it need not equal
   the exhaustive two-application Eq. 6 in general. We use the Lookahead because names it. Measured: in 9 of 4,144 logged
   epochs the Lookahead picks a strictly lower total-utility split than the exhaustive optimum (README correction notes).
   Then every hit counter is halved (paper §3.5). Initial partition (before the first boundary): 4 : 4 *(ours)*.
4. *Enforcement on the earlier masked victim selection.* A fill of a new line by application a computes a per-set allowed-way
   mask and calls the same `set_t::select_victim(…, way_mask)` as A2: free ways if any (lowest free way); else, if a holds
   fewer lines in the set than its allocation, the ways holding the other application's lines (PLRU = the largest age
   among them = "LRU among the others"); else a's own ways. This is the paper's §3.6 rule with SimX's PLRU age in place of
   true LRU (the PLRU state itself is unchanged, as for A2). A partition is therefore a *count per set*, not a fixed set of
   ways, and a repartition never moves or invalidates a line.
5. *Why no flush, and why it is schedulable.* A2 makes a mask a launch-boundary knob followed by the flush (§14.A item 4) so
   that a new static mask starts from empty ways. UCP-lite does not need that: lookups search every way, a line beyond its
   application's new allocation simply stays until a miss of the application that gained the way evicts it (the paper's
   gradual take-over), and no line is ever in a way it may not occupy (there are no forbidden ways, only per-set counts).
   So the UCP repartitions inside a launch are its normal operation, and switching UCP-lite on or off by `PS_SCHEDULE`
   inside a launch is legal (TEST `test_two_segment_schedule`); turning it on restarts the UMON cold with the 4 : 4 split,
   turning it off returns to plain PLRU over all ways. `l2_waymask` stays rejected by `PS_SCHEDULE`.
6. *State per launch.* The UMON, counters and partition restart at every launch reset (cold caches, §1) and at a
   re-parameterisation (epoch or class change).
7. *Counters / trace (observation only).* `PSUCP start` (cycle, N, class A, initial split), one `PSUCP epoch=<k>` line per
   boundary (cycle, pre-halving `hitsA`/`hitsB`, sampled accesses, `old` and `new` split), `PSUCP end` at the flush or
   switch-off (fills, fills of A, fills that evicted the other application's line, `violations` = victims outside the
   computed mask (must be 0), the occupancy of A over the whole L2). The TEST replays the Lookahead from every epoch line
   in Python, independently of the C++.

**Phase information: none.** UCP-lite reads only `l2_ucp()`, `ucp_epoch()`, `ucp_class()` of the config, the class of each
L2 access/fill, the L2's own tags and the cycle count; no mode, phase id, `ps_mark`, schedule position or range (static
check `test_no_phase_input`). The class ranges exist in these runs because the comparison needs the same class split A2
uses; the class is a property of the address, not of the phase.

**Limits.** (1) L2 only; two applications only. (2) No timing cost for the UMON or the lookahead (the paper estimates
1,920 B per UMON and an N²/2-step algorithm, off the al path). (3) Line-granular UMON on a sectored L2; the UMON does not see L2 atomic hits (the `commitAmo` path returns before the UMON hook; negligible for these kernels, BUG-QUEUE line 367). (4) The epoch
N is scaled, not the paper's 5 M cycles; results are given for three N. (5) CCWS-lite (C17) not implemented.

### 14.D Mechanism D — KV-aware L2 insertion / bypass, and its baselines (ps patch `0005-kv-insert`)

Patch `patches/vortex/ps/0005-kv-insert.patch` (5 SimX files), app `kernels/ps_wkv_micro/`, tests `tests/test_kv_insert.py`,
evidence `results/<run>/` (README there has every number), errata 35 (fill semantics).

**Knob.** `[dcr_phase] insert` (`VX_DCR_PHASE_INSERT_POLICY`), one 32-bit word, written by the host (`ps::PhaseCfg::insert_policy`)
or by a `PS_SCHEDULE` line (`insert=`). Encoding (`PhaseConfig::INS_*`): bits 1:0 = D mode (0 off, 1 insert-at-LRU, 2 bypass;
3 rejected), bit 4 = `rtl_like_fill` baseline (combinable with D), bit 8 = NVIDIA-practice window baseline (exclusive with
D), bits 15:12 = window set-aside ways (0 → ¾ of the ways), bits 24:16 = window hitRatio × 256 (0 → 1.0). Any other bit, D
mode 3, window + D, or window fields without the window bit abort SimX at the DCR write (`Cluster::dcr_write`). Only the L2 acts
on it (`Cache::ps_set_insert_capable`, set by `Cluster` for its L2; L1/L3 see 0), and only with the age-counter PLRU there
(`VX_CFG_L2_REPL_POLICY = 2`, every PS-BASE); any other L2 policy aborts (the "host helper rejects D with FIFO" is
done in SimX instead, so `kernels/common` and every app hash stay unchanged — deviation recorded in the check report).
Word 0 = stock: every new code path is behind `insert != 0`; the no-op regression is identical (8/8).

**What "class" D acts on.** The class of the fill = the class of the request that allocated the MSHR chain (§12.5); KV_STREAM
comes from the KV ranges (`kv[0]`, §12.3) or from `mem_hint` on a hint build. D therefore needs either the host's range (`-R kv`)
or a PS_HINTS=1 binary; uses ranges, so every compared configuration runs the same PS_HINTS=0 binary (rev 2.7).

**SimX replacement model (what is being modified).** SimX's "PLRU" is an exact age counter per line (`line_t::lru_ctr`):
every committed lookup ages all valid lines of the set by one and resets the hit line to 0; the victim is the valid line with
the largest age (ties → higher way), a free way first. A stock fill sets the installed line's age to 0 (MRU insert).
- *D insert-at-LRU (mode 1).* A KV_STREAM fill gets age = 1 + the oldest other valid line of the set (saturating), so it is the
  next victim of that set; the line is marked `ps_low`. The replay of the chain that filled it is the same access, not a reuse,
  so it ages the set like a miss instead of promoting the line; the first genuine (pipelined) hit clears `ps_low` and promotes
  it as stock. This is the LRU-insertion policy (LIP) of DIP [C33] applied to one class. Counter `l2_fill_demoted_<cls>`.
  Note (measured): with several low-priority lines in a set the newest has the largest age, so the newest KV line is the one
  evicted next; earlier low-priority lines survive behind it. Effect: KV occupies ≈ 1 way per set.
- *D bypass (mode 2).* At the L2 input (after range classification), a KV_STREAM plain load whose line the L2 neither holds,
  nor has in an MSHR chain or its pipe, and which has not been written since the launch began, is sent to memory on the L2's
  stock non-cacheable port (the `flags.io` path, `processBypassRequest`) and is never allocated; counter
  `l2_fill_bypassed_<cls>`. Stores, atomics and reads of such lines take the normal path (KV_STREAM fills that still happen are
  demoted as in mode 1). The "written since launch" guard (a set of line addresses, cleared at reset) is needed: without it a
  read can overtake a store still queued in front of the bank and read stale memory — found by the all-KV_STREAM stress test
  on ps_prefill_attn (misaligned access / corrupted response tag), fixed in the patch, re-tested.
- *rtl_like_fill (bit 4).* The fill leaves the replacement state of the way as it was (the victim's pre-fill age, i.e. the
  oldest; 0 for a never-used way) — the RTL rule "PLRU state advances on lookup hits only" (errata 15/35). Because in SimX nearly
  every L2 read fill drains through the fill-forward path, which does not update recency (errata 35, measured ≈ 12,290 forwards
  vs 16 pipelined replays per bank), this makes every class LRU-inserted (LIP for all). It is a *baseline*: "what the stock
  RTL insertion would do", reported next to the SimX default so that D's gain is not a SimX artefact of MRU fills.
- *NVIDIA-practice static window (bit 8).* Model of CUDA's L2 access-policy window [C46]: addresses in `wgt` are the persisting
  window; a line in it persists if `h(line) < hitRatio·256`, `h` = top 8 bits of line_addr × 0x9E3779B97F4A7C15 (a fixed
  multiplicative hash, so the persisting subset is a fixed, address-determined fraction like CUDA's hitRatio); the rest of the
  window and every line of `kv[0]`/`kv[1]` are streaming (inserted as D mode 1). Persisting lines may hold at most the set-aside
  number of ways per set (the analogue of the set-aside L2 portion); a persisting fill into a full set-aside replaces the oldest
  persisting line, every other fill replaces the oldest non-persisting line. The window is per-launch static: a schedule line
  with work_count > 0 that changes bits 8..24 aborts. Caches are cold at every launch (§1), so its effect is within one launch;
  across launches it only matters in persist_l2 sensitivity runs (§5.2; not run here). Modeling choices and their limits:
  set-aside as *ways per set* (not a capacity carved anywhere) and a hash instead of CUDA's unspecified selection; with
  hitRatio = 1 and no weight traffic (decode/mixed attention) it equals "all KV streaming", i.e. D LRU applied to kv0 *and*
  kv1 — the phase-blind version of D.

**Gains reported.** `gain_vs_X = cycles(X) / cycles(D) − 1` (> 0: D faster), cycles = kernel cycles of the run's single
launch (sum of its work rows; cold caches), against X = SimX default (mode 0), rtl_like_fill (bit 4, with D on top: 0x11/0x12
vs 0x10) and the static window (0x100). DRAM traffic is the issued bytes (`dram_rd_bytes`, `dram_wr_bytes`); write residency
is never used (errata 24 addendum). memhier's D prediction (`Knobs.kv_insert`) is printed next to SimX and carries no weight (§9.8).

**Results (PS-BASE-S1-4c cold, 1 MiB and 512 KiB L2; `results/<run>/gains.csv`, `memhier.csv`).**
- *Thrash microbenchmark* (`ps_wkv_micro -w 512 -c 1024 -r 16`, TEST sizing, no resize needed): weight-region L2 hit rate
  0.000 stock → 0.9375 (= 15/16, the ideal: only the cold first round misses) with D LRU, D bypass and the window; 0.704 with
  rtl_like_fill. Kernel cycles 1,155,095 (stock) / 1,050,009 (rtl_like) / 1,015,977 (window, D LRU, rtl+D LRU) / 902,767
  (D bypass, rtl+bypass). D LRU gain: +13.7 % vs SimX default, +3.4 % vs rtl_like, 0 vs window; D bypass: +28.0 % / +16.3 % /
  +12.5 %. DRAM reads 25.2 → 17.3 MB. Bypass is faster than LRU at the same hit rate because the KV stream no longer passes
  through the L2 banks/MSHRs (no allocation, no fill pipeline slot).
- *wkv sizes:* w256 (¼ L2): LRU +7.2 %, bypass +23.1 % vs SimX; w768 (weight =
  ¾ L2): LRU +16.5 %, bypass +28.7 %; at 512 KiB L2 the w768 block no longer fits, LRU 0, bypass +8.5 %.
- *Decode attention 3×3 and the three mixed cells (the real kernels):* |gain| ≤ 1.15 % vs the SimX default in every mode
  and both L2 sizes (largest: bypass B2_T256 at 512 KiB), ≤ 0.51 % vs rtl_like and ≤ 0.82 % vs the window except one cell
  (c64_s0_n4_T256 at 512 KiB: LRU +2.2 % vs rtl_like, +1.8 % vs window). Reason: one launch of these kernels has no
  re-referenced data competing with the KV stream (q, out and the code are tiny; KV is read once), so insertion policy cannot
  matter inside a launch; with cold caches per launch (§1) there is no cross-launch reuse either. rtl_like_fill itself moves
  these kernels by −2.2 … +1.4 %.
- *memhier* predicts 0 gain for all decode cells and +0.2–0.3 % for c32_s32_n8_T512 (both modes), i.e. the same
  qualitative picture (qualitative only, §9.8).
- What D needs to matter: intra-launch reuse of a non-KV working set of the order of the L2 (the wkv case), i.e. fused
  multi-op launches or persistent L2 across launches (sensitivity), not the current per-op attention launches.

#### 14.D.b Phase-ignorant baselines SRRIP and SHiP-Mem-lite (ps patch `0006-d-baselines`)

Patch `patches/vortex/ps/0006-d-baselines.patch` (3 SimX files: `phase_config.h`, `mem/cache.h`, `mem/cache.cpp`), app
`kernels/ps_scan_micro/`, tests `tests/test_d_baselines.py`, evidence `results/<run>/` (README there has every number).
Purpose: D (above) uses the request class; these two published replacement policies use only addresses and hit/miss
history, so "D vs SRRIP / SHiP" separates what the KV-class information buys from what any scan-resistant replacement buys.
Hawkeye [C36] and Mockingjay [C37] are named in R§4 but **not implemented** (effort): a limitation of this
comparison — both learn from a sampled Belady (OPTgen) history and could do better than SHiP on a reuse pattern like wkv.

**Knob.** Bits 6:5 of `[dcr_phase] insert` (`PhaseConfig::INS_REPL_*`): 0 off, 1 = SRRIP (word 0x20), 2 = SHiP-Mem-lite
(0x40), 3 rejected. Exclusive with D (bits 1:0), `rtl_like_fill` (bit 4) and the window (bit 8): any combination aborts at the
DCR write (`Cluster::dcr_write` → `insert_policy_error`), as does any non-PLRU L2 (the earlier guard; the baselines replace PLRU's
victim choice, so they are defined on the PLRU-configured L2 only). L2 only (`ps_set_insert_capable`). Bits 6:5 are
outside the window's per-launch-static field (bits 8..24), so the baselines **are schedulable**: a `PS_SCHEDULE` line may
switch them on/off inside a launch (TEST `test_two_segment`). Word 0 = stock; every new path is behind `ps_repl != 0`
(no-op regression identical).

**Phase information: none.** Neither policy reads the request class, the kv/wgt ranges, `mem_hint`, the window fields or any
schedule/phase input; the only class use is the index of the accounting counter `l2_fill_demoted_<cls>` (static check
`test_policies_phase_ignorant`). The ranges in these runs exist only so the class counters can report the weight / KV hit
rates.

**SRRIP [C34] as implemented.** Per line a 2-bit RRPV (`line_t::ps_rrpv`, M = 2, values 0 "near" … 3 "distant").
- Insertion: RRPV = 2 ("long", 2^M − 2), the static-RRIP rule. A refill (second 64-B sector of a resident 128-B L2 line)
  keeps the line's state — it is not a new insertion.
- Hit: RRPV = 0 (hit-priority promotion, RRIP-HP). A *committed* lookup hit only (the same call sites as SimX's
  `update_lru`: core hit, AMO hit, replays). The replay of the chain that filled a still-fresh line (`ps_fresh`) is the same
  access, not a re-reference, and does not promote — the same rule uses for D's `ps_low` (fill-forwarded reads never
  update replacement state in SimX, errata 35). A miss changes nothing (RRIP ages only during the victim search).
- Victim: a free way first; else RRIP's loop "find the first way with RRPV 3, else increment every RRPV and repeat". It is
  computed in closed form without mutating state (the fill can still stall on the write-back queue after the choice): the
  victim is the lowest way among the lines with the largest RRPV, and the increment applied to every valid line when the
  fill commits is 3 − that RRPV. Equivalent: equal increments keep the order, so the first line to reach 3 is the first
  line holding the maximum. "First way" = lowest way index (the paper scans from a fixed end; the choice only orders ties).
- Lines installed while neither baseline was active carry RRPV 2 (`line_t::reset()`), i.e. after a schedule switch they
  behave as if SRRIP had just inserted them; SimX's age counters keep being updated in baseline mode, so switching back to
  stock is also well defined.

**SHiP-Mem-lite [C35] as implemented.** SRRIP plus the SHiP signature history counter table, with the *memory-region*
signature (SHiP-Mem) and simplifications ("lite") chosen here:
- Signature = 16 KiB address region (`addr >> 14`). SHCT = 16 K direct-mapped 3-bit saturating counters, one table per
  cache shared by all banks (a region's lines are interleaved across banks, so per-bank tables would learn 1/banks as fast).
  16 K × 16 KiB = 256 MiB before two regions alias — larger than every footprint, so no hashing is needed and the
  table never aliases in these runs (a hardware table would be smaller; this is the predictor's best case).
- Per line: the outcome bit (`ps_outcome`, re-referenced since fill). Hit → outcome = 1 and SHCT[sig]++ (saturating at
  7). Eviction of a valid victim with outcome 0 (a dead line) → SHCT[sig]−− (saturating at 0). Invalidations (persist-mode
  host writes, AMO probes) and the end-of-launch flush are not replacement evictions and do not train.
- Insertion: RRPV 3 ("distant") if SHCT[sig] = 0, else 2 (as SRRIP). Distant insertions are counted in `l2_fill_demoted`.
- Initial value 1 at every launch reset (§1: caches are cold per launch; the predictor state is reset with them, and kept
  in persist_l2 mode like the lines). Modeling choice: an untrained region is predicted "re-referenced" (insert as SRRIP),
  and a single dead eviction is enough to predict it distant — the fastest learning a 3-bit counter allows, i.e. the
  baseline's best case on streaming regions; a region that is re-used recovers with each hit. Not taken from the paper.
- Instrumentation (stdout, SHiP runs only, observation only): `PSSHCT first idx= cycle= val=` at the first fill that uses an
  entry, `PSSHCT zero idx= cycle=` when an entry first reaches 0, and one `PSSHCT dump idx= val=` per used entry at the
  end-of-launch L2 flush.

**Scan-resistance microbenchmark `ps_scan_micro`.** = `ps_wkv_micro` (unchanged) plus `-p` warm-up passes over the
weight block before round 0. Why a new app and why the warm-up: in `ps_wkv_micro -w 512 -c 1024 -r 16` every weight
re-reference comes after a KV chunk ≥ L2 − weight bytes, so no weight line is ever hit while resident; SRRIP then inserts
every line at the same RRPV and evicts in way order, which is exactly the stock age-counter order for a hit-free stream —
measured: SRRIP cycles **identical** to stock on that point (dev run, and the RESULTS sweep). RRIP protects a block only
after it has been re-referenced once while resident. One warm-up pass (block read twice back to back; it fits the L2) gives
it that first hit; then 16 rounds interleave the re-used 512 KiB (= ½ L2) block with a one-pass 16 MiB (= 16 × L2) scan,
the TEST shape. Under LRU the warm-up hit does not help (the next 1 MiB chunk evicts the block). `ps_wkv_micro` is left
unchanged so the earlier evidence keeps its binary; the "resize rule" is applied as "reshape" (recorded as a deviation).

**Gains reported** (`results/<run>/gains.csv`; cycles = kernel cycles of the single cold launch, every mode re-run on this
patch's SimX so every ratio compares one simulator binary and one app binary):
`gain_<B>_vs_lru = cyc(0x0)/cyc(B) − 1` for B ∈ {SRRIP, SHiP} (> 0: the baseline is faster than stock LRU-like SimX), and
`gain_<D>_vs_<B> = cyc(B)/cyc(D) − 1` for D ∈ {LRU 0x1, bypass 0x2} (> 0: D is faster than the phase-ignorant baseline),
next to the earlier D gains vs SimX default / rtl_like / window recomputed on the same binary.

**Results (PS-BASE-S1-4c cold, 1 MiB and 512 KiB L2; `results/<run>/gains.csv`, 288 runs, all simulated in full).**
- *Scan TEST* (`ps_scan_micro -w 512 -c 1024 -r 16 -p 1`): weight-region L2 hit 0.059 stock → 0.935 SRRIP, 0.941 SHiP, 0.941
  D LRU / bypass. Cycles 1,175,623 stock / 1,038,064 SRRIP / 1,036,891 SHiP / 1,035,189 D LRU / 921,133 D bypass: on this
  shape the phase-ignorant baselines recover all of D LRU's gain (D LRU vs SRRIP +0.28 %), and only bypass stays ahead
  (+12.7 % vs SRRIP). Without the warm-up (`-p 0`) SRRIP = stock to the cycle. SHiP: every KV-region SHCT entry reaches 0
  within ≈ 49 k cycles of its chunk's first touch (< 2 chunk periods); weight regions end at 7.
- *wkv (the earlier thrash case, no warm-up):* SRRIP = stock exactly (gain 0.0 for w256/w512/w768, both L2 sizes; no weight line is
  ever hit while resident). SHiP +2.9 / +5.9 / +6.3 % vs stock at 1 MiB (its dead-region learning demotes part of the KV
  stream); D keeps its lead: D LRU +7.2 / +13.7 / +16.5 % vs SRRIP and +4.2 / +7.4 / +9.6 % vs SHiP; D bypass +23 – 29 % vs
  SRRIP, +19.7 – 21.0 % vs SHiP. At 512 KiB w768 (block does not fit): D LRU −4.3 % vs SHiP, bypass +3.9 %.
- *Decode 3×3 and the three mixed cells:* SRRIP and SHiP move cycles by −0.07 … +1.5 % vs stock; D vs either baseline
  −1.2 … +0.5 %. As for D (above), one launch of these kernels has no reuse to protect, so no insertion/replacement policy
  matters much; the small baseline gains (≤ 1.5 %) are ordering effects, not reuse.
- Takeaway for D's claim: on workloads with a resident, re-hit working set a phase-ignorant scan-resistant policy gets D
  LRU's benefit without class information; the class information pays where re-use has no in-cache first hit (wkv) and
  through bypass (no allocation at all), which no replacement policy can emulate.

### 14.B Mechanism B — two-class DRAM arbitration (ps patch `0007-dram-2class`)

Patch `patches/vortex/ps/0007-dram-2class.patch` (9 files: `sim/common/{dram_sim.cpp,dram_sim.h,ps_dram_b.h,ps_dram_ctrl.cpp}`,
`sim/simx/{mem/memory.{h,cpp},processor.cpp,processor_impl.h}`, `third_party/Makefile`), app `kernels/ps_mix_micro/`, tests
`tests/test_dram_2class.py`, evidence `results/<run>/` (every number is in its README).

**Scope: simulation only.** Every B result is a property of the simulated DRAM controller, not of the Vortex RTL. The controller is a
C++ Ramulator 2.0 model [D7] that SimX and the RTL simulators share (`sim/common/dram_sim.cpp`). The RTL top level discards the memory
attribute, and an FPGA's vendor controller would ignore class bits (R§2 fact 3). B therefore has no RTL or FPGA counterpart in this
project, and its area/energy is not modelled.

**Queue location (measured first, as the requires).** With stock FR-FCFS [C19] on `ps_mix_micro` at PS-BASE-S1-4c, DramSim's
front FIFO (`pending_txs`, sampled once per DRAM clock before the head is offered to Ramulator) holds 0.29 transactions on average
over all clocks and 0.74 over busy clocks. Ramulator refuses the head on only 0.3 % of accepts. Ramulator's read buffer holds 1.42
requests over all clocks and 5.31 over busy clocks (PSDRAMQ / PSDRAMB lines, `results/<run>/README.md`). The mean front depth is below 1, so a front-only
two-queue design would be close to a no-op: requests wait inside the controller. Priority therefore acts inside Ramulator (the
**Ramulator path**). The front FIFO also needs a class rule, for a reason found during development: see "Front admission" below.

**Classes.** BULK = `VX_PS_CLASS_BULK_DXA` (every DXA GMEM read, tagged where DxaCore builds it, patch mem-class). DEMAND = every other
class (ORDINARY, WEIGHT, KV_PERSIST, KV_STREAM). This is the same split as the MSHR merge priority `ps_class_prio` (§12). The class
travels from `DramSim::send_request` into each transaction and on to Ramulator's `Request::source_id`. Stock passes 0 there; nothing
else in the pinned Ramulator reads `source_id` except the Generic controller's per-core statistics.

**Knob.** `[dcr_phase] dram_sched` / `dram_aging`, reserved. Both are written by the host or by a `PS_SCHEDULE` line
(`dram_sched=` / `dram_aging=`); DramSim reads them whenever the latched config changes, so a schedule can switch B inside a launch.
- `dram_sched` bits 1:0 = mode: 0 = stock FR-FCFS, 1 = **strict** two-class, 2 = **work-conserving** two-class, 3 = rejected.
- `dram_sched` bits 15:8 = BULK slot cap C (0 → 8). A cap with mode 0 is rejected, and so is any other bit; a rejected value aborts
  SimX at the push (`DramSim::set_phase_config`).
- `dram_aging` = W in **DRAM controller clocks**; 0 = no aging. At PS-BASE (800 MT/s, 400 MHz core) one controller clock equals one
  core cycle; on the quarter-rate sensitivity machine (200 MT/s) it equals 4 core cycles.

**Controller (`PsTwoClass`, `sim/common/ps_dram_ctrl.cpp`).** A copy of Ramulator's `GenericDRAMController` (MIT licence; same
buffers, write-mode watermarks, active-buffer-first rule, closing-row check and refresh hook). It is built into
`third_party/ramulator_ext/libramulator-vortex.so` and registers with Ramulator's factory when that library loads. This is the same
out-of-tree mechanism the shipped GDDR6 model already uses, so the open question about out-of-tree registration (internals Q3)
is answered by that precedent, and the pinned Ramulator source is not patched (deviation in ``). DramSim selects
`PsTwoClass` for every channel and every mode.

With mode 0 the read/write-buffer choice is the configured scheduler's (`FRFCFS`: ready first, then oldest). The only other changes
are statistics sized for the 8 classes and observation counters, so mode 0 is the stock controller. This is tested: a static
line-for-line test against the Generic source, and the no-op regression. With mode 1/2 the read/write-buffer choice is
`select_two_class`, defined below. The **active buffer** keeps stock FR-FCFS in every mode. Its entries are requests whose row was
already opened for them (at most one per bank), that is, the continuations of requests that were already granted. Leaving them alone
avoids wasted activates, and it keeps rows from being left open in a way that would trip the closing-row check.

Definitions used by the rule and by the counters (per buffer, read and write separately):
- a request's **grant** = the clock its first command issues (the controller's `is_stat_updated` transition);
- the **BULK head** = the oldest BULK request in the buffer (buffer order = arrival order);
- the **BULK class wait** `b(t) = t − max(arrival of the BULK head, last BULK grant in this buffer)` = how long the BULK class has
  gone unserved. Using the later of the two clocks makes W a per-class service guarantee: one BULK grant at least every
  ≈ W + one service time while BULK waits. It is not an age bound on every BULK request. A per-request age bound cannot hold
  for a backlog of k BULK requests, because the k-th waits for the k − 1 before it.

Rule (`aged` = W > 0 and b(t) > W):
- **strict (1):** if `aged`, the BULK head. Otherwise FR-FCFS among DEMAND whenever any DEMAND request waits in the buffer, and
  FR-FCFS among BULK only when none does. The choice stands even when it is not ready this clock, so the controller is not
  work-conserving (counted: `strict_idle`).
- **work-conserving (2):** the first ready of: the aged BULK head, FR-FCFS among DEMAND, FR-FCFS among BULK. If nothing is ready: the
  head if aged, else the DEMAND choice, else the BULK choice.
- W = 0 gives pure class priority. This is the earlier "strict demand-first (no aging)" baseline, and it can starve BULK (measured).

**Front admission (DramSim, modes 1/2).** The front FIFO stays FIFO within each class. Between classes, `front_pick` applies the
same order as the controller: the BULK head first if its front class wait exceeds W, otherwise the oldest DEMAND transaction,
otherwise the BULK head. In addition, a BULK transaction is offered only while fewer than C BULK requests sit in the controller's
read and write buffers (C = 8 of the 32 + 1 read-buffer slots). Both rules come from a development measurement at a quarter of the
PS-BASE data rate, where the DRAM is the bottleneck:
- Without them, strict mode slowed DEMAND by 73 % (the GEMV span went from 558 k to 968 k cycles). The BULK requests that the
  controller held back filled its bounded read buffer (15.6 of 32 slots on average on channel 0). The single front FIFO then
  queued DEMAND behind BULK (head-of-line blocking), and DEMAND kept only the slots that were left.
- With class order alone the GEMV span was still 785 k cycles.
- With the cap as well, strict W = 0 gives 510 k cycles, 9 % faster than FR-FCFS.

The cap therefore reserves at least 3/4 of the buffer for DEMAND. Its value is a modeling choice; it is the knob's default and can be
set in bits 15:8. With mode 0 the head is always taken (`std::deque` replaces `std::queue`; only `front()`/`pop_front()` are used).

**Invariants (TEST, strict mode).** Counted by the controller, cumulative per channel, printed by `PS_DRAM_REPORT=1`:
- `viol` = BULK grants made while a DEMAND request waited in the same buffer and b ≤ W (or W = 0). Required: 0.
- `bwait_max` = the maximum b at a BULK grant.
- `svc_max` = the longest observed service time, defined as final-command clock − grant clock over all requests.
- Required for W > 0: `bwait_max` ≤ W + `svc_max`. An aged head is granted as soon as its first command can issue; the delay
  until then comes from the bank's preceding command sequence and the active-buffer continuations, and each of those is bounded by
  one service time.
- `aged_grants` counts BULK grants made with b > W.
- With mode 0 the same `viol` counter measures how often stock FR-FCFS grants BULK ahead of waiting DEMAND. It is > 0 on the TEST
  workload, so the invariant is not satisfied trivially by the workload.

**Latency metrics (RESULTS).**
- `PSDRAMLAT`: the exact distribution, in 1-cycle buckets, of the tracked read residency in the DRAM model per class (DEMAND,
  BULK). The interval is the one PsStats uses (§12.9): from `send_request` until the response is handed back. It is in core
  cycles, and the reported quantities are mean, P50, P99 and max (buckets ≥ 8191 are open, and the line says `(overflow)`).
- `PSDRAMB qwait_*`: arrival in the controller → final command, in controller clocks.
- "Decode-class DRAM wait" in the README = DEMAND `PSDRAMLAT`.
- Kernel cycles = the `PERF` cycles of the launch. `ps_mix_micro` also prints the DEMAND (GEMV warps) and BULK (DXA warp)
  completion spans from per-warp `mcycle` stamps.

**Known limits.**
1. B acts only where a BULK request competes with DEMAND in the same channel. Kernels without DXA have no BULK traffic, so modes 1/2
   take exactly the FR-FCFS decisions there (tested: identical cycles on `mixed_attn` and the stories260K decode). The model then
   predicts a gain of exactly 0 for those kernels.
2. The DXA variant of the decode-attention kernel fetches the decode step's own K/V through DXA, which is tagged BULK. B then
   deprioritises the al traffic. This is the failure case named in the M9 row of §9 (the earlier per-stream class fixes the tag). The small decode-DXA B gain measured (+0.4 … +2.5 % at lowbw) is a mis-tag artifact: with the K/V stream tagged KV_STREAM, B is cycle-identical to FR-FCFS in those cells, so that gain must not be cited as a B result.
3. Strict mode gives up bandwidth when the chosen request is not ready (`strict_idle`). On PS-BASE the DRAM is far from saturated
   (the GEMV alone uses ~21 % of the 64 B/cycle peak, and the DXA copy is limited by the single DXA engine to ~5 B/cycle), so
   B's effects at PS-BASE are small by construction. The quarter-rate machine is a sensitivity point, not a baseline.
4. No refresh is modelled (stock DramSim choice).
5. B sees only the DRAM controller. Contention in the L2 MSHRs, the crossbar and the LSU queues ahead of it is not arbitrated by
   class.

The B baselines (strict demand-first = mode 1 with W = 0; BLISS [C22]; PADC-lite [C25]) are 's. They plug into the same
controller as further modes.

### 14.B.b B baselines: strict demand-first, BLISS, PADC-lite (ps patch `0008-b-baselines`)

Patch `patches/vortex/ps/0008-b-baselines.patch` (8 files: `sim/common/{dram_sim.cpp,dram_sim.h,ps_dram_b.h,ps_dram_ctrl.cpp}`,
`sim/simx/{mem/memory.{h,cpp},processor.cpp,processor_impl.h}`), tests `tests/test_b_baselines.py`, evidence `results/<run>/`
(numbers in its README). Simulation only, as §14.B.

**Controller choice.** The pinned Ramulator ships BLISS as an `IBHScheduler` (`bliss_scheduler.cpp`) that needs
the BH controller and the BLISS plugin (`plugin/bliss/bliss.cpp`); DramSim uses a Generic-derived controller. The preferred
route was taken: BLISS's blacklisting logic is **ported into `PsTwoClass`** (the controller, itself a copy of the Generic
controller) as a further selectable mode. Every policy of this step and (FR-FCFS, strict demand-first, B strict /
work-conserving, BLISS, PADC-lite) therefore runs in one controller: the same read/write buffers, write-mode watermarks,
active-buffer-first rule, closing-row check and open-row policy. Only the read/write-buffer choice differs. The active buffer
keeps FR-FCFS in every mode (as in §14.B). The controller's mode-0..2 decision paths are unchanged, but the binary changed, so
**FR-FCFS and the earlier B modes were re-run on the binary**, and every comparison in `results/<run>/` is within that binary.
The no-op regression shows that mode 0 is still bit-identical to base and to ps 0002–0007.

**Encoding (`dram_sched`, the reserved field).** Bits 7:4 select a baseline policy; they are valid only with bits 1:0 = 0 and no bit
above 15:
- `0x?10` = BLISS. Bits 15:8 = blacklisting threshold T (0 → 4). `dram_aging` = clearing interval I in DRAM controller clocks
  (0 → 10000). The defaults are the pinned plugin's `blacklist_thresh` / `unblacklist_cycles` defaults.
- `0x?20` = PADC-lite. Bits 15:8 = usefulness threshold θ in percent (0 → 50, values up to 255 allowed). `dram_aging` = epoch length
  E in core cycles (0 → 10000).
- Policy values 3–15 are rejected ("baseline policy reserved"). So is a policy with mode bits or with bits ≥ 16. the earlier rejections
  (mode 3, bit 2/3, a cap with mode 0) are unchanged.
- **Strict demand-first (no aging)** needs no new encoding: it is `dram_sched=1`, `dram_aging=0` (§14.B, W = 0: pure class priority,
  with front class order and the BULK slot cap 8). Here it is run as mode `m1_w0`.

**BLISS (mode 3) [C22].** Sources = the request class carried in `source_id` (8 classes; at the DRAM only DEMAND classes and
BULK_DXA occur in the workloads). The paper's sources are applications or cores. A single GPU kernel has one application, and the
competition B is about is between traffic classes, so the class is the natural source here. It is also the only source identity
DramSim passes. The ported rule:
- *Streak and blacklisting* (pinned `BLISS::update`). Per channel there is one counter c and the previous source p. On every
  serviced request of source s: c ← c + 1; if s ≠ p then p ← s, c ← 0; if c ≥ T then s is blacklisted. c is the streak length − 1,
  so the **(T + 1)-th consecutive serviced request of one source blacklists it** ("exceeds T"). A blacklist bit stays set until the
  next clear.
- *Clearing.* Every bit is cleared on each controller clock that is a multiple of I (`m_clk % I == 0`). As in the pinned plugin, this
  happens before that clock's streak update, and the streak counter itself is not reset.
- *Selection* (pinned `BLISSScheduler::compare`, a left fold over the buffer): non-blacklisted ("safe") first, then ready, then the
  older (`arrive <=`). Every request in the read/write buffers is a read or a write, so safe = not blacklisted. A source id < 0
  counts as blacklisted, as in the pinned plugin, but no such request reaches these buffers.
- *Front FIFO:* stock FIFO (BLISS has no front rule).
- *Differences from the pinned code, and why:*
  - (i) A streak event is a **serviced request**, i.e. the issue of its final RD/WR command, wherever it issues (usually from the
    active buffer after its ACT). The pinned plugin counts every issued command with `request_found`, which includes ACT and PRE,
    so one row-miss request would count 2–3 times. The paper counts requests served. The pinned plugin even looks up RD/WR
    command ids it never uses, which suggests the same intent.
  - (ii) The blacklist is sized for 8 classes. The pinned plugin sizes it by the frontend's core count, which is 1 for DramSim's
    GEM5 frontend, so a class id > 0 would index out of range (the same issue as erratum 36b).
  - (iii) The safe and ready flags are recomputed, not kept in `scratchpad[0..1]`, because PsTwoClass keeps the grant clock in
    `scratchpad[0]` (§14.B).
- *Counter trace (TEST):* `PS_BLISS_TRACE=<prefix>` writes `<prefix>.<channel>` with one `S,clk,src,streak,bl_after` row per
  serviced request and one `C,clk` row per clear. `tests/test_b_baselines.py` replays it independently.
- *Counters:* `bliss_bl_grants` = grants of blacklisted-source requests. `bliss_unsafe_grants` = those granted while a
  non-blacklisted request waited in the same buffer; it must be 0, using the blacklist bits the selection saw.

**PADC-lite (mode 4) [C25].** PADC [C25] measures each core's prefetch accuracy per interval. It treats prefetches like demands
when accuracy is high and deprioritises them when it is low. Its adaptive prefetch dropping (APD) also removes old, low-accuracy
prefetches from the queue. The "lite" reduction used here:
- *Stream.* The DXA engine's GMEM reads (class BULK_DXA) form one stream at the DRAM. counts DXA bytes per worker slot, but the
  DRAM request carries only the class, so it cannot tell the slots apart. Per-stream classes are mechanism C.
- *Usefulness.* For epoch k (E core cycles, restarted at every launch), the ratio is

      u_k = Δ(core LMEM read bytes) / Δ(DXA bytes).

  Core LMEM read bytes are the LocalMem read requests × the LSU word (4 B), summed over cores. DXA bytes are the counters
  `dxa_bytes_s0..3`, summed (one L1 line per DXA GMEM read; every DXA transfer here lands in LMEM). u is the fraction of what DXA
  delivered that the cores then read. It is a device-wide, word-granular proxy for PADC's per-prefetch "used before eviction" bit.
  Words read twice count twice, so u can exceed 1 (GQA heads share K/V). Data that cores write into LMEM themselves and read back
  also counts. The proxy is therefore biased toward "useful", and it can only undercount a stream that is in fact used.
  Epochs without DXA bytes keep the previous state. The first epoch after PADC-lite is switched on only takes the counter base.
- *Decision.* low ⇔ 100·u_k < θ, evaluated at the end of each epoch, and held until the next judged epoch. Within an epoch the state
  is fixed.
  - Not low: FR-FCFS, i.e. BULK is treated like DEMAND, PADC's "accurate prefetches are demands".
  - Low: exactly strict demand-first, i.e. mode 1 with W = 0, front class order and BULK cap 8.
  - Counter `padc_low_viol` = BULK grants past a waiting DEMAND request while low; it must be 0.
- *No dropping.* A DXA read is not speculative: its data is written to LMEM and the kernel waits on the transfer's barrier. Dropping
  it would hang or corrupt the kernel. PADC-lite therefore only deprioritises. APD has no counterpart, and that is a stated limit.
- *θ = 50 % and E = 10000.* These are modeling choices for the default, not values taken from [C25]. θ = 255 % is used in the TEST
  to force the low state on a fully consumed stream.
- *Measured consequence (RESULTS, `results/<run>/README.md`).*
  - **ps_mix_micro:** the DXA stream is consumed in full (epoch u ≈ 1.0, min 0.75–0.91). At the default θ, PADC-lite never
    deprioritises on base (d2, d4) or lowbw d2. It then **reproduces FR-FCFS exactly** (identical cycles; tested on d2).
  - **lowbw d4:** 10 epochs were judged low (u min 0.16, a lag between delivery and consumption at the tile boundaries), and the
    kernel came out −0.5 %.
  - **Decode-attention DXA variant:** mean u is 1.8–7.2, because the GQA heads re-read the shared K/V. Single epochs fall to
    u ≈ 0 when DXA delivers a tile that is read in the next epoch, so PADC-lite deprioritises in 0–3 epochs per run.
  - So the per-epoch usefulness proxy is noisy at E = 10000. Over a whole run it says "useful" for every stream here, which is
    the correct PADC behaviour for an accurate prefetcher.
  - It also means a usefulness-driven baseline cannot separate the decode step's own K/V (al) from a background copy (not
    al): both are fully used. Usefulness is not ality. Phase information or a per-stream class is what
    separates them, and that is the PhaseShift argument this baseline is meant to test.
- *Trace (TEST):* `PS_PADC_TRACE=1` prints one `PSPADCEP epoch cycle lmem_rd_bytes dxa_bytes u theta low` line per epoch. The
  `PSPADC` summary and the `PSDRAMX` counters are printed with `PS_DRAM_REPORT=1`.

**Limits.**
1. BLISS with two classes alternates blacklisting between DEMAND and BULK runs. Its fairness rationale (no application
   monopolises the bus) is about applications, so for classes it is a heuristic baseline. It is not tuned (T and I are the pinned
   defaults). With two request sources and per-request streaks about 99.5 % of serviced requests come from a blacklisted source
   (35,276 of 35,452 on base, `:141`), so BLISS degenerates toward ready-first FR-FCFS here; read its columns as
   that limit. Results use T = 4 plus one T = 2 test point only (no T sweep), and BLISS state is not reset on a mid-run mode switch.
2. PADC-lite has one device-wide usefulness value and no APD.
3. Both baselines see only the DRAM controller, as B does (§14.B limit 5).

### 14.C Mechanism C — DXA stream classes, per-class depth, L2-prefetch-only (ps patch `0009-dxa-class`)

**What it is.** Mechanism C gives each DXA transfer stream a request class and lets the phase controller bound how many GMEM reads
of each class the DXA engine keeps in flight. The zero-hardware layer is the software DXA depth (the kernels' `-p` stages); this patch adds the hardware knobs. Everything is SimX-only: the RTL ignores the new META bits (below), as it ignores every
PhaseShift class tag (R§2 fact 3).

**META bit usage.** The descriptor META DCR word is 32 bits wide (`VX_DCR_DATA_BITS`). The stock fields are defined
twice, by widths whose running sum gives the offsets: `hw/rtl/dxa/VX_dxa_pkg.sv:36-50` (RTL) and `sw/common/dxa_meta.h:21-33`
(host encoder and SimX decoder). The the earlier pointers (`:37-51`, `:22-27`) are off by one line (now points here; BUG-QUEUE line 317); the design doc's
`VX_types.toml:282-291` pointer is wrong (those lines are OM blend constants), as R§2 says.

| bits | field | width | owner | read by |
|---|---|---|---|---|
| 2:0 | DIM (rank) | 3 | stock | RTL `VX_dxa_setup.sv`, SimX |
| 4:3 | ELEMSZ (log2 bytes) | 2 | stock | RTL, SimX |
| 6:5 | LAYOUT (row-major / K-major / Flat / BlockMajor) | 2 | stock | RTL, SimX |
| 8:7 | SWIZZLE | 2 | stock | nobody (reserved upstream) |
| 10:9 | INTERLEAVE | 2 | stock | nobody (reserved upstream) |
| 12:11 | L2PROMO | 2 | stock | nobody (reserved upstream) |
| **15:13** | **PS_CLASS**: `VX_PS_CLASS_*` of every GMEM read of the stream; 0 = unset = BULK_DXA | 3 | patch dxa-class | SimX `DxaCore` |
| **16** | **PS_L2ONLY**: L2-prefetch-only transfer | 1 | patch dxa-class | SimX `DxaCore` |
| 31:17 | reserved (ignored, as before) | 15 | — | nobody |

Evidence that bits 13..31 were free: `DXA_DESC_META_TOTAL_BITS` = 13 (`VX_dxa_pkg.sv:50`, `STATIC_ASSERT`ed ≤ 32 in
`VX_dxa_setup.sv:92`); `VX_dxa_setup.sv` reads `desc_data.meta` only at the DIM, ELEMSZ and LAYOUT offsets (lines 96-102); stock
SimX reads the same three; the stock encoder (`dxa.h` `pack_meta`, `set_layout`) never sets bits above 6. `dxa_core.cpp` has
`static_assert`s that PS_CLASS starts right after L2PROMO, is as wide as the class field (8 classes) and is followed by PS_L2ONLY.
Choices: (i) the fields sit directly above the stock ones so an upstream extension of the stock fields would collide visibly (the
static_assert) instead of silently; (ii) class 0 means "unset" and keeps BULK_DXA, so every existing descriptor (META bits 13..31
= 0) behaves exactly as before — the price is that a stream cannot be tagged ORDINARY on purpose, which no mechanism needs; (iii)
the class is per descriptor (= per stream), not per issue instruction, because the host already owns the descriptor and the
DXA issue instruction has no free operand bits. The host helper is `vortex::dxa::ps_set_stream(dev, slot, rank, elem_bytes,
layout, class, l2_only)`; it re-packs META from scratch like the stock `set_layout`.

**Per-class in-flight cap at work-list dispatch.** The work-list dispatch is the per-line GMEM issue `tick_worker_gmem_req`
(pristine `dxa_core.cpp:537-594`; the GMEM read is built at `:579-586`). The the earlier `:328-332` is the request-queue intake
(`drain_req_in`) and `:857-864` the GMEM arbiter's construction, so the cap went to the issue step instead, the only place that
sees one read at a time. Rule: with `n_c` = GMEM reads of class c issued by this `DxaCore` whose responses have not arrived,
the AGU does not issue the next read of a class-c stream while `dxa_depth[c] != 0 && n_c >= dxa_depth[c]` (the issue cycle is
counted as a cap stall). `dxa_depth[c]` = 0 is the stock behaviour (no cap; each worker still has its stock
`VX_CFG_DXA_MAX_INFLIGHT` = 8 slots, so caps ≥ 8 change nothing). Choices: the cap is per `DxaCore` (per socket; at PS-BASE one
worker per core, `VX_CFG_NUM_DXA_CORES` = 1, `VX_CFG_SOCKET_SIZE` = 1), summed over its workers, because the knob bounds a class's
memory pressure, not a worker's; out-of-range lines issue no read and are never capped; a held read waits at the head of the
work list (in order), as the stock slot-full stall does.

**Phase-dependent depth.** `dxa_depth[0..7]` are the [dcr_phase] knobs reserved (`VX_DCR_PHASE_DXA_DEPTH0/1`, 8 bits per
class) made schedule knobs (`dxa_depth0`..`dxa_depth7` in `PS_SCHEDULE`). `DxaCore` reads the latched copy at every issue,
so a schedule segment change acts from the next issue on; reads already in flight are not recalled (a lowered cap drains).

**L2-prefetch-only.** A PS_L2ONLY transfer enumerates the same work list and issues the same GMEM reads (same class, same cap). DXA
bypasses L1 and enters at the socket→L2 port at the lowest priority (`socket.cpp:206-233`), so the read allocates its line in the
L2 on a miss on its way, like any read. When the response comes back to the DXA it is dropped and the slot freed; no LMEM write is
built, out-of-range lines are skipped (nothing to fetch), and the transfer's barrier(s) are released when the last read has
returned (`release_all_barriers`, the stock release of an empty transfer; normally the last LMEM write carries notify_done).
Modeling choices: (i) the dropped response still crosses L2→DXA (a real prefetch-only port would not need the data; this
overstates its cost slightly); (ii) completion = last response, so a consumer that waits on the barrier knows the lines are in
the L2 (unless evicted since). The consumer then reads the data with ordinary loads (`ps_mix_micro -P`). **Untested:** L2-only combined with a multicast descriptor (the barrier path is `release_all_barriers` over the
receivers, as for an empty multicast transfer, but no TEST run exercises it).

**GMEM reads carry the class.** `mreq.flags.mem_class` = the stream class (was always BULK_DXA, patch mem-class). The class reaches
the L2 per-class counters, the MSHR merge rule (§12), mechanism D's KV insertion and mechanism B's DRAM controller, which treats
only BULK_DXA as BULK (§14.B). This fixes the mis-tag: the decode-attention DXA variant now tags its own K/V KV_STREAM
(`ps_decode_attn_gqa -v dxa -C 3`), i.e. DEMAND for B.

**Counters (observation only; `PS_DXA_REPORT=1` prints one `PSDXAC` line per launch).** Per class: GMEM reads issued, max reads in
flight (since construction, max over sockets), cap stalls (issue cycles held back); L2-only reads / dropped responses; LMEM write
beats and transfers (stock DXA counters); multicast transfers and multicast replay beats. The class-8 MPM counter
`dxa_max_inflight_s*` (patch ps-counters, per worker) is unchanged; with one stream class it is ≤ the class cap, which the TEST
checks too.

**Serial multicast.** A multicast transfer reads each line once and then writes it to the receivers' LMEM one receiver after the
other (`smem_wr`, the `mc_cta_idx` replay cursor): the 2nd..n-th receivers cost one extra LMEM write beat each (R§5 risk). It is
reported separately (replay beats, their share of all DXA LMEM beats, and kernel cycles) and the cap does not touch it (the cap
bounds reads only).

**What C cannot do.** It does not reorder DRAM or cache traffic (that is B and D); it can only slow a DXA stream (cap) or move its
data to the L2 instead of LMEM. A cap therefore helps a kernel only if the DXA stream's in-flight reads were hurting a more
al stream; on its own al path a cap can only cost cycles. Like prefetch throttling in the literature (FDP [C27]
throttles by measured accuracy and lateness; PADC [C25] by usefulness), C's value depends on telling the useful stream from the
harmful one; here the phase controller is given that information (phase), and the earlier FDP-lite is the phase-ignorant baseline.

**Measured (`results/<run>/README.md`; within one binary, every point simulated in full).**
- **Caps on decode attention's own DXA K/V stream cost cycles.** Cap 1: −0.6 … −3.7 %. Cap 4: −0.2 … −0.4 %. The one exception is
  lowbw B4/T1024, where cap 2 is +0.3 % and cap 4 is +0.4 %.
- **Caps on prefill attention help at PS-BASE** (cap 1–2: +1.2 … +2.5 %).
- **A cap of 2–4 on an interfering bulk stream shortens the decode-like GEMV span at lowbw by 0.2–2.9 %** (kernel +0.03 … +0.97 %).
  Cap 1 can cost up to 18.2 % of kernel cycles.
- **With the KV_STREAM tag, mechanism B reproduces FR-FCFS exactly on the DXA decode step**, because that step has no BULK traffic
  left. the earlier lowbw B gain on this kernel therefore came from mis-tagging its own K/V.
- **L2-only costs +3.4 … +3.9 % when the data is consumed right away.**
- **Serial multicast replay is 82–88 % of the multicast LMEM write beats.**
- **Not tested:** L2-only mode combined with multicast (only the `release_all_barriers` path covers it).

### 14.C.b C baseline: FDP-lite feedback throttle of the DXA depth (ps patch `0010-fdp-lite`)

Patch `patches/vortex/ps/0010-fdp-lite.patch` (9 files: `sim/common/phase_config.h`, `sim/simx/{barrier_unit.{h,cpp},
dxa/dxa_core.{h,cpp},processor.cpp,ps_schedule.h,socket.{h,cpp}}`; `scripts/mkschedule.py` mirrors the new knob), tests
`tests/test_fdp_lite.py`, evidence `results/<run>/` (numbers in its README). SimX only, like all of C.

**What it is.** FDP [C27] (Srinath, Mutlu, Kim, Patt, HPCA 2007) adjusts a hardware prefetcher's aggressiveness at the end of
every sampling interval from three measured quantities (accuracy, lateness, cache pollution) using fixed thresholds and a small
set of aggressiveness levels. FDP-lite transplants that loop onto the DXA engine: the "prefetcher" is the DXA engine of one socket
(`DxaCore`), its "aggressiveness" is the in-flight GMEM read depth that per-class cap already enforces, and the interval is
an epoch of E DxaCore cycles. It is **phase-ignorant**: it reacts only to what it measures, never to a phase id, the request class
of a stream or the schedule position (`test_policy_no_phase_input` checks the code text).

**Selection (my encoding choice).** `[dcr_phase]` flags bit 2 (`PhaseConfig::FLAG_DXA_FDP`) turns FDP-lite on; flags bits 31:16
give E (0 → 8192). The PS_SCHEDULE knob `dxa_fdp=E` (1..65535) sets both, `dxa_fdp=0` clears them; the general `flags` knob stays
rejected (persist_l2), as `range_acct_only` already does for bit 1. The constant lives in `phase_config.h`, not in `VX_types.toml`,
because only SimX reads it. Bit 2 clear (the reset value) leaves every code path of patch dxa-class unchanged; the only additions
on that path are a flag test per DxaCore cycle and two counters in `BarrierUnit` (no timing effect; no-op regression below).

**Depth levels (my choice).** Five levels, as FDP has five aggressiveness configurations; depth = `FDP_DEPTH[level]` =
{1, 2, 4, 6, 8} GMEM reads in flight per DxaCore, for every class (the cap of class c is that depth, bounded above by a
non-zero static `dxa_depth[c]`, so the earlier static caps remain upper bounds). 8 is the stock maximum measured (one worker's
8 slots; PS-BASE has one core per socket), so the top level equals stock there. The floor is 1, never 0: a depth of 0 would stall
the transfer forever (in the `dxa_depth` encoding, 0 means "uncapped"). Start level 2 (depth 4), the middle (my reading is that FDP
also starts from its middle configuration; verified pass against the author-hosted PDF of C27, `:108`). The mapping of FDP's distance/degree pairs to one in-flight depth is mine.

**Measured inputs per epoch (my definitions on DXA).** FDP's quantities do not exist one-to-one on a DMA copy engine; each is
replaced by the nearest DXA observable:
- *Accuracy* (FDP: useful prefetches / prefetches sent). A DXA copy is always "used" by construction, so per-line usefulness is
  meaningless. The proxy is the one PADC-lite uses (§14.B.b): acc = min(1, core LMEM read bytes / DXA GMEM read bytes) of the
  socket's cores (LocalMem read requests × one LSU word; DXA bytes = the `dxa_bytes_s*` counters, one L1 line per read).
  It is low when the engine fetches data that the cores do not (yet) read.
- *Lateness* (FDP: late prefetches / useful prefetches, a demand that hits a prefetch still in flight). On DXA a transfer is
  consumed through its completion barrier. `BarrierUnit::event_release` counts every release that drops a local barrier's pending
  async events to 0, and counts it **late** when at that moment every arrival is already in (count = 0) and at least one warp is
  suspended on the barrier, i.e. the consumer was waiting for the transfer. late fraction = late / releases. A barrier that
  waits for both a transfer and slower warps is counted timely; a multi-event barrier is judged at its last event.
- *Pollution* is **not modelled**: DXA copies write LMEM, not a cache, and an L2 pollution measure would need shadow tags. FDP's
  pollution-driven cases are therefore absent.
- *Interval update* (C27): every counter is half-decayed per epoch, s ← ½·s + ½·(count in the epoch), and acc / late are computed
  from the decayed counters, so a fill that is read one epoch later still counts. FDP's interval is defined by cache evictions;
  here it is E DxaCore cycles, as asks.

**Decision rule (FDP's table reduced to accuracy × lateness, no pollution).** At each epoch end, if the epoch had DXA reads:
acc < A_low → one level down; else (high or medium accuracy) late > T_late → one level up; else hold. An epoch without DXA reads
holds the level (nothing to judge). At most one step per epoch, clamped to [0, 4]. Thresholds A_low = 0.40, A_high = 0.75 (high vs
medium; the same action here because the cases that tell them apart involve pollution) and T_late = 0.01 are the values I take
from C27's threshold table; **the verified the thresholds, the five levels, the middle start level and Eq. 1 against the author-hosted PDF of C27** (`:108`; BUG-QUEUE lines 337, 343). FDP's own interval is 8192 L2 evictions, whereas FDP-lite uses an epoch of E cycles. In FDP's table, case 11 (low accuracy, not late) is "No Change", while FDP-lite decrements (modeling choice, `:123`). As I read FDP's
table, it increments for "high or medium accuracy, late" and holds for "high, not late, not polluting"; my
reduction keeps those cases and maps every low-accuracy case to "decrement", which is my simplification.

**Epoch length E (my choice).** Default 8192 cycles; the RESULTS sweep runs E ∈ {2048, 8192, 32768}. The measured kernels last
3·10⁵ … 2.6·10⁶ cycles, so E = 8192 gives 40–300 decisions per launch, i.e. enough to cross the 5 levels several times; 2048 is the
fastest-reacting setting (used for the functional runs, the most level changes) and 32768 the smoothest.

**Timing of a change.** `DxaCore::on_tick` runs FDP-lite before the DXA pipeline, and the level changes only inside
`ps_fdp_epoch_end` (every E-th FDP cycle since the start). A lowered depth does not cancel reads in flight; it holds new issues
until the count falls below the new depth. Every launch restarts FDP-lite (start level, new counter base), and so does every enable
through PS_SCHEDULE (epoch counting starts at the enable).

**Scope.** One controller per DxaCore (per socket); at PS-BASE that is per core. Counters: `PS_FDP_TRACE=1` prints one `PSFDPEP`
line per DxaCore epoch (raw deltas, decayed counters, level before/after, max in flight in the epoch); with `PS_DXA_REPORT=1`
a `PSFDP` summary line (epochs, judged, ups, downs, FDP cycles, depth-cycles, epochs per level) is printed when FDP-lite ran.

**Measured (`results/<run>/README.md`).** All runs are on one binary and fully simulated. Stock caps re-run here equal the earlier rows exactly (56/56).
- **Decode attention (DXA K/V on the al path).** FDP-lite costs −0.06 … −2.7 % at PS-BASE; the shorter the epoch, the
  higher the cost. At lowbw it is −2.9 … +0.6 %, and E = 8192/32768 gives +0.3 … +0.6 %, slightly better than the best static cap
  in 2 of 3 cells. The mean depth is 5.5–7.9: FDP-lite mostly climbs to the top, because the consumers do wait on the transfers.
- **Prefill attention, where static caps 1–2 help (+1.2 … +2.5 %).** FDP-lite reaches at most +1.6 % (E = 2048). It is 0.3 … 3.4 %
  behind the best static cap in every cell, and E = 32768 is negative on S128. The late consumers push it up, although a lower
  depth would be better.
- **ps_mix_micro.** It is within −0.6 … +0.2 % of the best static cap, and it avoids the cap-1 pathology (−18 %). The DEMAND span
  gains are +0.6 … +2.1 % at lowbw.
- **Reading.** A lateness-driven throttle raises the depth whenever a consumer waits. That is the right move for a stream that is on
  the al path, and the wrong one when the cost is contention with other traffic (prefill). The measured signals do not tell
  the two cases apart; a phase input could. The E sensitivity is large (up to 2.6 points), and no single E is best everywhere.

### 14.E Composition and the reduced ablation set (ps patch `0013-compose`)

Files: knob space and ablation set `experiments/knobs.yaml` (the single source; read by `scripts/knobs.py`, which mirrors
the SimX legality rules, resolves a (vector, kernel) cell into concrete host arguments + a `PS_SCHEDULE`, and validates the
file), patch `patches/vortex/ps/0013-compose.patch` (`sim/simx/processor.cpp` only), host option `-W` added to nine apps,
no-op regression `scripts/noop_regression.sh --all`, matrix `results/<run>/compat.csv`, tests `tests/test_compose.py`,
evidence and every number `results/<run>/README.md`.

**Knob space K (as implemented in ps 0001–0013, PS-BASE-S1-4c).** Two kinds of knob, by where they may change:
- *launch-boundary* (host arguments, written before launch 0; never in a schedule, §5.1): A1 = the kernel's key-tile variant,
  i.e. the per-launch LMEM usage (`-t` decode, `-w` mixed: large = app default Tt 32, small = 16; §14.A; the other P2 apps
  have no tile variant, so A1 is *not applicable* there); A2 = `-W <l1word>:<l2word>` way masks (§14.A encoding;
  the values in K are listed in knobs.yaml, e.g. `0x0:0x00F0000F` = WEIGHT fills in L2 ways 4–7, all other fills in 0–3);
  the KV range `-R kv` (decode, mixed), which is not a mechanism but D's class source.
- *policy-only* (`PS_SCHEDULE` lines, may switch inside a launch at a work-unit boundary at zero simulated cost, §5.1, §13):
  B = `dram_sched`/`dram_aging` (FR-FCFS; strict / work-conserving two-class with aging W, §14.B), C = `dxa_depth4` (in-flight
  cap of class BULK_DXA, the class of every untagged DXA stream; §14.C), D = `insert` bits 1:0 (insert-at-LRU / bypass of
  KV_STREAM fills, §14.D).
The phase-ignorant baselines (rtl_like_fill, window, SRRIP, SHiP-Mem-lite, BLISS, PADC-lite, FDP-lite, UCP-lite) are knobs
of K too (`baselines:`), for P6/P7 comparison rows; they are not in the reduced set.

**Composition rule A2 × D (patch 0013-compose; the BUG-QUEUE item "define mask×D if needs it").** Until an L2
mask aborted with *any* non-zero insert policy. needs A2 and D in the same vector (A+B+C+D rows cross
product), so the composition is now *defined*: with an L2 mask and D mode 1 or 2,
1. the victim of a new line is chosen by the masked selection exactly as A2 alone (lowest allowed free way, else the oldest
   allowed way; `select_victim(…, way_mask)`, §14.A item 2) — D does not choose victims;
2. D then acts as it does alone: a KV_STREAM fill (mode 1) is installed in that way at age 1 + the oldest other line of the
   set (the LIP insertion, §14.D), or (mode 2) a qualifying KV_STREAM load never allocates.
So the mask decides *where* a line may go and D decides *how long* a KV line stays; the two act on disjoint state (the
victim way vs the installed line's age / allocation) and both paths already existed in `processFill` (the D age code runs
after any victim selection). Only the abort condition changed: an L2 mask stays exclusive with every insert policy that
picks or ages victims by itself (rtl_like_fill bit 4, window bit 8, SRRIP/SHiP bits 6:5), and UCP-lite keeps all its
exclusivities (§14.A.b). With a zero mask or zero insert the code path is the old one (no-op below). Consequence: the earlier two
"mask + insert=1" rejection cases now run and PASS (tests/test_waymask.py was updated accordingly); the knob space
(D × B × C × A2 × A1) is fully legal.

**`-W` in every P2 app.** added `-W` to three apps; added it to `ps_gemv_f32` (and through the shared
`gemv_host.h` to `ps_twolaunch`), `ps_linear`, `ps_prefill_attn`, `ps_softmax_fixed`, `ps_rmsnorm`, `ps_residual`,
`ps_silu_gate`, `ps_rope` (one helper in `kernels/elt_common/elt_host.h`) and `ps_llama` (one block before launch 0, so the
mask is constant for the whole run — a mask change would need the flush of §14.A item 4). Host code only: without `-W` the
host path is the old one (no DCR write); no kernel source changed (device `kernel.vxbin` hashes recorded before/after); the
no-op regression proves both the default and the explicit all-zero block identical to base. Nothing was added under
`kernels/common`. Side fix: `ps_decode_attn_gqa` / `ps_prefill_attn` no longer compiled on the base tree
(`-C` used ps-only symbols); the `-C` block is now compiled only where `VX_PS_NUM_CLASSES` exists (default path unchanged).

**Phase labels (modeling choice).** The per-phase maps below need the phase of a launch. Each P2 cell is one kernel type,
so its label is the kernel type: `decode` (decode attention gqa/dxa), `prefill` (prefill attention lmem/dxa), `mixed`
(fused mixed-batch attention), `other` (GEMV, linear, softmax, RMSNorm, residual, SiLU-gate, RoPE — the weight / element-
wise ops of either phase), `model` (`ps_llama`, a whole decode-dominated model run: one label for all its launches,
because A is per run there; per-launch labels for policy knobs are possible with `launch=k` schedule lines and are left to P6).

**Per-phase maps (the "on" setting of each mechanism; evidence).**
| mech | decode | mixed | prefill | other | model | why |
|---|---|---|---|---|---|---|
| A | A1 large, A2 `0x0:0x00F0000F` | same | A2 same | A2 same | A2 same | large tiles were always fastest (−17…−48 % for smaller); the weight partition is the only A2 setting with a measured gain (+7…+16 %, wkv); on kernels without WEIGHT-class traffic it confines every fill to ½ of the L2 — the cost the ablation must show (−0.9…−2.3 %) |
| B | strict, W = 256 | same | off | strict, W = 256 | same | B protects DEMAND against BULK_DXA; in prefill the DXA stream *is* the al path, so B is off there; W = 256 is the earlier lower aging window |
| C | off | off | cap 2 | off | off | caps cost cycles on a stream on its own al path (decode); prefill caps 1–2 gained +1.2…+2.5 % (§14.C) |
| D | bypass (`-R kv`) | bypass (`-R kv`) | off | off | bypass | bypass was the strongest D mode (§14.D); prefill KV is re-read inside the launch (KV_PERSIST), so not streamed |
These are planning defaults for the ablation rows, not tuned optima; the per-interval optimum is P6's job (oracle). D in
`ps_llama` and in apps without `-R` is legal but *inert* (no KV_STREAM class: no range, PS_HINTS=0); compat.csv says so.

**Detectors.** *ideal* = the true label of every launch, applied before its first cycle (work 0): zero latency, zero error,
zero switching cost. *practical* = the earlier detector; the harness hook is `detectors.practical.source`. Until exists the
placeholder is the label the host declares for the launch it enqueues, which is exact for the single-kernel P2 cells, so
**full-practical and all-ideal resolve to the same knob vectors on this matrix** (asserted by the tests); they will differ
once the practical detector (P8) or multi-phase sequences (P7) supply labels that can be late or wrong.

**The reduced ablation set (proposal ablation matrix).** Each row is a policy; for one launch it resolves to
one knob vector of K:
- **static** — every PhaseShift knob at reset (no host knob argument, no schedule): stock PS-BASE behaviour; the reference
  of every "incremental" number. (The per-workload *best static* baseline is chosen over K by P6, not here.)
- **A**, **B**, **C**, **D** — only that mechanism's per-phase map, ideal detector; the other mechanisms off.
- **A+B** — the A and B maps together (the proposal's interaction row).
- **all-ideal** (proposal "A+B+C+D, ideal") — all four maps, ideal detector.
- **full-practical** — all four maps, practical detector (see above).
- **oracle** — "optimal per interval, zero-cost oracle": the hook drives. Its knob values come from an oracle file
  (`ablation.oracle.oracle_file`: per-launch A as host arguments plus the per-interval B/C/D schedule in mkschedule.py
  format). does not produce the oracle result; it checks the hook with `hook_check`: a K member different from
  all-ideal in every mechanism (A1 small, A2 `0x3:0x00FC0003`, `-R kv`; B work-conserving W = 1024, C cap 1, D insert-at-LRU
  from work 0, then at half of the kernel's static instruction count B strict W = 256, C cap 4, D bypass), so every policy
  knob switches inside the launch (also the BUG-QUEUE "2-segment schedule test on its own knob" for the composition).
  `ps_llama` switches at 2,000 instructions of every launch (the counter restarts per launch).

**Exclusivity probes.** Three knob vectors that are illegal by design are run on every P2 kernel and must abort with their
message: L2 mask + SRRIP (patch 0013 rule), L2 mask + UCP-lite (patch 0012), window + D (patch 0005). compat.csv marks those
cells `rejected-by-design` with the SimX error line; a PASS there would be a harness error.

**No-op regression `--all`.** REG (8 entries, §2) + the 13 P2 kernel cells on both trees + one "all modes 0" ps run per
kernel (an explicit all-zero [dcr_phase] block via `-W 0x0:0x0` *and* a schedule setting every schedulable knob to 0 at
work 0) compared with the plain base run: identical `instrs` and `cycles` required.

**Limits.** (1) The matrix is functional (PASS + invariants) at one machine; performance of the rows is reported, not
claimed (P6/P7). (2) A1 acts only in decode/mixed. (3) D is inert where no KV range exists. (4) practical = ideal here (above).
(5) The oracle hook is checked with a fixed K member, not with a DP schedule.

## 15. Serving metrics and statistics

Implementation: `serving/des.py`; tests: `tests/test_des.py`; demonstration sweep (not a gate number):
`results/<run>/`.

### 15.1 Simulator

A heapq discrete-event loop over three event kinds ordered, at equal times, ARRIVAL < ITER_END < SCHEDULE. A
SCHEDULE event forms an iteration from the waiting/running sets if the engine is idle; ITER_END applies its effects
(prefill progress, token emissions, completions) at the iteration's end time. The ordering makes every request with
arrival ≤ t visible to the iteration formed at t, which is the rule (§10).

**Batching semantics = (§10)** with the same FCFS waiting queue, B_max, chunked (budget c; running requests
first, oldest admission first; then FCFS admissions) or full-prefill (prefill-prioritising) modes. A completed
prompt emits the first output token at that iteration's end; each decode emits one token. A decode of a request
with prompt S that has generated g tokens attends over T = S + g entries (= stored + 1, §10).
**Modeling choice — unlimited KV capacity (no preemption).** The table covers B ≤ 16 and ΣT ≤ 32,768, far
below any realistic KV capacity of the modelled device, so the B_max = 16 bound binds first; omitting preemption
removes recompute paths that the table cannot cost faithfully anyway. The test cross-checks first-token and finish
times and iteration counts against `serving.scheduler.run_schedule` with unlimited capacity (exact equality).
Defaults: B_max = 16, c = 64 (the largest chunk mixed grid), clock 400 MHz (§11) to convert cycles to
seconds.

### 15.2 Iteration cost (interpolated; non-gate use)

Each iteration is summarised by x = [1, N_pf, P_pf, N_dec, ΣT, ΣT·1[N_dec = 1]] where N_pf = Σ n over prefill chunks,
P_pf = Σ n·(s0 + (n+1)/2) is the number of causal attention (query, key) pairs of the chunks (a chunk of n tokens
starting at offset s0: query j attends to s0 + j keys, summed over j = 1..n), N_dec = number of decodes and
ΣT = Σ of decode contexts. Per iteration kind k ∈ {prefill, decode, mixed} (§4) and configuration c, the cost is
T̂(x, c) = β_{k,c} · x (cycles), with β_{k,c} fitted by ordinary least squares on the workloads of kind k
(`iteration_costs.csv`, grid + trace iterations: 26 decode, 16 prefill, 16 mixed), using only the components that
vary for that kind (decode: 1, N_dec, ΣT, ΣT·1[N_dec=1]; prefill: 1, N_pf, P_pf; mixed: all but the last).

*Why this shape:* every launch of a layer is either weight-streaming (constant per iteration: the intercept),
per-token (GEMM rows, rope, element-wise: N_pf and N_dec) or attention (∝ pairs for prefill, ∝ context for
decode), consistent with E1–E6. *Empirical term:* shows ≈ 2× the per-context-token attention cost at B = 1
compared with B ≥ 2 (e.g. 44.5 k vs 19 k cycles/token between T = 128 and 512); without the indicator the decode
fit errs by up to 21 %, with it by ≤ 5.9 %. The cause is not established (plausibly the per-sequence work split
of the decode-attention launch across 4 cores); it is used as a fitted term only.

Fit quality (`CostTable.fit_report`, relative residual at the fit points): ≤ 5.9 % (decode), ≤ 6.2 % (prefill),
≤ 7.6 % (mixed) for every configuration except the C caps; **C1 decode errs by up to 38 %, C2 by 9.6 %**, so C1/C2
costs from this model are not trustworthy. **Resolution limit:** the error of the predicted ratio T̂(c)/T̂(base)
(`ratio_error_report`) is 0.40–1.0 % for the D and B configurations and 2.1–2.6 % for those with C4 — the same size
as or larger than the measured mechanism effects (≤ 0.6 % geomean). Policy differences produced with this
interpolated model below that limit are not evidence; gate-feeding comparisons need exact per-iteration costs.
Extrapolation outside the grid (B > 16, ΣT > 32,768, chunk > 64) is unvalidated; a non-positive prediction raises.

### 15.3 Policies

- **static c**: every iteration costs T̂(x, c).
- **hint-driven PhaseShift**: the engine knows each iteration's composition (the §12 hint), classifies it into a
  §4 region (prefill / mixed / decode-weight / decode-kv, decode KV-bound iff ΣT ≥ 14,314 for llama8b_s8 fp32, the
  ρ_KV = 1 point) and runs the configured config of that region; default mapping = `topk.json`
  best-by-region (prefill B_s1024, mixed D_byp, decode D_lru). A change of configuration between consecutive
  iterations adds C_sw cycles (§13; parameter, default 0) to the switching iteration.
- **DP-oracle schedule** (interface): per iteration index, else per region, either a configuration name or
  a ratio r with cost r · T̂(x, base) (r = oracle iteration cost incl. its own switches / base cost). The
  iteration sequence depends on timing, so an index-keyed schedule is only meaningful for the run that produced it;
  the region-keyed form is the portable one.

### 15.4 Workload and seeds

Poisson arrivals at rate λ (i.i.d. exponential gaps of mean 1/λ, first arrival after the first gap) and request
lengths drawn i.i.d. with replacement from a pool (e.g. a σ-scaled trace, §10). One `numpy.random.default_rng(seed)`
per seed generates gaps then pool indices, so a seed gives the same requests at every policy (common random numbers:
policy differences are paired). The cost model is deterministic, so a run is a deterministic function of
(pool, λ, n, seed, policy, engine config) (tested). ≥ 5 seeds are required (`sweep` refuses fewer).

### 15.5 Metrics (per run, over all its requests)

With t_1 < … < t_O the emission times of a request's O output tokens and a its arrival:
- **TTFT** = t_1 − a.
- **TBT**: every gap t_{j+1} − t_j, pooled over all requests of the run (so long outputs weigh more; this is the
  per-token latency a user observes).
- **TPOT** = (t_O − t_1)/(O − 1) per request (undefined for O = 1; such requests are excluded from TPOT
  percentiles and judged on TTFT only).
- P50/P95/P99 of each: `numpy.percentile` with linear interpolation.
- **SLO attainment** at level ℓ with (TTFT_ℓ, TPOT_ℓ): fraction of requests with TTFT ≤ TTFT_ℓ and TPOT ≤ TPOT_ℓ
  (DistServe's per-request criterion [cand:A4]; TPOT rather than per-gap TBT so one stall does not fail a request).
- **SLO levels**: scales k ∈ {2, 5, 10} of unloaded references on the base configuration: TTFT_ref = cost of a
  median-length prompt prefilled alone in c-token chunks, TPOT_ref = cost of a single decode at the median total
  context (DistServe likewise sets SLOs as multiples of a reference latency). Levels are fixed across policies.
- **Goodput** (DistServe [cand:A4]): the highest request rate at which SLO attainment ≥ 90 %. On the swept rate
  grid r_1 < … < r_m with mean attainment A(r): if A(r_k) is the first value < 0.9, goodput = linear interpolation
  of the 0.9 crossing between r_{k−1} and r_k (0 if k = 1); if none falls below, goodput = r_m and the row is
  flagged `censored` (a lower bound). Linear interpolation between grid points is a modeling choice; finer grids
  reduce its effect.

### 15.6 Confidence intervals

Seed-level (cluster) percentile bootstrap: requests within a run share a queue and are correlated, so the seed is the
independent unit. For a metric with per-seed values x_1..x_K (K ≥ 5), draw B = 10,000 resamples of K seeds with
replacement (generator seed 0, fixed for reproducibility), compute the mean of each, and report the point estimate
(mean over all seeds) with the 2.5th–97.5th percentiles of the resampled means. Goodput: resample seed indices
jointly across all rates (keeping each seed's attainment curve intact), recompute the mean curve and its goodput per
resample. With K = 5 the percentile bootstrap undercovers and can only take values built from 5 points; it is
reported as such, and more seeds should be used where an interval decides a claim.

## 16. Phase detector (not yet written)

To be written: PhaseMonitor counters, hybrid hint + counter rule, thresholds and the held-out evaluation.

## 17. Energy model (not yet written)

To be written: per-event energies (CACTI SRAM tables, logic power), the HBM pJ/bit source, E_token (E11).

## 18. Edge decode: Q8_0 GEMV, element-wise kernels and KV precision

This section states the definitions and the modeling choices behind the edge-decode study (single-sequence and small-batch
Q8_0 decode on small Vortex machines). G-EPB and G-EPK name the two go/no-go criteria defined in §18.3.

### 18.1 Machines and controls

- **E1 = `PS-BASE-S0`** (primary edge machine, unchanged from §11): 1 core, NT 32, NW 8, ISSUE_WIDTH 2, TCU (WGMMA) + DXA +
  MUFU, 256 KiB L2 (PLRU), one HBM2 channel at half rate (16 B/cycle peak, 9.99 B/cycle stream), +11 fixed DRAM cycles.
  Why edge-class: one small core and a weights-to-L2 ratio far above 1 (stories15M fp32 60.8 MB / 256 KiB ≈ 232; llama8b_s8 ≈
  1,790), with the S1 bandwidth per core. Limitation: the DRAM is a half-rate HBM2 model, not LPDDR (no LPDDR timing source in
  the ledger). **E4 = `PS-BASE-S1-4c`** (the G2 machine) connects EP to P6. **E1-lat190 = `EP-S0-lat190`** (+190 fixed DRAM
  cycles) is a sensitivity point that never decides a gate.
- **Controls (never machines):** `EP-CTRL-S0-L2big` = E1 with a 64 MiB L2 *and* `-DVX_CFG_L2_LATENCY=4`; `-P` variants add
  `-DPS_ALLOW_PERSIST_L2=1` (host allow macro of the persist_l2 knob, §5.2 /). Why the latency pin: `VX_config.toml:256`
  derives the SimX L2 pipeline latency as `4 + max(0, clog2(L2_SIZE) − clog2(1 MiB))`, i.e. 10 cycles at 64 MiB versus E1's 4;
  the generated `VX_config.h` wraps the macro in `#ifndef`, so the pin makes the control differ from E1 in capacity only.
  confirms this with the L2 pointer-chase probe (`ps_memprobe -m chase -s 131072 -w 2048 -n 2048`): cycles per hop and
  the core's average load latency at E1 and at the control must agree within 1 cycle.
- All EP SimX runs use the **ps tree** with every PhaseShift mode at reset (no-op-identical to base), so cold, resident
  and mechanism runs share one tree.

### 18.2 Token time, shares and the memory-side bounds

- **Token time** `C_tok` = Σ of the cycles of the launches of one decode step. stories15M B = 1 at T ≈ 8 is measured directly
  (`ps_llama` at E1, position 7, launch deltas of its PSL lines, which sum exactly to the PERF line, §14.5). Other workloads are
  **composed** from single launches: `C_tok ≈ L · Σ_{op∈layer} C_op(B, T) + C_rmsnorm,final + C_cls`. Why acceptable: SimX resets
  every cache at every launch (errata 24), so launch costs add except for DRAM-model history (≤ 0.7 %); (a2) measured
  0.05 % for an isolated-launch layer × L against the full step. stories15M T ≈ 128 / 250 = the direct T ≈ 8 token with its 6
  attention launches replaced by single `ps_decode_attn_gqa -v gqa -M 256` launches at T (the kernel and KV row count
  `ps_llama` itself uses, so variants are not mixed; also runs the single launch at T = 8 against the in-model launch).
  Limitation: in the composed stories15M points the soft-float RoPE launches keep their position-7 cost (its angle-dependent
  cost is not re-measured); this does not enter the gates, which remove RoPE (next item).
- **Shares** `f_k = C_k / C_tok` (`cur`). **Repair-adjusted `max` shares** divide by `C_tok − C_rope − C_silu`:
  repairs the soft-float RoPE/SiLU artifact (§14.0, §14.5), removing both launches entirely is the most favourable case for a
  weight-stream mechanism, so the earlier gates use `max`. `f_W` = every GEMV/GEMM launch incl. the classifier; `f_attn` = decode
  attention.
- **Amdahl bound.** If a mechanism can cut launch class k by at most a fraction h_k, `C_tok' ≥ C_tok (1 − Σ_k f_k h_k)`, so
  `S_max = 1 / (1 − Σ_k f_k h_k) − 1` (speedup as in G2; a 10 % speedup is a 9.09 % time reduction).
- **DRAM-side headroom of one GEMV**, `h_mem = 1 − C_res / C_cold`. `C_cold` = cold single launch at E1 (the machine).
  `C_res` = the second launch of the same app with every weight line already in L2: `C(-L 2 -P) − C(-L 1 -P)` of `ps_twolaunch`
  on `EP-CTRL-S0-L2big-P` (the app prints cumulative totals, so launch 2 is a difference of two runs; validated on the earlier rows:
  125,167 − 66,026 = 59,141 vs the traced 59,123). Both runs of a pair use perf class 5, so the TEST "launch-2 L2 read misses ≤
  5 % of launch 1" (the earlier criterion) is evaluated on the same rows; cycles do not depend on the perf class. Why it is the
  ceiling of H1–H3: a perfect L2 prefetcher or arbiter can at best turn every weight load into an L2 hit; L1-miss/L2-hit
  latency and in-core work remain. It does **not** bound software LMEM staging, which can also hide L1/L2 latency.
- **Init-walk correction.** Every launch starts with an L2 init walk of `sets_per_bank` cycles (errata 34) that persist launches
  skip. measures the per-launch fixed cost F with a minimal launch (`ps_twolaunch -m 32 -n 32 -v rows -L 1`, one warp,
  4 KiB) at every config (with `-P` at the `-P` controls) and reports `h_mem_corr = 1 − (C_res + F_E1 − F_L2big-P) / C_cold`
  beside `h_mem`. **The gate uses the uncorrected, larger h_mem** (favourable to EP), as pre-registered.
- **Capacity headroom (bypass/insertion bound)** `h_cap = 1 − (C_cap − (F_L2big − F_E1)) / C_cold`, with `C_cap` the cold
  single launch on `EP-CTRL-S0-L2big` (no persist). With a 64 MiB L2 nothing is evicted within the launch, so h_cap bounds what
  any insertion/bypass policy can save by keeping non-weight data resident while the weights stream (their own misses are
  compulsory in both runs). Assumptions (stated in G-EPB.md): mechanism D's bypass (NC path, patch 0005) does not make a
  weight's own miss faster than allocate-on-miss; h_mem stays the strict ceiling for any L2 policy, bypass included.
- **Issue floor** `h_issue = 1 − (instrs / ISSUE_WIDTH) / C_cold` (1 core; information only, no gate).
- **Assumption h_mem(P32) ≥ h_mem(P8):** Q8_0 moves 3.6–3.8× fewer bytes per weight with more in-core work per byte, so its
  DRAM-side share should be smaller; uses P32 measures every track directly.
- **B = 4 proxy.** the earlier h_mem,k for the B = 4 GEMM launches (`ps_linear -v tcu`, WGMMA + DXA) is the B = 1 fp32 GEMV value
  of the same matrix shape (labelled; measures B = 4 directly).

### 18.3 Gate aggregation and near-threshold rule

- Every gate aggregate over a workload set W is `G = geomean_{w∈W}(1 + s_w) − 1` (the project's region-gap convention; defined
  for s_w = 0).: `s_w = S_max,w` over W_0 = {stories15M B1 T ≈ 8 (direct), ≈ 128, ≈ 250 (composed); llama8b_s8 B ∈ {1, 4}
  × T ∈ {128, 2048} (composed)}, PASS iff G_0 ≥ 10 %. G-EPB: `s_w = S_byp,w` with h_cap (opens iff ≥ 10 %). G-EPK: `S_kv,w =
  1/(1 − f_attn,w^max) − 1` (KV precision cannot make attention faster than zero; opens iff ≥ 10 %). Strata T ≤ 128 / 250–512 /
  2048 are always reported.
- **Near-threshold rule:** an aggregate within `max(ε_gate, 1 %)` of its threshold is flagged as near-threshold whichever side it falls.
  ε_gate = max(ε_EP, 1.04 %) for sets with llama8b_s8 workloads (1.04 % = the adopted layer-multiplication bound); ε_EP is
  measured only, so at the band is 1.04 percentage points (8.96–11.04 %).

### 18.4 Precision tracks

P32 (fp32, token-exact vs llama2.c `run` [cand:D15]), P8 (Q8_0 weights as llama2.c `runq.c`: GS = 32 for stories15M, 1.125
B/weight; GS = 64 for llama8b_s8, 1.0625 B/weight; fp32 KV), P8K8 (+ int8 KV), P16T/P8T (stock TCU paths). measures P32
only. Primary track: P8 (pre-registered).

### 18.5 Q8_0 GEMV, activation quantizer, `ps_llama -q q80` and two-launch persist mode

Everything here is llama2.c `runq.c` [cand:D15] arithmetic; the golden model is `model/ref_q80.py`, the host C++ port is
`kernels/common/ps_q80.h`, the device kernels are `kernels/gemv_q80/kernel.cpp`.

**Format and groups.** A tensor is split into groups of GS values: GS int8 values q and one fp32 scale s per group, x ≈ q·s. A
weight matrix W (M, N) is two arrays, `Wq` M×N int8 and `Ws` M×(N/GS) fp32, the order of runq.c's `init_quantized_tensors`.
GS = 64 backed off by halving until it divides the model `dim` (export.py:193-195): stories15M (dim 288) GS = 32, **including
its w2 (K = 768) and the classifier**; llama8b_s8 (dim 512) GS = 64. Bytes per weight (GS + 4)/GS = 1.125 / 1.0625, i.e. 3.56× / 3.76×
fewer than fp32. The apps read GS from the file header (`ps_llama`) or take it with `-G` (default: the same back-off rule on N).

**Quantizer (runq.c `quantize`).** wmax = max |x|; scale = wmax / 127.0f (float32 division); q = (int8_t) round(x / scale), the
quotient in float32 and `round` on its double (half away from zero). The double `round` of a float is exact, so the device uses musl
`roundf` and gets identical results. A group with scale 0 would divide by zero in runq.c (NaN to int8: undefined behaviour); the host
quantizer, the NumPy port and the device kernel all give q = 0 there. Three implementations (NumPy port, host C++, device kernel)
are compared bit for bit in the Tests; re-quantizing the dequantized stories15M export reproduces its int8 values and scales
exactly (so the exported file is runq.c-format Q8_0, and the quantizer rounding agrees with export.py's on that data).

**GEMV (runq.c `matmul`).** Per output row and group: ival = Σ_k xq_k·wq_k in int32, then val += ((float)ival · ws) · xs with every
operation a separate float32 rounding, groups in ascending order. Exactness: |ival| ≤ GS·127² = 516,128 (GS = 32) or 1,032,256
(GS = 64) < 2²⁴, so the int32 sum is exact and float(ival) is exact. `gemv_q80_rows` (thread per row, grid-stride over rows) keeps
runq.c's order and is compiled with `#pragma clang fp contract(off)` (no fused multiply-add), so its output is **BIT-identical** to
runq.c's `matmul`, to the host loop `matmul_row_q80` and to `ref_q80.matmul_q80`; every run checks this (`PSBITEXACT`) and the
device output digest equals the NumPy port's (`PSDIGEST`, errata 37: passing error checks alone do not exclude silent SimX
corruption). `gemv_q80_lanes` (warp per row, lane per group, `vx_shfl_bfly` reduction) sums in a different order, so it is checked
with the error model below, not bit-exactly.

**Error model (PASS criterion).** rel_i = |y_i − ref_i| / S_i with ref the exact float64 value of the Q8_0 product and
S_i = Σ_g |t_g|, t_g = ival_g·ws_g·xs_g. Each term passes through 2 products and at most G adds (G = N/GS), so
|fl(y_i) − y_i| ≤ γ_{G+2} S_i, γ_m = m·u/(1 − m·u), u = 2⁻²⁴, and two correct implementations differ by ≤ 2 γ_{G+2} S_i. The tolerance
is max(1e-4, 2 γ_{G+2}) (1e-4 for every shape here: G ≤ 28). the earlier max(1e-4, 2 γ_{n/GS}) is the same rule up to two extra
roundings; this one is the exact bound.

**Small batch.** B = 1..4 activation vectors per launch share every weight word: a thread loads and unpacks each weight word once
and multiplies it with the B activation words (B accumulators), i.e. the weight stream is read once for the whole batch (the layout
of `gemm_small`). Output Y[B][M].

**Layout rule.** The rule (§14.5) is applied to the int8 row stride of N bytes (`rows` iff at most DCACHE_NUM_WAYS of the NT rows
of a warp share an L1 set). With int8 rows it picks `rows` for every shape of this study at E1 (pressure 1–4 ≤ 4 ways), so `lanes` is
measured only for completeness.

**Weight word and dequantization cost (the input).** Weights are read as 32-bit words (4 weights per `lw`), each byte is
sign-extended with a left shift (byte to the top) and an arithmetic right shift by 24; the activation byte likewise. `dequant_share.py`
classifies the instructions of the group loop `group_step<B, GS>` from the device image: at B = 1, GS = 32 the loop is 255
instructions per 32 weights, of which 45 % are byte-unpack shifts, 26 % int32 multiply/add, 16 % register spills, 2 % fp32 scale work;
the "dequant share" (unpack + scale) is 47 %. This is a heuristic input (static mix of an inner loop with no internal branch), not
a measured dynamic split; the dynamic instructions per weight come from the SimX rows (`gemv_q80.csv`).

**Synthetic llama8b_s8 weights.** Each Q8_0 app draws the same seeded fp32 weights its P32 twin uses (`ps::Rng`, xorshift32, W first,
then the activation vectors) and quantizes them on the host; SimX timing does not depend on the int8 values (no data-dependent
latency in the GEMV kernels), so these give the cycles of real weights. The activation quantizer is the one kernel that is value-
dependent in principle (musl `roundf` branches on the value), so its cycles/instructions were measured on two RNG seeds and on real
stories15M activations (`quant_spread.csv`).

**Persist mode and the launch-2 miss test.** `ps_gemv_q80` and `ps_linear` have `-L <n>` (back-to-back launches on the same buffers) and
`-P` (persist_l2 DCR, set BEFORE the first host write so that host-write invalidation is active for every write, `gemv_host.h:180-184`
order). Before each launch after the first the whole output buffer (for the TCU path the padded C) is overwritten with NaN from the host
(host DMA writes go straight to device RAM and touch no cache), so every launch's output is checked and digested on its own. Resident
control: C_res = C(-L 2 -P) − C(-L 1 -P) on `EP-CTRL-S0-L2big-P`; the TEST criterion is launch-2 L2 read misses (perf class 5,
`l2cache.miss_r`) ≤ 5 % of launch 1. A negative control (the same two launches on the 64 MiB L2 without persist) re-misses.

### 18.6 SimX cache-bank ordering fix (patch `0014-cache-ordering` on base AND ps; errata 38)

**Why the simulator was changed.** Stock SimX `CacheBank` (`sim/simx/mem/cache.cpp`) frees an MSHR entry when its replay
*enters* the bank pipe, so the hit-path ordering rule (defer a same-sector access behind a chain that still holds a write) can no
longer see that replay. A Core request already in the pipe ahead of it then performs first: a younger store overtakes an older
load (WAR; the cause of A2 failures and, inferred, the earlier seed-1 failure), a younger store an older store (WAW), or a
younger load an older write replay such as a deferred write-through merge (RAW, pipes deeper than 2 entries). The fill-forward
drain has the mirror problem: it can answer a read chain head while an older conflicting write is still in the pipe. The wrong
value is architecturally visible, so this is a correctness defect of the model, not a timing choice.

**The rule the fix enforces.** Two accesses to the same (set, tag, sector) of a bank *conflict* if at least one is a store or
atomic; two stores conflict only if their byte enables overlap. Conflicting accesses must perform (merge or read the line) in bank
admission order (`adm_seq`), which for one input port is program order. Byte precision is used only for store/store pairs: loads
reach the cache with byteen = 0 and a line-aligned address (the LSU sets byteen only for stores and AMOs; the coalescer drops the
lane offsets), and an L2 read is an L1 fill of the whole sector, so a read's footprint is its sector.

**Mechanism (two gates on existing decisions, no new data path).**
- *Replay-entry gate.* The oldest ready replay does not enter the pipe while a conflicting Core (or AMO-probe) request is inside
  the pipe; the bank then takes no input that cycle. The request reaches the pipe exit while the replay is still an MSHR entry, so
  the unchanged stock rule defers it into the replay stream, which drains oldest-first. Invariant: no replay in the pipe has a
  conflicting Core request ahead of it, so at every pipe exit the MSHR holds every older unperformed same-sector access, which is
  what the stock rule assumes.
- *Forward gate.* A read chain head is not forwarded while an older conflicting request is in the pipe; it takes the stock fallback
  (the replay path), as on response backpressure.
- *No deadlock or starvation.* The gate stalls only the bank input, and only while a conflicting request is in the pipe. The pipe
  head never waits on the input side: its stalls are output backpressure and the MSHR slot reserved at admission. No request enters
  during the hold, so the hold ends within the pipe latency plus output stalls.
- *One exception: the AMO vanished-sector refill.* When it is at the pipe head and the MSHR is full, it waits on
  `mshr_.full()`, and only a replay dequeue, which is input-side, frees a slot. Gate G can therefore hold that dequeue
  behind a conflicting Core request and deadlock. The path is compiled out (`VX_CFG_EXT_A_ENABLED` = 0 in every current
  configuration) and must be fixed before any A-extension configuration is used. The same path is the one ordering
  residual: its refill re-enters the MSHR with the old sequence number.

**Cost and what it changes.** Only accesses that would have been reordered, plus same-sector read/write pairs that merely
coexisted in the window, are affected: an input hold (one episode can last several cycles, up to the pipe latency plus output stalls), a deferral that the stock rule already defines, or a forward that
becomes a replay. Every other cycle is stock. Cycle counts can therefore change slightly in any launch, and a launch's output
changes only if the stock run read a wrong value. The measured distribution over the re-validated gate rows is in
`results/<run>/README.md`.

**Definition change of "base".** From on, `vortex-trees/base` = pristine Vortex 50cb174 + infra patches 0000-0002 + this one
correctness patch (`patches/vortex/base/0014-cache-ordering.patch`, functionally identical to ps 0014). Base and ps must carry it
together no-op identity (ps with every mode at reset ≡ base) breaks exactly in the launches that hit the hazard. Rows
measured stay valid as rows of the unfixed trees (their run keys hold the old tree fingerprints); no number is reused
across the fingerprint change.

**Order checker (instrumentation, off by default).** With `PS_ORDER_CHECK=1` every bank records each admitted access per sector
and counts a violation whenever an access performs while an older conflicting access of the same sector is unperformed. It also
prints the number of gate holds and forward blocks (`PSORDER` lines). It reads state only, and the patch is the same with or
without it. Unlike the shadow memories it does not depend on data values or commit order, so it also sees a reorder inside
an L2 bank.

### 18.7 Element-wise baseline repair: RoPE angle table and SiLU exp variants

Why: RoPE and SiLU-gate cost 26 % of a stories15M token and 15-39 % of llama8b_s8 decode iterations only because the toolchain's libm
(musl) evaluates `powf`/`cosf`/`sinf`/`expf` in soft-float double (§14.0 cost note). The first feature-smoke gates used the most favourable `max` share;
this study builds the repaired baselines so that every later edge-decode number is measured against a baseline that does not carry that artifact.
Default behaviour of every app is unchanged (musl, per-launch angles); the repairs are opt-in flags, in new files, in a new ps_llama
translation unit (TU 10, linked last, so TUs 1-9 and the default module image are unchanged).

**RoPE angle table (bit-identical).** The angle of rope_kv depends only on (position p, pair j): freq = 1/powf(10000, 2j/hs),
val = p·freq, (cos, sin) = (cosf(val), sinf(val)). `rope_table` (kernels/rope/kernel_tab.cpp) evaluates those very source
expressions with the same device libm for every p < seq_len and j < hs/2, one thread per entry, and stores interleaved (cos, sin)
at tab[(p·hs/2 + j)·2] (seq_len·hs/2·8 B: 49 KB for stories15M, 16 KB for stories260K, 1 MiB for llama8b_s8 at 2048 positions).
`rope_kv_tab` is rope_kv with fcr/fci loaded from the table; the rotation and KV-write lines are the same source text, so q, K and V
slots are bit-identical to rope_kv's (compiler contraction is decided per expression, identically). The setup launch is a model-load cost: it
is reported separately (PSL line `pos=-1 op=rope_table`, PSSETUP line, amortised per token over seq_len) and is never inside C_tok or
any PSTOK count. Limitation: the table is the same musl-computed values, so it inherits musl's last-bit behaviour (below); it is a
*bit-identical* repair of the cost, not a more accurate RoPE. `ps_llama -R table` works in tf, free and prefill mode (the prefill
pass passes B = P rows with positions 0..P-1 to the same kernel).

**How the table is verified (no silent corruption).** showed that a SimX launch can corrupt a private stack word under some
memory timings and still pass a tolerance check (removed the cache-ordering defect, but a table is reused by every later token).
Four independent checks, all in the apps' output: (1) *host recomputation entry by entry* (`kernels/rope/rope_table_ref.h`): freq, val
in float32 from a correctly rounded powf (long double `powl`, rounded to float; the measured musl's powf as correctly
rounded on every exponent the models use), then cos/sin of that float angle in long double. Criterion per entry (PSCHECK
`rope_table_vs_host`, scale = ulp of the entry): |device - exact| <= 0.5 + 2^-9 ulp. It is not exactly 0.5 ulp because musl's sinf/cosf
round a double polynomial whose error is up to 2^-34.1 (__cosdf), i.e. 2^-10 ulp: the first run found one entry of stories260K at 0.50023
ulp whose exact value sits 2.3e-4 ulp from a rounding tie (the neighbouring float). `PSTABLE` counts entries whose bits differ from the
correctly rounded float and lists each with its exact error; every one must be a near-tie neighbour. (2) *per-launch device probe*
(`ps_rope -A`): rope_kv run on q = (1, 0) pairs returns (cos, sin) of its own per-token evaluation exactly (1·c - 0·s = c); its digest must
equal the table's. (3) *timing independence*: the table digest must be identical on REG-CFG, E1 and PS-BASE-S1-4c-lat0 (different cores,
L2 and DRAM timing) - a setup-launch corruption would break this. (4) outputs: every fixture's q, K/V slots and the whole cache arrays are
compared by FNV digest between rope_kv and rope_kv_tab. `ps_llama -R table` runs (1) on the table it just built and refuses to continue
on a failure. An entry corrupted by one bit would be missed by (1) only if its exact value lies within 2^-9 ulp of a tie (< 0.4 %), and
by (2)-(4) never. NaN-poisoned table and output buffers make an unwritten entry visible.

**SiLU exp variants (numerics-changing).** `silu_gate_expf_poly` uses `psk::ExpSw` = `ps_expf` (float-only Cody-Waite reduction +
degree-7 polynomial, all FMAs, <= 2 ulp); `silu_gate_expf_mufu` uses `vx_expf_fast` = ex2(x·fl(log2 e)) on the MUFU unit (needs
`-DVX_CFG_EXT_MUFU_ENABLE`, i.e. the PS-BASE machines; the host refuses it elsewhere). Same formula and operation order as run.c.
Bound (derived in kernels/silu_gate/main.cpp; u = 2^-24, 1 ulp <= 2u): a relative error e of E = exp(-v) moves 1/(1+E) by e·E/(1+E);
the add, the reciprocal and the two products add 4·u/2 = 2u, so bound_i = e(x_i)·E_i/(1+E_i) + 2u. Poly: e = 4u -> <= 6u = 3.6e-7 for every v.
MUFU: e(x) = (c_L|x| + 4)u with c_L = 1 + |fl(log2 e) - log2 e|/(log2 e·u) (the prescale constant and its product rounding enter the exponent of
2^y; d(2^y)/2^y = ln2·dy and ln2·log2 e = 1) plus ex2's 2 ulp: 1.2e-6 at |v| = 12, 6.8e-6 at |v| = 88. A check's tolerance is the maximum
of bound_i over its inputs (musl keeps 1e-5); dev_vs_fixture adds NumPy's own bound (5u·E/(1+E) + 2u <= 7u).
*Near-tie token rule.* The variants change the logits in the last bits, so P32 `ps_llama -E poly|mufu` is judged rule only:
the argmax must equal the golden token or the disagreement must be a near-tie in both senses (|l1 - l2|/|l1| <= 1e-4 and |l1 - lg|/|l1| <= 1e-4),
and the logits stay within 1e-3 norm-wise of NumPy (§14.5). Such runs print `PSEP1: ... numerics_changing=1` and must be labelled
"numerics-changing repair" wherever used; musl stays the default (an open question).

**Bit-exact digests.** Every PSCHECK line is followed by its PSDIGEST (FNV-1a over the IEEE bit patterns); adds PSDIGEST lines for
the whole K and V cache arrays (`kcache_full`, `vcache_full`, sentinel included) and, in ps_llama, `PSDIGEST: logits_dev:p<pos>` for every
position's logits, so identity claims are digest comparisons, not tolerance statements.

### 18.8 Software weight-streaming variants, shuffle semantics, Q8_0 over-fetch diagnosis

This study builds the strong software baseline SW* that measures hardware headroom against: ILP variants of the fp32 and Q8_0 GEMV, a DXA→LMEM
weight-streaming GEMV (`kernels/gemv_dxa`, apps `ps_gemv_dxa`), the stock TCU linear paths at B <= 4, and two-launch persist mode in every
app SW* can pick. Every new kernel is a **new entry in a new translation unit** (`kernels/gemv_f32/kernel_ilp.cpp`, `kernels/gemv_q80/kernel_sw.cpp`,
`kernels/gemv_dxa/kernel.cpp`) linked after the stock one, so the stock entries and ps_llama's reused copies are untouched (`results/<run>/stock_sha256.json`,
`test_stock_sources_untouched`). Consequence: the *app source hash* of ps_gemv_f32 / ps_twolaunch / ps_gemv_q80 / ps_linear changes, so the literal
"stock run keys unchanged" check of the plan cannot hold; it is replaced by cycle **and** instruction identity of the stock entries against 's
rows (all 9 fp32 shapes and all Q8_0 B = 1 and B = 2 points equal to the cycle; Q8_0 B = 4 at the two w13 shapes has identical instructions and -1.61 % / +0.55 % cycles because
the earlier rows were measured cache-ordering fix, the earlier -0.47 .. +0.17 % bound having been established on the gate rows, not on these).

**ILP variants (numerics).** fp32 `gemv_rows_ilp2/4`, `gemv_lanes_ilp2` keep U independent accumulator chains and issue the loads of U float4 chunks together;
the partial sums are added at the end, so the result differs from the stock kernels in the last bits and is judged by the same rule (2 gamma_K S_i, default
tolerance max(1e-4, 2 gamma_N)); no bit-identity is claimed. Q8_0 `gemv_q80_rows_ilp2/4` form the int32 sums of U consecutive groups from loads issued together and
then run the fp32 scale steps in group order; the int32 group sums are exact (|sum| <= GS * 127^2 = 2.1e6 for GS <= 128, below 2^24, so `(float)ival` is exact and the
sum is independent of its order), therefore **all Q8_0 variants (rows, rows_inl, rows_ilp2/4, gemv_dxa) are bit-identical to runq.c's matmul** and to each other
(`PSBITEXACT mismatches=0`, equal `PSDIGEST` across kernels for the same inputs, `test_ep3_results`). `rows_inl` is `rows` with the group step forced inline.

**Shuffle semantics used by the DXA Q8_0 kernel (verified on SimX).** SimX's `alu_unit.cpp` slices the operands of `vx_shfl_*` to the lane width L = 32 (E1, E4):
mask = (bc >> 12) & (L-1), cval = (bc >> 6) & (L-1), bval = bc & (L-1); minLane = i & mask, maxLane = minLane | (cval & ~mask); BFLY reads lane i ^ bval if <= maxLane;
IDX reads lane minLane | (bval & ~mask) if <= maxLane; an inactive source returns the reader's own value. With mask = ~(S-1) & 0x3f and cval = S-1 this is a
**segmented butterfly** over aligned segments of S lanes (a group of GS/4 lanes holds one Q8_0 group: each lane multiplies its weight word with the activation word,
then log2(GS/4) butterfly steps leave the exact int32 group sum in every lane of the segment), and `shfl_idx(v, j*GW, 31, 0x20)` broadcasts lane j*GW to the warp, which
lets every lane add the groups of a 32-word step to its fp32 accumulator in group order. The probe kernel `q80_shfl_probe` (host `ps_gemv_q80 -v shfl_probe`) checks, per
thread of a full CTA (8 warps x 32 lanes), segmented sums for S = 4, 8, 16, 32, `bfly` xor 3 inside segments of 8, xor 16 across the warp, `idx` of lanes 0/8/16/24 across
the warp, `idx` 3 inside segments of 8 and `idx` 2 inside segments of 4 against the model above: **0 mismatches** (`PSSHFL: threads=256 slots=12 mismatches=0`, E1 and E4). These semantics need NUM_ALU_LANES = NT (true for E1 and E4;
the host refuses NT other than 8/16/32 and GS/4 > NT).

**`gemv_dxa` design.** One CTA = one core (all NW = 8 warps); a *row group* is NW rows (one per warp); the weights of a row group are staged as K-tiles of KT columns by one 2-D DXA
transfer per tile (plus one for the Q8_0 scales) into a ring of `depth` in {1, 2, 4} LMEM stages; x (and the Q8_0 activation scales) are copied once into LMEM; B <= 4 accumulators per
warp row share every staged tile. LMEM budget: 64 KiB per core, KT = the largest multiple of 16 floats / GS bytes that fits x + depth stages. Protocol (the earlier `gqa_dxa_dbuf`
pattern): warp 0 issues tile t + depth - 1 into the stage that held tile t - 1 (every warp finished it at the previous phase barrier), all warps wait on the stage barrier
(`expect_tx` + DXA completions), compute, meet at a phase barrier. depth = 1 has no overlap. The fp32 flavour is the `gemv_lanes` loop on LMEM (bit-identical to `gemv_lanes` when
KT >= N, 2 gamma_N otherwise). **DXA counters:** per launch the engine's `transfers` and `lmem_writes` equal the host's `PSDXAEXP` prediction exactly (the beat enumeration of `dxa_core.cpp`
`enumerate_work_list`: one beat = at most one GMEM line and one 64 B LMEM word); the engine's `gmem_reads` counter also counts the same-line repeats it reports as `gmem_dedup`, so
the predicted distinct line reads equal `gmem_reads - gmem_dedup` (25 launches, `results/<run>/dxac.csv`). Edge tiles (KT does not divide N, or M is not a multiple of NW) are enumerated
and staged in full (`staged_bytes` can exceed the matrix); rows beyond M are not stored.

**P8T track.** `ps_linear` built with `-DITYPE=int8 -DOTYPE=int32` runs the *stock* WGMMA + DXA device code with int8 inputs and int32 accumulation (config `EP-SMOKE-S0-tcuint8`; W and X quantized with
the Q8_0 quantizer; the check is the exact integer product, `PSCHECK vs_int_exact`). The Q8_0 group scales are **not applied** (the stock path has no per-group epilogue), so P8T cycles are a *lower bound*
on a complete Q8_0 TCU GEMV; no P8T number is compared with P8 or P32 without that label. The 32-row WGMMA tile is padded (M = B is padded to 32), so tcu cycles do not depend on B in 1..4.

**Q8_0 DRAM over-fetch diagnosis.** found Q8_0 `rows` reading 1.6-3.7x its weight bytes from DRAM (fp32 1.00x). Method: `ps_gemv_q80 -A` programs the ps-tree range classifier (no policy is on, so the
timing is that of the unlabelled run: cycles equal in every pair) with Wq -> class `wgt`, Ws -> `kvp`, the per-thread stack window (0xFFFF0000 - NC*NW*NT*8 KiB; VX_MEM_STACK_LOG2_SIZE = 13) -> `kvs`,
everything else (code, x, y, arguments) = `ord`; perf class 8 then splits L1->L2 read bytes, L2 misses and DRAM read bytes by source (`diag2.csv`). Controls: the same kernels at the 64 MiB-L2 control;
touch kernels (the weight-word loads of `rows` / `lanes` with no arithmetic, no scales, no stack frame). Result (E1): DRAM weight bytes are 1.00x Wq for every kernel (1.05x at stories15M qkvo); the excess of `rows` is the **stack window**: 76 / 501 / 2,447 KB of the 178 / 818 / 3,648 KB DRAM
reads at stories15M qkvo / llama8b_s8 qo / w2 (43 / 61 / 67 %; 28 / 74 / 74 KB at the 64 MiB L2), i.e. the register spills of the group step (B accumulators, U group sums, unpack temporaries;: 16-36 % of the
instructions) go L1 -> L2 -> DRAM because E1's 256 KiB L2 loses stack lines that a 64 MiB L2 keeps (which of capacity pressure from the streamed weights or stack set conflicts is not resolved; DRAM write bytes were not
attributed). `rows_inl` removes ~25 % of it, `rows_ilp4` doubles it (so the ILP variants lose), the spill-free DXA Q8_0 kernel reads 1.16-1.7x of Wq + Ws. The weight lines themselves are re-fetched by the L1
(L1->L2 = 2.0 / 8.1 / 5.2x Wq for `rows`; the weights-only `touch_rows` 2.0 / 16.0 / 16.0x, `touch_rows4` 1.7 / 4.2 / 4.05x, coalesced `touch_lanes` 1.00x): the 256 resident rows (8 warps x 32) with row strides of
512 B and 1792 B fall into 8 and 16 of the 64 sets of the 16 KiB 4-way L1, so nearly every 4-byte load misses; the L2 absorbs these (touch-kernel DRAM 1.00x of Wq). The layout rule (`rows_set_pressure`)
counts the rows of one warp only. Pre-diagnosis hypotheses not supported: a compulsory-byte overhead (weights are read once), the warp-per-row access pattern itself (its excess is its own spills).

**h_mem\* of the SW* candidates** (feeds; rule §18.2): h_mem\* = 1 - C_res / C_cold with C_cold = the cold single launch of the same app and arguments at E1 and
C_res = C(-L 2 -P) - C(-L 1 -P) at `EP-CTRL-S0-L2big-P` (the persist two-launch protocol of §18.5, every launch's output NaN-poisoned and checked); h_mem\*_corr adds the 184-cycle per-launch walk
difference (E1 - L2big-P) back to C_res. Persist mode exists in `ps_twolaunch` / `ps_gemv_f32` (the fp32 ILP entries through the same `-v`), `ps_gemv_q80`, `ps_gemv_dxa` and `ps_linear`
(small and tcu paths, fp16 and int8). Criterion: launch-2 L2 read misses <= 5 % of launch 1 in every pair (`res.csv`).

**Not claimed.** E1 numbers use synthetic weights (timing is data-independent). No energy. B > 4 not measured. `gemv_dxa` Q8_0 is correct but slow because exact in-order fp32 group accumulation
costs a shuffle + add per group (instruction counts in `dxa.csv`); a faster Q8_0 DXA kernel that keeps bit-exactness is open work, not a claim about what DXA staging can do.

### 18.9 KV-precision decode attention: int8 KV (and fp16 KV with Zfhmin)

This study makes the KV cache narrower and measures what that buys in decode attention and per token. New entries live in **new translation units** linked after the stock ones
(`kernels/decode_attn_gqa/kernel_kv.cpp`, `kernels/rope/kernel_kv8.cpp`; ps_llama TUs 11 and 12, `k_kv8r.cpp`, `k_kv8a.cpp`), so the stock device code is byte-identical
(`tests/test_ep_kv8.py::test_stock_sources_untouched`); the app source hashes change, so stock identity is shown by instruction equality and the cycle deviation of the
re-run stock rows against the earlier (`results/<run>/identity.csv`).

**Quantization scheme (modeling choice).** Symmetric, **per (position, KV head)**: the group is the D = head_size values of one head of one position, scale = max|v| / 127.0f
(float32), q = (int8) round(v / scale) (float32 division, round half away from zero), q = 0 for an all-zero group; dequantization is the float32 product (float) q * scale.
This is runq.c's `quantize` with GS = D (the same rule as the weights and activations, so one quantizer is tested once: `ps_q80.h`, `model/ref_q80.py`, device kernel).
Why per position and head: decode appends one K and one V row per step, so a scale that depends only on that row is known at write time and the cache never has to be
re-quantized (a running per-channel or per-layer scale would); a head is the unit the attention reads together, and its scale costs 4 B per D bytes (8.3 % at D = 48, 6.25 % at D = 64).
Per-channel K scales (outlier channels) would be more accurate and are not modeled.

**Layouts and bytes.** K8, V8: int8 [B][Tmax][H_kv][D] (stock layout, narrower element); scales: float32 [B][H_kv][Tmax] (position-contiguous per head, so a key tile's Tt scales are one
contiguous run: coalesced loads, a single 1-D-shaped DXA transfer). kv16: fp16 [B][Tmax][H_kv][D], no scales, host conversion round to nearest even. **Measured and reported bytes count
the scales** (E3's "metadata ignored" is not used for them): analytic KV bytes of one step = 2 B T H_kv (D esz + 4 [kv8 only]), ratio to fp32 (D + 4) / (4 D) = 0.2708 (D = 48),
0.2656 (D = 64) for kv8, 0.5 for kv16. The analytic region thresholds use E3's convention (scales ignored: llama8b_s8 P8K8 KV-bound iff sum T >= 15,210) and are
reported next to the scales-counted ones (measured bytes: the scales are counted) (14,315; stories15M 4,948 / 4,567); `attsum.csv` has both. Arithmetic intensity E5: AI = 2g/q FLOP per byte (g = kv_mul, q = bytes per cached element),
so llama8b_s8 (g = 4) fp32 KV 2 FLOP/B, int8 KV 8 FLOP/B with the scales ignored, 2gD/(D + 4) = 7.5 FLOP/B with them counted.

**Kernels.** `attn_gqa_kv8` / `attn_dxa_kv8` (and `attn_*_kv16`): kernel.cpp's fused online-softmax attention unchanged except the operand source: K and V are loaded narrow and dequantized in the core right
after the load, kd = (float) q * ks[t] (kv8) or (float) h (kv16). Plain kernel: K rows are read as 32-bit words (4 int8 per load, sign-extended by shifts; D % 4 = 0 required), V
element-wise (thread per column, a warp's loads coalesce), the tile's scales are copied to LMEM by the CTA's threads. DXA kernel: K and V tiles are 4-D descriptors with 1- (2-) byte elements, the scales 3-D descriptors
(4-byte elements, tile Tt x 1 x 1); all four transfers share the stage barrier (`expect_tx` 4 for kv8, 2 for kv16), the dequantization happens after the LMEM read. LMEM layout in `common_kv.h`
(words: qs | acc | ps | stat | ksc | vsc | Kb | Vb, 64 B regions). `rope_kv_tab_q8`: q rotated as `rope_kv_tab`; one extra work item per (sequence, KV head, K or V) rotates (K table), finds max|x|,
quantizes and writes the int8 slot and the scale (two passes over D values; the rotation is recomputed in pass 2 from the same operands, so both passes see bit-identical values). This translation unit is compiled with
**fp contract off** (`#pragma clang fp contract(off)`): every product and sum is a separate float32 rounding, which is what the NumPy port evaluates, so the quantizer input is bit-identical to the port's.
Zfhmin: the kv16 entries are compiled only when the compiler defines `__riscv_zfhmin` (config `EP-SMOKE-S0-zfhmin` = E1 + `-DVX_CFG_EXT_ZFHMIN_ENABLE`, an ISA extension only: the fp32 kernels time identically,
checked in `identity.csv`); `_Float16` loads compile to `flh` + `fcvt.s.h`; a build without it refuses the variant before launch (ISA flag `VX_ISA_EXT_ZFHMIN`).

**Error model (tolerance of the TEST).** The host reference is the float64 attention over the DEQUANTIZED operands (K_d = fl((float) q * s), V_d likewise; for kv16 the exact half values). The device operand equals the
reference operand bit for bit (one correctly rounded float32 product in both; fp16 -> fp32 is exact), and the kernel's accumulation order is kernel.cpp's, so the bound of §14.2 holds **unchanged**:
tol = max(1e-4, 2 E_s + 2 (s_max - s_min) u + gamma(n_acc + n_l + 8 (n_tiles - 1) + 10)) per head, u = 2^-24 (main.cpp derives it as before; the quantization itself adds no term because it is part of the
operand, not of the arithmetic). The error of the cache against the fp32-KV attention is a property of the cache and is printed as `PSKVQ` (information, never pass / fail): per element |K_d - K| <= s/2 = max|K|/254 (int8),
<= 2^-11 |K| (fp16, normal range); `test_attn_kv*` assert only sanity bounds (int8 < 0.1, fp16 < 5e-3 of S_i).

**Bit-identity claims.** `rope_kv_tab_q8`: K8 / V8 bytes and both scale arrays equal a host recomputation from the device's own table (`PSCHECK k8_exact v8_exact ks_exact vs_exact`, 0 errors), and the dumped inputs and slots are
reproduced bit for bit by the NumPy port (`tests/test_ep_kv8.py::test_rope_kv8_vs_host_and_numpy`). The NumPy model `Llama2cRefKV8` uses the correctly rounded table (`rope_table_np`) in
non-fused float32; the device table equals it except for the isolated near-tie neighbours (§18.7), so the full-model comparison is by the token rule and the 1e-3 logits bound, not bit identity.

**Token-time effect (EP-RQ3).** stories15M, B = 1: direct at T = 8 (position 7 of `ps_llama -q q80 -R table` and `-q q80 -Q kv8`), composed for the other T as
C_tok(T) = C_tok(T = 8) + L (A(T) - A(8)) with A the single-launch attention cycles of the variant (`gqa` / `gqa_kv8`, the entries ps_llama runs) and L = 6 (`compose.csv`). llama8b_s8 has no P8 token composition here
(composes P8K8 tokens in its grid); `llama_bound.csv` applies the measured attention-cycle ratio at the earlier P32 attention share, labelled P32. Nothing compares a P8K8 number with a P8 or P32 number without the label.

### 18.10 Edge static characterization sweep, SW*, composed tokens

This study builds the strongest static software baseline from the single-launch costs and asks how much memory-side headroom is left. Everything is
measured at E1 = `PS-BASE-S0` (the machine) on the `ps` tree with the cache-ordering fix; the scripts are `results/<run>/{make_batch.py, sw_select.py, compose.py, analyze.py}` (the plan's `select.py`
is `sw_select.py`: a module named `select` shadows the standard library when its directory is on `sys.path`).

**Launch classes and SW\*.** A launch class is (track, model, matrix, B) with track in {P32, P8} (P8K8 uses P8's weight kernels) and nine matrix shapes (stories15M qkvo / w13 / w2 / cls,
llama8b_s8 qo / kv / w13 / w2 / cls). Candidates: fp32 (B = 1) `ps_twolaunch` rows / lanes / rows_ilp2 / rows_ilp4 / lanes_ilp2; fp32 `gemv_dxa` depth 1 / 2 / 4 (B = 1) and depth 2 / 4 (B = 2, 4);
fp32 `ps_linear` `small` (B >= 2, the `gemm_small`); Q8_0 `rows` / `rows_inl` / `rows_ilp2` / `rows_ilp4` / `gemv_dxa` depth 1 / 2 / 4 (B = 1) and `rows`, `rows_inl`, `gemv_dxa` depth 2 (B = 2, 4;
the ILP kernels lose at every measured B). The stock TCU paths P16T (fp16 inputs) and P8T (int8, scales not applied) are separate tracks (§18.4): they are measured and listed in
`candidates.csv` under their own label and never enter P32's or P8's minimum. SW* = the minimum-cycle candidate whose run PASSED; candidates within 0.5 % of the minimum are tied
and the simplest wins (fewest DXA stages, then the default: `rows`/`lanes` by the layout rule for fp32 B = 1, `gemm_small` for fp32 B >= 2, `rows` for Q8_0). The choice is
per launch class, one per launch: free on Vortex (kernel entry per launch), the G2b-strength static baseline. Because SW* is an **empirical** minimum over the two layouts, the
layout rule's known blind spot (it counts the rows of one warp per L1 set and ignores the other resident warps; BUG-QUEUE) does not bias SW*; `layout_rule.csv` reports how often the rule
picked the measured-faster layout. A "launch of B vectors" is one kernel launch; B repeated B = 1 launches are NOT a candidate of the primary number, but `S_star_repeat` / `G_repeat` in
`tokens.csv` / `summary.json` report the gate with min(SW*(B), B x SW*(1)) per class (the strongest conceivable software reading), labelled sensitivity.

**Token composition.** C_tok = sum over the launch groups of one decode step of (count x cold single-launch cycles): per layer 2 rmsnorm, 4 + 3 GEMMs (q, k, v, o, w1, w3, w2), rope, attention, 2 residual, silu
(+ 4 `quantize_q80` in P8 / P8K8: before q/k/v, o, w1/w3, w2), then rmsnorm, classifier (+ quantizer): 14 L + 2 launches (18 L + 3 with the quantizer), L = 6 (stories15M), 32 (llama8b_s8).
Caches reset at every launch, so launch costs add; the residue is eps_EP. Element-wise launches are measured at the workload's B (`-b B`); RoPE is the table launch (cost = `-R table`
run minus the `-R setup` run of the same shape: the 14 M-cycle table is a model-load cost, never in C_tok), SwiGLU is the fastest correct exp variant (`mufu`, labelled numerics-changing, `S_star_silu_musl`
reports the gate with the musl default), attention is the earlier single launch at (B, T) (SW* = `gqa` or `dxa`; P8K8: `gqa_kv8` / `dxa_kv8` and the `rope_kv_tab_q8` writer), T = 8 / 128 / 250 (stories15M, `-M 256`)
and 128 / 512 / 2048 (llama8b_s8). No T interpolation. stories15M B = 1 T ~ 8 is composed like the others (see "Not done": `ps_llama -V`).

**Headroom of SW*.** h_mem*_k = 1 - C_res(SW*_k) / C(SW*_k): C the cold single launch of the SW* app, C_res = C(`-L 2 -P`) - C(`-L 1 -P`) of the same app and arguments on `EP-CTRL-S0-L2big-P` (perf 5), the
two-launch persist mode (second launch with every weight line in a 64 MiB L2; launch-2 L2 read misses must be <= 5 % of launch 1's: checked for every pair). It is measured for **every** SW*
launch class (54 classes), so no proxy replaces a measurement. Uncorrected for the persist walk (the larger, gate-conforming value; the walk is 184 cycles, <= 0.2 pp).
S*_w = 1 / (1 - U_w) - 1 with U_w = sum_k f_k h_mem*_k, f_k = (count x cycles of the GEMM launch group k) / C_tok over the GEMM / GEMV launches (layers and classifier); the shares are those of the SW* token
(f_k,w(SW*)), including all the other launches in the denominator. ** = geomean_w(1 + S*_w) - 1 over the 18 W_EP workloads, primary (E1, P8); PASS iff >= 10 %**, aggregation §18.3. If the value is
within max(eps_gate, 1 %) of 10 % the verdict is "NEAR-THRESHOLD" whichever side it falls (§18.3). Strata and P32 / P8K8 are reported (`strata.csv`).

**eps_EP.** The composition error: |composed - direct| / direct of the ps_llama kernel set at E1, stories15M B = 1, positions 0-7 (T = 1..8) of the stored direct runs (P32 with `-R table -E musl|poly|mufu`, P8, P8K8)
and, when the run finished, the two decode positions (T = 129, 130) after the 128-row prefill; the composition uses the *default* kernels of that run (stock GEMV rule, `gemm_small`, Q8_0 `rows`, `attn_gqa`) because
those are what `ps_llama` runs. eps_EP = the 95th percentile (nearest rank: element ceil(0.95 n) of the sorted values). It is a property of the composition rule (inter-launch cost: caches reset, DRAM-model history),
not of the SW* kernels, which are validated per launch (every launch PASSED). eps_gate = eps_EP for stories15M workloads and max(eps_EP, 1.04 %) for llama8b_s8 (the adopted bound; BUG-QUEUE 408 stays open).

**Other quantities.** EP-RQ2: R_q = C_tok(SW*, P8) / C_tok(SW*, P32). EP-RQ3: C_tok(P8K8) / C_tok(P8) (labelled; composed for llama8b_s8, from per-launch costs). EP-RQ4 proxy: DRAM read bytes per token = sum of count x DRAM
read bytes of the launches (perf 7 rows). trigger heuristic: dequantization share of the Q8_0 group step (the earlier static instruction mix, 47 % at GS = 32, 43 % at GS = 64) x f_W(SW*) -> S6 = 1/(1 - f_W x share) - 1, geomean >= 10 %
(heuristic, not a bound; computed, not acted on). Roofline view: weight B/cycle and FLOP/cycle per GEMV candidate (`roofline.csv`).

**Caveats carried into every statement.** (1) The kv8 free-run golden diverges from the runq golden at position 54 (deviation 5); every kv8 token comparison here is within positions 0-7, where they agree, and
nothing depends on later positions. (2) llama8b_s8 P8K8 token effects are composed from per-launch costs. (3) the earlier RoPE table check for llama8b_s8 used head size 128 windows (0..63, 1984..2047); the W_EP model has
head size 64, so verifies the hs 64 table at **all 2048 positions** (`ropefull`), see the README. (4) Stock-entry identity rule: instructions identical, cycles within 0.1 %; re-runs / re-uses
stock entries on the same tree and compares instructions (`identity.csv`).

**Not done (recorded).** (1) `ps_llama -V sw_star_<track>.json` does not exist (lists it among the flags EP adds, but no step implemented it) and the SW\* kernels (DXA, ILP, inlined Q8_0) are not in ps_llama's device module; therefore
validates the composition rule on the stored direct ps_llama runs of the default kernel set (eps_EP above) and the SW\* kernels per launch, and composes the stories15M B = 1 T ~ 8 workload like the others. never uses a direct token.
(2) No E4 columns (the gate is E1). (3) The reduced-precision track P8T has no scale epilogue and P16T converts weights to fp16: both are separate tracks and are not SW\* candidates of P32 / P8.
**Model of the negative h\*.** h\* = 1 - C_res / C_cold compares a launch on the 64 MiB-L2 control with one on E1; the two machines differ in L2 capacity and its set mapping only, so h\* can be slightly negative when a kernel's own conflict / stack
pattern is worse on the larger cache (observed once, P8 llama8b_s8 classifier B = 1, `rows_inl`, -9.0 %); the gate uses the measured value, `G_h0clamp` reports the clamped aggregate.

### 18.11 Spill-free Q8_0 batched GEMV, SW*', capacity / residency split with the init-walk term

This section follows the decision of 2026-11-14: fix the Q8_0 batched stack spill in software, re-measure SW*, then re-evaluate G-EPB on the spill-free baseline.
Scripts: `results/<run>/{make_batch.py, static_spill.py, analyze.py}` (analyze reuses `results/<run>/{ep4lib, sw_select, compose, analyze}.py` by path and edits none of them); device code `kernels/gemv_q80/kernel_nsp.cpp`
(a translation unit appended LAST in `app.conf`, so every earlier entry keeps its code and address; the `ident` rows check this).

**Root cause (compiler output).** In `rows_inl` / `group_step` the word loop and the batch loop of one group are fully unrolled. The int32 group sum is exact in any order (|iv| <= GS x 127^2 = 1,032,256 < 2^31 for GS <= 64),
so -O3 re-associates `iv[b] += dot4(...)` over the whole group into one add tree: every load and every product of the group is computed before any add. The live set (GS x B products + (1 + B) GS/4 loaded words, 32-256 values)
exceeds rv32's ~27 allocatable integer registers, so each value is stored to the stack and reloaded once per group (13-398 stores per group per thread). Stack interleaving (`VX_CFG_LSU_STACK_INTERLEAVE_ENABLE`, 4-byte slot of
all 32 lanes = one 128 B block) gives a per-core stack footprint **S_stack = N_warps x N_threads x F = 256 F bytes** (F = per-thread frame bytes: F/4 slots, each a 128 B block per warp, so 32 F per warp, 8 warps),
up to ~295 KB at B = 4 / GS = 64, more than the 256 KiB L2 of E1. This is the P8 "capacity" headroom (finding 7).

**The fix (F1, `rows_nsp`): a register fence.** `asm volatile("" : "+r"(iv[b]))` after each weight word emits no instruction. It makes iv[b] a value that must be in a register at that point and may have changed, so the adds cannot be deferred past the
word and the live set drops to ~1 weight word + B activation words + B sums. Only the order of exact integer additions is constrained; the float steps (`acc = acc + ((float)iv * s) * xs`, `#pragma clang fp contract(off)`) are untouched and stay in
group order, so the result is bit-identical to runq.c (`PSBITEXACT: runq_exact mismatches=0` is the acceptance criterion, no tolerance). F2 (`rows_nsp_ilp2`) is F1 with 2 groups in flight (spill-free only while U x B <= 4, so B <= 2; at B = 3, 4 the entry
does nothing and the host check FAILS on purpose). `rows_nsp_nu` (sensitivity, alternative form) is the plain-C alternative: word loop `#pragma nounroll`, no asm. Each (B, GS in {32, 64, generic}) path is a `noinline` function of its own
(`nsp_rows_B<b>_GS<g>`, `nsp_rows2_...`, `nsp_nu_...`) so each can be read in the linked ELF.

**S1 (static spill rule, `static_spill.py`).** `llvm-objdump -d` of the linked `kernel.elf` of every config used. A loop range = [target, pc] of a backward branch / jump within the function. The thread-per-row `for (row ...)` loop is compiled with SIMT
divergence control (vx_split / vx_join / vx_pred) that adds several back edges, so a loop range that contains an OUTPUT store (a store whose base register is not sp: the y[] write, once per row) is a *row-level* range; every other range is a
*group-level* loop (the group loop, its tail copy, and for the generic-GS path the word loop). Stack store = sw / sh / sb / fsw with base sp, reload = lw / lh / lb / lhu / lbu / flw with base sp. S1 passes for a path iff **0 stack stores in every group-level loop
range and <= 8 stack reloads in each** (loop-invariant pointers). Row-level stores / reloads (a handful of loop-invariant row-loop variables in the frame, once per row) are reported, not bounded. The same pass computes the earlier instruction mix
(unpack / mac / load / spill / scale / other, `dequant_share` = unpack + scale) over the largest group-level loop heuristic.

**D1 (dynamic, per class, E1).** Stack-class DRAM read bytes (`-A` run, perf 8, class `kvs`; `-A` only labels address ranges and is timing-neutral) <= 1.5 x S_stack, S_stack = 256 (F_path + F_entry) with F from the ELF prologue
`addi sp, sp, -F`. Why: it allows each stack line to come from DRAM about once (compulsory misses, including write-allocate fills) plus 50 % margin and rules out repeated capacity re-misses; a ratio to the weight bytes is not used because a compulsory
8-32 KB stack is up to 40 % of the smallest matrix without being a spill problem. For a SW*' kernel that is not an `rows_nsp*` path the entry frame of the stock function is used (a looser bound; stated in the table).

**D2 (time, aggregate).** S_cap(P8) <= 2.0 % (definition below), geomean over the 18 W_EP workloads. Outcome: FIXED iff S1 and D1 (all 27 SW*' classes) and D2; PARTIAL iff S1 and D1 but not D2; NOT FIXED iff S1 or D1 fails for a SW*' class.

**SW*' and '.** SW*' = the per-class minimum over the old candidates plus F1 / F2 (27 P8 classes, B = 1 included), the same 0.5 % near-tie rule; on a tie the new kernels rank AFTER `rows` and the other old kernels (more code: conservative
toward the old kernels). C_tok(SW*') is the earlier composition with only the P8 GEMV classes replaced; ' uses h_mem*_k = 1 - C_res / C(SW*'_k) exactly as (rule, population W_EP, threshold 10 %, near-threshold band max(eps_gate, 1 %) unchanged). P32 is unchanged
(recomputed as a check); P8K8 uses the P8 GEMV classes.

**Capacity / residency split with the init-walk term.** Per launch class k, at E1 and at `EP-CTRL-S0-L2big-P`:
cap_k = 1 - (C(-L 1 -P) + delta) / C_cold(E1) with delta = 184 cycles; res_k = (C(-L 1 -P) - C_res) / C_cold(E1), C_res = C(-L 2 -P) - C(-L 1 -P).
Why delta: measured C_cold(L2big) - 65,280 = C(-L 1 -P) + 184 to the cycle on all 9 shapes (the non-persist launch pays a 64 MiB init walk and a slightly larger fixed cost, the persist launch does not), so cap_k equals the earlier walk-corrected h_cap and the
persist launch 1 holds no walk (residency is not inflated: this removes the earlier init-walk caveat). The `hcap` group re-checks the identity on the new kernel (3 classes, within 1 cycle). U_x,w = sum_k f_k,w x_k with the shares f_k of C_tok(SW*') and
S_x,w = 1 / (1 - U_x,w) - 1 (Amdahl: a launch class with time share f that can lose at most a fraction x bounds the token speedup this way), aggregated as geomean_w(1 + S) - 1; the parts do not add up to G (convex map).
**G-EPB evaluation 2** = the aggregate of S_cap,w on SW*' (h_cap,k = cap_k; opens iff >= 10 %, near-threshold band max(eps_gate, 1 %)): shares are the measured f_k,w(SW*') (not the earlier f^max) and the population is W_EP, not W_0 (both recorded
deviations of the plan); the W_0 subset, P32, P8K8, strata and B splits are reported. Evaluation 2a is the same quantity on the earlier spilling SW* (stored rows only). **G_res** (entry check, Q-b) = geomean_w(1 + S_res,w) - 1 of the residency part, P8.

**Fallbacks and sensitivities.** F3 `rows_nsp_nu` (word loop `#pragma nounroll`, no asm; alternative form) is measured at the 27 classes as a sensitivity (SW* over the old kernels + F3 only: `sw_star_nounroll.json`, `tokens_nounroll.csv`, resident pairs of its winning classes) and is the first fallback of the plan's chain (F1 -> F3 -> F4 B-tiling -> F8 DXA-staged weights, the last one was not pursued). F4 `rows_nsp_tile` (kernel_tile.cpp) computes B = 4 as two B = 2 passes per group (the weight words are loaded twice, from L1) with the fence kept; it is a B = 4 candidate only. "Repeated B = 1 launches" (the earlier `G_repeat`) is recomputed with the same rule on SW*'. E1-lat190 is not re-run (user answer Q-e): its columns would be stale and are left empty.
**Post-hoc readings, labelled, never the pre-registered rule.** (1) D1 per wave: a launch is executed as `grid` blocks (`launch: gemv grid=..`) of 256 threads, each calling the kernel entry once per block; the weight stream of the other blocks evicts the stack lines from the 256 KiB L2, so the frame footprint is re-fetched once per block. `d1_wave_pass` applies the 1.5 x S_stack bound to stack bytes per block (the 64 MiB control, `attr_ctrl`, reads the footprint once). (2) S1 row-level vs group-level ranges (above) is a definition forced by the linked code (the plan says "any loop body"; counting the row-loop bookkeeping would also count the once-per-row y store loop).
**Shared build directory hazard (measured, README anomaly 1).** Two PhaseShift batches admitted together by the memory-admitted CPU lane (and NEEDS-RUN/A2, whose config is `PS-BASE-S1-4c`) share build/vortex-ps: a job's libsimx.so can be replaced between its build and its run, and the run PASSES on the wrong machine. Single-core instruction counts per launch are machine-independent, so `analyze.py consistency()` requires every row of one (app, arguments minus -L / -P / -A) to agree on instructions per launch and flags the minority row; flagged rows are re-run (`rerun` group, new `rep`, 900 s timeout) and the superseded rows stay in `rows.jsonl` and are listed in `summary.json`.
