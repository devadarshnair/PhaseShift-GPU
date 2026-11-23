"""Iteration-level continuous-batching scheduler (semantics in methodology §10).

vLLM-like [cand:A6] iteration scheduling with paged KV blocks, in two prefill modes:

* **full prefill** (`chunk_size=None`, prefill-prioritising like vLLM's original scheduler): an iteration either
  prefills whole prompts of newly admitted requests (no decodes that iteration) or, when nothing can be admitted,
  decodes one token for every running request. Iterations are pure prefill or pure decode.
* **chunked prefill** (`chunk_size=c`, Sarathi-Serve-style stall-free batching [cand:A5]): every iteration has a
  token budget c. Running requests are served first, oldest admission first (a decode costs 1 token, a partially
  prefilled prompt gets its next chunk of up to the remaining budget; if the budget runs out, the remaining running
  requests skip this iteration); the rest of the budget admits waiting requests FCFS with prompt chunks, unless this
  iteration had to preempt. Iterations can be mixed (prefill chunks + decodes).

Per iteration: requests that arrived by the iteration's start time join the FCFS waiting queue; admission is bounded
by `max_batch` (B_max, running = admitted and unfinished requests), by the token budget and by KV capacity; then
the iteration's duration comes from a pluggable `cost(iteration, hw_config)` callable and time advances.

**KV accounting** (E3 generalised to paging): a request that stores T_i KV entries holds ⌈T_i/b⌉ blocks of b
tokens, so the cache holds Σ_i ⌈T_i/b⌉·b token slots ≤ `kv_capacity_blocks`·b, asserted after every allocation.
A prefill chunk [s0, s0+n) stores n entries; the iteration that completes a prompt also emits the request's first
output token. A decode processes the most recent output token: it attends over T = stored + 1 entries (context
including itself), stores that token's KV and emits the next token. A request with S prompt and O output tokens
therefore runs one (possibly chunked) prefill and O − 1 decodes and stores at most S + O − 1 entries; the KV of the
last token is never computed.

**Reservation policies:** `kv_reservation="ondemand"` (vLLM): blocks are allocated as tokens are stored; admission
needs only the blocks for the scheduled chunk; when a running request cannot get a block, the most recently admitted
running request is preempted (recompute preemption: its blocks are freed, it goes to the front of the waiting queue
and later re-prefills prompt + generated tokens, which emits its next token; output token counts are unaffected).
`kv_reservation="full"`: a request is admitted only if blocks for its whole footprint ⌈(S+O−1)/b⌉ are free; they are
reserved at admission, so preemption never happens (uses the trace's output length as an oracle; a bound, not a
real server policy).

Fixed-schedule assumption (methodology §10): `run_schedule` generates the iteration sequence once with a reference
cost; `cost_schedule` re-costs that same sequence under any other hardware configuration.
"""
from __future__ import annotations

import dataclasses
import math
from collections import deque
from collections.abc import Sequence as AbcSequence
from typing import Any, Callable, Sequence

import numpy as np

from serving.traces import Request


@dataclasses.dataclass(frozen=True)
class SchedulerConfig:
    max_batch: int = 256                    # B_max: max running (admitted, unfinished) requests
    token_budget: int = 8192                # full mode: max prompt tokens per prefill iteration (vLLM max_num_batched_tokens)
    chunk_size: int | None = None           # None = full prefill; c = chunked prefill with token budget c per iteration
    block_size: int = 16                    # b, tokens per KV block
    kv_capacity_blocks: int = 1 << 20       # total KV blocks
    kv_reservation: str = "ondemand"        # "ondemand" (vLLM, recompute preemption) | "full" (reserve S+O-1 up front)
    max_iterations: int = 50_000_000        # safety stop

    def __post_init__(self):
        if self.max_batch < 1 or self.token_budget < 1 or self.block_size < 1 or self.kv_capacity_blocks < 1:
            raise ValueError(f"invalid scheduler config {self}")
        if self.chunk_size is not None and self.chunk_size < 1:
            raise ValueError("chunk_size must be >= 1 or None")
        if self.kv_reservation not in ("ondemand", "full"):
            raise ValueError(f"unknown kv_reservation {self.kv_reservation!r}")


@dataclasses.dataclass(frozen=True, slots=True)
class PrefillChunk:
    rid: int
    start: int          # s0: offset of the chunk in the request's prefill sequence (queries [s0, s0+n) attend to [0, s0+n))
    n_tokens: int       # n (≤ c in chunked mode)
    last: bool          # completes the prefill (and emits an output token)
    recompute: bool     # re-prefill after a preemption (prompt + previously generated tokens)


@dataclasses.dataclass(frozen=True, slots=True)
class DecodeItem:
    rid: int
    context: int        # T: KV entries attended, including the token being processed


@dataclasses.dataclass(frozen=True, slots=True)
class Iteration:
    idx: int
    t_start: float
    t_end: float
    prefill: tuple[PrefillChunk, ...]
    decode: tuple[DecodeItem, ...]
    kv_blocks: int          # blocks allocated (incl. full-mode reservations) after this iteration's allocations
    kv_tokens: int          # Σ_i T_i stored after this iteration
    n_running: int          # running requests during this iteration (after admissions and preemptions)
    n_waiting: int          # arrived but not running
    preempted: tuple[int, ...] = ()

    @property
    def n_prefill_tokens(self) -> int:
        return sum(p.n_tokens for p in self.prefill)

    @property
    def n_decode_tokens(self) -> int:
        return len(self.decode)

    @property
    def kind(self) -> str:
        """Methodology §4: prefill = prefill tokens and no decode tokens; mixed = both; otherwise decode."""
        if self.prefill and self.decode:
            return "mixed"
        return "prefill" if self.prefill else "decode"

    @property
    def duration(self) -> float:
        return self.t_end - self.t_start


@dataclasses.dataclass
class RequestRecord:
    rid: int
    arrival: float
    input_len: int
    output_len: int
    first_admit: float | None = None
    first_token: float | None = None
    finish: float | None = None
    output_tokens: int = 0          # tokens emitted (= output_len at the end)
    prefill_tokens: int = 0         # prefill tokens processed, incl. recompute
    recompute_tokens: int = 0       # prefill tokens re-processed because of preemptions
    preemptions: int = 0


# Columns of the compact schedule store: per-iteration scalars, and flat per-item arrays with offsets.
_IT_F = ("t_start", "t_end")
_IT_I = ("kv_blocks", "kv_tokens", "n_running", "n_waiting")
_PF = ("rid", "start", "n_tokens", "last", "recompute")
_DC = ("rid", "context")


class _Recorder:
    """Append-only columnar storage (array.array), ~8 bytes per field per item; a 1M-iteration schedule with 4M
    decode items needs ~100 MB instead of the GBs that one Python object per item would take."""

    def __init__(self):
        from array import array
        self.it_f = {k: array("d") for k in _IT_F}
        self.it_i = {k: array("q") for k in _IT_I}
        self.pf = {k: array("q") for k in _PF}
        self.dc = {k: array("q") for k in _DC}
        self.pre = array("q")
        self.pf_off, self.dc_off, self.pre_off = array("q", [0]), array("q", [0]), array("q", [0])

    def add(self, it: "Iteration"):
        for k in _IT_F:
            self.it_f[k].append(getattr(it, k))
        for k in _IT_I:
            self.it_i[k].append(getattr(it, k))
        for p in it.prefill:
            for k in _PF:
                self.pf[k].append(int(getattr(p, k)))
        for d in it.decode:
            self.dc["rid"].append(d.rid)
            self.dc["context"].append(d.context)
        self.pre.extend(it.preempted)
        self.pf_off.append(len(self.pf["rid"]))
        self.dc_off.append(len(self.dc["rid"]))
        self.pre_off.append(len(self.pre))

    def freeze(self) -> dict[str, np.ndarray]:
        c = {k: np.frombuffer(v, dtype=np.float64).copy() for k, v in self.it_f.items()}
        c.update({k: np.frombuffer(v, dtype=np.int64).copy() for k, v in self.it_i.items()})
        c.update({f"pf_{k}": np.frombuffer(v, dtype=np.int64).copy() for k, v in self.pf.items()})
        c.update({f"dc_{k}": np.frombuffer(v, dtype=np.int64).copy() for k, v in self.dc.items()})
        c["preempted"] = np.frombuffer(self.pre, dtype=np.int64).copy()
        for k in ("pf_off", "dc_off", "pre_off"):
            c[k] = np.frombuffer(getattr(self, k), dtype=np.int64).copy()
        return c


class IterationView(AbcSequence):
    """Read-only sequence of `Iteration`s materialised on access from the columnar store."""

    def __init__(self, cols: dict[str, np.ndarray]):
        self._c = cols
        self._n = len(cols["t_start"])

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, i):
        if isinstance(i, slice):
            return [self[j] for j in range(*i.indices(self._n))]
        if i < 0:
            i += self._n
        if not 0 <= i < self._n:
            raise IndexError(i)
        c = self._c
        a, b = c["pf_off"][i], c["pf_off"][i + 1]
        pf = tuple(PrefillChunk(int(c["pf_rid"][j]), int(c["pf_start"][j]), int(c["pf_n_tokens"][j]),
                                bool(c["pf_last"][j]), bool(c["pf_recompute"][j])) for j in range(a, b))
        a, b = c["dc_off"][i], c["dc_off"][i + 1]
        dc = tuple(DecodeItem(int(r), int(t)) for r, t in zip(c["dc_rid"][a:b].tolist(), c["dc_context"][a:b].tolist()))
        a, b = c["pre_off"][i], c["pre_off"][i + 1]
        return Iteration(i, float(c["t_start"][i]), float(c["t_end"][i]), pf, dc, int(c["kv_blocks"][i]),
                         int(c["kv_tokens"][i]), int(c["n_running"][i]), int(c["n_waiting"][i]),
                         tuple(c["preempted"][a:b].tolist()))

    def __iter__(self):
        for i in range(self._n):
            yield self[i]


@dataclasses.dataclass
class ScheduleResult:
    """`iterations` is a lazy view; `cols` holds the same data as NumPy arrays for vectorised costing:
    per iteration `t_start, t_end, kv_blocks, kv_tokens, n_running, n_waiting`; prefill chunks `pf_rid, pf_start,
    pf_n_tokens, pf_last, pf_recompute` and decodes `dc_rid, dc_context`, item k of iteration i at
    `[pf_off[i], pf_off[i+1])` / `[dc_off[i], dc_off[i+1])`; `preempted` with `pre_off`."""
    cols: dict[str, np.ndarray]
    requests: dict[int, RequestRecord]
    config: SchedulerConfig

    @property
    def iterations(self) -> IterationView:
        return IterationView(self.cols)

    @property
    def n_iterations(self) -> int:
        return len(self.cols["t_start"])

    @property
    def makespan(self) -> float:
        return float(self.cols["t_end"][-1]) if self.n_iterations else 0.0

    def per_iteration(self) -> dict[str, np.ndarray]:
        """Vectorised per-iteration aggregates: prefill tokens, decode count, Σ decode context, kind code
        (0 = decode, 1 = prefill, 2 = mixed; methodology §4)."""
        c = self.cols

        def seg_sum(vals, off):          # Σ of vals over [off[i], off[i+1]) for every iteration
            cs = np.concatenate(([0], np.cumsum(vals)))
            return cs[off[1:]] - cs[off[:-1]]

        pf_items, dc_items = np.diff(c["pf_off"]), np.diff(c["dc_off"])
        n_pf = seg_sum(c["pf_n_tokens"], c["pf_off"])
        ctx = seg_sum(c["dc_context"], c["dc_off"])
        kind = np.where((pf_items > 0) & (dc_items > 0), 2, np.where(pf_items > 0, 1, 0))
        return {"n_prefill_tokens": n_pf, "n_decode": dc_items, "sum_context": ctx, "kind": kind,
                "duration": c["t_end"] - c["t_start"]}


CostFn = Callable[[Iteration, Any], float]


class _Seq:
    """Mutable per-request state while scheduled."""
    __slots__ = ("req", "target", "prefilled", "generated", "blocks", "admit_seq", "recompute", "rec")

    def __init__(self, req: Request, rec: RequestRecord):
        self.req, self.rec = req, rec
        self.target = req.input_len     # prefill length (prompt, or prompt + generated after a preemption)
        self.prefilled = 0
        self.generated = 0
        self.blocks = 0
        self.admit_seq = -1
        self.recompute = False

    @property
    def in_prefill(self) -> bool:
        return self.prefilled < self.target

    @property
    def stored(self) -> int:
        """KV entries stored. During a prefill: the prefilled tokens. After it: S + generated − 1, which holds for the
        first prefill (target S, then generated = 1) and for a recompute (target S + g, then generated = g + 1)."""
        if self.in_prefill:
            return self.prefilled
        return self.req.input_len + self.generated - 1

    def footprint_max(self) -> int:
        return self.req.input_len + self.req.output_len - 1


def _blocks(tokens: int, b: int) -> int:
    return -(-tokens // b)


def run_schedule(requests: Sequence[Request], cfg: SchedulerConfig, cost: CostFn, hw_config: Any = None
                 ) -> ScheduleResult:
    """Simulate the scheduler over `requests` (any order; processed by arrival, then rid)."""
    b, cap = cfg.block_size, cfg.kv_capacity_blocks
    reqs = sorted(requests, key=lambda r: (r.arrival, r.rid))
    records = {r.rid: RequestRecord(r.rid, r.arrival, r.input_len, r.output_len) for r in reqs}
    if len(records) != len(reqs):
        raise ValueError("duplicate request ids")
    for r in reqs:
        if r.input_len < 1 or r.output_len < 1:
            raise ValueError(f"request {r.rid}: lengths must be >= 1")
        if _blocks(r.input_len + r.output_len - 1, b) > cap:
            raise ValueError(f"request {r.rid}: footprint {r.input_len + r.output_len - 1} tokens exceeds the KV "
                             f"capacity of {cap} blocks x {b}")
    pending = deque(reqs)                       # not yet arrived
    waiting: deque[_Seq] = deque()              # arrived, not running (FCFS; preempted go to the front)
    running: list[_Seq] = []                    # admission order (oldest first)
    free = cap
    admit_counter = 0
    rec_store = _Recorder()
    n_it = 0
    t = 0.0
    done = 0
    n_total = len(reqs)
    preempted: list[int] = []      # preemptions since the last recorded iteration (attached to the next record)

    def alloc(s: _Seq, new_stored: int) -> bool:
        """Grow s to hold `new_stored` entries; False if the free pool is too small (nothing changes then)."""
        nonlocal free
        need = _blocks(new_stored, b) - s.blocks
        if need <= 0:
            return True
        if need > free:
            return False
        free -= need
        s.blocks += need
        return True

    def preempt(s: _Seq, preempted: list[int]):
        nonlocal free
        running.remove(s)
        free += s.blocks
        s.blocks = 0
        s.rec.preemptions += 1
        new_target = s.req.input_len + s.generated if s.generated > 0 else s.req.input_len
        s.target, s.prefilled, s.recompute = new_target, 0, True
        waiting.appendleft(s)
        preempted.append(s.req.rid)

    def admit(s: _Seq):
        nonlocal admit_counter, free
        s.admit_seq = admit_counter
        admit_counter += 1
        running.append(s)
        if s.rec.first_admit is None:
            s.rec.first_admit = t
        if cfg.kv_reservation == "full":
            need = _blocks(s.footprint_max(), b) - s.blocks
            free -= need
            s.blocks += need

    def can_admit(s: _Seq, n_tokens: int) -> bool:
        if cfg.kv_reservation == "full":
            return _blocks(s.footprint_max(), b) <= free
        return _blocks(n_tokens, b) <= free

    while done < n_total:
        if n_it >= cfg.max_iterations:
            raise RuntimeError(f"max_iterations {cfg.max_iterations} reached")
        while pending and pending[0].arrival <= t:
            r = pending.popleft()
            waiting.append(_Seq(r, records[r.rid]))
        if not running and not waiting:
            t = max(t, pending[0].arrival)
            continue

        prefill: list[tuple[_Seq, int, int]] = []    # (seq, start, n)
        decode: list[_Seq] = []
        n_pre_before = len(preempted)

        def serve_running(budget: int | None) -> int:
            """Schedule running requests oldest first (decode: 1 token; partial prefill: next chunk), preempting the
            newest running request when a block is missing. Returns the budget left (None = unlimited)."""
            i = 0
            while i < len(running):
                s = running[i]
                if budget is not None and budget <= 0:
                    break
                n = 1 if not s.in_prefill else (s.target - s.prefilled if budget is None
                                                  else min(s.target - s.prefilled, budget))
                new_stored = s.stored + n
                while not alloc(s, new_stored):
                    victim = running[-1]
                    preempt(victim, preempted)
                    if victim is s:
                        break
                if s not in running:          # preempted itself
                    break
                if s.in_prefill:
                    prefill.append((s, s.prefilled, n))
                else:
                    decode.append(s)
                if budget is not None:
                    budget -= n
                i += 1
            return budget

        if cfg.chunk_size is None:
            # full prefill, prefill-prioritising: admit whole prompts if possible; else decode everyone
            budget = cfg.token_budget
            while waiting and len(running) < cfg.max_batch:
                s = waiting[0]
                n = s.target
                if prefill and n > budget:
                    break
                if not can_admit(s, n):
                    break
                waiting.popleft()
                admit(s)
                ok = alloc(s, n)
                assert ok
                prefill.append((s, 0, n))
                budget -= n
            if not prefill:
                serve_running(None)
        else:
            budget = serve_running(cfg.chunk_size)
            # like vLLM, no admissions in an iteration that had to preempt (the victim would be re-admitted at once)
            while budget > 0 and len(preempted) == n_pre_before and waiting and len(running) < cfg.max_batch:
                s = waiting[0]
                n = min(s.target, budget)
                if not can_admit(s, n):
                    break
                waiting.popleft()
                admit(s)
                ok = alloc(s, n)
                assert ok
                prefill.append((s, 0, n))
                budget -= n

        if not prefill and not decode:
            if len(preempted) == n_pre_before:
                raise RuntimeError("scheduler made no progress (bug)")
            continue                  # only preemptions happened; they are recorded with the next iteration

        used = cap - free
        assert 0 <= used <= cap, (used, cap)
        assert sum(s.blocks for s in running) == used
        it = Iteration(
            idx=n_it, t_start=t, t_end=t,
            prefill=tuple(PrefillChunk(s.req.rid, st, n, st + n == s.target, s.recompute) for s, st, n in prefill),
            decode=tuple(DecodeItem(s.req.rid, s.stored + 1) for s in decode),
            kv_blocks=used, kv_tokens=0, n_running=len(running), n_waiting=len(waiting),
            preempted=tuple(preempted))
        # apply effects
        for s, st, n in prefill:
            s.prefilled += n
            s.rec.prefill_tokens += n
            if s.recompute:
                s.rec.recompute_tokens += n
        for s in decode:
            s.generated += 1
            s.rec.output_tokens += 1
        emit_first = [s for s, st, n in prefill if not s.in_prefill]
        for s in emit_first:
            s.generated += 1
            s.rec.output_tokens += 1
            s.recompute = False
        kv_tokens = sum(s.stored for s in running)
        for s in running:
            assert _blocks(s.stored, b) <= s.blocks
        dur = float(cost(dataclasses.replace(it, kv_tokens=kv_tokens), hw_config))
        if not (dur >= 0 and math.isfinite(dur)):
            raise ValueError(f"cost returned {dur} for iteration {it.idx}")
        t_end = t + dur
        rec_store.add(dataclasses.replace(it, t_end=t_end, kv_tokens=kv_tokens))
        n_it += 1
        preempted = []
        for s in emit_first:
            if s.rec.first_token is None:
                s.rec.first_token = t_end
        for s in list(running):
            if s.generated >= s.req.output_len:
                s.rec.finish = t_end
                running.remove(s)
                free += s.blocks
                s.blocks = 0
                done += 1
        t = t_end

    assert free == cap
    return ScheduleResult(rec_store.freeze(), records, cfg)


def cost_schedule(iterations: Sequence[Iteration], cost: CostFn, hw_config: Any = None) -> np.ndarray:
    """Fixed-schedule assumption: the cost of the same iteration sequence under another hardware configuration."""
    return np.array([float(cost(it, hw_config)) for it in iterations], dtype=float)


# ------------------------------------------------------------------------------------------------ helpers
def iteration_region(it: Iteration, kv_bytes_per_token: float, weight_bytes: float) -> str:
    """Methodology §4 region of one iteration: prefill / mixed / decode-weight / decode-kv, where a decode iteration
    is KV-bound iff ρ_KV = Σ_i M_KV/token · T_i / (q_w·N_params) ≥ 1 (the E6 crossover)."""
    k = it.kind
    if k != "decode":
        return k
    rho = kv_bytes_per_token * sum(d.context for d in it.decode) / weight_bytes
    return "decode-kv" if rho >= 1 else "decode-weight"


def kv_blocks_for_bytes(kv_bytes: float, kv_bytes_per_token: float, block_size: int) -> int:
    """Number of whole KV blocks of `block_size` tokens that fit in `kv_bytes` (E3 per token × b per block)."""
    return int(kv_bytes // (kv_bytes_per_token * block_size))


def linear_cost(a: float = 0.0, per_prefill_token: float = 0.0, per_decode_seq: float = 0.0,
                per_context_token: float = 0.0) -> CostFn:
    """A toy cost for tests and smoke runs (not a hardware model): a + p·N_prefill + d·N_decode + k·Σ T."""
    def f(it: Iteration, _cfg=None) -> float:
        return (a + per_prefill_token * it.n_prefill_tokens + per_decode_seq * it.n_decode_tokens
                + per_context_token * sum(d.context for d in it.decode))
    return f
