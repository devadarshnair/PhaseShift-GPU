"""Tests: trace loaders, length scaling σ, iteration scheduler.

TEST list from the plan: hand-built 3-request trace -> exact iteration list; chunking conserves prompt tokens;
KV capacity never exceeded; output tokens per request equal GeneratedTokens. Plus loader/statistics checks against
an independent pandas computation and the scheduler invariants of methodology §10.
"""
import csv
import json
import math

import numpy as np
import pandas as pd
import pytest

from serving import scheduler as sch
from serving import traces as tr
from serving.scheduler import DecodeItem, PrefillChunk, SchedulerConfig, run_schedule
from serving.traces import Request

UNIT = sch.linear_cost(a=1.0)     # every iteration takes 1 s


# ------------------------------------------------------------------------------------------------ hand-built trace
def _three():
    return [Request(0, 0.0, 4, 3), Request(1, 0.0, 2, 2), Request(2, 2.5, 3, 1)]


def _summary(res):
    return [(it.t_start, it.t_end,
             tuple((p.rid, p.start, p.n_tokens, p.last) for p in it.prefill),
             tuple((d.rid, d.context) for d in it.decode)) for it in res.iterations]


def test_three_requests_full_prefill_exact():
    """Full prefill, prefill-prioritising. t=0: prefill r0 (4) + r1 (2), both emit token 1. t=1: decode r0 (T=5),
    r1 (T=3) -> r1 done (2 tokens). t=2: decode r0 (T=6) -> r0 done (3 tokens). t=3: r2 arrived at 2.5: prefill 3 ->
    emits its only token, done."""
    res = run_schedule(_three(), SchedulerConfig(max_batch=8, token_budget=100, block_size=2, kv_capacity_blocks=100),
                       UNIT)
    assert _summary(res) == [
        (0.0, 1.0, ((0, 0, 4, True), (1, 0, 2, True)), ()),
        (1.0, 2.0, (), ((0, 5), (1, 3))),
        (2.0, 3.0, (), ((0, 6),)),
        (3.0, 4.0, ((2, 0, 3, True),), ()),
    ]
    rec = res.requests
    assert [rec[i].first_token for i in range(3)] == [1.0, 1.0, 4.0]
    assert [rec[i].finish for i in range(3)] == [3.0, 2.0, 4.0]
    # Σ⌈T_i/2⌉ after this iteration's stores (before finished requests free theirs): it0 T=(4,2) -> 2+1;
    # it1 T=(5,3) -> 3+2; it2 T=6 -> 3; it3 T=3 -> 2. kv_tokens = Σ T_i at the same point.
    assert [it.kv_blocks for it in res.iterations] == [3, 5, 3, 2]
    assert [it.kv_tokens for it in res.iterations] == [6, 8, 6, 3]
    assert [it.kind for it in res.iterations] == ["prefill", "decode", "decode", "prefill"]


def test_three_requests_chunked_exact():
    """Sarathi chunk c=3. t=0: r0 chunk [0,3). t=1: r0 [3,4) (last) + r1 [0,2) (last). t=2: decode r0 (T=5), r1 (T=3);
    r1 done; r2 not yet arrived. t=3: decode r0 (T=6) + r2 chunk [0,2) (mixed); r0 done. t=4: r2 [2,3) last, done."""
    res = run_schedule(_three(), SchedulerConfig(max_batch=8, chunk_size=3, block_size=2, kv_capacity_blocks=100), UNIT)
    assert _summary(res) == [
        (0.0, 1.0, ((0, 0, 3, False),), ()),
        (1.0, 2.0, ((0, 3, 1, True), (1, 0, 2, True)), ()),
        (2.0, 3.0, (), ((0, 5), (1, 3))),
        (3.0, 4.0, ((2, 0, 2, False),), ((0, 6),)),
        (4.0, 5.0, ((2, 2, 1, True),), ()),
    ]
    assert [it.kind for it in res.iterations] == ["prefill", "prefill", "decode", "mixed", "prefill"]
    assert [res.requests[i].finish for i in range(3)] == [4.0, 3.0, 5.0]


def test_idle_gap_jumps_to_next_arrival():
    res = run_schedule([Request(0, 10.0, 1, 1)], SchedulerConfig(), UNIT)
    assert _summary(res) == [(10.0, 11.0, ((0, 0, 1, True),), ())]


def test_full_reservation_waits_instead_of_preempting():
    # capacity 4 blocks of 2 = 8 slots; r0 footprint 4+3-1 = 6 slots (3 blocks), r1 2+2-1 = 3 (2 blocks): r1 waits.
    res = run_schedule(_three(), SchedulerConfig(max_batch=8, block_size=2, kv_capacity_blocks=4,
                                                 kv_reservation="full"), UNIT)
    assert _summary(res)[0] == (0.0, 1.0, ((0, 0, 4, True),), ())
    assert all(not it.preempted for it in res.iterations)
    assert res.iterations[0].kv_blocks == 3                              # reserved for the whole footprint


def test_request_larger_than_cache_rejected():
    with pytest.raises(ValueError, match="exceeds the KV capacity"):
        run_schedule([Request(0, 0.0, 10, 10)], SchedulerConfig(block_size=4, kv_capacity_blocks=4), UNIT)


def test_config_validation():
    with pytest.raises(ValueError):
        SchedulerConfig(chunk_size=0)
    with pytest.raises(ValueError):
        SchedulerConfig(kv_reservation="lazy")
    with pytest.raises(ValueError, match="cost returned"):
        run_schedule(_three(), SchedulerConfig(), lambda it, c: float("nan"))


# ------------------------------------------------------------------------------------------------ invariants
def _replay_check(reqs, res):
    """Independent replay of the iteration records: KV slots, budgets, B_max, arrivals, token conservation."""
    cfg = res.config
    b, cap = cfg.block_size, cfg.kv_capacity_blocks
    by_id = {r.rid: r for r in reqs}
    stored = {}                     # rid -> stored KV entries of running requests
    out = {r.rid: 0 for r in reqs}
    pre = {r.rid: 0 for r in reqs}
    t_prev = 0.0
    finished = set()
    for it in res.iterations:
        assert it.t_start >= t_prev - 1e-12 and it.t_end >= it.t_start
        t_prev = it.t_end
        for rid in it.preempted:
            stored.pop(rid, None)
        rids = [p.rid for p in it.prefill] + [d.rid for d in it.decode]
        assert len(rids) == len(set(rids))
        for rid in rids:
            assert by_id[rid].arrival <= it.t_start and rid not in finished
        for p in it.prefill:
            if p.start == 0:
                stored[p.rid] = 0
            assert stored[p.rid] == p.start                # chunks are contiguous
            stored[p.rid] += p.n_tokens
            pre[p.rid] += p.n_tokens
        for d in it.decode:
            assert d.context == stored[d.rid] + 1          # attends over cached + itself
            stored[d.rid] += 1
        # KV capacity: Σ ⌈T_i/b⌉ ≤ capacity after this iteration's stores (ondemand: = allocated blocks)
        need = sum(-(-v // b) for v in stored.values())
        assert need <= cap and it.kv_blocks <= cap
        if cfg.kv_reservation == "ondemand":
            assert need == it.kv_blocks
        assert it.kv_tokens == sum(stored.values())
        assert it.n_running <= cfg.max_batch and len(rids) <= cfg.max_batch
        if cfg.chunk_size is not None:
            assert it.n_prefill_tokens + it.n_decode_tokens <= cfg.chunk_size
        else:
            assert it.kind != "mixed"
            assert it.n_prefill_tokens <= cfg.token_budget or len(it.prefill) == 1
        for p in it.prefill:
            if p.last:
                out[p.rid] += 1
        for d in it.decode:
            out[d.rid] += 1
        for rid in list(stored):
            if out[rid] == by_id[rid].output_len:
                finished.add(rid)
                del stored[rid]
    return out, pre


@pytest.fixture(scope="module")
def conv_head():
    return tr.load_trace("azure_conv", limit=400)


@pytest.mark.parametrize("chunk", [None, 64, 256, 512, 2048])
def test_conservation_large_cache(conv_head, chunk):
    """No preemption: Σ prefill chunk tokens = S exactly and emitted tokens = GeneratedTokens for every request."""
    cfg = SchedulerConfig(max_batch=32, token_budget=4096, chunk_size=chunk, block_size=16, kv_capacity_blocks=1 << 16)
    cost = sch.linear_cost(a=2e-3, per_prefill_token=2e-5, per_decode_seq=1e-4, per_context_token=1e-7)
    res = run_schedule(conv_head, cfg, cost)
    out, pre = _replay_check(conv_head, res)
    n_dec = np.bincount(res.cols["dc_rid"], minlength=len(conv_head))
    for r in conv_head:
        rec = res.requests[r.rid]
        assert rec.preemptions == 0
        assert pre[r.rid] == r.input_len == rec.prefill_tokens            # chunking conserves prompt tokens
        assert out[r.rid] == r.output_len == rec.output_tokens            # = GeneratedTokens
        assert n_dec[r.rid] == r.output_len - 1
        assert rec.arrival <= rec.first_admit <= rec.first_token <= rec.finish
    np.testing.assert_allclose(sch.cost_schedule(res.iterations, cost), [it.duration for it in res.iterations])
    if chunk is not None and chunk <= 512:
        assert any(it.kind == "mixed" for it in res.iterations)


@pytest.mark.parametrize("chunk", [None, 128])
@pytest.mark.parametrize("reservation", ["ondemand", "full"])
def test_kv_capacity_never_exceeded_under_pressure(conv_head, chunk, reservation):
    """Tight cache (smallest capacity that fits the largest request): preemption (ondemand) or admission waits (full);
    the KV slot sum never exceeds capacity and every request still emits exactly GeneratedTokens tokens."""
    reqs = conv_head[:150]
    b = 16
    cap = max(-(-(r.input_len + r.output_len - 1) // b) for r in reqs) + 8
    cfg = SchedulerConfig(max_batch=64, token_budget=8192, chunk_size=chunk, block_size=b, kv_capacity_blocks=cap,
                          kv_reservation=reservation)
    res = run_schedule(reqs, cfg, sch.linear_cost(a=1e-3, per_prefill_token=1e-5, per_decode_seq=1e-4))
    out, pre = _replay_check(reqs, res)
    assert all(out[r.rid] == r.output_len == res.requests[r.rid].output_tokens for r in reqs)
    n_pre = sum(rec.preemptions for rec in res.requests.values())
    if reservation == "ondemand":
        assert n_pre > 0                                    # the scenario really exercises preemption
        for r in reqs:
            rec = res.requests[r.rid]
            assert pre[r.rid] == rec.prefill_tokens >= r.input_len
            if rec.preemptions == 0:
                assert rec.prefill_tokens == r.input_len and rec.recompute_tokens == 0
    else:
        assert n_pre == 0
        assert all(pre[r.rid] == r.input_len for r in reqs)


def test_max_batch_respected(conv_head):
    res = run_schedule(conv_head[:200], SchedulerConfig(max_batch=4, chunk_size=256), sch.linear_cost(a=1e-3))
    assert max(it.n_running for it in res.iterations) == 4
    _replay_check(conv_head[:200], res)


def test_deterministic(conv_head):
    cfg = SchedulerConfig(max_batch=16, chunk_size=512)
    cost = sch.linear_cost(a=1e-3, per_prefill_token=1e-5)
    assert _summary(run_schedule(conv_head, cfg, cost)) == _summary(run_schedule(conv_head, cfg, cost))


def test_fixed_schedule_recosting(conv_head):
    """Methodology §10 (a): the sequence generated with a reference cost is re-costed, not re-scheduled."""
    ref_cost = sch.linear_cost(a=1e-3, per_prefill_token=1e-5, per_decode_seq=1e-4)
    res = run_schedule(conv_head[:100], SchedulerConfig(max_batch=16, chunk_size=256), ref_cost)
    other = lambda it, c: c * ref_cost(it)
    np.testing.assert_allclose(sch.cost_schedule(res.iterations, other, 2.0), 2 * np.array(
        [it.duration for it in res.iterations]))


def test_iteration_region():
    it = sch.Iteration(0, 0, 1, (), (DecodeItem(0, 100), DecodeItem(1, 50)), 0, 0, 2, 0)
    assert sch.iteration_region(it, kv_bytes_per_token=10, weight_bytes=1500) == "decode-kv"     # ρ = 1 (≥ 1)
    assert sch.iteration_region(it, kv_bytes_per_token=10, weight_bytes=1501) == "decode-weight"
    mixed = sch.Iteration(0, 0, 1, (PrefillChunk(2, 0, 8, True, False),), it.decode, 0, 0, 3, 0)
    assert sch.iteration_region(mixed, 10, 1) == "mixed"
    assert sch.kv_blocks_for_bytes(1 << 20, 1280, 16) == 51


# ------------------------------------------------------------------------------------------------ loaders
def test_azure_loader_matches_pandas(ps_root):
    for name in ("azure_code", "azure_conv"):
        reqs = tr.load_trace(name)
        df = pd.read_csv(ps_root / tr.TRACES[name][0])
        assert len(reqs) == len(df)                       # 8,819 / 19,366 data rows
        ts = pd.to_datetime(df.TIMESTAMP)
        arr = (ts - ts.iloc[0]).dt.total_seconds().to_numpy()
        np.testing.assert_allclose([r.arrival for r in reqs], arr, atol=1e-6)
        assert [r.input_len for r in reqs] == df.ContextTokens.tolist()
        assert [r.output_len for r in reqs] == df.GeneratedTokens.tolist()
        assert [r.rid for r in reqs] == list(range(len(reqs)))
    assert len(tr.load_trace("azure_code")) == 8819 and len(tr.load_trace("azure_conv")) == 19366


def test_azure_timestamp_parse():
    a = tr._azure_ticks("2023-11-16 18:17:03.9799600")
    b = tr._azure_ticks("2023-11-16 18:17:04.0319600")
    assert b - a == 520_000                               # 100 ns ticks, exact
    assert tr._azure_ticks("2023-11-16 18:17:04") - a == 200_400


@pytest.mark.parametrize("name,n", [("mooncake_conversation", 12031), ("mooncake_toolagent", 23608),
                                    ("mooncake_synthetic", 3993)])
def test_mooncake_loader(ps_root, name, n):
    reqs = tr.load_trace(name)
    rows = [json.loads(l) for l in open(ps_root / tr.TRACES[name][0])]
    assert len(reqs) == len(rows) == n
    t0 = rows[0]["timestamp"]
    assert [r.arrival for r in reqs] == [(d["timestamp"] - t0) / 1000.0 for d in rows]     # ms -> s
    assert all(r.hash_ids == tuple(d["hash_ids"]) for r, d in zip(reqs, rows))
    assert all(len(r.hash_ids) == math.ceil(r.input_len / tr.MOONCAKE_BLOCK_TOKENS) for r in reqs)
    assert reqs[-1].arrival <= 3600.0


def test_trace_stats_vs_pandas(ps_root):
    for name in tr.TRACES:
        st = tr.trace_stats(tr.load_trace(name))
        path, kind = tr.TRACES[name]
        if kind == "azure":
            df = pd.read_csv(ps_root / path).rename(columns={"ContextTokens": "i", "GeneratedTokens": "o"})
        else:
            df = pd.read_json(ps_root / path, lines=True).rename(columns={"input_length": "i", "output_length": "o"})
        for col, key in (("i", "input"), ("o", "output")):
            for p in (50, 90, 99):
                assert st[f"{key}_p{p}"] == pytest.approx(df[col].quantile(p / 100))
            assert st[f"{key}_mean"] == pytest.approx(df[col].mean())
            assert st[f"{key}_max"] == df[col].max() and st[f"{key}_min"] == df[col].min()


def test_trace_stats_csv_current(ps_root):
    """results/<run>/trace_stats.csv equals a fresh computation (regenerate with `python -m serving.traces`)."""
    path = ps_root / "results" / "P1.3" / "trace_stats.csv"
    rows = {r["trace"]: r for r in csv.DictReader(open(path))}
    assert set(rows) == set(tr.TRACES)
    for name, row in rows.items():
        st = tr.trace_stats(tr.load_trace(name))
        for k in ("n_requests", "input_p50", "input_p90", "output_p50", "output_p90", "input_mean", "output_mean"):
            assert float(row[k]) == pytest.approx(st[k], rel=1e-5), (name, k)


# ------------------------------------------------------------------------------------------------ σ scaling
def test_scale_lengths_rounding_and_floor():
    reqs = [Request(0, 0.0, 100, 10), Request(1, 1.0, 3, 1), Request(2, 2.0, 25, 5)]
    out = tr.scale_lengths(reqs, 0.1)
    assert [(r.input_len, r.output_len) for r in out] == [(10, 1), (1, 1), (3, 1)]    # 2.5 -> 3 (half-up), min 1
    assert [r.arrival for r in out] == [0.0, 1.0, 2.0]
    with pytest.raises(ValueError):
        tr.scale_lengths(reqs, 0.0)


def test_scale_lengths_max_context():
    reqs = [Request(0, 0.0, 1000, 300), Request(1, 0.0, 100, 50), Request(2, 0.0, 10, 1000)]
    clip = tr.scale_lengths(reqs, 1.0, max_context=256)
    assert [(r.input_len, r.output_len) for r in clip] == [(1, 255), (100, 50), (1, 255)]
    assert all(r.input_len + r.output_len <= 256 for r in clip)
    drop = tr.scale_lengths(reqs, 1.0, max_context=256, policy="drop")
    assert [r.rid for r in drop] == [1]


def test_sigma_matches_median_ratio():
    reqs = tr.load_trace("mooncake_conversation")
    kv_ref, l2_ref = 131072.0, 50 * 2 ** 20            # illustrative numbers (test only)
    kv_tgt, l2_tgt = 13824.0, 2 ** 20
    sigma = tr.sigma_for_kv_l2(kv_ref, l2_ref, kv_tgt, l2_tgt)
    assert sigma == pytest.approx((kv_ref / l2_ref) / (kv_tgt / l2_tgt))
    target = tr.median_kv_l2_ratio(reqs, kv_ref, l2_ref)
    got = tr.median_kv_l2_ratio(tr.scale_lengths(reqs, sigma), kv_tgt, l2_tgt)
    assert abs(got - target) <= 2 * kv_tgt / l2_tgt       # within the rounding of two lengths


def test_columnar_store_roundtrip(conv_head):
    """The lazy IterationView and the vectorised per_iteration() aggregates agree with each other."""
    res = run_schedule(conv_head[:120], SchedulerConfig(max_batch=16, chunk_size=256, block_size=16,
                                                        kv_capacity_blocks=300), sch.linear_cost(a=1e-3))
    its = list(res.iterations)
    assert len(its) == len(res.iterations) == res.n_iterations and res.iterations[-1] == its[-1]
    agg = res.per_iteration()
    code = {"decode": 0, "prefill": 1, "mixed": 2}
    assert agg["n_prefill_tokens"].tolist() == [it.n_prefill_tokens for it in its]
    assert agg["n_decode"].tolist() == [it.n_decode_tokens for it in its]
    assert agg["sum_context"].tolist() == [sum(d.context for d in it.decode) for it in its]
    assert agg["kind"].tolist() == [code[it.kind] for it in its]
    assert [it.idx for it in its] == list(range(len(its)))
    assert res.iterations[2:4] == its[2:4]
    assert res.makespan == its[-1].t_end
