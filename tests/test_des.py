"""TEST: hand-computed toy cases for TTFT/TBT, token conservation, determinism given the seed; plus the
 cross-check, policy/oracle interface, goodput and bootstrap units table fit."""
import json
import math

import numpy as np
import pytest

from serving import des
from serving.des import (EngineConfig, HintPolicy, LinearCost, OraclePolicy, StaticPolicy, goodput_rate,
                         bootstrap_ci, poisson_workload, run_metrics, simulate)
from serving.scheduler import SchedulerConfig, run_schedule
from serving.traces import Request

# Toy cost (clock 1 Hz so cycles = seconds): prefill 1 + 0.1/token, decode 1 + 0.5/decode, mixed 1 + 0.1·Npf + 0.5·Ndec.
TOY = LinearCost({("prefill", "base"): [1, 0.1, 0, 0, 0, 0], ("decode", "base"): [1, 0, 0, 0.5, 0, 0],
                  ("mixed", "base"): [1, 0.1, 0, 0.5, 0, 0],
                  ("prefill", "fast"): [0.5, 0.1, 0, 0, 0, 0], ("decode", "fast"): [0.5, 0, 0, 0.5, 0, 0],
                  ("mixed", "fast"): [0.5, 0.1, 0, 0.5, 0, 0]}, ["base", "fast"])
FULL = EngineConfig(max_batch=8, chunk_size=None, clock_hz=1.0)
CHUNK = EngineConfig(max_batch=8, chunk_size=4, clock_hz=1.0)


def test_toy_full_prefill_two_requests():
    # A: arr 0, S=10, O=3; B: arr 0.5, S=5, O=2.
    # it0 [0,2]: prefill A(10) → A tok1 @2. it1 [2,2.5]: prefill B (1+0.5) → wait: B arrived 0.5 ≤ 2, admitted →
    # it1 [2,3.5] B tok1 @3.5. it2 decode A,B (1+1=2) [3.5,5.5]: A tok2, B tok2 (B done). it3 decode A [5.5,7]: A tok3.
    r = simulate([Request(0, 0.0, 10, 3), Request(1, 0.5, 5, 2)], StaticPolicy(), TOY, FULL)
    a, b = r.requests[0], r.requests[1]
    assert a.token_times == pytest.approx([2.0, 5.5, 7.0])
    assert b.token_times == pytest.approx([3.5, 5.5])
    assert a.ttft == pytest.approx(2.0) and b.ttft == pytest.approx(3.0)
    assert list(a.tbts) == pytest.approx([3.5, 1.5]) and list(b.tbts) == pytest.approx([2.0])
    assert a.tpot == pytest.approx(2.5)
    m = run_metrics(r, {"x": (3.0, 2.0)})
    # pooled TBTs {3.5, 1.5, 2.0}: P50 = 2.0; TTFT {2, 3}: P50 = 2.5; attain: A tpot 2.5 > 2 fails, B (3 ≤ 3, 2 ≤ 2) ok → 0.5
    assert m["tbt_p50"] == pytest.approx(2.0) and m["ttft_p50"] == pytest.approx(2.5)
    assert m["attain_x"] == pytest.approx(0.5)


def test_toy_chunked_three_requests():
    # chunk 4. A arr0 S6 O2; B arr0 S2 O1; C arr 10 S1 O1.
    # it0 t0: admit A chunk 4 (budget 0) → prefill cost 1+0.4=1.4 → [0,1.4]
    # it1: A rest 2 (budget 2), admit B 2 → prefill 4 tok: 1.4 → [1.4,2.8]: A tok1, B tok1 (B done)
    # it2: A decode → decode 1.5 → [2.8,4.3]: A tok2 done. idle; C arrives 10: prefill 1 tok 1.1 → 11.1
    r = simulate([Request(0, 0, 6, 2), Request(1, 0, 2, 1), Request(2, 10.0, 1, 1)], StaticPolicy(), TOY, CHUNK)
    assert r.requests[0].token_times == pytest.approx([2.8, 4.3])
    assert r.requests[1].token_times == pytest.approx([2.8])
    assert r.requests[2].token_times == pytest.approx([11.1])
    assert math.isnan(r.requests[1].tpot)
    assert [it[2] for it in r.iterations] == ["prefill", "prefill", "decode-weight", "prefill"]


def test_toy_mixed_iteration_and_hint_switch_cost():
    # chunk 4; A arr0 S1 O3; B arr 1.0 S4 O1. it0 [0,1.1] A tok1. it1 t=1.1: A decode (1) + B chunk 3 → mixed
    # fast: 0.5+0.3+0.5 = 1.3 plus switch 0.25 (prefill→mixed changes config) ... mapping prefill:base, mixed:fast.
    pol = HintPolicy({"prefill": "base", "mixed": "fast", "decode-weight": "fast", "decode-kv": "fast"}, csw_cycles=0.25)
    r = simulate([Request(0, 0, 1, 3), Request(1, 1.0, 4, 1)], pol, TOY, CHUNK)
    assert [it[2] for it in r.iterations][:2] == ["prefill", "mixed"]
    assert r.iterations[1][1] == pytest.approx(1.1 + 1.55)
    # it2 t=2.65: A decode + B last 1 token → mixed fast, no switch: 0.5+0.1+0.5 = 1.1 → 3.75: A tok3, B tok1
    assert r.requests[0].token_times == pytest.approx([1.1, 2.65, 3.75])
    assert r.requests[1].token_times == pytest.approx([3.75])


@pytest.mark.parametrize("cfg", [FULL, CHUNK, EngineConfig(max_batch=2, chunk_size=3, clock_hz=1.0)])
def test_token_conservation_and_p13_crosscheck(cfg):
    pool = [Request(0, 0, 7, 5), Request(1, 0, 3, 1), Request(2, 0, 12, 4), Request(3, 0, 1, 9)]
    reqs = poisson_workload(pool, rate=0.7, n=40, seed=3)
    r = simulate(reqs, StaticPolicy(), TOY, cfg)
    assert r.n_prefill_tokens == sum(q.input_len for q in reqs)
    assert r.n_decode_tokens == sum(q.output_len - 1 for q in reqs)
    for q in reqs:
        rr = r.requests[q.rid]
        assert len(rr.token_times) == q.output_len
        assert rr.ttft > 0 and np.all(rr.tbts > 0)
    # same engine semantics as (unlimited KV): identical first-token and finish times
    def cost(it, _):
        return TOY.cycles(des.IterSummary(it.idx, tuple((p.rid, p.start, p.n_tokens) for p in it.prefill),
                                          tuple(d.context for d in it.decode)), "base")
    sc = SchedulerConfig(max_batch=cfg.max_batch, chunk_size=cfg.chunk_size, token_budget=cfg.token_budget)
    ref = run_schedule(reqs, sc, cost)
    for q in reqs:
        assert ref.requests[q.rid].first_token == pytest.approx(r.requests[q.rid].token_times[0])
        assert ref.requests[q.rid].finish == pytest.approx(r.requests[q.rid].finish)
    assert ref.n_iterations == len(r.iterations)


def test_deterministic_given_seed():
    pool = [Request(0, 0, 7, 5), Request(1, 0, 3, 2)]
    a = simulate(poisson_workload(pool, 0.5, 30, 11), StaticPolicy(), TOY, CHUNK)
    b = simulate(poisson_workload(pool, 0.5, 30, 11), StaticPolicy(), TOY, CHUNK)
    c = simulate(poisson_workload(pool, 0.5, 30, 12), StaticPolicy(), TOY, CHUNK)
    assert {k: v.token_times for k, v in a.requests.items()} == {k: v.token_times for k, v in b.requests.items()}
    assert {k: v.token_times for k, v in a.requests.items()} != {k: v.token_times for k, v in c.requests.items()}


def test_oracle_interface(tmp_path):
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"by_index": ["fast"], "by_region": {"decode-weight": {"ratio": 0.5},
                                                                    "prefill": "base", "mixed": "base"}}))
    pol = des.load_oracle_schedule(p)
    r = simulate([Request(0, 0, 1, 2)], pol, TOY, CHUNK)
    # it0 by_index fast: 0.5+0.1 = 0.6; it1 decode region ratio 0.5 × base (1.5) = 0.75
    assert r.requests[0].token_times == pytest.approx([0.6, 1.35])
    with pytest.raises(KeyError):
        simulate([Request(0, 0, 1, 2)], OraclePolicy(by_region={"prefill": "base"}), TOY, CHUNK)


def test_goodput_and_bootstrap():
    assert goodput_rate([1, 2, 3], [1.0, 0.95, 0.85]) == pytest.approx(2.5)
    assert goodput_rate([1, 2], [0.5, 0.4]) == 0.0
    assert goodput_rate([1, 2], [1.0, 0.99]) == 2.0
    est, lo, hi = bootstrap_ci(np.array([1.0, 2, 3, 4, 5]))
    assert est == 3.0 and 1.0 <= lo < 3.0 < hi <= 5.0
    assert bootstrap_ci(np.ones(5)) == (1.0, 1.0, 1.0)
    with pytest.raises(ValueError):
        des.sweep([Request(0, 0, 1, 1)], [StaticPolicy()], TOY, [1.0], [0, 1, 2, 3], 5, {"x": (1, 1)})


def test_p61_table_fit():
    t = des.CostTable.from_p61()
    assert "base" in t.configs and len(t.configs) == 15
    rep = t.fit_report()
    # C1 (and C2) caps are poorly fit (non-linear decode behaviour; §15.2); every other column within 7 %
    assert all(v["max_abs_rel"] < 0.07 for k, v in rep.items() if not k.endswith(("|C1", "|C2"))), rep
    assert rep["decode|C1"]["max_abs_rel"] > 0.2          # documented limitation, keep it visible
    rat = t.ratio_error_report()
    assert all(rat[c] < 0.011 for c in ("D_lru", "D_byp", "B_df", "B_s256", "B_s1024")), rat
    # positive and monotone in context for a decode iteration
    it1 = des.IterSummary(0, (), (128,) * 4)
    it2 = des.IterSummary(0, (), (2048,) * 4)
    assert 0 < t.cycles(it1, "base") < t.cycles(it2, "base")
