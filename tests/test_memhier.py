"""Tests: limiting cases (infinite cache -> compulsory traffic only; zero cache -> all
traffic to DRAM; KV insertion-at-LRU leaves the weight hit rate unchanged when the KV footprint grows;
persist_across_launch=False -> no cross-launch hits even with an infinite cache), monotonicity in cache size and
bandwidth, and the M/D/1 priority formula against a 10^5-request discrete simulation within 5 %. Plus the parameter
derivations from VX_config.toml, knob validation, tiling arithmetic, representative-layer exactness and the
scheduler glue (methodology §9)."""
import dataclasses
import math
import random
import re

import pytest

from model import memhier as mh
from model import opgen, roofline
from serving import scheduler as sch
from serving.traces import Request

INF = math.inf


@pytest.fixture(scope="module")
def s15(ps_root):
    return opgen.from_llama2c_bin(ps_root / "data" / "models" / "stories15M.bin")


@pytest.fixture(scope="module")
def s260(ps_root):
    return opgen.from_llama2c_bin(ps_root / "data" / "models" / "stories260K.bin")


@pytest.fixture(scope="module")
def llama8(ps_root):
    return opgen.from_hf_config(ps_root / "configs" / "unsloth__Meta-Llama-3.1-8B.json")


@pytest.fixture(scope="module")
def dsv2(ps_root):
    return opgen.from_hf_config(ps_root / "configs" / "deepseek-ai__DeepSeek-V2-Lite.json")


def scen(m):
    cap = m.max_seq or 1 << 30
    return [opgen.prefill(1, min(256, cap)), opgen.decode(1, min(200, cap)), opgen.decode(8, min(256, cap)),
            opgen.mixed([(min(64, cap), 0), (32, min(100, cap))], [(5, min(180, cap))])]


def knobsets(h):
    half = int(h.lmem_bytes // 2)
    return [mh.Knobs(), mh.Knobs(lmem_use=half), mh.Knobs(lmem_use=half, dxa_depth=2),
            mh.Knobs(kv_insert="lru"), mh.Knobs(kv_insert="bypass", kv_insert_scope="all"),
            mh.Knobs(b_two_class=True), mh.Knobs(a2_weight_ways=h.l2_ways // 2),
            mh.Knobs(lmem_use=int(h.lmem_bytes), dxa_depth=1, b_two_class=True, kv_insert="lru")]


def coster(m, q, machine="vortex-S1-8c", **hk):
    h = mh.hier_for(machine)
    persist = hk.pop("persist", None)
    h = dataclasses.replace(h, **hk)
    return mh.MemHier(m, machine, hier=h, q_w=q, q_a=q, q=q, persist_across_launch=persist)


def compulsory_reads(m, seqs, q):
    t = opgen.totals(opgen.gen_ops(m, list(seqs), q_w=q, q_a=q, q=q))
    return t["w_rd"] + t["a_rd"] + t["kv_rd"]


# --------------------------------------------------------------------------------------------- limiting cases


@pytest.mark.parametrize("persist_machine", [("vortex-S1-8c", False), ("h100-like", False)])
def test_infinite_cache_no_persist_gives_compulsory_reads(s15, llama8, persist_machine):
    machine, _ = persist_machine
    for m, q in ((s15, 4), (llama8, 2)):
        c = coster(m, q, machine, l2_bytes=INF, persist=False)
        for seqs in scen(m):
            comp = compulsory_reads(m, seqs, q)
            for k in knobsets(c.h):
                r = c.cost_seqs(seqs, k)
                if k.kv_insert == "bypass":
                    # bypassed (never allocated) KV re-reads go to DRAM whatever the capacity; every other class
                    # behaves exactly as with normal insertion (compulsory reads only)
                    dem = mh.demoted_classes(k)
                    ref = c.cost_seqs(seqs, dataclasses.replace(k, kv_insert="mru"))
                    assert all(r.dram_rd[cl] == pytest.approx(r.l2_rd[cl], rel=1e-12) for cl in dem)
                    assert all(r.dram_rd[cl] == pytest.approx(ref.dram_rd[cl], rel=1e-12)
                               for cl in mh.CLASSES if cl not in dem)
                    assert sum(r.dram_rd.values()) >= comp * (1 - 1e-12)
                    continue
                assert sum(r.dram_rd.values()) == pytest.approx(comp, rel=1e-12)
                assert r.compulsory_rd == pytest.approx(comp, rel=1e-12)
                assert sum(r.hit_cross.values()) == 0.0


def test_no_persist_means_no_cross_launch_hits_even_with_infinite_cache(s15):
    c = coster(s15, 4, l2_bytes=INF, persist=False)
    for seqs in scen(s15):
        r = c.cost_seqs(seqs, mh.Knobs(), detail=True)
        assert all(v == 0.0 for v in r.hit_cross.values())
        assert all(all(v == 0.0 for v in L.hit_cross.values()) for L in r.launches)
    # the same machine with persistence does reuse across launches
    cp = coster(s15, 4, l2_bytes=INF, persist=True)
    assert sum(cp.cost_seqs(opgen.decode(4, 128)).hit_cross.values()) > 0


def test_infinite_cache_persist_only_new_embedding_rows_miss(s15):
    c = coster(s15, 4, l2_bytes=INF, persist=True)
    for seqs in scen(s15):
        ops = opgen.gen_ops(s15, list(seqs), q_w=4, q_a=4, q=4)
        emb = sum(o.w_rd for o in ops if o.kind == "embed")
        r = c.cost_seqs(seqs)
        assert sum(r.dram_rd.values()) == pytest.approx(emb, rel=1e-12)


def test_zero_cache_sends_every_request_to_dram(s15, llama8):
    for m, q in ((s15, 4), (llama8, 2)):
        for machine in ("vortex-S1-4c", "h100-like"):
            c = coster(m, q, machine, l2_bytes=0.0)
            for seqs in scen(m):
                comp = compulsory_reads(m, seqs, q)
                for k in knobsets(c.h):
                    r = c.cost_seqs(seqs, k)
                    for cl in mh.CLASSES:
                        assert r.dram_rd[cl] == pytest.approx(r.l2_rd[cl], rel=1e-12, abs=1e-9)
                    assert sum(r.dram_rd.values()) >= comp * (1 - 1e-12)


def test_untimed_flush_and_infinite_cache_hide_all_writes(s15):
    c = coster(s15, 4, l2_bytes=INF, flush_untimed=True)
    assert c.cost_seqs(opgen.prefill(1, 64)).dram_wr == 0.0
    c2 = coster(s15, 4, l2_bytes=INF, flush_untimed=False)
    ops = opgen.gen_ops(s15, opgen.prefill(1, 64), q_w=4, q_a=4, q=4)
    assert c2.cost_seqs(opgen.prefill(1, 64)).dram_wr == pytest.approx(sum(o.a_wr + o.kv_wr for o in ops))


# --------------------------------------------------------------------------------------------- mechanism D / A2


def _weight_miss(c, seqs, k):
    return c.cost_seqs(seqs, k).dram_rd["weight"] - sum(
        o.w_rd for o in opgen.gen_ops(c.m, list(seqs), q_w=c.q_w, q_a=c.q_a, q=c.q) if o.kind == "embed")


def test_kv_insert_at_lru_keeps_weight_hits_as_kv_grows(s260):
    """stories260K weights (1.06 MB) fit a 4 MiB persistent L2. With normal insertion, growing decode KV pushes the
    step footprint over the L2 and the weights start missing; with KV inserted at LRU (or bypassed) the weight hit
    rate does not change with the KV footprint."""
    c = coster(s260, 4, l2_bytes=4 * 2**20, persist=True)
    growing = [opgen.decode(B, T) for B, T in ((1, 16), (4, 128), (16, 256), (64, 512), (128, 512))]
    for ins in ("lru", "bypass"):
        k = mh.Knobs(kv_insert=ins)
        misses = [_weight_miss(c, s, k) for s in growing]
        assert all(x == pytest.approx(misses[0], abs=1e-6) for x in misses), (ins, misses)
        assert misses[0] == pytest.approx(0.0, abs=1e-6)
    mru = [_weight_miss(c, s, mh.Knobs()) for s in growing]
    assert mru[0] == pytest.approx(0.0, abs=1e-6) and mru[-1] > 0.5 * c.m.d * c.m.d * 4


def test_kv_insert_lru_costs_one_way_and_bypass_never_hits(s260):
    # ideal step (cliff_ways = 0) so the capacities can be checked exactly; the default ramp is tested below
    h = dataclasses.replace(mh.vortex_hier(4), l2_bytes=8 * 2**20, persist_across_launch=True, cliff_ways=0)
    C, way = h.l2_bytes, h.l2_bytes / h.l2_ways
    win = {"weight": C - way / 2, "kv_stream": 10 * C}
    assert mh.hit_rate("weight", win, h, mh.Knobs()) == 0.0
    assert mh.hit_rate("weight", win, h, mh.Knobs(kv_insert="bypass")) == 1.0
    assert mh.hit_rate("weight", win, h, mh.Knobs(kv_insert="lru")) == 0.0          # needs C - one way
    win2 = {"weight": C - 1.5 * way, "kv_stream": 10 * C}
    assert mh.hit_rate("weight", win2, h, mh.Knobs(kv_insert="lru")) == 1.0
    assert mh.hit_rate("kv_stream", {"kv_stream": 1.0}, h, mh.Knobs(kv_insert="bypass")) == 0.0
    assert mh.hit_rate("kv_stream", {"kv_stream": way}, h, mh.Knobs(kv_insert="lru")) == 1.0
    assert mh.hit_rate("kv_stream", {"kv_stream": way + 1}, h, mh.Knobs(kv_insert="lru")) == 0.0
    # phase-aware scope: prefill-reused KV ("kv") is demoted only with scope "all"
    assert mh.hit_rate("kv", {"kv": way * 2}, h, mh.Knobs(kv_insert="bypass")) == 1.0
    assert mh.hit_rate("kv", {"kv": way * 2}, h, mh.Knobs(kv_insert="bypass", kv_insert_scope="all")) == 0.0


def test_a2_way_mask_partitions_capacity(s260):
    h = dataclasses.replace(mh.vortex_hier(4), l2_bytes=8 * 2**20, cliff_ways=0)
    C = h.l2_bytes
    k = mh.Knobs(a2_weight_ways=6)
    wcap, ocap = C * 6 / 8, C * 2 / 8
    big_kv = {"weight": wcap, "kv_stream": 100 * C, "act": 1e6}
    assert mh.hit_rate("weight", big_kv, h, k) == 1.0                     # KV cannot displace weights
    assert mh.hit_rate("weight", {"weight": wcap + 1}, h, k) == 0.0
    assert mh.hit_rate("act", {"act": ocap, "weight": 100 * C}, h, k) == 1.0
    assert mh.hit_rate("act", {"act": ocap + 1}, h, k) == 0.0
    assert wcap + ocap == C
    # end to end: with a weight partition that holds the weights, weight misses stay 0 as KV grows
    c = coster(s260, 4, l2_bytes=4 * 2**20, persist=True)
    for s in (opgen.decode(1, 16), opgen.decode(128, 512)):
        assert _weight_miss(c, s, mh.Knobs(a2_weight_ways=4)) == pytest.approx(0.0, abs=1e-6)


def test_d_helps_prefill_reuse_in_mixed_launch(s15):
    """Phase-aware D in a fused mixed launch: decode KV streamed next to a re-used prefill chunk evicts it under
    normal insertion; demoting only decode KV keeps the prefill KV re-reads in L2 (Vortex, cold per launch)."""
    c = coster(s15, 4, "vortex-S1-8c")
    seqs = opgen.mixed([(256, 0)], [(31, 256)])         # mechanism test; the size of the gain is cliff-sensitive (M2)
    base, lru = c.cost_seqs(seqs), c.cost_seqs(seqs, mh.Knobs(kv_insert="lru"))
    assert lru.dram_rd["kv"] < base.dram_rd["kv"] and lru.cycles < base.cycles
    assert lru.dram_rd["kv_stream"] == base.dram_rd["kv_stream"]        # decode KV is read once either way


# --------------------------------------------------------------------------------------------- monotonicity


def _random_cases(rng, models, n):
    out = []
    for _ in range(n):
        m, q = rng.choice(models)
        cap = m.max_seq or 4096
        kind = rng.choice(["prefill", "decode", "mixed"])
        if kind == "prefill":
            seqs = opgen.prefill(rng.randint(1, 3), rng.randint(2, min(cap, 300)))
        elif kind == "decode":
            seqs = opgen.decode(rng.randint(1, 32), rng.randint(1, min(cap, 1000)))
        else:
            seqs = opgen.mixed([(rng.randint(2, 128), rng.randint(0, 64))], [(rng.randint(1, 16), rng.randint(1, min(cap, 900)))])
        out.append((m, q, seqs))
    return out


@pytest.mark.parametrize("machine", ["vortex-S1-4c", "h100-like"])
@pytest.mark.parametrize("hit_model", ["cliff", "proportional"])
def test_monotone_in_cache_size(s15, llama8, machine, hit_model):
    rng = random.Random(4)
    sizes = [0.0, 2**16, 2**18, 2**20, 2**22, 2**24, 2**26, INF]
    for m, q, seqs in _random_cases(rng, [(s15, 4), (llama8, 2)], 6):
        h0 = dataclasses.replace(mh.hier_for(machine), hit_model=hit_model)
        for persist in (False, True):
            for k in rng.sample(knobsets(h0), 3):
                prev = None
                for C in sizes:
                    c = mh.MemHier(m, machine, hier=dataclasses.replace(h0, l2_bytes=C), q_w=q, q_a=q, q=q,
                                   persist_across_launch=persist)
                    r = c.cost_seqs(seqs, k)
                    cur = (r.cycles, r.dram_bytes)
                    if prev:
                        assert cur[0] <= prev[0] * (1 + 1e-9) and cur[1] <= prev[1] * (1 + 1e-12), (C, k, prev, cur)
                    prev = cur


@pytest.mark.parametrize("machine", ["vortex-S1-8c", "a100-like"])
def test_monotone_in_bandwidth(s15, llama8, machine):
    rng = random.Random(5)
    base = roofline.machines()[machine]
    h = mh.hier_for(machine)
    for m, q, seqs in _random_cases(rng, [(s15, 4), (llama8, 2)], 6):
        for k in rng.sample(knobsets(h), 3):
            prev = None
            for f in (0.25, 0.5, 1, 2, 4, 16):
                mc = dataclasses.replace(base, dram_bytes_per_cycle=base.dram_bytes_per_cycle * f)
                t = mh.MemHier(m, mc, hier=h, q_w=q, q_a=q, q=q).cost_seqs(seqs, k).cycles
                if prev is not None:
                    assert t <= prev * (1 + 1e-9), (f, k, prev, t)
                prev = t


# --------------------------------------------------------------------------------------------- queueing


@pytest.mark.parametrize("rho_hi,rho_lo,seed", [(0.2, 0.4, 0), (0.3, 0.3, 1), (0.1, 0.5, 2)])
def test_md1_priority_formula_matches_simulation(rho_hi, rho_lo, seed):
    s = 3.0
    sim_hi, sim_lo = mh.simulate_priority_queue(rho_hi / s, rho_lo / s, s, n=100_000, seed=seed)
    w_hi, w_lo = mh.md1_wait_priority(rho_hi, rho_lo, s)
    assert sim_hi == pytest.approx(w_hi, rel=0.05)
    assert sim_lo == pytest.approx(w_lo, rel=0.05)
    sim_f = mh.simulate_priority_queue(rho_hi / s, rho_lo / s, s, n=100_000, seed=seed, priority=False)
    w_f = mh.md1_wait_fifo(rho_hi + rho_lo, s)
    assert sim_f[0] == pytest.approx(w_f, rel=0.05) and sim_f[1] == pytest.approx(w_f, rel=0.05)


def test_md1_conservation_and_limits():
    s = 2.0
    for rh, rl in ((0.1, 0.2), (0.3, 0.5), (0.05, 0.9)):
        w_hi, w_lo = mh.md1_wait_priority(rh, rl, s)
        # work-conserving non-preemptive disciplines keep Σ ρ_i W_i (Kleinrock's conservation law)
        assert rh * w_hi + rl * w_lo == pytest.approx((rh + rl) * mh.md1_wait_fifo(rh + rl, s), rel=1e-12)
        assert w_hi < mh.md1_wait_fifo(rh + rl, s) < w_lo
    assert mh.md1_wait_priority(0.0, 0.5, s)[1] == pytest.approx(mh.md1_wait_fifo(0.5, s))
    assert mh.md1_wait_fifo(1.0, s) == INF and mh.md1_wait_priority(0.5, 0.5, s)[1] == INF


def test_b_lowers_critical_wait_in_mixed_launch(s15):
    c = coster(s15, 4, "vortex-S1-8c")
    seqs = opgen.mixed([(256, 0)], [(15, 256)])
    a = c.cost_seqs(seqs, mh.Knobs(), detail=True)
    b = c.cost_seqs(seqs, mh.Knobs(b_two_class=True), detail=True)
    att_a = [L for L in a.launches if "attn_qk" in L.names][0]
    att_b = [L for L in b.launches if "attn_qk" in L.names][0]
    assert att_b.w_crit <= att_a.w_crit and att_b.w_bulk >= att_b.w_crit
    assert b.dram_rd == a.dram_rd                                        # B changes timing, never traffic


def test_littles_law_limit_with_unbounded_bandwidth(s15):
    """With effectively infinite DRAM and L2 bandwidth, queueing vanishes and a memory-bound launch takes
    (hit lines · hit latency + miss lines · miss latency) / outstanding slots (Little's law)."""
    base = roofline.machines()["vortex-S1-4c"]
    mc = dataclasses.replace(base, dram_bytes_per_cycle=1e15)
    h = dataclasses.replace(mh.vortex_hier(4), l2_bytes_per_cycle=1e15)
    c = mh.MemHier(s15, mc, hier=h, q_w=4, q_a=4, q=4)
    r = c.cost_seqs(opgen.decode(1, 100), detail=True)
    L = [x for x in r.launches if x.names == "lm_head"][0]
    miss = sum(L.dram_rd.values()) / h.line
    hit = sum(L.l2_rd.values()) / h.line - miss
    slots = min(h.cores * min(h.l1_mshr, h.warps * h.mlp_per_warp), h.l2_banks * h.l2_mshr)
    assert L.t_lsu == pytest.approx((hit * h.hit_lat + miss * h.miss_lat) / slots, rel=1e-6)
    assert L.t_mem == pytest.approx(L.t_lsu, rel=1e-6) and L.bound in ("latency", "compute")


# --------------------------------------------------------------------------------------------- parameters, knobs


def test_vortex_params_match_vx_config(ps_root):
    t = (ps_root / "third_party" / "vortex" / "VX_config.toml").read_text()
    lit = lambda k: int(re.search(rf"^{k} = (\d+)", t, re.M).group(1))
    h = mh.vortex_hier(8)
    assert h.line == lit("VX_CFG_MEM_BLOCK_SIZE") == 64
    assert h.l1_bytes == lit("VX_CFG_DCACHE_SIZE") and h.l1_mshr == lit("VX_CFG_DCACHE_MSHR_SIZE")
    assert h.l2_bytes == lit("VX_CFG_L2_SIZE") and h.l2_ways == lit("VX_CFG_L2_NUM_WAYS")
    assert h.l2_mshr == lit("VX_CFG_L2_MSHR_SIZE") and h.dram_channels == lit("VX_CFG_PLATFORM_MEMORY_NUM_BANKS")
    assert lit("lmem_bytes_per_bank") == 2048 and lit("lmem_min_bytes") == 16384 and h.lmem_bytes == 65536
    assert lit("VX_CFG_PLATFORM_MEMORY_LATENCY") == 0 and lit("VX_CFG_PLATFORM_CLOCK_RATE") == 400
    assert "DCACHE_LATENCY = \"expr: 2 + max(0, clog2($VX_CFG_DCACHE_SIZE) - clog2(16384))\"" in t
    assert "L2_LATENCY = \"expr: 4 + max(0, clog2($VX_CFG_L2_SIZE) - clog2(1048576))\"" in t
    assert (h.l1_lat, h.l2_lat, h.dram_lat, h.miss_lat) == (2, 4, 20.0, 26.0)
    assert [mh.vortex_hier(c).l2_banks for c in (1, 4, 8, 16)] == [2, 8, 16, 16]
    assert h.persist_across_launch is False and h.flush_untimed is True
    for name in roofline.machines():
        hh = mh.hier_for(name)
        assert hh.persist_across_launch == (not name.startswith(("vortex", "psbase"))), name
        assert hh.sources, name
    with pytest.raises(KeyError):
        mh.hier_for("tpu-like-v9")


def test_gpu_params_bandwidth_delay_rule():
    for p in ("a100", "h100"):
        h, m = mh.gpu_hier(p), roofline.machines()[f"{p}-like"]
        slots = min(h.cores * h.l1_mshr, h.l2_banks * h.l2_mshr)
        assert slots * h.line / h.miss_lat >= 2 * m.dram_bytes_per_cycle * (1 - 1e-9)
    assert mh.gpu_hier("a100").dram_lat == pytest.approx(190 + 50 * 1.41)


def test_knob_validation():
    h = mh.vortex_hier(4)
    for bad in (mh.Knobs(dxa_depth=2), mh.Knobs(lmem_use=h.lmem_bytes + 1), mh.Knobs(lmem_use=-1),
                mh.Knobs(kv_insert="lip"), mh.Knobs(kv_insert_scope="prefill"), mh.Knobs(a2_weight_ways=0),
                mh.Knobs(a2_weight_ways=h.l2_ways), mh.Knobs(lmem_use=1024, dxa_depth=1.5)):
        with pytest.raises(ValueError):
            bad.validate(h)
    grid = mh.knob_grid(h)
    assert len(grid) == len(set(grid)) and all(k.validate(h) is k for k in grid)
    assert not any(k.dxa_depth and not k.lmem_use for k in grid)
    with pytest.raises(ValueError):
        dataclasses.replace(h, hit_model="magic")


def test_capacity_hit_function():
    assert mh.capacity_hit(10, 10) == 1.0 and mh.capacity_hit(11, 10) == 0.0
    # default cliff = ramp one way wide CENTRED on the capacity (unbiased): continuous,
    # so a one-line change cannot jump; an exactly fitting window gets ½
    assert mh.capacity_hit(8, 10, "cliff", 4) == 1.0 and mh.capacity_hit(10, 10, "cliff", 4) == 0.5
    assert mh.capacity_hit(11, 10, "cliff", 4) == 0.25 and mh.capacity_hit(12, 10, "cliff", 4) == 0.0
    h = mh.vortex_hier(4)
    way = h.l2_bytes / h.l2_ways
    assert h.cliff_ways == 1.0
    assert mh.hit_rate("weight", {"weight": h.l2_bytes - way / 2}, h, mh.Knobs()) == 1.0
    assert mh.hit_rate("weight", {"weight": h.l2_bytes - way / 4}, h, mh.Knobs()) == pytest.approx(0.75)
    assert mh.hit_rate("weight", {"weight": h.l2_bytes}, h, mh.Knobs()) == 0.5
    assert mh.hit_rate("weight", {"weight": h.l2_bytes + way / 2}, h, mh.Knobs()) == 0.0
    # mean effective capacity of the ramp = C: ∫ h dF over the ramp equals the step's
    import numpy as np
    F = np.linspace(h.l2_bytes - way, h.l2_bytes + way, 40001)
    hs = np.array([mh.capacity_hit(f, h.l2_bytes, "cliff", way) for f in F])
    step = (F <= h.l2_bytes).astype(float)
    assert abs(np.trapezoid(hs - step, F)) < way * 1e-3
    assert mh.capacity_hit(20, 10, "proportional") == 0.5 and mh.capacity_hit(5, 0) == 0.0
    assert mh.capacity_hit(0, 0) == 1.0 and mh.capacity_hit(1e30, INF) == 1.0


# --------------------------------------------------------------------------------------------- tiling


def test_gemm_tile_and_reread(s15):
    c = coster(s15, 4)
    # stock: two co-resident CTAs share ½ of the 16 KiB L1 -> 4 KiB each; 4096 / (32 · 8 B) = 16
    assert c._n_cta(mh.Knobs()) == 2 and c._tile_budget(mh.Knobs()) == 4096
    assert c._gemm_tile(mh.Knobs()) == 16
    # accumulator cap: 16 accumulators × 4 warps × 32 threads = 2048 -> T ≤ floor_pow2(√2048) = 32
    assert c._acc_cap_gemm() == 32
    assert c._gemm_tile(mh.Knobs(lmem_use=65536)) == 32
    big = coster(s15, 4, acc_per_thread=10**6)
    assert big._gemm_tile(mh.Knobs(lmem_use=65536)) == 256        # 64 KiB / 256 B, buffer-limited
    assert big._gemm_tile(mh.Knobs(lmem_use=65536, dxa_depth=1)) == 128
    # zero cache: q_proj weight traffic = w_rd · ceil(M / T), activation traffic = a_rd · ceil(N / T)
    cz = coster(s15, 4, l2_bytes=0.0)
    r = cz.cost_seqs(opgen.prefill(1, 100), mh.Knobs(), detail=True)
    q = [L for L in r.launches if L.names == "q_proj"][0]
    w = 4 * s15.d * s15.d
    a = 4 * 100 * s15.d
    assert q.dram_rd["weight"] == pytest.approx(w * math.ceil(100 / 16))
    assert q.dram_rd["act"] == pytest.approx(a * math.ceil(s15.d / 16))


@pytest.mark.parametrize("machine", ["vortex-S1-4c", "vortex-S1-8c", "vortex-S1-16c", "a100-like", "h100-like"])
def test_tiles_fit_l1_share_lmem_and_registers(s15, llama8, machine):
    """Required 1 and 2: the co-resident CTAs' implicit tiles fit the L1 share, LMEM tiles fit
    the LMEM, and GEMM / attention accumulators fit the registers, for every knob vector of the grid."""
    h = mh.hier_for(machine)
    for m, q in ((s15, 4), (llama8, 2)):
        c = mh.MemHier(m, machine, q_w=q, q_a=q, q=q)
        threads = h.warps_per_cta * h.threads_per_warp
        for k in mh.knob_grid(h, inserts=("mru",), two_class=(False,)):
            n, beta = c._n_cta(k), c._tile_budget(k)
            if k.lmem_use == 0:
                assert n * beta <= h.l1_tile_frac * h.l1_bytes * (1 + 1e-12), (k, n, beta)
            else:
                assert n * k.lmem_use <= h.lmem_bytes and beta * (k.dxa_depth + 1) == k.lmem_use
            T = c._gemm_tile(k)
            assert T * T <= h.acc_per_thread * threads, (k, T)
            assert T * T / threads <= 128                         # adversarial bound
            info = dict(dk=m.D_h, dv=m.D_h, kv=2 * m.H_kv * m.D_h * q, kt=512)
            _, tq = c._attn_reread(512, 0, 0, True, info, k)
            assert tq * m.g * m.D_h <= h.acc_per_thread * threads, (k, tq)


def test_prefill_attention_reread_factor(s15):
    c = coster(s15, 4)
    info = dict(dk=48, dv=48, kv=2 * 6 * 48 * 4)      # stories15M: H_kv = 6, D_h = 48, fp32
    # tile row = g·dk·q_a + KV per key per KV head = 48·4 + 2·48·4 = 576 B; 4 KiB (stock, per CTA) / 576 -> Tq = 4
    r, tq = c._attn_reread(64, 0, 0, True, info, mh.Knobs())
    assert tq == 4
    assert r == pytest.approx(sum(4 * (t + 1) for t in range(16)) / 64)         # causal: Σ keys per tile / 64 = 8.5
    r_full, _ = c._attn_reread(64, 0, 0, False, info, mh.Knobs())
    assert r_full == 16
    assert c._attn_reread(1, 99, 0, True, info, mh.Knobs()) == (1.0, 1)          # decode reads its KV once


# --------------------------------------------------------------------------------------------- structure, glue


@pytest.mark.parametrize("persist", [False, True])
def test_representative_layers_exact(s15, dsv2, persist):
    for m, q, seqs in ((s15, 4, opgen.mixed([(40, 10)], [(3, 90)])), (dsv2, 2, opgen.mixed([(64, 0)], [(4, 300)])),
                       (dsv2, 2, opgen.decode(8, 512))):
        a = mh.MemHier(m, "vortex-S1-8c", q_w=q, q_a=q, q=q, persist_across_launch=persist)
        b = mh.MemHier(m, "vortex-S1-8c", q_w=q, q_a=q, q=q, persist_across_launch=persist, use_layer_classes=False)
        for k in (mh.Knobs(), mh.Knobs(lmem_use=32768, dxa_depth=1, kv_insert="lru")):
            ra, rb = a.cost_seqs(seqs, k), b.cost_seqs(seqs, k)
            assert ra.cycles == pytest.approx(rb.cycles, rel=1e-9)
            assert ra.dram_bytes == pytest.approx(rb.dram_bytes, rel=1e-9)
            assert ra.flops == pytest.approx(opgen.totals(opgen.gen_ops(m, list(seqs), q_w=q, q_a=q, q=q))["flops"],
                                             rel=1e-12)


def test_split_launches(s15):
    ops = opgen.gen_ops(s15, opgen.mixed([(16, 0)], [(2, 50)]), q_w=4, q_a=4, q=4, layers=[0])
    L = mh.split_launches(ops, fused_attention=True, fused_mixed=True)
    core = lambda x: x[0].name in ("attn_qk", "attn_softmax", "attn_av")
    att = [x for x in L if core(x)]
    assert [x[0].name for x in L].count("attn_norm") == 1            # attn_norm / attn_residual are not attention
    assert len(att) == 1 and len(att[0]) == 6                         # qk/softmax/av × 2 groups, one launch
    L2 = mh.split_launches(ops, fused_attention=True, fused_mixed=False)
    assert len([x for x in L2 if core(x)]) == 2
    assert sum(len(x) for x in L) == len(ops) == sum(len(x) for x in L2)
    assert len(mh.split_launches(ops, fused_attention=False)) == len(ops)
    with pytest.raises(ValueError):
        mh.parse_group("prefill n=3")


def test_iteration_seqs_collapse_is_exact(s15, ps_root):
    it = sch.Iteration(idx=0, t_start=0.0, t_end=0.0,
                       prefill=(sch.PrefillChunk(0, 32, 16, False, False), sch.PrefillChunk(1, 0, 8, True, False)),
                       decode=tuple(sch.DecodeItem(r, T) for r, T in ((2, 5), (3, 9), (4, 9), (5, 100))),
                       kv_blocks=0, kv_tokens=0, n_running=6, n_waiting=0)
    col = mh.iteration_seqs(it, s15)
    ex = mh.iteration_seqs(it, s15, collapse_decodes=False)
    assert col[:2] == (opgen.Seq("prefill", 16, 32, 1, False), opgen.Seq("prefill", 8, 0, 1, True))
    assert sorted((s.ctx + 1, s.count) for s in ex[2:]) == [(5, 1), (9, 2), (100, 1)]
    assert sum(s.count for s in col[2:]) == 4 and sum((s.ctx + 1) * s.count for s in col[2:]) == 123
    ta = opgen.totals(opgen.gen_ops(s15, list(col), q_w=4, q_a=4, q=4))
    tb = opgen.totals(opgen.gen_ops(s15, list(ex), q_w=4, q_a=4, q=4))
    for key in ("flops", "kv_rd", "kv_wr", "w_rd", "a_rd", "a_wr"):
        assert ta[key] == pytest.approx(tb[key], rel=1e-12), key
    c = mh.MemHier(s15, "vortex-S1-8c", q_w=4, q_a=4, q=4)
    assert c.cost_seqs(col).cycles == pytest.approx(c.cost_seqs(ex).cycles, rel=1e-9)
    # windowed model: decodes are grouped exactly, never collapsed
    mistral = opgen.from_hf_config(ps_root / "configs" / "mistralai__Mistral-7B-v0.1.json")
    assert mh.iteration_seqs(it, mistral)[2:] == ex[2:]


def test_cost_function_drives_scheduler(s15):
    c = mh.MemHier(s15, "vortex-S1-8c", q_w=4, q_a=4, q=4)
    reqs = [Request(0, 0.0, 40, 5), Request(1, 0.001, 20, 3), Request(2, 0.002, 64, 4)]
    cfg = sch.SchedulerConfig(max_batch=4, chunk_size=32, block_size=16)
    res = sch.run_schedule(reqs, cfg, c, mh.Knobs())
    its = list(res.iterations)
    assert res.n_iterations == len(its) > 0
    for it in its:
        assert it.duration == pytest.approx(c.cost_seqs(mh.iteration_seqs(it, s15)).seconds, rel=1e-12)
    alt = sch.cost_schedule(its, c, mh.Knobs(kv_insert="lru", lmem_use=32768, dxa_depth=1))
    assert len(alt) == len(its) and (alt > 0).all()


def test_deterministic_and_memoised(s15):
    c = mh.MemHier(s15, "vortex-S1-8c", q_w=4, q_a=4, q=4)
    s = opgen.mixed([(64, 0)], [(3, 90)])
    a, b = c.cost_seqs(s, mh.Knobs(b_two_class=True)), c.cost_seqs(s, mh.Knobs(b_two_class=True))
    assert a is b
    d = mh.MemHier(s15, "vortex-S1-8c", q_w=4, q_a=4, q=4).cost_seqs(s, mh.Knobs(b_two_class=True), detail=True)
    assert d.cycles == a.cycles and d.dram_rd == a.dram_rd


def test_opgen_options_pass_through(dsv2):
    """MLA prefill form and MoE routing reach opgen (minor 1: may evaluate both MLA forms)."""
    seqs = opgen.prefill(1, 128)
    for form in ("absorbed", "decompressed"):
        for routing in ("uniform", "zipf"):
            c = mh.MemHier(dsv2, "h100-like", mla_prefill_form=form, routing=routing)
            ref = opgen.totals(opgen.gen_ops(dsv2, seqs, mla_prefill_form=form, routing=routing))
            r = c.cost_seqs(seqs)
            assert r.flops == pytest.approx(ref["flops"], rel=1e-12)
            assert r.compulsory_rd == pytest.approx(ref["w_rd"] + ref["a_rd"] + ref["kv_rd"], rel=1e-12)
    assert mh.MemHier(dsv2, "h100-like", mla_prefill_form="decompressed").cycles(seqs) != \
        mh.MemHier(dsv2, "h100-like").cycles(seqs)


def test_methodology_section9_filled(ps_root):
    """§9 explains every approximation with its failure mode (M1–M14), persist semantics and the
    calibration targets; every Hier tunable is named there."""
    meth = (ps_root / "docs" / "methodology.md").read_text()
    sec = meth.split("\n## 9. ")[1].split("\n## 10. ")[0]
    assert sec.splitlines()[0].startswith("Analytical memory-hierarchy model")
    for n in range(1, 15):
        assert f"**M{n} " in sec, f"M{n}"
    assert sec.count("*Fails*") >= 13                      # M13 is exact by construction (tested)
    for needle in ("### 9.2 `persist_across_launch`", "### 9.5 Calibration targets", "W0/(1 − ρ_crit)",
                   "Little's law", "r_w = ⌈M_e/T⌉", "C/ways", "B_max ≤ c"):
        assert needle in sec, needle
    for f in dataclasses.fields(mh.Hier):
        if f.name in ("name", "sources", "persist_across_launch"):
            continue
        assert f.name in sec or f.name in ("cores", "warps", "line", "l1_bytes", "lmem_bytes", "l1_lmem_shared",
                                           "l2_bytes", "l2_ways", "dram_channels", "l2_bytes_per_cycle"), f.name


# --------------------------------------------------------------------------------------------- (review checks)


def test_no_knob_reads_below_compulsory_cold(s15, llama8):
    """Adversarial case: with persistence off no knob gets credit for traffic the program does not have."""
    for m, q in ((s15, 4), (llama8, 2)):
        c = mh.MemHier(m, "vortex-S1-8c", q_w=q, q_a=q, q=q)
        for seqs in (opgen.prefill(1, 200), opgen.decode(8, 200), opgen.mixed([(128, 0)], [(7, 200)])):
            for k in mh.knob_grid(c.h, a2=(None, 4)):
                r = c.cost_seqs(seqs, k)
                assert sum(r.dram_rd.values()) >= r.compulsory_rd * (1 - 1e-12), k


@pytest.mark.parametrize("machine", ["vortex-S1-8c", "h100-like", "a100-like"])
@pytest.mark.parametrize("variant", [{}, {"warps_per_cta": 8}])
def test_one_line_l2_change_robustness(s15, llama8, machine, variant):
    """Required 3: power-of-two tiles, dimensions and caches made the stock Llama-8B prefill
    window exactly 1,048,576 B = the L2. With concurrent tiles and outputs in the windows, removing or adding one
    64 B line changes no example-scenario cost by more than 1 % (cliff model; knob speedups likewise). Attempt 3:
    also on the anchored GPU-like machines (h100 q_proj/o_proj weights are exactly 32 MiB = the L2) and with
    8-warp CTAs (P ≤ cb puts the whole-matrix weight window on the ramp)."""
    base = dataclasses.replace(mh.hier_for(machine), **variant)
    for m, q in ((s15, 4), (llama8, 2)):
        cap = m.max_seq or 4096
        scen4 = [opgen.prefill(1, min(512, cap)), opgen.decode(16, min(4096, cap)),
                 opgen.mixed([(min(256, cap), 0)], [(15, min(4096, cap))])]
        cs = {d: mh.MemHier(m, machine, hier=dataclasses.replace(base, l2_bytes=base.l2_bytes + d), q_w=q, q_a=q, q=q)
              for d in (-64, 0, 64)}
        for seqs in scen4:
            for k in mh._example_knobs(base).values():
                cyc = {d: c.cycles(seqs, k) for d, c in cs.items()}
                assert cyc[-64] / cyc[0] < 1.01 and cyc[64] / cyc[0] > 0.99, (m.name, seqs, k.label, cyc)


def test_write_fraction_untimed_flush(s15):
    """Minor 3: a dirty line written uniformly during a launch is evicted (timed) iff more than C bytes follow it,
    so the timed fraction is (1 − C/F)⁺ with F = the launch's reads (incl. re-reads) + writes."""
    c = coster(s15, 4, "vortex-S1-8c")
    r = c.cost_seqs(opgen.decode(1, 100), detail=True)
    L = [x for x in r.launches if x.names == "lm_head"][0]
    wr = 4 * s15.vocab
    F = sum(L.l2_rd.values()) + wr
    assert L.dram_wr == pytest.approx(wr * max(0.0, 1 - c.h.l2_bytes / F), rel=1e-12)


def test_lsu_and_dxa_share_l2_mshrs(llama8):
    """Minor 4: both issue paths draw on the same L2 MSHRs."""
    h = dataclasses.replace(mh.vortex_hier(8), l2_mshr=4)
    c = mh.MemHier(llama8, "vortex-S1-8c", hier=h)
    r = c.cost_seqs(opgen.mixed([(64, 0)], [(4, 300)]), mh.Knobs(lmem_use=32768, dxa_depth=4), detail=True)
    both = [L for L in r.launches if len(L.n_slots) == 2]
    assert both
    for L in r.launches:
        assert sum(L.n_slots.values()) <= h.l2_banks * h.l2_mshr * (1 + 1e-12)


def test_bound_label_bandwidth_when_at_bandwidth_bound(s15):
    """Minor 5: Vortex decode launches sit at the DRAM bandwidth bound and are labelled so."""
    r = coster(s15, 4).cost_seqs(opgen.decode(1, 256), detail=True)
    for L in r.launches:
        if L.names in ("q_proj", "gate_proj", "lm_head"):
            assert L.bound == "bandwidth" and L.t_mem <= 1.05 * L.t_bw, (L.names, L.bound)


def test_b_structural_limit_conservation():
    """Required 4: with one issue path, no posted writes and uncapped waits, priority only moves
    waiting between classes (ρ_c·W_c + ρ_b·W_b is conserved), so a launch's time is identical with B on or off."""
    m = opgen.ModelShape("toy", d=64, L=1, H_q=1, H_kv=1, D_h=64, d_ff=64, vocab=64, tie_embeddings=True)
    h = dataclasses.replace(mh.vortex_hier(4), l1_mshr=1, mlp_per_warp=0.25)       # few slots: latency-bound
    c = mh.MemHier(m, "vortex-S1-4c", hier=h)
    op = opgen.linear(0, "x", 1, 64, 64, 4, 4)
    streams = [mh.Stream("weight", "lsu", "crit", 1e6), mh.Stream("kv", "lsu", "bulk", 2e6)]
    t = {}
    for b in (False, True):
        r = c._launch_time([op], mh.Knobs(b_two_class=b), streams, dict.fromkeys(mh.CLASSES, 0.0), {"fpu": 0.0}, 0.0)
        t[b] = r
    assert t[True].bound == "latency"
    assert t[True].w_crit < t[False].w_crit < t[True].w_bulk
    assert t[True].cycles == pytest.approx(t[False].cycles, rel=1e-9)


def test_b_gain_does_not_grow_with_latency(s15, ps_root):
    """Adversarial case (inverted): an earlier analysis claimed B's gain grows with DRAM latency. In this model it
    does not (its only gains are reads ahead of posted writes and the outstanding caps), and the README says so."""
    base = mh.hier_for("vortex-S1-8c")
    seqs = opgen.mixed([(256, 0)], [(15, 256)])
    g = []
    for lat in (20, 300):
        c = mh.MemHier(s15, "vortex-S1-8c", hier=dataclasses.replace(base, dram_lat=lat), q_w=4, q_a=4, q=4)
        g.append(c.cycles(seqs) / c.cycles(seqs, mh.Knobs(b_two_class=True)))
    assert g[1] <= g[0] + 1e-9, g
    readme = (ps_root / "results" / "P1.4" / "README.md").read_text()
    assert "grows with DRAM latency" not in readme
    assert "not evidence about B" in readme and "not monotone in latency" in readme


# --------------------------------------------------------------------------------------------- (review checks)


@pytest.mark.parametrize("machine", ["h100-like", "a100-like"])
def test_gpu_prefill_roofline_anchor(llama8, machine):
    """Required: the GPU-like effective L2 bandwidth is anchored to the roofline (assumption,
    not measured) so that the best knob_grid vector for Llama-3.1-8B prefill at S = 512 is within 1.5× of
    max(FLOP bound, DRAM bound). The unanchored value is kept for the §9.6 sensitivity row and misses the anchor."""
    seqs = opgen.prefill(1, 512)
    t = opgen.totals(opgen.gen_ops(llama8, seqs))
    mc = roofline.machines()[machine]
    bound = max(t["flops"] / mc.flops_per_cycle["tcu"],
                (t["w_rd"] + t["a_rd"] + t["kv_rd"] + t["a_wr"] + t["kv_wr"]) / mc.dram_bytes_per_cycle)
    h = mh.hier_for(machine)
    assert h.l2_bytes_per_cycle == h.l2_banks * h.line * mh.GPU_L2_BW_ANCHOR
    assert "anchored to the roofline: an assumption, not a measurement" in h.sources["l2_bytes_per_cycle"]
    c = mh.MemHier(llama8, machine, hier=h)
    best = min(c.cycles(seqs, k) for k in mh.knob_grid(h))
    assert best / bound <= 1.5, best / bound
    assert c.cycles(seqs) >= best                       # stock = no shared-memory tiling, may stay slow (§9.1)
    if machine == "h100-like":                          # unanchored: L2-bound far above the roofline
        hu = mh.gpu_hier("h100", l2_bw_anchor=1)
        cu = mh.MemHier(llama8, machine, hier=hu)
        assert min(cu.cycles(seqs, k) for k in mh.knob_grid(hu)) / bound > 3
    assert mh.vortex_hier(8).l2_bytes_per_cycle == mh.vortex_hier(8).l2_banks * 64     # Vortex unchanged


def test_lockstep_wave_boundary_share(s15):
    """Minor case 2: 1/P of activation re-reads (and cb/P of weight re-reads) have their partner tile
    in the next wave and use the serial window. With an L2 that holds the lockstep K-chunks but not the serial
    activation window, exactly that share of q_proj's activation re-reads misses."""
    c = coster(s15, 4, l2_bytes=2**17, cliff_ways=0)     # K-chunk window 64 KiB fits, serial window 322 KiB does not
    k = mh.Knobs()
    T, P = c._gemm_tile(k), c.h.cores * c._n_cta(k)
    r = c.cost_seqs(opgen.prefill(1, 100), k, detail=True)
    q = [L for L in r.launches if L.names == "q_proj"][0]
    a = 4 * 100 * s15.d
    reread = (math.ceil(s15.d / T) - 1) * a
    assert q.dram_rd["act"] == pytest.approx(a + reread / P, rel=1e-9)
