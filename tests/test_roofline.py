"""Tests: E6 crossover for 7B-class MHA ≈ 27 K tokens (R§1), E7 min() properties, ridge
arithmetic, E2/E5 consistency with opgen, E6 step bound, §4 ρ_KV, and that every machine number has a source."""
import math
import random
from pathlib import Path

import pytest

from model import opgen, roofline
from model.roofline import Machine, attainable, machines

ROOT = Path(__file__).resolve().parents[1]

# common 7B MHA shape of methodology E6 (Llama-2-7B-like): L=32, H=H_KV=32, D_h=128, d=4096, d_ff=11008, vocab 32000
MHA7B = opgen.ModelShape(name="mha7b", d=4096, L=32, H_q=32, H_kv=32, D_h=128, d_ff=11008, vocab=32000,
                         tie_embeddings=False)


def test_e6_crossover_7b_mha():
    assert opgen.kv_bytes_per_token(MHA7B, 2) == 524_288
    bt = roofline.kv_weight_crossover_bt(MHA7B, 2, 2, n_params=7e9)
    assert bt == pytest.approx(1.4e10 / 524_288) and round(bt) == 26_703          # methodology E6 arithmetic
    bt_real = roofline.kv_weight_crossover_bt(MHA7B, 2, 2)                          # N_params from the shape: 6.74e9
    assert opgen.n_params(MHA7B) == 6_738_415_616
    assert abs(bt_real - 27_000) / 27_000 < 0.06                                    # "≈ 27 K tokens" (R§1)
    # at the crossover the region test of methodology §4 sits exactly at ρ_KV = 1
    B = 8
    T = bt_real / B
    assert roofline.rho_kv(MHA7B, [T] * B) == pytest.approx(1.0, rel=1e-3)   # E3 rounds T up to whole tokens
    assert roofline.rho_kv(MHA7B, [int(T) + 1] * B) > 1 > roofline.rho_kv(MHA7B, [int(T) - 1] * B)


def test_e7_min_properties():
    rng = random.Random(7)
    for _ in range(2000):
        peak, bw = 10 ** rng.uniform(9, 16), 10 ** rng.uniform(9, 13)
        ai = 10 ** rng.uniform(-3, 4)
        p = attainable(ai, peak, bw)
        assert p <= peak and p <= ai * bw * (1 + 1e-12) and p in (peak, ai * bw)
        assert attainable(ai * 1.5, peak, bw) >= p                                   # monotone in AI
        ridge = peak / bw
        assert attainable(ridge, peak, bw) == pytest.approx(peak)
        assert (p == peak) == (ai >= ridge) or math.isclose(ai, ridge)


def test_ridge_arithmetic():
    ms = machines()
    for mc in ms.values():
        for u in ("fpu", "tcu"):
            assert mc.ridge(u) == pytest.approx(mc.peak(u) / mc.bw)
            assert mc.ridge(u) == pytest.approx(mc.flops_per_cycle[u] / mc.dram_bytes_per_cycle)
    v8 = ms["vortex-S1-8c"]
    assert v8.flops_per_cycle == {"fpu": 8 * 32 * 2, "tcu": 8 * 512} and v8.dram_bytes_per_cycle == 64
    assert v8.ridge("fpu") == 8.0 and v8.ridge("tcu") == 64.0 and not v8.measured
    assert [(ms[f"vortex-S1-{c}c"].ridge("tcu"), ms[f"vortex-S1-{c}c"].ridge("fpu")) for c in (4, 8, 16)] == \
        [(32, 4), (64, 8), (128, 16)]
    assert ms["vortex-S1-16c"].ridge("tcu") == 2 * ms["vortex-S1-8c"].ridge("tcu")
    a, h = ms["a100-like"], ms["h100-like"]
    assert a.bw == pytest.approx(1555e9) and a.clock_hz == 1410e6 and h.bw == pytest.approx(3350e9)
    assert a.peak("tcu") == pytest.approx(108 * 2048 * 1410e6) and a.peak("tcu") == pytest.approx(311.87e12, rel=1e-4)
    assert a.peak("fpu") == pytest.approx(19.49e12, rel=1e-3) and h.peak("fpu") == pytest.approx(66.9e12, rel=1e-3)
    assert a.ridge("tcu") == pytest.approx(200.56, rel=1e-4) and h.ridge("tcu") == pytest.approx(319.56, rel=1e-4)
    # a machine without a TCU falls back to the FPU
    m = Machine("x", 1e9, {"fpu": 10}, 5)
    assert m.ridge("tcu") == m.ridge("fpu") == 2


def test_vortex_tcu_geometry_small_threads():
    """FEDP grid derivation (VX_tcu_pkg.sv:66-74): NT=4 -> TC_M=2, TC_N=2, TC_K=2 words = 4 FP16 -> 16 MAC/cycle."""
    assert roofline.vortex_s1(1, threads=4, warps=4).flops_per_cycle == {"fpu": 8, "tcu": 32}
    assert roofline.vortex_s1(1, threads=32, warps=8).flops_per_cycle["tcu"] == 2 * 8 * 4 * 8


def test_e2_e5_consistent_with_opgen():
    rng = random.Random(8)
    for _ in range(200):
        n, din, dout = rng.randint(1, 5000), rng.randint(16, 16384), rng.randint(16, 16384)
        qw, qa = rng.choice([0.5, 1, 2, 4]), rng.choice([1, 2, 4])
        assert roofline.ai_linear(n, din, dout, qw, qa) == pytest.approx(opgen.linear(0, "x", n, din, dout, qw, qa).ai)
    assert roofline.ai_linear(1, 8192, 8192, 2, 2) == pytest.approx(roofline.ai_linear_small_batch(1, 2), rel=1e-3)
    assert roofline.ai_linear(1e9, 4096, 4096, 2, 2) == pytest.approx(4096 / 2, rel=1e-5)   # saturates at d/q_a
    assert [roofline.ai_attn_decode(g, q) for g, q in ((1, 2), (8, 2), (8, 0.5))] == [1, 8, 32]   # E5 examples
    m = opgen.from_hf_config(ROOT / "configs" / "unsloth__Meta-Llama-3.1-8B.json")
    assert opgen.attention_ai_kv(opgen.gen_ops(m, opgen.decode(4, 777))) == roofline.ai_attn_decode(m.g, 2)


def test_e6_decode_step_bound():
    ms = machines()
    m = opgen.from_hf_config(ROOT / "configs" / "unsloth__Meta-Llama-3.1-8B.json")
    a = ms["a100-like"]
    r = roofline.decode_step_lb(m, 1, 1024, a)
    npar = opgen.n_params(m)
    assert r["t_mem"] == pytest.approx((2 * npar + 1024 * 131072) / 1555e9)
    assert r["t_comp"] == pytest.approx((2 * npar + 4 * 32 * 1024 * 4096) / a.peak("tcu"))
    assert r["bound"] == "memory" and r["t_step"] == r["t_mem"]
    assert roofline.decode_step_lb(m, 4096, 128, a)["bound"] == "compute"
    # per-op bounds: Σ_ops max(t_comp, t_mem) ≥ max(Σ t_comp, Σ t_mem); the ops move exactly E6's bytes except that
    # the embedding gather reads one row instead of the whole (untied) table
    ops = opgen.gen_ops(m, opgen.decode(1, 1024))
    lb = [roofline.op_time_lb(o, a) for o in ops]
    assert sum(x["t"] for x in lb) >= max(sum(x["t_comp"] for x in lb), sum(x["t_mem"] for x in lb))
    t = opgen.totals(ops)
    assert t["w_rd"] + t["kv_rd"] == pytest.approx(2 * (npar - m.vocab * m.d + m.d) + 1024 * 131072)


def test_every_machine_number_has_a_source():
    for mc in machines().values():
        for key in ("fpu", "tcu", "dram_bytes_per_cycle", "clock_hz"):
            src = mc.sources.get(key, "")
            assert src, (mc.name, key)
            assert ("VX_config.toml" in src or "profiles/" in src or "assumption" in src
                    or "results/P3.1/machine_params.csv" in src), (mc.name, key)
        if mc.measured:   # PS-BASE machines: every number is measured and points at its CSV row
            assert all("measured: results/P3.1/machine_params.csv" in mc.sources[k]
                       for k in ("fpu", "tcu", "dram_bytes_per_cycle")), mc.name
            continue
        if mc.name.startswith("vortex"):
            assert "unmeasured" in mc.sources["fpu"] and "assumption" in mc.sources["tcu"]
            assert "unmeasured" in mc.sources["dram_bytes_per_cycle"]
            assert "runtime-reported" in mc.sources["dram_bytes_per_cycle"]
        else:
            assert "assumption" in mc.sources["fpu"] and "assumption" in mc.sources["tcu"]
            assert "profiles/" in mc.sources["dram_bytes_per_cycle"] and "profiles/" in mc.sources["clock_hz"]


def test_main_writes_outputs(tmp_path):
    roofline.main(["--out", str(tmp_path)])
    for f in ("machines.csv", "crossover.csv", "fig/roofline_stories15M.png", "fig/roofline_Llama-3.1-8B.png",
              "fig/roofline_points.csv"):
        assert (tmp_path / f).stat().st_size > 0, f
    assert "unmeasured" in (tmp_path / "machines.csv").read_text()


def test_dram_bandwidth_derivation():
    """Peak DRAM bandwidth = channels × channel width × data rate (third_party/vortex/docs/simulation.md:198-200).
    Vortex defaults (VX_config.toml:80,85,93,94): 2 HBM2 channels × 16 B × 800 MT/s = 25.6 GB/s = 64 B per 400 MHz
    cycle; the runtime-reported VX_CFG_PLATFORM_MEMORY_PEAK_BW (460 GB/s, :84) is not used. The same formula
    reproduces the A100/H100 profiles' stated bandwidths from their own keys."""
    pd = roofline.vortex_platform_defaults()
    assert pd == dict(num_banks=2, mem_type="HBM2", clock_mhz=400.0, data_rate_mts=800.0, runtime_peak_bw_mb_s=460000.0)
    assert roofline.dram_peak_bw(2, "HBM2", 800) == 25.6e9
    assert machines()["vortex-S1-4c"].bw == 25.6e9
    for prof, bw in (("a100", 1555e9), ("h100", 3350e9)):
        p = roofline._profile(prof)["platform"]
        derived = roofline.dram_peak_bw(p["VX_CFG_PLATFORM_MEMORY_NUM_BANKS"], p["VX_CFG_PLATFORM_MEMORY_TYPE"],
                                        p["VX_CFG_PLATFORM_MEMORY_DATA_RATE"])
        assert derived == pytest.approx(bw, rel=1e-3)                     # 32 × 16 B × 3038 MT/s = 1,555.5 GB/s
        assert derived == pytest.approx(p["VX_CFG_PLATFORM_MEMORY_PEAK_BW"] * 1e6, rel=1e-3)
    assert roofline.dram_peak_bw(16, "GDDR6", 14000) == 16 * 2 * 14000e6     # simulation.md example: 448 GB/s


def test_rho_kv_moe_uses_per_step_weight_bytes():
    m = opgen.from_hf_config(ROOT / "configs" / "mistralai__Mixtral-8x7B-v0.1.json")
    ops = opgen.gen_ops(m, opgen.decode(1, 2048))
    wb = opgen.totals(ops)["w_rd"]
    assert wb < 0.4 * 2 * opgen.n_params(m)                                # top-2 of 8 experts streamed
    r_all, r_step = roofline.rho_kv(m, [2048]), roofline.rho_kv(m, [2048], weight_bytes=wb)
    assert r_step == pytest.approx(r_all * 2 * opgen.n_params(m) / wb)
