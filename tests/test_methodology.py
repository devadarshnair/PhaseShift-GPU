"""Documentation tests: methodology.md has E1-E11 headings, the §4 numeric region thresholds, the DP initial
condition J(1,c) and the required sections; the errata file carries a file:line pointer or "unverified" in every item."""
import re

import pytest


@pytest.fixture(scope="module")
def meth(ps_root):
    return (ps_root / "docs" / "methodology.md").read_text()


@pytest.mark.parametrize("n", range(1, 12))
def test_equation_headings(meth, n):
    assert re.search(rf"^### E{n} — ", meth, re.M), f"E{n} heading missing"


def test_each_equation_has_symbols_why_status(meth):
    blocks = re.split(r"^### ", meth, flags=re.M)[1:]
    eq = [b for b in blocks if re.match(r"E\d+ — ", b)]
    assert len(eq) == 11
    for b in eq:
        assert "**Symbols:**" in b and "**Why this shape:**" in b and "**Status:**" in b, b[:20]


def test_section_headings_1_to_8(meth):
    for n, word in [(1, "Notation"), (2, "E1–E11"), (3, "Baselines"), (4, "Regions"), (5, "Oracle"),
                    (6, "Sampling"), (7, "Scaling"), (8, "Two-level")]:
        assert re.search(rf"^## {n}\. .*{word}", meth, re.M), (n, word)


def test_region_thresholds(meth):
    sec4 = meth.split("## 4. Regions")[1].split("## 5.")[0]
    assert "prefill tokens and no decode tokens" in sec4
    assert "both prefill and decode tokens" in sec4
    assert "ρ_KV = M_KV,total / (q_w·N_params) ≥ 1" in sec4
    assert "four regions" in sec4.lower()
    assert "majority of its baseline time" in sec4


def test_oracle_dp(meth):
    assert "`J(1, c) = T(1, c)`" in meth
    assert "J(i, c) = T(i, c) + min_c′ [ J(i−1, c′) + C_sw(c′, c) ]" in meth
    assert "C_sw(c, c) = 0" in meth
    assert "min_c J(n, c)" in meth
    assert "O(n·|C|²)" in meth
    assert "work units" in meth


def test_section10_complete(meth):
    """methodology §10 complete (scheduler semantics, fixed-schedule assumption, σ, G4 handling)."""
    sec = meth.split("\n## 10. ")[1].split("\n## 11. ")[0]
    assert "PLACEHOLDER" not in sec.splitlines()[0]
    for needle in ("### 10.1 Traces", "### 10.2 Iteration scheduler", "### 10.3 Fixed-schedule assumption",
                   "### 10.4 Trace length scaling σ", "Σ_i ⌈T_i/b⌉·b", "σ = (M_ref / L2_ref) / (M_tgt / L2_tgt)",
                   "median decode length of Azure `conv` after the σ scaling", "unscaled value"):
        assert needle in sec, needle


def test_required_facts_present(meth):
    assert "Decode is two regimes" in meth
    assert "Capacity is not the decode lever" in meth
    assert "1,056,540" in meth and "1 MiB L2" in meth
    assert "Caches are cold at every launch" in meth
    assert "Per-kernel best static" in meth


def test_errata_items_have_pointer_or_unverified(ps_root):
    text = (ps_root / "docs" / "vortex_errata.md").read_text()
    items = re.findall(r"^(\d+)\. \*\*(.+?)(?=^\d+\. \*\*|^## |\Z)", text, re.M | re.S)
    assert len(items) >= 18
    for num, body in items:
        has_ptr = re.search(r"[\w/.\-]+\.(?:cpp|h|sv|md|toml|mk|txt|yaml|in)\b:\d+|Makefile:\d+", body)
        assert has_ptr or "unverified" in body.lower(), f"errata item {num} lacks file:line or 'unverified'"


# -----------------------------------------------------------------------------------
def test_interval_dependent_switch_cost(meth):
    assert "`C_sw,i(c′, c) = 0` if interval i is the first\n  interval of a new kernel launch (K(i) ≠ K(i−1))" in meth
    assert "`J(i, c) = T(i, c) + min_c′ [ J(i−1, c′) + C_sw,i(c′, c) ]`" in meth
    assert "K(i) the kernel\nlaunch that interval i belongs to" in meth
    assert "`T_DP ≤ T_kernel`: the per-launch schedule" in meth


def test_e5_and_prefill_interpolation_wording(meth):
    assert "E3 per token ÷ L × T" in meth
    assert "T_prefill(S) = a + b·S + c·S²" in meth


def _dp(T, K, csw):
    """The methodology §5 DP with C_sw,i = 0 at launch starts. T[i][c] costs, K[i] launch ids, csw[c'][c]."""
    n, C = len(T), len(T[0])
    J = list(T[0])
    for i in range(1, n):
        J = [T[i][c] + min(J[p] + (0 if K[i] != K[i - 1] else csw[p][c]) for p in range(C)) for c in range(C)]
    return min(J)


def test_dp_ordering_holds_on_random_instances():
    import random
    rng = random.Random(1)
    for _ in range(300):
        n, C = rng.randint(1, 12), rng.randint(1, 4)
        T = [[rng.uniform(1, 10) for _ in range(C)] for _ in range(n)]
        K, k = [], 0
        for i in range(n):
            if i and rng.random() < 0.3:
                k += 1
            K.append(k)
        csw = [[0 if a == b else rng.uniform(0, 20) for b in range(C)] for a in range(C)]
        t_zero = sum(min(r) for r in T)
        t_dp = _dp(T, K, csw)
        t_kernel = sum(min(sum(T[i][c] for i in range(n) if K[i] == kk) for c in range(C)) for kk in set(K))
        t_static = min(sum(T[i][c] for i in range(n)) for c in range(C))
        assert t_zero <= t_dp + 1e-9 <= t_kernel + 2e-9 <= t_static + 3e-9


def test_errata_pointer_fixes(ps_root):
    errata = (ps_root / "docs" / "vortex_errata.md").read_text()
    assert "`:132`:\n    `.write (init || (lookup_valid && lookup_hit))`" in errata
    assert "table rows at `:72-73`, example `CONFIGS` at `:80`" in errata


def test_errata_pointers_match_pristine_clone(ps_root):
    """Read-only check of the corrected pointers against third_party/vortex (skipped if the clone is absent)."""
    v = ps_root / "third_party" / "vortex"
    if not v.is_dir():
        pytest.skip("pristine clone absent")
    line = lambda f, n: (v / f).read_text().splitlines()[n - 1]
    assert ".write (init || (lookup_valid && lookup_hit))" in line("hw/rtl/cache/VX_cache_repl.sv", 132)
    assert "state advances on lookup hits only" in line("hw/rtl/cache/VX_cache_repl.sv", 111)
    assert "VX_CFG_L2_CACHE_SIZE" in line("docs/simulation.md", 72)
    assert "VX_CFG_L3_CACHE_SIZE" in line("docs/simulation.md", 73)
    assert line("docs/simulation.md", 80).startswith("$ CONFIGS=")


def test_launch_history_and_kernel_metric_rules(meth):
    """§2 consequences 5 (launch-history rule) and 6 (kernel correctness metric), the measured
    cold-cache confirmation in consequence 4, and the §8 pointer."""
    sec2 = meth.split("## 2. ")[1].split("\n## 3. ")[0]
    assert "**Measured**" in sec2 and "coldcache.csv" in sec2
    assert re.search(r"^5\. \*\*Launch history", sec2, re.M)
    for needle in ("first launch of an app run", "identical launch sequences", "only when stated"):
        assert needle in sec2, needle
    assert re.search(r"^6\. \*\*Kernel correctness metric", sec2, re.M)
    assert "kernels/common/ps_common.h" in sec2 and "K ≤ 838" in sec2
    assert "§2, consequence 5" in meth.split("## 8. ")[1].split("\n## 9. ")[0]
