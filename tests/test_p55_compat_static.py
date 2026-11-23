"""Every cell of results/<run>/compat.csv agrees with scripts/knobs.py legality.

Static, no SimX: a "pass" ablation cell must have no legality error for any segment, a "rejected-by-design" probe cell must
have at least one (the simulator abort the probe expects), and the csv must hold exactly one row per (vector, kernel)."""
import csv
import sys
from collections import Counter
from pathlib import Path

import pytest

PS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PS / "scripts"))
import knobs  # noqa: E402

CSV = PS / "results" / "P5.5" / "compat.csv"


@pytest.fixture(scope="module")
def cells():
    if not CSV.is_file():
        pytest.skip("results/P5.5/compat.csv not present")
    doc = knobs.load()
    return doc, list(csv.DictReader(CSV.open()))


def test_compat_csv_cells_agree_with_knobs_legality(cells):
    doc, rows = cells
    by = {(r["vector"], r["kernel"]): r for r in rows}
    assert len(by) == len(rows) == 156
    assert Counter(r["status"] for r in rows) == {"pass": 117, "rejected-by-design": 39}
    seen = 0
    for v in list(doc["ablation"]) + list(doc["probes"]):
        for k in knobs.kernels(doc):
            r = by[(v, k["id"])]
            cell = knobs.resolve(doc, v, k, half=1000 if v == "oracle" else None)
            errs = knobs.cell_errors(doc, cell)
            if r["status"] == "pass":
                assert not cell.probe and not errs, (v, k["id"], errs)
            else:
                assert r["status"] == "rejected-by-design" and cell.probe and errs, (v, k["id"])
            seen += 1
    assert seen == 156
