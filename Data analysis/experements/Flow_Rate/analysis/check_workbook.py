"""
File-integrity check for the Flow-Rate workbook (instructions section 3).

Run standalone:   python check_workbook.py
Or call run(wb) from generate_all.py.

It does NOT alter the workbook.  It reports:
  1. which input blocks are still empty (the yellow cells),
  2. any Excel formula errors (#DIV/0!, #VALUE!, ...),
  3. an independent "by hand" recomputation of one value per sheet, compared
     against the cell Excel produced (only possible once data is entered),
  4. whether the Reynolds numbers land in a sane turbulent range (~1e3-1e5).
"""

from __future__ import annotations

import flow_common as fc
import numpy as np

ERROR_TOKENS = {"#DIV/0!", "#VALUE!", "#NUM!", "#REF!", "#NAME?", "#NULL!", "#N/A"}

# (sheet, column, row0, row1, human description)
INPUT_BLOCKS = [
    ("Venturi", "D", 4, 13, "Venturi Δh (measured)"),
    ("Venturi", "I", 4, 13, "Venturi Q_real (measured)"),
    ("Diaphragm", "D", 4, 13, "Diaphragm Δh (measured)"),
    ("Diaphragm", "I", 4, 13, "Diaphragm Q_real (measured)"),
    ("Nozzle", "D", 4, 13, "Nozzle Δh (measured)"),
    ("Nozzle", "I", 4, 13, "Nozzle Q_real (measured)"),
    ("Pitot", "C", 4, 13, "Pitot Δh (measured)"),
    ("Pitot", "K", 4, 13, "Pitot Q rotameter (measured)"),
    ("Rotameter", "C", 4, 13, "Rotameter reading, part 1"),
    ("Rotameter", "D", 4, 13, "Rotameter mass-flow, part 1"),
    ("Rotameter", "C", 17, 21, "Rotameter abs-pressure, part 2"),
    ("Rotameter", "D", 17, 21, "Rotameter reading, part 2"),
    ("Rotameter", "E", 17, 21, "Rotameter mass-flow, part 2"),
]


def _empty_inputs(wb):
    out = []
    for sheet, col, r0, r1, desc in INPUT_BLOCKS:
        arr = fc.read_column(wb[sheet], col, r0, r1)
        n = int(np.sum(np.isfinite(arr)))
        total = r1 - r0 + 1
        if n == 0:
            out.append(f"{desc} [{sheet}!{col}{r0}:{col}{r1}] — 0/{total} filled")
        elif n < total:
            out.append(f"{desc} [{sheet}!{col}{r0}:{col}{r1}] — only {n}/{total} filled (gap?)")
    return out


def _formula_errors(wb):
    out = []
    for ws in wb.worksheets:
        for row in ws.iter_rows():
            for c in row:
                if c.value is not None and str(c.value).strip() in ERROR_TOKENS:
                    out.append(f"{ws.title}!{c.coordinate} = {c.value}")
    return out


def _hand_checks(wb):
    """Independent recomputation of one value per sheet (needs data)."""
    out = []
    s = wb["Setup"]
    rho_w = fc.read_cell(s, "C9")     # 1000
    g = fc.read_cell(s, "C10")        # 9.81
    rho_a = fc.read_cell(s, "C8")     # 1.184

    # Venturi ΔP = ρ_water · g · Δh  (first filled row)
    v = wb["Venturi"]
    dh = fc.read_column(v, "D", 4, 13)
    dP = fc.read_column(v, "E", 4, 13)
    idx = np.where(np.isfinite(dh) & np.isfinite(dP))[0]
    if idx.size:
        i = idx[0]
        expect = rho_w * g * dh[i]
        ok = np.isclose(expect, dP[i], rtol=1e-3, atol=1e-6)
        out.append(f"Venturi ΔP row {4+i}: hand={expect:.4g} Pa vs Excel={dP[i]:.4g} Pa "
                   f"-> {'OK' if ok else 'MISMATCH'}")

    # Pitot u_Bernoulli = sqrt(2·ΔP/ρ_air)
    p = wb["Pitot"]
    pdP = fc.read_column(p, "D", 4, 13)
    u = fc.read_column(p, "G", 4, 13)
    idx = np.where(np.isfinite(pdP) & np.isfinite(u))[0]
    if idx.size:
        i = idx[0]
        expect = np.sqrt(2 * pdP[i] / rho_a)
        ok = np.isclose(expect, u[i], rtol=1e-3, atol=1e-6)
        out.append(f"Pitot u_Bernoulli row {4+i}: hand={expect:.4g} m/s vs "
                   f"Excel={u[i]:.4g} m/s -> {'OK' if ok else 'MISMATCH'}")
    return out


def _re_range(wb):
    res = []
    allvals = []
    for sheet, col in [("Venturi", "K"), ("Diaphragm", "K"), ("Nozzle", "K"),
                       ("Pitot", "N")]:
        arr = fc.read_column(wb[sheet], col, 4, 13)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            allvals.append(arr)
            res.append(f"{sheet}: Re in [{arr.min():.4g}, {arr.max():.4g}]")
    if allvals:
        a = np.concatenate(allvals)
        sane = np.all((a > 1e2) & (a < 1e6))
        res.append(f"overall {'OK (1e2-1e6)' if sane else 'OUT OF RANGE — check inputs!'}")
    return res


def run(wb=None):
    wb = wb or fc.load_workbook()
    empty = _empty_inputs(wb)
    errors = _formula_errors(wb)
    data_present = len(empty) < len(INPUT_BLOCKS)  # at least one block has data
    hand = _hand_checks(wb) if data_present else []
    re = _re_range(wb) if data_present else []

    # Console output
    if errors:
        print(f"  [!] {len(errors)} formula error(s):")
        for e in errors:
            print("      -", e)
    else:
        print("  formula errors: none")
    if empty:
        print(f"  [!] {len(empty)} input block(s) still empty:")
        for e in empty:
            print("      -", e)
    else:
        print("  inputs: all blocks filled")
    for h in hand:
        print("  hand-check:", h)
    for r in re:
        print("  Re:", r)
    if not data_present:
        print("  (no measurements entered yet — graphs will be placeholders)")

    return {"data_present": data_present, "empty_inputs": empty,
            "formula_errors": errors, "hand_checks": hand, "re_range": re}


if __name__ == "__main__":
    run()
