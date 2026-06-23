"""
Generate every figure for the Flow-Rate lab report in one shot.

    python generate_all.py

It runs the file-integrity check first (section 3 of the instructions), then
each fig*.py in order, collects the fitted slopes / C-values / Re-ranges, and
writes them to ``figures/RESULTS_SUMMARY.txt`` for the discussion section.

While the workbook's yellow input cells are still empty, every figure is a
clearly-labelled "Awaiting measurements" placeholder and the summary says so.
Fill the cells, re-run, and the real graphs + numbers appear automatically.
"""

from __future__ import annotations

import datetime as _dt

import flow_common as fc
import check_workbook
import fig01_rotameter_calibration as fig01
import fig02_04_flow_calibration as fig02_04
import fig05_pitot_velocity as fig05
import fig06_08_C_vs_Re as fig06_08
import fig09_qreal_vs_qideal as fig09
import fig10_rotameter_pressure as fig10


def _fmt(v, nd=4):
    try:
        return f"{float(v):.{nd}g}"
    except (TypeError, ValueError):
        return str(v)


def main():
    print("=" * 70)
    print("Flow-Rate lab — generating all figures")
    print("Workbook:", fc.WORKBOOK)
    print("Figures :", fc.FIGDIR)
    print("=" * 70)

    wb = fc.load_workbook()

    print("\n--- File-integrity check ---")
    integrity = check_workbook.run(wb)

    print("\n--- Figures ---")
    r01 = fig01.main(wb)
    r02_04 = fig02_04.main(wb)
    r05 = fig05.main(wb)
    r06_08 = fig06_08.main(wb)
    r09 = fig09.main(wb)
    r10 = fig10.main(wb)

    # ----------------------------------------------------------------- summary
    lines = []
    lines.append("Flow-Rate lab — results summary")
    lines.append(f"Generated: {_dt.datetime.now():%Y-%m-%d %H:%M:%S}")
    lines.append(f"Workbook : {fc.WORKBOOK.name}")
    lines.append("=" * 60)

    lines.append("\nFILE-INTEGRITY CHECK")
    lines.append(f"  data present : {integrity['data_present']}")
    if integrity["empty_inputs"]:
        lines.append("  EMPTY input blocks (fill these yellow cells):")
        for item in integrity["empty_inputs"]:
            lines.append(f"    - {item}")
    if integrity["formula_errors"]:
        lines.append("  FORMULA ERRORS:")
        for item in integrity["formula_errors"]:
            lines.append(f"    - {item}")
    if integrity["hand_checks"]:
        lines.append("  Hand-check (Python vs. Excel):")
        for item in integrity["hand_checks"]:
            lines.append(f"    - {item}")

    lines.append("\nGRAPH #1 — Rotameter calibration (atmospheric)")
    if r01.get("status") == "ok":
        lines.append(f"  Q = {_fmt(r01['slope'])}·reading + {_fmt(r01['intercept'])}"
                     f"   (chi2/nu={_fmt(r01['chi2_red'])})")
    else:
        lines.append("  (no data yet)")

    lines.append("\nGRAPHS #2-4 — Flow-rate calibration (Q ∝ √ΔP fits)")
    for label, r in (r02_04 or {}).items():
        if r.get("status") == "ok":
            lines.append(f"  {label:10s}: Q_ideal={_fmt(r['a_ideal'])}·√ΔP "
                         f"(chi2/nu={_fmt(r['chi2_ideal'])}), "
                         f"Q_real={_fmt(r['a_real'])}·√ΔP "
                         f"(chi2/nu={_fmt(r['chi2_real'])})")
        else:
            lines.append(f"  {label:10s}: (no data yet)")

    lines.append("\nGRAPH #5 — Pitot velocity (u = a·√ΔP)")
    if r05.get("status") == "ok":
        for k, v in r05["fits"].items():
            lines.append(f"  {k:8s}: a={_fmt(v['slope'])} (chi2/nu={_fmt(v['chi2_red'])})")
    else:
        lines.append("  (no data yet)")

    lines.append("\nGRAPHS #6-8 — Discharge coefficient C vs. Re")
    for label, r in (r06_08 or {}).items():
        if r.get("status") == "ok":
            lines.append(f"  {label:10s}: C in [{_fmt(r['C_min'])}, {_fmt(r['C_max'])}] "
                         f"(mean {_fmt(r['C_mean'])}); "
                         f"Re in [{_fmt(r['Re_min'])}, {_fmt(r['Re_max'])}]")
        else:
            lines.append(f"  {label:10s}: (no data yet)")

    lines.append("\nGRAPH #9 — Q_real = f(Q_ideal); slope = C")
    if r09.get("status") == "ok":
        for label, v in r09["slopes"].items():
            lines.append(f"  {label:10s}: C={_fmt(v['C'])} "
                         f"(intercept={_fmt(v['intercept'])}, chi2/nu={_fmt(v['chi2_red'])}, "
                         f"n={v['n']})")
    else:
        lines.append("  (no data yet)")

    lines.append("\nGRAPH #10 — Normalised rotameter flow vs. √(P0/Pi)")
    if r10.get("status") == "ok":
        lines.append(f"  slope={_fmt(r10['slope'])}, intercept={_fmt(r10['intercept'])}, "
                     f"chi2/nu={_fmt(r10['chi2_red'])}  "
                     f"[{r10['ratio_mode']}, P_ref={r10['p_ref_bar']} bar_a]")
    else:
        lines.append("  (no data yet)")

    text = "\n".join(lines) + "\n"
    fc.FIGDIR.mkdir(parents=True, exist_ok=True)
    out = fc.FIGDIR / "RESULTS_SUMMARY.txt"
    out.write_text(text, encoding="utf-8")

    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)
    print(f"Summary written to: {out}")
    print(f"All figures in    : {fc.FIGDIR}")


if __name__ == "__main__":
    main()
