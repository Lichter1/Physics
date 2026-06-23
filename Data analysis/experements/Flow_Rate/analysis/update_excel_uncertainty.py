"""
Update the workbook so the REAL flow-rate error (Q_real, measured by the
LDN 1009 GAPL mass flow meter) follows the manufacturer datasheet instead of a
flat constant.

Datasheet (LDN 1009 GAPL), combined in quadrature (RSS):
    ΔQ_real = sqrt( (4%·Q)^2 + (0.5%·FS)^2 + (2%·Q)^2 )
with FS (range-end value) = 250 Nl/min.

What this script changes (in place, preserving formatting):
  * Setup: adds an "LDN 1009 spec" block (rows 32-36) with the % terms + FS.
  * Venturi / Diaphragm / Nozzle, J4:J13 (Q_real_err): flat `Setup!C14`
        -> per-point quadrature formula referencing the new Setup cells.
  * Pitot, M4:M13 (u-from-flow error): the flow-error term that used the flat
        `Setup!C14` is replaced by the same per-point quadrature on K.

It is idempotent: re-running detects the spec block and only refreshes formulas.
openpyxl drops Excel's cached values on save; the figure scripts recompute via
the `formulas` engine, and Excel/LibreOffice will recompute on open.

Run:  python update_excel_uncertainty.py
"""

from __future__ import annotations

from copy import copy

import flow_common as fc
import openpyxl

INSTRUMENT_SHEETS = ["Venturi", "Diaphragm", "Nozzle"]
ROWS = range(4, 14)  # 4..13

# Setup cells for the LDN 1009 spec.
SPEC = {
    "B32": ("מפרט מד הספיקה LDN 1009 GAPL (שגיאת ספיקה אמיתית ΔQ_real)", None, None),
    "B33": ("שגיאה מהקריאה (חלק מהערך הנמדד)", 0.04, "-"),
    "B34": ("שגיאה מערך קצה הטווח (חלק מ-FS)", 0.005, "-"),
    "B35": ("ערך קצה הטווח FS", 250, "L/min"),
    "B36": ("שגיאת חזרתיות (חלק)", 0.02, "-"),
}
# Quadrature error in terms of a flow value reference (cell like $I4 or $K4):
QUAD = ("SQRT((Setup!$C$33*{q})^2+(Setup!$C$34*Setup!$C$35)^2+"
        "(Setup!$C$36*{q})^2)")


def _copy_style(src, dst):
    dst.font = copy(src.font)
    dst.fill = copy(src.fill)
    dst.border = copy(src.border)
    dst.alignment = copy(src.alignment)
    dst.number_format = src.number_format
    dst.protection = copy(src.protection)


def main():
    path = fc.WORKBOOK
    print(f"Editing: {path}")
    wb = openpyxl.load_workbook(path, data_only=False)  # keep formulas + styles
    setup = wb["Setup"]

    # ----- Setup spec block (idempotent) -----
    if setup["C35"].value == 250:
        print("  Setup LDN-spec block already present — refreshing formulas only.")
    else:
        for addr, (label, value, unit) in SPEC.items():
            r = addr[1:]
            setup[f"B{r}"] = label
            _copy_style(setup["B12"], setup[f"B{r}"])  # label style
            if value is not None:
                setup[f"C{r}"] = value
                _copy_style(setup["C12"], setup[f"C{r}"])  # yellow input style
            if unit is not None:
                setup[f"D{r}"] = unit
                _copy_style(setup["D12"], setup[f"D{r}"])
        print("  Added Setup LDN-spec block (rows 32-36).")

    # ----- Venturi / Diaphragm / Nozzle: J = quadrature on I (Q_real) -----
    for sheet in INSTRUMENT_SHEETS:
        ws = wb[sheet]
        for r in ROWS:
            ws[f"J{r}"] = (f'=IF($I{r}="","",' + QUAD.format(q=f"$I{r}") + ")")
        print(f"  {sheet}: J4:J13 -> datasheet quadrature.")

    # ----- Pitot: M uses quadrature on K instead of flat Setup!C14 -----
    pit = wb["Pitot"]
    for r in ROWS:
        q = QUAD.format(q=f"$K{r}")
        pit[f"M{r}"] = (
            f'=IF($K{r}="","",SQRT('
            f'(4/(PI()*(Setup!$C$21/1000)^2)/60000*{q})^2'
            f'+(8*($K{r}/60000)/(PI()*(Setup!$C$21/1000)^3)*Setup!$C$13)^2))'
        )
    print("  Pitot: M4:M13 -> datasheet quadrature on K.")

    # Clarify the now-legacy constant so nobody trusts it.
    if "(לא בשימוש" not in str(setup["B14"].value):
        setup["B14"] = str(setup["B14"].value) + "  (לא בשימוש — ראה מפרט LDN למטה)"

    wb.save(path)
    print("Saved.  (Excel will recompute on open; figures recompute via `formulas`.)")


if __name__ == "__main__":
    main()
