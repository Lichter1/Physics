"""
Graphs #6-#8 — Discharge coefficient C vs. Reynolds number (Venturi, Diaphragm,
Nozzle).

Manual §4.c.  One graph per instrument: the pointwise discharge coefficient
C = Q_real / Q_ideal as a function of Re.  DISCRETE points only — no global
regression line (C is not a linear function of Re; manual §7).

The workbook's C-table (rows 19-28: col C = C, col D = Re) holds exactly these
numbers.  We read C and Re from there, and because the workbook has no "C_err"
column we propagate the error ourselves from the Q_real / Q_ideal errors in the
main table:
        C = Qr / Qi
        ΔC = |C| · sqrt( (ΔQr/Qr)^2 + (ΔQi/Qi)^2 )
Re error comes straight from the main table's Re_err column (L).

ISO 5167 reference curves (β = D_min/D_max = 6/12 = 0.5, from Setup!D18:D20):
  • Venturi   — ISO 5167-4 machined convergent:  C ≈ 0.984 (flat; valid Re > 5×10⁴)
  • Diaphragm — ISO 5167-2 Reader-Harris-Gallagher (corner taps): C ≈ 0.60–0.62
  • Nozzle    — ISO 5167-3 ISA 1932 nozzle:  C ≈ 0.93–0.97
Set SHOW_ISO_REFERENCE = False to hide them.
"""

from __future__ import annotations

import flow_common as fc

# ----------------------------- CONFIG -------------------------------------
INSTRUMENTS = [
    # (sheet, output name, label)
    ("Venturi",   "fig06_venturi_C_vs_Re",   "Venturi"),
    ("Diaphragm", "fig07_diaphragm_C_vs_Re", "Diaphragm"),
    ("Nozzle",    "fig08_nozzle_C_vs_Re",    "Nozzle"),
]
MAIN_ROWS = fc.INSTRUMENT_ROWS   # (4, 13) -> Qi/Qr/Re live here
CTABLE_ROWS = fc.CTABLE_ROWS     # (19, 28) -> C and Re live here
COL = dict(q_ideal="G", q_ideal_err="H", q_real="I", q_real_err="J",
           Re="K", Re_err="L")
CT_COL = dict(C="C", Re="D")

CONNECT_POINTS = True            # thin guide line between points (readability only)
SHOW_ISO_REFERENCE = True        # overlay ISO 5167 reference curve on each graph
BETA = 0.5                       # D_min / D_max = 6 mm / 12 mm  (from Setup sheet)

# Optional manual overrides: {instrument_label: (Re_array, C_array)}.
# Leave empty to use the auto-generated ISO 5167 curves when SHOW_ISO_REFERENCE=True.
LITERATURE: dict[str, tuple] = {}
# --------------------------------------------------------------------------


def _iso_C(Re_arr, instrument: str, beta: float = 0.5):
    """ISO 5167 discharge coefficient vs Reynolds number.

    Venturi   — ISO 5167-4 machined convergent: C = 0.984 (Re > 5e4, essentially flat).
                Outside the ISO validity range the dashed line is shown as a guide only.
    Diaphragm — ISO 5167-2 Reader-Harris-Gallagher (RHG), corner taps (L1 = L2' = 0).
    Nozzle    — ISO 5167-3 ISA 1932 nozzle.
    """
    import numpy as np
    Re = np.asarray(Re_arr, float)

    if instrument == "Venturi":
        # ISO 5167-4 §5.3: C = 0.984 for machined convergent, Re ≥ 5×10⁴.
        # Extended as a constant reference line across the full Re axis.
        return np.full_like(Re, 0.984)

    elif instrument == "Diaphragm":
        # ISO 5167-2 Eq.(1), corner taps → L1 = L2' = 0, so the tap-geometry
        # terms vanish and the equation reduces to:
        #   C = 0.5961 + 0.0261β² − 0.216β⁸
        #       + 0.000521(10⁶β/Re)^0.7
        #       + (0.0188 + 0.0063 A) β^3.5 (10⁶/Re)^0.3
        # where A = (19000 β / Re)^0.8
        b = beta
        A = (19000.0 * b / Re) ** 0.8
        C = (0.5961
             + 0.0261 * b**2
             - 0.216  * b**8
             + 0.000521 * (1e6 * b / Re) ** 0.7
             + (0.0188 + 0.0063 * A) * b**3.5 * (1e6 / Re) ** 0.3)
        return C

    elif instrument == "Nozzle":
        # ISO 5167-3 Eq.(1), ISA 1932 nozzle:
        #   C = 0.9900 − 0.2262 β^4.1
        #       − (0.00175 β² − 0.0033 β^4.15)(10⁶/Re)^1.15
        b = beta
        C = (0.9900
             - 0.2262 * b**4.1
             - (0.00175 * b**2 - 0.0033 * b**4.15) * (1e6 / Re) ** 1.15)
        return C

    import numpy as np
    return np.full_like(Re, float("nan"))


def main(wb=None):
    wb = wb or fc.load_workbook()
    fc.apply_style()
    import matplotlib.pyplot as plt
    import numpy as np

    results = {}
    for sheet, name, label in INSTRUMENTS:
        ws = wb[sheet]
        # C and Re straight from the C-table (the authoritative source).
        C = fc.read_column(ws, CT_COL["C"], *CTABLE_ROWS)
        Re_ct = fc.read_column(ws, CT_COL["Re"], *CTABLE_ROWS)
        # Main table for the error propagation + Re_err (row-aligned with C-table).
        Qi = fc.read_column(ws, COL["q_ideal"], *MAIN_ROWS)
        Qi_e = fc.read_column(ws, COL["q_ideal_err"], *MAIN_ROWS)
        Qr = fc.read_column(ws, COL["q_real"], *MAIN_ROWS)
        Qr_e = fc.read_column(ws, COL["q_real_err"], *MAIN_ROWS)
        Re = fc.read_column(ws, COL["Re"], *MAIN_ROWS)
        Re_e = fc.read_column(ws, COL["Re_err"], *MAIN_ROWS)

        title = f"Figure {name[3:5]} — {label}: discharge coefficient C vs. Re"
        if not fc.has_data(C, Re_ct):
            fc.placeholder(name, title)
            print(f"[{name}] no data yet -> placeholder written")
            results[label] = {"figure": name, "status": "no-data"}
            continue

        m = fc.finite_mask(C, Re_ct, Qi, Qr)
        c = C[m]
        re = Re_ct[m]
        # Error propagation for C (workbook has no C_err column).
        with np.errstate(divide="ignore", invalid="ignore"):
            c_err = np.abs(c) * np.sqrt((Qr_e[m] / Qr[m]) ** 2 +
                                        (Qi_e[m] / Qi[m]) ** 2)
        re_err = Re_e[m]
        re_err = np.where(np.isfinite(re_err), re_err, np.nan)

        fig, ax = plt.subplots()
        if CONNECT_POINTS:
            order = np.argsort(re)
            ax.plot(re[order], c[order], "-", color=fc.COLORS[label.lower()],
                    lw=0.8, alpha=0.5, zorder=2)
        fc.errorbar_series(ax, re, c, xerr=re_err, yerr=c_err,
                           color=fc.COLORS[label.lower()],
                           label=f"{label} (measured)")

        # ISO 5167 reference curve (manual / user override takes priority).
        if label in LITERATURE:
            lit_re, lit_c = LITERATURE[label]
            ax.plot(lit_re, lit_c, "--", color=fc.COLORS["literature"], lw=1.5,
                    label="Literature / standard")
        elif SHOW_ISO_REFERENCE:
            re_lo = max(float(np.nanmin(re)) * 0.8, 1e3)
            re_hi = float(np.nanmax(re)) * 1.2
            re_ref = np.linspace(re_lo, re_hi, 300)
            c_ref = _iso_C(re_ref, label, beta=BETA)

            # Validity note per instrument (ISO minimum Re).
            if label == "Venturi":
                iso_label = rf"ISO 5167-4 ($C=0.984$, machined; Re$\geq5\times10^4$)"
                # shade outside-validity region
                valid_lo = 5e4
                if re_lo < valid_lo:
                    ax.axvspan(re_lo, min(valid_lo, re_hi), alpha=0.07,
                               color="#aaaaaa", zorder=0, label="Outside ISO validity")
            elif label == "Diaphragm":
                iso_label = rf"ISO 5167-2 RHG ($\beta={BETA}$, corner taps)"
            else:  # Nozzle
                iso_label = rf"ISO 5167-3 ISA 1932 ($\beta={BETA}$)"

            ax.plot(re_ref, c_ref, "--", color=fc.COLORS["literature"], lw=1.5,
                    label=iso_label)

        ax.set_xlabel(r"Reynolds number  $Re$  [-]")
        ax.set_ylabel(r"Discharge coefficient  $C = Q_{real}/Q_{ideal}$  [-]")
        #ax.set_title(title)
        ax.legend(loc="best", fontsize=8.5)

        # Print ISO value at the midpoint Re for a quick sanity check.
        re_mid = float(np.nanmedian(re))
        c_iso_mid = float(_iso_C([re_mid], label, BETA)[0])
        paths = fc.save_fig(fig, name)
        print(f"[{name}] C range=[{np.nanmin(c):.3g}, {np.nanmax(c):.3g}]  "
              f"Re range=[{np.nanmin(re):.0f}, {np.nanmax(re):.0f}]  "
              f"C_ISO(Re={re_mid:.0f})={c_iso_mid:.4f}  -> {paths[0].name}")
        results[label] = {"figure": name, "status": "ok",
                          "C_mean": float(np.nanmean(c)),
                          "C_min": float(np.nanmin(c)), "C_max": float(np.nanmax(c)),
                          "Re_min": float(np.nanmin(re)), "Re_max": float(np.nanmax(re)),
                          "C_ISO_midRe": c_iso_mid}
    return results


if __name__ == "__main__":
    main()
