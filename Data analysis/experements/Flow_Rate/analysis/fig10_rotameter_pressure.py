"""
Graph #10 — Normalised rotameter flow rate vs. √(P0/Pi).

Manual §4.f.  For the variable-pressure test (5 points, 1-2.5 bar_a) at constant
true flow rate, convert each rotameter reading into a flow rate using the
calibration equation from graph #1, then plot that flow rate against √(P0/Pi).
A linear trendline gives the slope, whose meaning is discussed in the report.

Data:  sheet 'Rotameter', rows 17-21
        C = pressure  [bar]   (the sheet header labels it "P0"; see note below)
        D = rotameter reading [cm]
        E = constant true flow rate [L/min]   (reference; not plotted)

IMPORTANT — pressure units and the definition of √(P0/Pi):
    1. The sheet header for column C currently reads "לחץ גייג׳י" = GAUGE
       pressure.  Absolute pressure is therefore  P_abs = P_gauge + atmosphere.
       That is controlled by PRESSURE_IS_GAUGE / ATM_BAR below; turn the flag
       off if your column already holds absolute pressure.
    2. The manual's ratio P0/Pi needs a reference pressure too.  Two common
       conventions are provided; pick the one your manual (§4.f) uses via
       RATIO_MODE:
        RATIO_MODE = "ref_over_p"  ->  X = sqrt(P_ref / P_abs)
        RATIO_MODE = "p_over_ref"  ->  X = sqrt(P_abs / P_ref)
       P_REF_BAR defaults to 1.013 bar_a (atmospheric — the part-1 calibration
       state).
    >>> Verify both points against the manual before quoting the slope. <<<
"""

from __future__ import annotations

import flow_common as fc
import fig01_rotameter_calibration as fig01

# ----------------------------- CONFIG -------------------------------------
SHEET = "Rotameter"
ROWS = fc.ROTAMETER_PRES_ROWS  # (17, 21)
COL_PRESSURE = "C"  # pressure [bar] (gauge — see PRESSURE_IS_GAUGE)
COL_READING = "D"   # rotameter reading [cm]
COL_QTRUE = "E"     # constant true flow rate [L/min] (reference only)

PRESSURE_IS_GAUGE = True    # sheet header says "גייג׳י" -> add atmosphere for absolute
ATM_BAR = 1.0             # atmospheric pressure for gauge -> absolute [bar]
RATIO_MODE = "ref_over_p"   # "ref_over_p" or "p_over_ref"  (see header)
P_REF_BAR = 1.0          # reference pressure P0 [bar_a]  (atmospheric)

READING_ERR = 0.05          # rotameter-reading error [cm]; None -> no Y error bar
PRESSURE_ERR_BAR = None      # gauge-pressure reading error [bar]; None -> read Setup!C15
NAME = "fig10_rotameter_pressure"
TITLE = "Figure 10 — Normalised rotameter flow rate vs. √(P0/Pi)"
# --------------------------------------------------------------------------


def main(wb=None):
    wb = wb or fc.load_workbook()
    ws = wb[SHEET]
    p_raw = fc.read_column(ws, COL_PRESSURE, *ROWS)
    reading = fc.read_column(ws, COL_READING, *ROWS)

    cal = fig01.get_calibration(wb)
    have_inputs = fc.has_data(p_raw, reading)
    if not have_inputs or cal is None:
        why = "no pressure/reading data" if not have_inputs else \
              "graph-#1 calibration not available (atmospheric data missing)"
        fc.placeholder(NAME, TITLE,
                       message=f"Awaiting data — {why}.\nFill the yellow cells, then re-run.")
        print(f"[{NAME}] {why} -> placeholder written")
        return {"figure": NAME, "status": "no-data", "reason": why}

    fc.apply_style()
    import matplotlib.pyplot as plt
    import numpy as np

    m = fc.finite_mask(p_raw, reading)
    p_raw, reading = p_raw[m], reading[m]
    # Convert gauge -> absolute pressure if needed.
    p_abs = p_raw + ATM_BAR if PRESSURE_IS_GAUGE else p_raw

    # Y: rotameter reading -> flow rate via the graph-#1 calibration.
    slope_cal, intercept_cal = cal["slope"], cal["intercept"]
    q_calc = slope_cal * reading + intercept_cal
    y_err = (abs(slope_cal) * READING_ERR) if READING_ERR is not None else None

    # X: √(P0/Pi) per the chosen convention, with propagated error.
    if RATIO_MODE == "ref_over_p":
        ratio = P_REF_BAR / p_abs
        xlabel = r"$\sqrt{P_{ref}/P_i}$  [-]"
    elif RATIO_MODE == "p_over_ref":
        ratio = p_abs / P_REF_BAR
        xlabel = r"$\sqrt{P_i/P_{ref}}$  [-]"
    else:
        raise ValueError(f"Unknown RATIO_MODE: {RATIO_MODE!r}")
    x = np.sqrt(ratio)
    # Gauge-pressure reading error: use the documented Setup!C15 unless overridden.
    # (A gauge offset is additive, so Δ(P_abs) = Δ(P_gauge).)
    p_err_bar = PRESSURE_ERR_BAR
    if p_err_bar is None:
        p_err_bar = fc.read_cell(wb["Setup"], "C15")
        if not np.isfinite(p_err_bar):
            p_err_bar = None
    # dX = 0.5 * X * (ΔP/P)   (same relative form for either orientation)
    x_err = 0.5 * x * (p_err_bar / p_abs) if p_err_bar else None

    fig, ax = plt.subplots()
    fc.errorbar_series(ax, x, q_calc, xerr=x_err, yerr=y_err,
                       color=fc.COLORS["real"], label="Calculated flow rate")

    a, b, _ = fc.fit_linear(x, q_calc)
    # χ² via effective variance: the X (pressure) error folds in through slope a.
    _, _, chi2_red = fc.chi_square(
        x, q_calc, lambda t: a * t + b,
        dmodel=lambda t: np.full_like(np.asarray(t, float), a),
        xerr=x_err, yerr=y_err, n_params=2)
    xs = np.linspace(float(np.min(x)) * 0.98, float(np.max(x)) * 1.02, 100)
    ax.plot(xs, a * xs + b, "-", color=fc.COLORS["fit"], lw=1.5,
            label=fc.fmt_eq(a, b, xsym="x", ysym="Q", unit="L/min", chi2_red=chi2_red))

    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"Calculated flow rate  $Q$  [L/min]")
    #ax.set_title(TITLE)
    ax.legend(loc="best")
    gauge_note = f"gauge+{ATM_BAR}" if PRESSURE_IS_GAUGE else "absolute"
    # ax.text(0.99, 0.02,
    #         f"P={gauge_note} bar, {RATIO_MODE}, P_ref={P_REF_BAR} bar_a  "
    #         f"— verify vs. manual §4.f",
    #         transform=ax.transAxes, ha="right", va="bottom",
    #         fontsize=7.5, color="#666666")

    paths = fc.save_fig(fig, NAME)
    print(f"[{NAME}] slope={a:.4g}  intercept={b:.4g}  chi2_nu={chi2_red:.3g}  "
          f"-> {paths[0].name}")
    return {"figure": NAME, "status": "ok", "slope": a, "intercept": b,
            "chi2_red": chi2_red, "ratio_mode": RATIO_MODE, "p_ref_bar": P_REF_BAR}


if __name__ == "__main__":
    main()
