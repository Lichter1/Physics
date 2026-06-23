"""
Graph #1 — Rotameter calibration (atmospheric conditions).

Manual §3.5.1 part 1.  Rotameter float height Δh (X) vs. mass-flow-meter
flow rate (Y), 10 points.  A linear trendline gives the calibration equation
that converts future rotameter readings (in cm) into a flow rate (used again
by graph #10).

Data:  sheet 'Rotameter', rows 4-13
        X = column E (rotameter reading [cm])
        Y = column D (mass flow meter [L/min])

Run:    python fig01_rotameter_calibration.py
Returns (when imported) a dict with the fitted calibration coefficients.
"""

from __future__ import annotations

import flow_common as fc

# ----------------------------- CONFIG -------------------------------------
SHEET = "Rotameter"
ROWS = fc.ROTAMETER_CAL_ROWS          # (4, 13)
COL_READING = "E"                     # X — rotameter float height Δh [cm]
COL_FLOW = "D"                        # Y — mass flow meter [L/min]

THROUGH_ORIGIN = True                # True -> force calibration through (0,0)
X_READING_ERR = 0.05                  # set a number (cm) to show X error bars
NAME = "fig01_rotameter_calibration"
TITLE = "Figure 1 — Rotameter calibration (atmospheric)"
XLABEL = r"Rotameter reading  $\Delta h$  [cm]"
YLABEL = "Flow rate  Q  [L/min]"
# --------------------------------------------------------------------------


def get_calibration(wb=None):
    """Read the part-1 points and return the fit, or None if no data yet.

    Shared with graph #10, which needs this calibration equation.
    Returns a dict: x, y, slope, intercept, r2, through_origin.
    """
    wb = wb or fc.load_workbook()
    ws = wb[SHEET]
    x = fc.read_column(ws, COL_READING, *ROWS)
    y = fc.read_column(ws, COL_FLOW, *ROWS)
    m = fc.finite_mask(x, y)
    if not m.any():
        return None
    x, y = x[m], y[m]
    if THROUGH_ORIGIN:
        slope, r2 = fc.fit_through_origin(x, y)
        intercept = 0.0
    else:
        slope, intercept, r2 = fc.fit_linear(x, y)
    return dict(x=x, y=y, slope=slope, intercept=intercept, r2=r2,
                through_origin=THROUGH_ORIGIN)


def main(wb=None):
    wb = wb or fc.load_workbook()
    cal = get_calibration(wb)
    if cal is None:
        fc.placeholder(NAME, TITLE)
        print(f"[{NAME}] no data yet -> placeholder written")
        return {"figure": NAME, "status": "no-data"}

    fc.apply_style()
    import matplotlib.pyplot as plt
    import numpy as np

    yerr = fc.ldn_error(wb, cal["y"])   # LDN 1009 datasheet error on the flow Y
    fig, ax = plt.subplots()
    fc.errorbar_series(
        ax, cal["x"], cal["y"],
        xerr=X_READING_ERR, yerr=yerr,
        color=fc.COLORS["real"], label="Measured points",
        markersize=5
    )

    slope, intercept0 = cal["slope"], cal["intercept"]
    n_params = 1 if cal["through_origin"] else 2
    _, _, chi2_red = fc.chi_square(
        cal["x"], cal["y"], lambda x: slope * x + intercept0,
        dmodel=lambda x: np.full_like(np.asarray(x, float), slope),
        xerr=X_READING_ERR, yerr=yerr, n_params=n_params)

    xline = fc.smooth_x(cal["x"])
    yline = slope * xline + intercept0
    intercept = None if cal["through_origin"] else intercept0
    ax.plot(xline, yline, "-", color=fc.COLORS["fit"], lw=1.6,
            label=fc.fmt_eq(slope, intercept, xsym="h", ysym="Q",
                            unit="L/min", chi2_red=chi2_red))

    # Zoom inset centred on the middle point to make X-error bars visible.
    mid = len(cal["x"]) // 3
    xc, yc = cal["x"][mid], cal["y"][mid]
    xpad = max(X_READING_ERR * 6, (cal["x"].max() - cal["x"].min()) * 0.04)
    ypad_arr = np.asarray(yerr) if np.ndim(yerr) > 0 else np.full(len(cal["y"]), float(yerr))
    ypad =  (cal["y"].max() - cal["y"].min()) * 0.04
    xlim_ins = (xc - xpad, xc + xpad)
    ylim_ins = (yc - ypad, yc + ypad)

    def draw_inset(axins):
        fc.errorbar_series(
            axins, cal["x"], cal["y"],
            xerr=X_READING_ERR, yerr=yerr,
            color=fc.COLORS["real"], label=None, markersize=4,
        )
        xs = fc.smooth_x(cal["x"])
        axins.plot(xs, slope * xs + intercept0, "-", color=fc.COLORS["fit"], lw=1.2)

    fc.add_zoom_inset(ax, draw_inset, xlim=xlim_ins, ylim=ylim_ins,
                      bounds=(0.6, 0.05, 0.30, 0.30))

    ax.set_xlabel(XLABEL)
    ax.set_ylabel(YLABEL)
    #ax.set_title(TITLE)
    ax.legend(loc="best")
    paths = fc.save_fig(fig, NAME)
    print(f"[{NAME}] slope={slope:.4g}  intercept={intercept0:.4g}  "
          f"chi2_nu={chi2_red:.3g}  -> {paths[0].name}")
    return {"figure": NAME, "status": "ok", "slope": slope,
            "intercept": intercept0, "chi2_red": chi2_red}


if __name__ == "__main__":
    main()
