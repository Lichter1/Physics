"""
Graph #9 — Actual flow rate vs. ideal flow rate (all three instruments).

Manual §4.d.  ONE graph, three point series:  Q_real = f(Q_ideal) for Venturi,
Diaphragm and Nozzle.  A linear regression (through the origin by default) is
fitted to each series; the SLOPE of each line IS that instrument's discharge
coefficient C (ref. report fig. 13).  The three slopes are returned so they can
be quoted in the discussion/conclusions.

Data:  sheet 'Summary_QvQ'
        Venturi   : X = C (Qideal), Y = D (Qreal)
        Diaphragm : X = F (Qideal), Y = G (Qreal)
        Nozzle    : X = I (Qideal), Y = J (Qreal)
Error bars come from the source sheets' Q_ideal_err (H) and Q_real_err (J).
"""

from __future__ import annotations

import flow_common as fc

# ----------------------------- CONFIG -------------------------------------
SUMMARY_SHEET = "Summary_QvQ"
ROWS = fc.INSTRUMENT_ROWS  # (4, 13)

# Each instrument: where Qideal/Qreal sit in Summary_QvQ, and where the matching
# errors sit in the instrument's own sheet (cols H = Qideal_err, J = Qreal_err).
SERIES = [
    # (label, x_col, y_col, source_sheet, color, marker)
    ("Venturi",   "C", "D", "Venturi",   fc.COLORS["venturi"],   "o"),
    ("Diaphragm", "F", "G", "Diaphragm", fc.COLORS["diaphragm"], "s"),
    ("Nozzle",    "I", "J", "Nozzle",    fc.COLORS["nozzle"],    "^"),
]
SRC_ERR = dict(q_ideal_err="H", q_real_err="J")

THROUGH_ORIGIN = True   # slope = C is the physical meaning -> origin by default
NAME = "fig09_qreal_vs_qideal"
TITLE = "Figure 9 — Actual vs. ideal flow rate (slope = discharge coefficient C)"
# --------------------------------------------------------------------------


def main(wb=None):
    wb = wb or fc.load_workbook()
    ws = wb[SUMMARY_SHEET]

    # Check at least one series has data.
    any_data = False
    for _, xc, yc, *_ in SERIES:
        if fc.has_data(fc.read_column(ws, xc, *ROWS), fc.read_column(ws, yc, *ROWS)):
            any_data = True
            break
    if not any_data:
        fc.placeholder(NAME, TITLE)
        print(f"[{NAME}] no data yet -> placeholder written")
        return {"figure": NAME, "status": "no-data"}

    fc.apply_style()
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots()
    slopes = {}
    xmax = 0.0
    for label, xc, yc, src, color, marker in SERIES:
        x = fc.read_column(ws, xc, *ROWS)
        y = fc.read_column(ws, yc, *ROWS)
        xe = fc.read_column(wb[src], SRC_ERR["q_ideal_err"], *ROWS)
        ye = fc.read_column(wb[src], SRC_ERR["q_real_err"], *ROWS)
        m = fc.finite_mask(x, y)
        if not m.any():
            continue
        x, y, xe, ye = x[m], y[m], xe[m], ye[m]
        xmax = max(xmax, float(np.nanmax(x)))
        fc.errorbar_series(ax, x, y, xerr=xe, yerr=ye, fmt=marker,
                           color=color, label=label)
        if THROUGH_ORIGIN:
            a, _ = fc.fit_through_origin(x, y)
            b, n_params = 0.0, 1
        else:
            a, b, _ = fc.fit_linear(x, y)
            n_params = 2
        _, _, chi2_red = fc.chi_square(
            x, y, lambda t: a * t + b,
            dmodel=lambda t: np.full_like(np.asarray(t, float), a),
            xerr=xe, yerr=ye, n_params=n_params)
        slopes[label] = {"C": a, "intercept": b, "chi2_red": chi2_red,
                         "n": int(m.sum())}
        xs = np.linspace(0, float(np.nanmax(x)), 100)
        ax.plot(xs, a * xs + b, "-", color=color, lw=1.3,
                label=rf"  C$_{{{label[0]}}}$ = {a:.4g}  ($\chi^2_\nu$={chi2_red:.2f})")

    # 1:1 reference line (C = 1) for context.
    if xmax > 0:
        ax.plot([0, xmax], [0, xmax], ":", color="#999999", lw=1.0,
                label="1:1  (C = 1)")

    ax.set_xlabel(r"Ideal flow rate  $Q_{ideal}$  [L/min]")
    ax.set_ylabel(r"Actual flow rate  $Q_{real}$  [L/min]")
    #ax.set_title(TITLE)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", fontsize=9)

    paths = fc.save_fig(fig, NAME)
    print(f"[{NAME}] slopes(C): " +
          ", ".join(f"{k}={v['C']:.4g}" for k, v in slopes.items()) +
          f"  -> {paths[0].name}")
    return {"figure": NAME, "status": "ok", "slopes": slopes,
            "through_origin": THROUGH_ORIGIN}


if __name__ == "__main__":
    main()
