"""
Graphs #2-#4 — Flow-rate calibration (Venturi, Diaphragm, Nozzle).

Manual §4.a.  One graph per instrument (three figures total).  X = pressure
difference ΔP, Y = flow rate.  Each graph shows TWO series:
    * ideal flow rate  Q_ideal  (theory)  with its own error,
    * actual/measured flow rate  Q_real (rotameter) with its own error.

Both curves get a √ΔP trendline (Q ∝ √ΔP — the physically expected shape, not
a straight line).  Error bars are drawn on both axes; because the ΔP error is
usually tiny on the full scale, a zoomed inset of the high-ΔP points is added
so those error bars are visible (manual requirement; ref. report figs. 8-10).

Data:  sheet 'Venturi' | 'Diaphragm' | 'Nozzle', rows 4-13
        X      = E (ΔP [Pa])         X err = F
        ideal  = G (Q_ideal [L/min]) err   = H
        real   = I (Q_real  [L/min]) err   = J
"""

from __future__ import annotations

import flow_common as fc

# ----------------------------- CONFIG -------------------------------------
INSTRUMENTS = [
    # (sheet, output name, nice label)
    ("Venturi",   "fig02_venturi_calibration",   "Venturi"),
    ("Diaphragm", "fig03_diaphragm_calibration", "Diaphragm"),
    ("Nozzle",    "fig04_nozzle_calibration",    "Nozzle"),
]
ROWS = fc.INSTRUMENT_ROWS  # (4, 13)
COL = dict(dP="E", dP_err="F", q_ideal="G", q_ideal_err="H",
           q_real="I", q_real_err="J")

# Inset: how many of the highest-ΔP points to frame, plus padding (fraction).
INSET_N_POINTS = 3
INSET_PAD = 0.06
SHOW_INSET = False  # True to show the zoomed-in ΔP error bars (manual requirement)
# --------------------------------------------------------------------------


def _read(ws):
    d = {k: fc.read_column(ws, c, *ROWS) for k, c in COL.items()}
    return d


def _plot_instrument(ax, d, inset_axes_target=None):
    """Draw both series + both √ΔP trendlines on ``ax``."""
    import numpy as np

    m = fc.finite_mask(d["dP"], d["q_ideal"], d["q_real"])
    x = d["dP"][m]
    xerr = d["dP_err"][m]
    qi, qi_e = d["q_ideal"][m], d["q_ideal_err"][m]
    qr, qr_e = d["q_real"][m], d["q_real_err"][m]

    fc.errorbar_series(ax, x, qi, xerr=xerr, yerr=qi_e,
                       color=fc.COLORS["ideal"], label="Ideal  $Q_{ideal}$")
    fc.errorbar_series(ax, x, qr, xerr=xerr, yerr=qr_e, fmt="s",
                       color=fc.COLORS["real"], label="Measured  $Q_{real}$")

    xs = fc.smooth_x(x)
    sqrt = np.sqrt
    a_i, _ = fc.fit_sqrt(x, qi)
    a_r, _ = fc.fit_sqrt(x, qr)
    # χ² with the local slope d(a√ΔP)/dΔP = a/(2√ΔP) folding in the ΔP error.
    _, _, chi2_i = fc.chi_square(x, qi, lambda t: a_i * sqrt(t),
                                 dmodel=lambda t: a_i / (2 * sqrt(t)),
                                 xerr=xerr, yerr=qi_e, n_params=1)
    _, _, chi2_r = fc.chi_square(x, qr, lambda t: a_r * sqrt(t),
                                 dmodel=lambda t: a_r / (2 * sqrt(t)),
                                 xerr=xerr, yerr=qr_e, n_params=1)
    ax.plot(xs, a_i * sqrt(xs), "-", color=fc.COLORS["ideal"], lw=1.4,
            label=rf"$Q_{{ideal}}={a_i:.3g}\sqrt{{\Delta P}}$  ($\chi^2_\nu$={chi2_i:.2f})")
    ax.plot(xs, a_r * sqrt(xs), "--", color=fc.COLORS["real"], lw=1.4,
            label=rf"$Q_{{real}}={a_r:.3g}\sqrt{{\Delta P}}$  ($\chi^2_\nu$={chi2_r:.2f})")
    return dict(x=x, xerr=xerr, qi=qi, qi_e=qi_e, qr=qr, qr_e=qr_e,
                a_i=a_i, a_r=a_r, chi2_i=chi2_i, chi2_r=chi2_r)


def main(wb=None):
    wb = wb or fc.load_workbook()
    fc.apply_style()
    import matplotlib.pyplot as plt
    import numpy as np

    results = {}
    for sheet, name, label in INSTRUMENTS:
        ws = wb[sheet]
        d = _read(ws)
        title = f"Figure {name[3:5]} — {label}: flow-rate calibration"
        if not fc.has_data(d["dP"], d["q_real"]):
            fc.placeholder(name, title)
            print(f"[{name}] no data yet -> placeholder written")
            results[label] = {"figure": name, "status": "no-data"}
            continue

        fig, ax = plt.subplots()
        info = _plot_instrument(ax, d)
        ax.set_xlabel(r"Pressure difference  $\Delta P$  [Pa]")
        ax.set_ylabel(r"Flow rate  $Q$  [L/min]")
        #ax.set_title(title)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.legend(loc="upper left", fontsize=9)
# 
        # Zoom inset on the highest-ΔP points so tiny X-errors are visible.
        if SHOW_INSET and info["x"].size >= INSET_N_POINTS:
            order = np.argsort(info["x"])
            sel = order[-INSET_N_POINTS:]
            xlo, xhi = info["x"][sel].min(), info["x"][sel].max()
            span = max(xhi - xlo, 1e-9)
            xlim = (xlo - INSET_PAD * span, xhi + INSET_PAD * span)
            yvals = np.concatenate([info["qi"][sel], info["qr"][sel]])
            ypad = 0.10 * (yvals.max() - yvals.min() + 1e-9)
            ylim = (yvals.min() - ypad, yvals.max() + ypad)

            def draw(axins, info=info):
                fc.errorbar_series(axins, info["x"], info["qi"], xerr=info["xerr"],
                                   yerr=info["qi_e"], color=fc.COLORS["ideal"],
                                   label=None, markersize=5)
                fc.errorbar_series(axins, info["x"], info["qr"], xerr=info["xerr"],
                                   yerr=info["qr_e"], fmt="s", color=fc.COLORS["real"],
                                   label=None, markersize=5)
                xs = np.linspace(*xlim, 50)
                axins.plot(xs, info["a_i"] * np.sqrt(xs), "-",
                           color=fc.COLORS["ideal"], lw=1.2)
                axins.plot(xs, info["a_r"] * np.sqrt(xs), "--",
                           color=fc.COLORS["real"], lw=1.2)

            fc.add_zoom_inset(ax, draw, xlim=xlim, ylim=ylim)

        paths = fc.save_fig(fig, name)
        print(f"[{name}] a_ideal={info['a_i']:.4g}  a_real={info['a_r']:.4g}  "
              f"-> {paths[0].name}")
        results[label] = {"figure": name, "status": "ok",
                          "a_ideal": info["a_i"], "a_real": info["a_r"],
                          "chi2_ideal": info["chi2_i"], "chi2_real": info["chi2_r"]}
    return results


if __name__ == "__main__":
    main()
