"""
Graph #5 — Velocity calibration (Pitot tube).

Manual §4.b.  X = √ΔP, Y = velocity.  THREE curves on one graph:
    1. u Bernoulli  — local point velocity            (col G, err H)
    2. u 1/7-profile — cross-section average           (col I, err J)
    3. u rotameter   — cross-section average from Q     (col L, err M)

Under the Bernoulli approximation u ∝ √ΔP, so each series gets a LINEAR
trendline through the origin (f(x)=a·x) with the equation shown.  A zoomed
inset of the high-range segment makes the separation between the three curves
and their error bars visible (manual requirement; ref. report figs. 11-12).

Data:  sheet 'Pitot', rows 4-13
        X = E (√ΔP [√Pa])   X err = F
"""

from __future__ import annotations

import flow_common as fc

# ----------------------------- CONFIG -------------------------------------
SHEET = "Pitot"
ROWS = fc.PITOT_ROWS  # (4, 13)
COL = dict(sqrt_dP="E", sqrt_dP_err="F",
           u_bern="G", u_bern_err="H",
           u_prof="I", u_prof_err="J",
           u_rota="L", u_rota_err="M")
SERIES = [
    # (key, err_key, color, label, marker)
    ("u_bern", "u_bern_err", fc.COLORS["bernoulli"], "Bernoulli (point)", "o"),
    ("u_prof", "u_prof_err", fc.COLORS["profile17"], "1/7 profile (avg)", "s"),
    ("u_rota", "u_rota_err", fc.COLORS["rotameter"], "Rotameter (avg)", "^"),
]
INSET_N_POINTS = 3
INSET_PAD = 0.06
SHOW_INSET = False
NAME = "fig05_pitot_velocity"
TITLE = "Figure 5 — Pitot tube: velocity calibration"
# --------------------------------------------------------------------------


def main(wb=None):
    wb = wb or fc.load_workbook()
    d = {k: fc.read_column(wb[SHEET], c, *ROWS) for k, c in COL.items()}

    if not fc.has_data(d["sqrt_dP"], d["u_bern"]):
        fc.placeholder(NAME, TITLE)
        print(f"[{NAME}] no data yet -> placeholder written")
        return {"figure": NAME, "status": "no-data"}

    fc.apply_style()
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots()
    fits = {}
    drawn = []  # (x, y, xerr, yerr, color, marker) for the inset
    for key, errkey, color, label, marker in SERIES:
        m = fc.finite_mask(d["sqrt_dP"], d[key])
        x, y = d["sqrt_dP"][m], d[key][m]
        xerr, yerr = d["sqrt_dP_err"][m], d[errkey][m]
        fc.errorbar_series(ax, x, y, xerr=xerr, yerr=yerr, fmt=marker,
                           color=color, label=label)
        a, _ = fc.fit_through_origin(x, y)
        _, _, chi2 = fc.chi_square(
            x, y, lambda t: a * t,
            dmodel=lambda t: np.full_like(np.asarray(t, float), a),
            xerr=xerr, yerr=yerr, n_params=1)
        fits[key] = (a, chi2)
        xs = fc.smooth_x(x)
        ax.plot(xs, a * xs, "-", color=color, lw=1.3,
                label=rf"  fit: u = {a:.3g}·√ΔP  ($\chi^2_\nu$={chi2:.2f})")
        drawn.append((x, y, xerr, yerr, color, marker, a))

    ax.set_xlabel(r"$\sqrt{\Delta P}$  [$\sqrt{\mathrm{Pa}}$]")
    ax.set_ylabel(r"Velocity  $u$  [m/s]")
    #ax.set_title(TITLE)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", fontsize=8.5, ncol=1)

    if SHOW_INSET and drawn[0][0].size >= INSET_N_POINTS:
        xall = drawn[0][0]
        order = np.argsort(xall)
        sel = order[-INSET_N_POINTS:]
        xlo, xhi = xall[sel].min(), xall[sel].max()
        span = max(xhi - xlo, 1e-9)
        xlim = (xlo - INSET_PAD * span, xhi + INSET_PAD * span)
        yvals = np.concatenate([s[1][sel] for s in drawn if s[0].size == xall.size])
        ypad = 0.10 * (yvals.max() - yvals.min() + 1e-9)
        ylim = (yvals.min() - ypad, yvals.max() + ypad)

        def draw(axins):
            xs = np.linspace(*xlim, 50)
            for x, y, xerr, yerr, color, marker, a in drawn:
                fc.errorbar_series(axins, x, y, xerr=xerr, yerr=yerr, fmt=marker,
                                   color=color, label=None, markersize=5)
                axins.plot(xs, a * xs, "-", color=color, lw=1.1)

        fc.add_zoom_inset(ax, draw, xlim=xlim, ylim=ylim)

    paths = fc.save_fig(fig, NAME)
    summary = {k: {"slope": v[0], "chi2_red": v[1]} for k, v in fits.items()}
    print(f"[{NAME}] slopes: " +
          ", ".join(f"{k}={v[0]:.4g}" for k, v in fits.items()) +
          f"  -> {paths[0].name}")
    return {"figure": NAME, "status": "ok", "fits": summary}


if __name__ == "__main__":
    main()
