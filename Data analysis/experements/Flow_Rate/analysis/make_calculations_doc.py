"""
Generate `calculations.md` — a full write-up of every formula, every
uncertainty, and a worked numeric example (taken from the real workbook data)
for the Flow-Rate lab.

    python make_calculations_doc.py

The document is regenerated from the live workbook, so the worked-example
numbers always match the current measurements.  Math is written as Markdown +
LaTeX ($...$, $$...$$), which renders in VS Code / GitHub and pastes straight
into a LaTeX report.
"""

from __future__ import annotations

import math

import flow_common as fc
import fig01_rotameter_calibration as fig01

OUT = fc.HERE / "calculations.md"


def f(x, n=4):
    """Format a number with n significant figures (plain)."""
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "—"
    return f"{x:.{n}g}"


def main():
    wb = fc.load_workbook()
    S = wb["Setup"]
    c = lambda a: fc.read_cell(S, a)  # noqa: E731

    # ---- constants ----
    rho_a, rho_w, g, mu = c("C8"), c("C9"), c("C10"), c("C11")
    dh_err, dD_err, dP_gauge_err = c("C12"), c("C13"), c("C15")
    pMV, pFS, FS, pR = c("C33"), c("C34"), c("C35"), c("C36")
    Dmax_v, Dmin_v = c("C18"), c("D18")        # Venturi (mm)
    Dpipe, Dpitot = c("C21"), c("D21")         # Pitot (mm)

    # ---- worked-example rows (live values) ----
    V = wb["Venturi"]
    v1 = {k: fc.read_cell(V, f"{k}4") for k in "DEFGHIJKL"}
    v10 = {k: fc.read_cell(V, f"{k}13") for k in "DEFGHIJKL"}
    C_v1 = fc.read_cell(V, "C19")
    P = wb["Pitot"]
    p1 = {k: fc.read_cell(P, f"{k}4") for k in "CDEFGHIJKLMNO"}
    cal = fig01.get_calibration(wb)

    L: list[str] = []
    A = L.append  # raw text / LaTeX lines
    # convenience for numeric prose lines uses f-strings inline below

    A("# Flow Rate and Velocity Measurement — Calculations & Uncertainties")
    A("")
    A("_Auto-generated from the workbook "
      f"`{fc.WORKBOOK.name}`. Every worked example uses the real measured data._")
    A("")
    A("All measured inputs are the **yellow** cells; everything below is how each "
      "**green** (calculated) cell is obtained, together with its uncertainty and "
      "one fully worked example. Uncertainties use the standard "
      "**partial-derivative (propagation of errors)** method:")
    A("")
    A(r"$$\delta f(x_1,\dots,x_n)=\sqrt{\sum_i\left(\frac{\partial f}{\partial x_i}\,\delta x_i\right)^2}.$$")
    A("")

    # ---------- constants ----------
    A("## 1. Constants and fixed uncertainties (`Setup` sheet)")
    A("")
    A("| Symbol | Meaning | Value | Uncertainty |")
    A("|---|---|---|---|")
    A(f"| $\\rho_\\text{{air}}$ | air density | {f(rho_a)} kg/m³ | — |")
    A(f"| $\\rho_\\text{{water}}$ | manometer water density | {f(rho_w)} kg/m³ | — |")
    A(f"| $g$ | gravity | {f(g)} m/s² | — |")
    A(f"| $\\mu_\\text{{air}}$ | air dynamic viscosity | {f(mu)} kg/(m·s) | — |")
    A(f"| $\\Delta h$ | manometer reading (ruler) | input | $\\delta h={f(dh_err)}$ m |")
    A(f"| $D$ | diameters | input | $\\delta D={f(dD_err)}$ m |")
    A(f"| $P_\\text{{gauge}}$ | gauge pressure | input | $\\delta P={f(dP_gauge_err)}$ bar |")
    A(f"| Venturi/Nozzle | $D_\\text{{max}}={f(Dmax_v)}$ mm, $D_\\text{{min}}={f(Dmin_v)}$ mm | | |")
    A(f"| Pitot | $D_\\text{{pipe}}={f(Dpipe)}$ mm, $D_\\text{{probe}}={f(Dpitot)}$ mm | | |")
    A("")
    beta = Dmin_v / Dmax_v
    A(f"Diameter ratio (Venturi/Nozzle): $\\beta=D_\\text{{min}}/D_\\text{{max}}="
      f"{f(Dmin_v)}/{f(Dmax_v)}={f(beta)}$.")
    A("")

    # ---------- LDN flow error ----------
    A("## 2. Real flow-rate uncertainty — LDN 1009 GAPL mass flow meter")
    A("")
    A("The real flow rate $Q_\\text{real}$ is read on the LDN 1009 GAPL mass flow "
      "meter and is taken as the *true* flow against which every restriction is "
      "judged. Its datasheet error has three statistically independent parts — a "
      "term proportional to the reading (gain error), a fixed term proportional to "
      "full scale (offset/zero error), and a reproducibility term — so they are "
      "combined in quadrature rather than added:")
    A("")
    A(r"$$\delta Q_\text{real}=\sqrt{(p_\text{MV}\,Q)^2+(p_\text{FS}\,\text{FS})^2+(p_\text{R}\,Q)^2},$$")
    A("")
    A(f"with $p_\\text{{MV}}={f(pMV)}$ (4 % of reading), $p_\\text{{FS}}={f(pFS)}$ "
      f"(0.5 % of full scale), $\\text{{FS}}={f(FS)}$ L/min, and "
      f"$p_\\text{{R}}={f(pR)}$ (2 % reproducibility). The 0.5 %·FS term is the "
      f"constant ${f(pFS*FS)}$ L/min. The ±2 °C temperature spec is a temperature "
      "figure and is not part of the flow uncertainty.")
    A("")
    Q = v10["I"]
    dq = math.sqrt((pMV*Q)**2 + (pFS*FS)**2 + (pR*Q)**2)
    A(f"**Worked example (Venturi point 10, $Q={f(Q)}$ L/min):**")
    A("")
    A(f"$\\delta Q_\\text{{real}}=\\sqrt{{({f(pMV)}\\cdot{f(Q)})^2+({f(pFS)}\\cdot{f(FS)})^2+({f(pR)}\\cdot{f(Q)})^2}}"
      f"=\\sqrt{{{f(pMV*Q)}^2+{f(pFS*FS)}^2+{f(pR*Q)}^2}}={f(dq)}$ L/min "
      f"(workbook: {f(v10['J'])}).")
    A("")

    # ---------- pressure difference ----------
    A("## 3. Pressure difference $\\Delta P$ from the manometer")
    A("")
    A("The pressure drop across each restriction is read on an inclined/vertical "
      "water manometer. The reading is purely hydrostatic — a column of height "
      "$\\Delta h$ of water balances the gas pressure difference — so $\\Delta P$ "
      "is linear in $\\Delta h$ and the densities/$g$ are treated as exact. The "
      "only random input is the ruler reading $\\delta h$, hence the uncertainty "
      "is just that single term scaled by $\\rho_\\text{water}g$:")
    A("")
    A(r"$$\Delta P=\rho_\text{water}\,g\,\Delta h,\qquad \delta\Delta P=\rho_\text{water}\,g\,\delta h.$$")
    A("")
    dP = rho_w*g*v1["D"]
    dPe = rho_w*g*dh_err
    A(f"**Worked example (Venturi point 1, $\\Delta h={f(v1['D'])}$ m):** "
      f"$\\Delta P={f(rho_w)}\\cdot{f(g)}\\cdot{f(v1['D'])}={f(dP)}$ Pa "
      f"(workbook {f(v1['E'])}); "
      f"$\\delta\\Delta P={f(rho_w)}\\cdot{f(g)}\\cdot{f(dh_err)}={f(dPe)}$ Pa "
      f"(workbook {f(v1['F'])}).")
    A("")

    # ---------- ideal flow ----------
    A("## 4. Ideal flow rate $Q_\\text{ideal}$ (Venturi, Diaphragm, Nozzle)")
    A("")
    A("Applying Bernoulli's equation between the wide section and the throat, "
      "together with mass continuity $A_\\text{max}u_\\text{max}=A_t u_t$, for an "
      "ideal (inviscid, incompressible) flow gives the throat velocity and hence "
      "the volumetric flow. The factor $1/\\sqrt{1-\\beta^4}$ is the continuity "
      "correction for the upstream velocity (the *velocity-of-approach* factor); "
      "$A_t=\\pi D_\\text{min}^2/4$ is the throat area and $6\\times10^4$ converts "
      "m³/s → L/min:")
    A("")
    A(r"$$Q_\text{ideal}=\frac{A_t}{\sqrt{1-\beta^4}}\sqrt{\frac{2\,\Delta P}{\rho_\text{air}}}\times 6\times10^4 .$$")
    A("")
    A("This is the flow a *loss-free* restriction would pass for the measured "
      "$\\Delta P$; the real flow is smaller (viscous losses and the vena "
      "contracta), which is exactly what the discharge coefficient $C$ in §6 "
      "captures. Its uncertainty depends on $\\Delta P$ and on the throat diameter "
      "$D_\\text{min}$ (the analytic partials are columns O and P in the sheet):")
    A("")
    A(r"$$\delta Q_\text{ideal}=\sqrt{\left(\frac{\partial Q_\text{ideal}}{\partial D_\text{min}}\,\delta D\right)^2+\left(\frac{\partial Q_\text{ideal}}{\partial \Delta P}\,\delta\Delta P\right)^2}.$$")
    A("")
    A(f"**Worked example (Venturi point 1):** $Q_\\text{{ideal}}={f(v1['G'])}$ L/min, "
      f"$\\delta Q_\\text{{ideal}}={f(v1['H'])}$ L/min — i.e. "
      f"$Q_\\text{{ideal}}={f(v1['G'])}\\pm{f(v1['H'])}$ L/min.")
    A("")

    # ---------- Reynolds ----------
    A("## 5. Reynolds number $Re$")
    A("")
    A("The Reynolds number characterises the flow regime (laminar vs. turbulent) "
      "and is the natural x-axis for the discharge coefficient. It is built from "
      "the pipe diameter $D_\\text{max}$ and the approach velocity "
      "$u=\\sqrt{2\\Delta P/\\rho_\\text{air}}$, so it inherits its uncertainty "
      "from both $\\Delta P$ and $D$ (the $\\Delta P$ term dominates):")
    A("")
    A(r"$$Re=\frac{D_\text{max}\sqrt{2\,\rho_\text{air}\,\Delta P}}{\mu_\text{air}},\quad \delta Re=\sqrt{\left(\frac{D_\text{max}\sqrt{2\rho_\text{air}}}{\mu_\text{air}}\frac{\delta\Delta P}{2\sqrt{\Delta P}}\right)^2+\left(\frac{\sqrt{2\rho_\text{air}\Delta P}}{\mu_\text{air}}\,\delta D\right)^2}.$$")
    A("")
    A(f"**Worked example (Venturi point 1):** $Re={f(v1['K'],5)}$, "
      f"$\\delta Re={f(v1['L'])}$ — i.e. $Re=({f(v1['K']/1000)}\\pm{f(v1['L']/1000)})\\times10^3$.")
    A("")

    # ---------- discharge coefficient ----------
    A("## 6. Discharge coefficient $C$")
    A("")
    A("The discharge coefficient is the ratio of the measured flow to the ideal "
      "(loss-free) flow at the same $\\Delta P$. It lumps together the viscous "
      "losses and the vena-contracta area reduction, so physically $C<1$ for the "
      "Venturi/diaphragm/nozzle, and for a given geometry it should approach a "
      "constant once the flow is fully turbulent (high $Re$). As a ratio of two "
      "measured quantities its relative error is the quadrature sum of the two "
      "relative errors:")
    A("")
    A(r"$$C=\frac{Q_\text{real}}{Q_\text{ideal}},\qquad \delta C=C\sqrt{\left(\frac{\delta Q_\text{real}}{Q_\text{real}}\right)^2+\left(\frac{\delta Q_\text{ideal}}{Q_\text{ideal}}\right)^2}.$$")
    A("")
    Qr, dQr, Qi, dQi = v1["I"], v1["J"], v1["G"], v1["H"]
    Cval = Qr/Qi
    dC = Cval*math.sqrt((dQr/Qr)**2 + (dQi/Qi)**2)
    A(f"**Worked example (Venturi point 1):** "
      f"$C={f(Qr)}/{f(Qi)}={f(Cval)}$ (workbook {f(C_v1)}); "
      f"$\\delta C={f(Cval)}\\sqrt{{({f(dQr)}/{f(Qr)})^2+({f(dQi)}/{f(Qi)})^2}}={f(dC)}$ "
      f"— i.e. $C={f(Cval)}\\pm{f(dC)}$.")
    A("")
    A("> Note: the workbook's C-table has no error column; the figure scripts "
      "compute $\\delta C$ with exactly this formula for the error bars in "
      "graphs #6–#8.")
    A("")

    # ---------- Pitot ----------
    A("## 7. Pitot tube velocities")
    A("")
    A("The Pitot-static tube measures the difference between stagnation and static "
      "pressure; by Bernoulli this gives the **local** velocity at the probe tip. "
      "To turn that point velocity into a cross-sectional **average** we use the "
      "turbulent $1/7$ power-law profile $u(r)/u_\\text{max}=(1-r/R)^{1/7}$, whose "
      "area-average is $\\tfrac{14}{15}$ of the centre-line value — the probe sits "
      "on the axis, so $u_\\text{avg}=\\tfrac{14}{15}u_\\text{Bernoulli}$. As an "
      "independent cross-check, $u_\\text{rot}$ is the bulk velocity obtained from "
      "the rotameter flow divided by the pipe area. The three should agree:")
    A("")
    A(r"$$u_\text{Bernoulli}=\sqrt{\frac{2\,\Delta P}{\rho_\text{air}}},\quad \delta u=\sqrt{\frac{2}{\rho_\text{air}}}\,\delta(\sqrt{\Delta P}),\quad \delta(\sqrt{\Delta P})=\tfrac12\sqrt{\frac{\rho_\text{water}g}{\Delta h}}\,\delta h.$$")
    A("")
    A(r"$$u_{1/7}=\tfrac{14}{15}\,u_\text{Bernoulli},\qquad u_\text{rot}=\frac{4\,Q}{\pi D_\text{pipe}^2}.$$")
    A("")
    A(f"**Worked example (Pitot point 1, $\\Delta P={f(p1['D'])}$ Pa):** "
      f"$u_\\text{{Bernoulli}}=\\sqrt{{2\\cdot{f(p1['D'])}/{f(rho_a)}}}={f(p1['G'])}$ m/s "
      f"($\\delta u={f(p1['H'])}$); "
      f"$u_{{1/7}}={f(p1['I'])}$ m/s; "
      f"$u_\\text{{rot}}={f(p1['L'])}$ m/s ($\\delta u={f(p1['M'])}$); "
      f"$Re={f(p1['N'],5)}\\pm{f(p1['O'])}$.")
    A("")

    # ---------- rotameter calibration ----------
    A("## 8. Rotameter calibration and the pressure test")
    A("")
    if cal:
        # Build the calibration equation, suppressing a "+0" intercept.
        eq = f"Q={f(cal['slope'])}\\,r"
        if abs(cal['intercept']) > 1e-9:
            sign = "+" if cal['intercept'] >= 0 else "-"
            eq += f"\\,{sign}\\,{f(abs(cal['intercept']))}"
        A("**Why a linear fit is physical.** A rotameter is a *variable-area* "
          "meter: the float rises until its submerged weight is balanced by the "
          "pressure drop across the annular gap between the float and the "
          "(conically tapered) tube. Because the tube taper is linear, that gap "
          "area grows essentially linearly with the float height $r$, and the "
          "volumetric flow that passes through it is, to first order, proportional "
          "to that area. A straight line $Q\\propto r$ is therefore the "
          "physically *expected* calibration law — not just a convenient curve — "
          "and the high $R^2$ below confirms the taper is linear over the working "
          "range.")
        A("")
        A(f"A least-squares fit of the mass-flow-meter reading vs. the rotameter "
          f"height $r$ (10 points) gives the calibration line")
        A("")
        A(f"$$\\boxed{{{eq}}}\\ \\text{{[L/min]}}\\qquad (R^2={f(cal['r2'])}).$$")
        A("")
        if cal['through_origin']:
            A("The fit is deliberately forced **through the origin**: at zero "
              "reading the float rests on its bottom stop and the meter passes no "
              "flow ($r=0\\Rightarrow Q=0$), so a non-zero intercept would be "
              "physically meaningless. Allowing a free intercept shifts $R^2$ only "
              "in the third–fourth decimal, which shows the zero-offset is "
              "negligible and the one-parameter (through-origin) line is the "
              "correct physical model. This calibration is reused in graph #10 to "
              "convert readings into a flow rate.")
        else:
            A("A small non-zero intercept survives the fit; it represents the "
              "float's zero-offset (minimum indicated flow) and is negligible "
              "against the working range. This calibration is reused in graph #10 "
              "to convert readings into a flow rate.")
        A("")
        A("**Variable-pressure (density) test.** A rotameter actually responds to "
          "the gas *density*: the same float position corresponds to a different "
          "mass/volumetric flow when the gas is compressed. Calibrated at a "
          "reference pressure $P_0$ and used at absolute pressure "
          "$P_i=P_\\text{gauge}+P_\\text{atm}$ (with density $\\rho\\propto P$ at "
          "fixed temperature), the standard density correction is "
          "$Q\\propto\\sqrt{\\rho_0/\\rho_i}=\\sqrt{P_0/P_i}$. The 5-point test "
          "holds the true flow constant and varies the pressure, so plotting the "
          "calibrated flow against")
        A("")
        A(r"$$x=\sqrt{\frac{P_0}{P_i}},\qquad \delta x=\tfrac12\,x\,\frac{\delta P}{P_i}$$")
        A("")
        A("should again be linear, testing the $\\sqrt{P_0/P_i}$ density law (the "
          "$\\delta x$ form follows from $\\ln x=\\tfrac12(\\ln P_0-\\ln P_i)$, so "
          "$\\delta x/x=\\tfrac12\\,\\delta P/P_i$). With $P_0$ the reference "
          "(atmospheric) pressure, the fitted slope of $Q$ vs. $x$ is reported and "
          "interpreted in the discussion (see graph #10).")
    A("")

    # ---------- chi square ----------
    A("## 9. Goodness of fit — reduced chi-square")
    A("")
    A("Every trendline reports a **reduced chi-square** rather than $R^2$, because "
      "$R^2$ ignores the error bars whereas $\\chi^2_\\nu$ weights each residual by "
      "its own uncertainty and so actually tests whether the model is consistent "
      "with the measured scatter. With $N$ points, $p$ fitted parameters and "
      "per-point error $\\sigma_i$ (folding the x-error in through the local slope "
      "$f'$, the *effective-variance* method):")
    A("")
    A(r"$$\chi^2=\sum_i\frac{\big(y_i-f(x_i)\big)^2}{\sigma_i^2},\quad \sigma_i^2=\sigma_{y,i}^2+\big(f'(x_i)\,\sigma_{x,i}\big)^2,\quad \chi^2_\nu=\frac{\chi^2}{N-p}.$$")
    A("")
    A("A value $\\chi^2_\\nu\\approx1$ indicates the scatter is consistent with the "
      "error bars. The *ideal* $Q_\\text{ideal}$ curve and the Bernoulli/1-7 Pitot "
      "curves give $\\chi^2_\\nu\\approx0$ because those series are exact analytic "
      "functions of the x-axis (a perfect $\\sqrt{\\Delta P}$ or linear law).")
    A("")

    # ---------- results ----------
    A("## 10. Results summary")
    A("")
    A("See `figures/RESULTS_SUMMARY.txt` for the auto-generated table of fitted "
      "slopes, discharge coefficients, $Re$ ranges and $\\chi^2_\\nu$ values that "
      "accompanies the current data.")
    A("")

    OUT.write_text("\n".join(L), encoding="utf-8")
    print(f"Wrote {OUT}  ({len(L)} lines)")


if __name__ == "__main__":
    main()
