# Flow Rate and Velocity Measurement — Calculations & Uncertainties

_Auto-generated from the workbook `מדידת ספיקה ומהירות לוח מדידה copy.xlsx`. Every worked example uses the real measured data._

All measured inputs are the **yellow** cells; everything below is how each **green** (calculated) cell is obtained, together with its uncertainty and one fully worked example. Uncertainties use the standard **partial-derivative (propagation of errors)** method:

$$\delta f(x_1,\dots,x_n)=\sqrt{\sum_i\left(\frac{\partial f}{\partial x_i}\,\delta x_i\right)^2}.$$

## 1. Constants and fixed uncertainties (`Setup` sheet)

| Symbol | Meaning | Value | Uncertainty |
|---|---|---|---|
| $\rho_\text{air}$ | air density | 1.184 kg/m³ | — |
| $\rho_\text{water}$ | manometer water density | 1000 kg/m³ | — |
| $g$ | gravity | 9.81 m/s² | — |
| $\mu_\text{air}$ | air dynamic viscosity | 1.837e-05 kg/(m·s) | — |
| $\Delta h$ | manometer reading (ruler) | input | $\delta h=0.0005$ m |
| $D$ | diameters | input | $\delta D=0.0001$ m |
| $P_\text{gauge}$ | gauge pressure | input | $\delta P=0.01$ bar |
| Venturi/Nozzle | $D_\text{max}=12$ mm, $D_\text{min}=6$ mm | | |
| Pitot | $D_\text{pipe}=12$ mm, $D_\text{probe}=3$ mm | | |

Diameter ratio (Venturi/Nozzle): $\beta=D_\text{min}/D_\text{max}=6/12=0.5$.

## 2. Real flow-rate uncertainty — LDN 1009 GAPL mass flow meter

The real flow rate $Q_\text{real}$ is read on the LDN 1009 GAPL mass flow meter and is taken as the *true* flow against which every restriction is judged. Its datasheet error has three statistically independent parts — a term proportional to the reading (gain error), a fixed term proportional to full scale (offset/zero error), and a reproducibility term — so they are combined in quadrature rather than added:

$$\delta Q_\text{real}=\sqrt{(p_\text{MV}\,Q)^2+(p_\text{FS}\,\text{FS})^2+(p_\text{R}\,Q)^2},$$

with $p_\text{MV}=0.04$ (4 % of reading), $p_\text{FS}=0.005$ (0.5 % of full scale), $\text{FS}=250$ L/min, and $p_\text{R}=0.02$ (2 % reproducibility). The 0.5 %·FS term is the constant $1.25$ L/min. The ±2 °C temperature spec is a temperature figure and is not part of the flow uncertainty.

**Worked example (Venturi point 10, $Q=140$ L/min):**

$\delta Q_\text{real}=\sqrt{(0.04\cdot140)^2+(0.005\cdot250)^2+(0.02\cdot140)^2}=\sqrt{5.6^2+1.25^2+2.8^2}=6.385$ L/min (workbook: 6.385).

## 3. Pressure difference $\Delta P$ from the manometer

The pressure drop across each restriction is read on an inclined/vertical water manometer. The reading is purely hydrostatic — a column of height $\Delta h$ of water balances the gas pressure difference — so $\Delta P$ is linear in $\Delta h$ and the densities/$g$ are treated as exact. The only random input is the ruler reading $\delta h$, hence the uncertainty is just that single term scaled by $\rho_\text{water}g$:

$$\Delta P=\rho_\text{water}\,g\,\Delta h,\qquad \delta\Delta P=\rho_\text{water}\,g\,\delta h.$$

**Worked example (Venturi point 1, $\Delta h=0.004$ m):** $\Delta P=1000\cdot9.81\cdot0.004=39.24$ Pa (workbook 39.24); $\delta\Delta P=1000\cdot9.81\cdot0.0005=4.905$ Pa (workbook 4.905).

## 4. Ideal flow rate $Q_\text{ideal}$ (Venturi, Diaphragm, Nozzle)

Applying Bernoulli's equation between the wide section and the throat, together with mass continuity $A_\text{max}u_\text{max}=A_t u_t$, for an ideal (inviscid, incompressible) flow gives the throat velocity and hence the volumetric flow. The factor $1/\sqrt{1-\beta^4}$ is the continuity correction for the upstream velocity (the *velocity-of-approach* factor); $A_t=\pi D_\text{min}^2/4$ is the throat area and $6\times10^4$ converts m³/s → L/min:

$$Q_\text{ideal}=\frac{A_t}{\sqrt{1-\beta^4}}\sqrt{\frac{2\,\Delta P}{\rho_\text{air}}}\times 6\times10^4 .$$

This is the flow a *loss-free* restriction would pass for the measured $\Delta P$; the real flow is smaller (viscous losses and the vena contracta), which is exactly what the discharge coefficient $C$ in §6 captures. Its uncertainty depends on $\Delta P$ and on the throat diameter $D_\text{min}$ (the analytic partials are columns O and P in the sheet):

$$\delta Q_\text{ideal}=\sqrt{\left(\frac{\partial Q_\text{ideal}}{\partial D_\text{min}}\,\delta D\right)^2+\left(\frac{\partial Q_\text{ideal}}{\partial \Delta P}\,\delta\Delta P\right)^2}.$$

**Worked example (Venturi point 1):** $Q_\text{ideal}=14.26$ L/min, $\delta Q_\text{ideal}=1.026$ L/min — i.e. $Q_\text{ideal}=14.26\pm1.026$ L/min.

## 5. Reynolds number $Re$

The Reynolds number characterises the flow regime (laminar vs. turbulent) and is the natural x-axis for the discharge coefficient. It is built from the pipe diameter $D_\text{max}$ and the approach velocity $u=\sqrt{2\Delta P/\rho_\text{air}}$, so it inherits its uncertainty from both $\Delta P$ and $D$ (the $\Delta P$ term dominates):

$$Re=\frac{D_\text{max}\sqrt{2\,\rho_\text{air}\,\Delta P}}{\mu_\text{air}},\quad \delta Re=\sqrt{\left(\frac{D_\text{max}\sqrt{2\rho_\text{air}}}{\mu_\text{air}}\frac{\delta\Delta P}{2\sqrt{\Delta P}}\right)^2+\left(\frac{\sqrt{2\rho_\text{air}\Delta P}}{\mu_\text{air}}\,\delta D\right)^2}.$$

**Worked example (Venturi point 1):** $Re=6296.9$, $\delta Re=397$ — i.e. $Re=(6.297\pm0.397)\times10^3$.

## 6. Discharge coefficient $C$

The discharge coefficient is the ratio of the measured flow to the ideal (loss-free) flow at the same $\Delta P$. It lumps together the viscous losses and the vena-contracta area reduction, so physically $C<1$ for the Venturi/diaphragm/nozzle, and for a given geometry it should approach a constant once the flow is fully turbulent (high $Re$). As a ratio of two measured quantities its relative error is the quadrature sum of the two relative errors:

$$C=\frac{Q_\text{real}}{Q_\text{ideal}},\qquad \delta C=C\sqrt{\left(\frac{\delta Q_\text{real}}{Q_\text{real}}\right)^2+\left(\frac{\delta Q_\text{ideal}}{Q_\text{ideal}}\right)^2}.$$

**Worked example (Venturi point 1):** $C=14/14.26=0.9814$ (workbook 0.9814); $\delta C=0.9814\sqrt{(1.398/14)^2+(1.026/14.26)^2}=0.1208$ — i.e. $C=0.9814\pm0.1208$.

> Note: the workbook's C-table has no error column; the figure scripts compute $\delta C$ with exactly this formula for the error bars in graphs #6–#8.

## 7. Pitot tube velocities

The Pitot-static tube measures the difference between stagnation and static pressure; by Bernoulli this gives the **local** velocity at the probe tip. To turn that point velocity into a cross-sectional **average** we use the turbulent $1/7$ power-law profile $u(r)/u_\text{max}=(1-r/R)^{1/7}$, whose area-average is $\tfrac{14}{15}$ of the centre-line value — the probe sits on the axis, so $u_\text{avg}=\tfrac{14}{15}u_\text{Bernoulli}$. As an independent cross-check, $u_\text{rot}$ is the bulk velocity obtained from the rotameter flow divided by the pipe area. The three should agree:

$$u_\text{Bernoulli}=\sqrt{\frac{2\,\Delta P}{\rho_\text{air}}},\quad \delta u=\sqrt{\frac{2}{\rho_\text{air}}}\,\delta(\sqrt{\Delta P}),\quad \delta(\sqrt{\Delta P})=\tfrac12\sqrt{\frac{\rho_\text{water}g}{\Delta h}}\,\delta h.$$

$$u_{1/7}=\tfrac{14}{15}\,u_\text{Bernoulli},\qquad u_\text{rot}=\frac{4\,Q}{\pi D_\text{pipe}^2}.$$

**Worked example (Pitot point 1, $\Delta P=19.62$ Pa):** $u_\text{Bernoulli}=\sqrt{2\cdot19.62/1.184}=5.757$ m/s ($\delta u=0.7196$); $u_{1/7}=5.373$ m/s; $u_\text{rot}=3.242$ m/s ($\delta u=0.2406$); $Re=4452.6\pm557.8$.

## 8. Rotameter calibration and the pressure test

**Why a linear fit is physical.** A rotameter is a *variable-area* meter: the float rises until its submerged weight is balanced by the pressure drop across the annular gap between the float and the (conically tapered) tube. Because the tube taper is linear, that gap area grows essentially linearly with the float height $r$, and the volumetric flow that passes through it is, to first order, proportional to that area. A straight line $Q\propto r$ is therefore the physically *expected* calibration law — not just a convenient curve — and the high $R^2$ below confirms the taper is linear over the working range.

A least-squares fit of the mass-flow-meter reading vs. the rotameter height $r$ (10 points) gives the calibration line

$$\boxed{Q=14.31\,r}\ \text{[L/min]}\qquad (R^2=0.9949).$$

The fit is deliberately forced **through the origin**: at zero reading the float rests on its bottom stop and the meter passes no flow ($r=0\Rightarrow Q=0$), so a non-zero intercept would be physically meaningless. Allowing a free intercept shifts $R^2$ only in the third–fourth decimal, which shows the zero-offset is negligible and the one-parameter (through-origin) line is the correct physical model. This calibration is reused in graph #10 to convert readings into a flow rate.

**Variable-pressure (density) test.** A rotameter actually responds to the gas *density*: the same float position corresponds to a different mass/volumetric flow when the gas is compressed. Calibrated at a reference pressure $P_0$ and used at absolute pressure $P_i=P_\text{gauge}+P_\text{atm}$ (with density $\rho\propto P$ at fixed temperature), the standard density correction is $Q\propto\sqrt{\rho_0/\rho_i}=\sqrt{P_0/P_i}$. The 5-point test holds the true flow constant and varies the pressure, so plotting the calibrated flow against

$$x=\sqrt{\frac{P_0}{P_i}},\qquad \delta x=\tfrac12\,x\,\frac{\delta P}{P_i}$$

should again be linear, testing the $\sqrt{P_0/P_i}$ density law (the $\delta x$ form follows from $\ln x=\tfrac12(\ln P_0-\ln P_i)$, so $\delta x/x=\tfrac12\,\delta P/P_i$). With $P_0$ the reference (atmospheric) pressure, the fitted slope of $Q$ vs. $x$ is reported and interpreted in the discussion (see graph #10).

## 9. Goodness of fit — reduced chi-square

Every trendline reports a **reduced chi-square** rather than $R^2$, because $R^2$ ignores the error bars whereas $\chi^2_\nu$ weights each residual by its own uncertainty and so actually tests whether the model is consistent with the measured scatter. With $N$ points, $p$ fitted parameters and per-point error $\sigma_i$ (folding the x-error in through the local slope $f'$, the *effective-variance* method):

$$\chi^2=\sum_i\frac{\big(y_i-f(x_i)\big)^2}{\sigma_i^2},\quad \sigma_i^2=\sigma_{y,i}^2+\big(f'(x_i)\,\sigma_{x,i}\big)^2,\quad \chi^2_\nu=\frac{\chi^2}{N-p}.$$

A value $\chi^2_\nu\approx1$ indicates the scatter is consistent with the error bars. The *ideal* $Q_\text{ideal}$ curve and the Bernoulli/1-7 Pitot curves give $\chi^2_\nu\approx0$ because those series are exact analytic functions of the x-axis (a perfect $\sqrt{\Delta P}$ or linear law).

## 10. Results summary

See `figures/RESULTS_SUMMARY.txt` for the auto-generated table of fitted slopes, discharge coefficients, $Re$ ranges and $\chi^2_\nu$ values that accompanies the current data.
