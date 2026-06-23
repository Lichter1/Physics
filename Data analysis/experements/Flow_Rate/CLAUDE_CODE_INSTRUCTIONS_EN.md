# Instructions — Building the Graphs for the "Flow Rate and Velocity Measurement" Lab Report

This document is for the Claude Code agent. It contains all the background
knowledge, the exact location of data inside the Excel workbook, and a
precise, numbered list of every graph required for the report. The goal:
produce all graphs (submission-quality PNG/PDF, plus the Python/MATLAB code
that generates them) directly from the data in the attached Excel file, with
error bars, trendlines, and clear formatting — exactly as the lab manual
requires and at the quality level demonstrated in the reference report.

---

## 0. Attached files and the role of each

| File | Role |
|---|---|
| **מדידת_ספיקה_ומהירות_לוח_מדידה_1.xlsx** | The main working file. Contains all measured data (input, yellow cells) and all calculations (output, green cells — live Excel formulas). **This is the single source of truth for all numeric values.** |
| **מדידת_מהירות_וספיקה....pdf** (lab manual / handout) | The official source of requirements: which graphs are required, which variable goes on which axis, which physical assumptions apply. Section 4 ("Results Processing" / עיבוד תוצאות) is the official, binding list of graphs. |
| **מעבדה_ספיקה_ומהירות_98...pdf** (peer reference report) | **Reference for presentation quality only.** Use it to understand axis labeling, legends, trendline style, error bars, and inset/zoom conventions. **Do not** copy any numbers, values, or results from it — its data belongs to a different experiment (different group, different year) and is not relevant to our numbers. |
| **Course lecture / slides** (הרצאה 6.pdf) | Additional theoretical background, mainly for the C/EC-vs-Reynolds-number graph (see graph #9 below) — use it to pull the expected literature curve/range for comparison. |

If any file from this list is actually missing from the working environment —
**stop and ask for it**, do not proceed on assumptions.

---

## 1. Experiment background — what this is, and why

The experiment evaluates **five instruments** for measuring the flow rate /
velocity of air flowing through a pipe:

1. **Rotameter** — the reference instrument. This is the instrument that all
   other instruments are calibrated against. You read the float height →
   convert to flow rate.
2. **Venturi tube** — a flow-restriction instrument (gradual constriction),
   based on Bernoulli's equation.
3. **Diaphragm (Orifice plate)** — a plate with a bore, a sharp constriction
   (larger energy losses than the Venturi).
4. **Nozzle** — gradual constriction at the inlet + wide expansion at the
   outlet, higher accuracy than the diaphragm.
5. **Pitot tube** — does not measure flow rate directly but rather **local
   point velocity**, via static/dynamic pressure.

**The physical principle shared by Venturi/Diaphragm/Nozzle:** flow through a
restriction in the pipe creates a pressure difference (ΔP) that is related to
the velocity/flow rate via Bernoulli's equation + the continuity equation.
From this, an "ideal flow rate" (Q_ideal) formula is derived, assuming
inviscid ideal flow. In practice there are losses (viscosity, vortices,
friction), so the actual flow rate (Q_real, as measured by the rotameter)
differs from the ideal one. The ratio between them is the **discharge
coefficient C = Q_real / Q_ideal**, which is primarily a function of the
Reynolds number (Re) and the diameter ratio β.

**The physical principle of the Pitot tube:** it measures the difference
between static and dynamic pressure at a single point in the pipe, from which
a local point velocity is extracted. To compare this to the *average*
velocity across the cross-section, there are two additional methods: (a)
assuming a turbulent "1/7 power law" velocity profile, (b) going back to the
flow rate measured by the rotameter and dividing by the cross-sectional area.

---

## 2. What is actually measured vs. what is calculated — and where it is in the Excel file

This is the most important distinction. **Everything labeled "measured" is
human input (a yellow cell in the Excel file). Everything else is formula
output (a green cell) — do not calculate these yourselves; read the value
that Excel has already computed.**

### 2.1 `Setup` sheet
Global constants (not tied to a specific measurement point): air/water
density, gravitational acceleration, air viscosity, fixed instrument errors
(Δh, ΔD, ΔQ), diameters (Dmax/Dmin) for each instrument, and the maximum
target flow rate (Qmax) for each instrument. **Every formula in the file
draws from here** — do not duplicate values in your own work; reference the
Setup cells through what is already computed in the other sheets.

### 2.2 `Rotameter` sheet
- **Measured (human input):** the rotameter reading and the mass flow meter
  reading, in two parts — 10 points under atmospheric conditions (manual
  §3.5.1 part 1), 5 points at variable absolute pressure 1–2.5 bar_a (manual
  §3.5.1 part 2).
- Usage: basis for graph #1 and graph #10 (see below).

### 2.3 `Venturi` / `Diaphragm` / `Nozzle` sheets (identical structure for all three)
| Column | Name | Measured / Calculated |
|---|---|---|
| B | Point number (1–10) | — |
| C | Target flow rate (Qmax/10 × point) | Calculated — guidance only, **not a measured value** |
| **D** | **Measured Δh [m]** | **Measured (input)** — the height difference read on the manometer |
| E | ΔP [Pa] | Calculated = ρ_water·g·Δh |
| F | ΔP_err [Pa] | Calculated (fixed error) |
| G | Q_ideal [L/min] | Calculated from Bernoulli + continuity |
| H | Q_ideal_err [L/min] | Calculated (analytic derivative, see columns O–Q) |
| **I** | **Q_real [NL/min]** | **Measured (input)** — the volum flow meter [Nl/min] reading *at the same point in time* |
| J | Q_real_err [L/min] | Calculated (= the fixed ΔQ_real from Setup) |
| K | Re | Calculated |
| L | Re_err | Calculated |
| O–Q | Partial derivatives for Q_ideal error | Intermediate, do not edit, can be hidden |
| Rows 19–28 (separate block below) | Table of C = Q_real/Q_ideal per point | Calculated |

**The only two input cells in each row are D (Δh) and I (Q_real)** — every
other column is output.

### 2.4 `Pitot` sheet
| Column | Name | Measured / Calculated |
|---|---|---|
| B | Point number | — |
| **C** | **Measured Δh [m]** | **Measured (input)** — the Pitot manometer |
| D | ΔP [Pa] | Calculated |
| E | √ΔP [√Pa] | Calculated |
| F | √ΔP_err | Calculated |
| G | u Bernoulli [m/s] | Calculated — eq. 4.10 in the report / 4.9 in the manual |
| H | u_err | Calculated |
| I | u average (1/7 profile) [m/s] | Calculated = (14/15)·u_Bernoulli |
| J | u_err | Calculated |
| **K** | **Q rotameter [L/min]** | **Measured (input)** — rotameter reading *taken in parallel* with the Pitot reading |
| L | u average (rotameter) [m/s] | Calculated = Q / (pipe cross-sectional area) |
| M | u_err | Calculated |
| N | Re | Calculated (using u Bernoulli) |
| O | Re_err | Calculated |

The two input cells: **C (Δh) and K (Q rotameter)**.

### 2.5 `Summary_QvQ` sheet
Consolidates the Q_ideal and Q_real columns from the three Venturi/Diaphragm/
Nozzle sheets into one convenient table for graph #9. **Entirely calculated
/ linked — there is no input on this sheet.**

---

## 3. File integrity check — perform **before** producing any graphs

Before producing a single graph:

1. Open the Excel file and confirm that **all** input columns (D and I in
   each of Venturi/Diaphragm/Nozzle; C and K in Pitot; all reading cells in
   Rotameter) are filled with 10 (or 5, for Rotameter part 2) numeric values,
   with no empty cells in the middle of the range.
2. Confirm there is no formula error in any cell anywhere in the file
   (`#DIV/0!`, `#VALUE!`, `#NUM!`, etc.). If there is — stop, report the
   exact location (sheet + cell), and do not proceed with partial values.
3. **Manually verify one or two calculations** from each sheet against the
   physical formulas in the manual (sections 4.1–4.5) — i.e., compute one
   value "by hand" (a small Python script, not inside Excel) and compare it
   to the corresponding cell. For example: check that ΔP in the first row of
   Venturi equals exactly `ρ_water * g * Δh` using the values from Setup. If
   there is a discrepancy — report it and do not proceed.
4. Confirm that the Re numbers across all sheets fall within a reasonable
   range for turbulent flow in a lab-scale pipe (roughly 10³–10⁵; if you see
   negative, zero, or wildly out-of-range values — that's a sign of a
   problem in the input data, not in the formula).
5. Only after all checks pass — proceed to producing the graphs.

**Important:** the Excel file is the binding "source of truth." If there is
a conflict between a value in the Excel file and what the reference report
shows (for example, some worked example) — the Excel file wins, because it
is calculated from our own data, not from another report's data. The
reference report is for formatting only, not for numbers.

---

## 4. General formatting requirements (apply to **every** graph)

Per the lab manual (§4.g) and consistent with the presentation quality in the
reference report:

- **Every graph must include error bars** (on both the X and Y axes,
  wherever an error has been calculated) — taken from the corresponding
  `_err` columns in the Excel file.
- Axes labeled with variable + units (e.g. `ΔP [Pa]`, `Q [L/min]`,
  `u [m/s]`).
- A title for each graph (figure + number + short description), a clear
  legend whenever there is more than one series.
- When the X error is so small it isn't visible at the main graph's scale —
  add a **zoomed inset** of a small region showing the error bars clearly
  (as done in the reference report, figures 7 and 12). This is not a
  cosmetic add-on — the manual explicitly requires it.
- Trendlines only when physically justified (see details for each graph
  below) — do not default to a linear trendline "just because."
- Output: an image file (PNG, high resolution ≥150dpi, or vector PDF/SVG)
  **and** the source code that produced it (Python/matplotlib preferred,
  MATLAB also acceptable), so a parameter can be changed and it can be
  re-run.
- Reasonable file names, for example: `fig08_venturi_calibration.png`.

---

## 5. Required graph list — complete and numbered

The numbering here is for traceability in the work (not the final figure
numbering in the report). Each item points precisely to where it is anchored
in the manual (manual §4) and which sheet/columns in the Excel file to use.

### Graph #1 — Rotameter calibration (atmospheric conditions)
- **Source in manual:** §3.5.1 part 1.
- **Data:** `Rotameter` sheet, the first 10 rows — rotameter reading
  (column C) vs. mass flow meter flow rate (column D).
- **Axes:** X = rotameter reading (scale units as recorded), Y = flow rate
  [L/min].
- **Trendline:** a trendline (likely linear, or whatever is physically
  appropriate for a rotameter calibration) — this is the calibration curve
  that will be used to convert future rotameter readings into flow rate.
- Note: this is a graph required by the manual (§3.5.1) but **not**
  explicitly numbered in the official §4 list — confirm with the report
  author whether it should be included as a supporting/appendix figure or
  as a standalone figure in the body of the report.

### Graphs #2–#4 — Flow-rate calibration graph (Venturi, Diaphragm, Nozzle)
- **Source in manual:** §4.a — *"For the Venturi, Diaphragm, and Nozzle
  instruments, produce a flow-rate calibration graph... X axis is the
  pressure difference... Y axis is the flow rate... each calibration graph
  must show two curves: the flow rate as measured in the experiment, and
  the ideal flow rate."*
- **Data:** `Venturi` / `Diaphragm` / `Nozzle` sheet, respectively.
  - X = column E (ΔP [Pa]), X error = column F.
  - Series 1 (ideal flow rate) = column G, error = column H.
  - Series 2 (actual/measured flow rate) = column I, error = column J.
- **3 separate graphs** — one per instrument (do not merge all three into
  one graph).
- **Trendline:** for the "ideal" curve — a theoretical curve (per the
  formula, smooth and continuous). For the "actual" curve — a square-root
  type trendline (Q ∝ √ΔP), consistent with the expected physical behavior
  (and not a straight line!) — as done in the reference report (figures
  8–10).
- **Additional requirement:** error bars on both series on both axes. If the
  X error bar is too small to be visible at the chart's scale — add a
  point-zoom inset (like figure 7 in the reference report).

### Graph #5 — Velocity calibration graph (Pitot)
- **Source in manual:** §4.b — *"X axis is the square root of the pressure
  difference... Y axis is the flow velocity... the graph must contain 3
  curves: velocity per Bernoulli at the measurement point, average velocity
  per the 1/7 profile (including computing Re), average velocity per the
  rotameter reading."*
- **Data:** `Pitot` sheet.
  - X = column E (√ΔP), X error = column F.
  - Series 1 (Bernoulli, point velocity) = column G, error = column H.
  - Series 2 (1/7 profile, cross-section average) = column I, error =
    column J.
  - Series 3 (rotameter, cross-section average) = column L, error =
    column M.
- **Trendline:** linear for all three curves (u ∝ √ΔP is expected to be
  approximately linear under the Bernoulli approximation) — as shown in the
  reference report (figure 11), with the line equation (f(x)=a·x) displayed
  on the graph.
- **Inset required:** the manual/example note that it's hard to see
  differences/errors at full scale — add a zoomed inset of a high-range
  segment (see figure 12 in the reference report) that clearly shows the
  separation between the three curves and the theoretical error bars.

### Graphs #6–#8 — Coefficient C / EC vs. Reynolds number (Venturi, Diaphragm, Nozzle)
- **Source in manual:** §4.c — *"For the Venturi, Diaphragm, and Nozzle
  instruments, build a graph for each showing the pointwise variation of the
  C (discharge coefficient) or EC (flow coefficient) as a function of the Re
  number. Present it so it can be compared against a graph from the
  literature for the same component."*
- **Data:** the C table in each sheet (Venturi/Diaphragm/Nozzle, the rows
  below the main table — column C = coefficient, column D = corresponding
  Re).
- **3 separate graphs**, discrete points (not a continuous line) — C as a
  function of Re.
- **Comparison to literature:** the manual asks that this allow comparison
  with manufacturer/literature graphs (see figure 14 in the reference report
  — C curves vs. Re_d from professional standards, for each instrument). If
  the lecture slides (manual §1, "Lecture 6") containing these graphs are
  available to you — pull the theoretical curve/range from there and display
  it **on the same graph** for visual comparison (shared X axis = Re). If no
  literature graphs were provided — show only the measured points, and note
  in the caption that a comparison curve is missing.
- No global trendline is needed on these points (C is not a linear function
  of Re) — a local connecting line is fine if it aids readability, but not a
  regression line.

### Graph #9 — Actual flow rate as a function of ideal flow rate (all three instruments, slope = C)
- **Source in manual:** §4.d — *"For the same instruments, produce a graph
  of QReal=f(Qideal) (3 curves on the same graph)."*
- **Data:** `Summary_QvQ` sheet — three verified column pairs: Venturi
  (Qideal = column C, Qreal = column D), Diaphragm (Qideal = column F,
  Qreal = column G), Nozzle (Qideal = column I, Qreal = column J).
- **One graph, 3 point series** (not 3 separate graphs) — X = Q_ideal,
  Y = Q_real, for each instrument in a different color/marker.
- **Trendline:** a linear regression line (through the origin, or with an
  intercept) for each of the three series separately, with the line equation
  displayed on the graph — **the slope of each line is the discharge
  coefficient C of that instrument** (and that is the point of this graph —
  as explained in the reference report, figure 13). The three slopes must be
  extracted and reported separately (they will be used in the
  discussion/conclusions section).
- Error bars on X and Y for all points (the errors come from the
  corresponding Q_ideal_err / Q_real_err columns in each source sheet).

### Graph #10 — Normalized rotameter flow rate as a function of √(P0/Pi)
- **Source in manual:** §4.f — *"For the rotameter flow-rate test under
  variable pressure conditions (1–2.5 bar_a), calculate the rotameter flow
  rate using the calibration equation from section e [= graph #1 here], and
  plot the variation of this flow rate as a function of √(P0/Pi). Fit a
  trendline and use it to calculate the slope."*
- **Data:** `Rotameter` sheet, second part (the 5 points at variable
  absolute pressure) — compute P0/Pi from the recorded absolute pressure,
  then feed the rotameter reading into the calibration equation obtained in
  graph #1 to get the "normalized/calculated flow rate" (Y).
- **X = √(P0/Pi)**, **Y = calculated flow rate [L/min]** (5 points only).
- **Trendline:** linear, with an explicit slope calculation (carry this
  value into the discussion/conclusions section — the manual explicitly
  asks "what is the meaning of the slope and of the X axis").
- Errors: per the existing rotameter-reading/pressure error figures; if the
  file does not contain a ready-made error column for this part — calculate
  it per the general error method in the manual (section 10.1 in the
  reference report, the partial-derivatives method) and state this clearly
  in the code.

---

## 6. Recommended order of work

1. File integrity check (section 3 above) — before anything else.
2. Write one helper function to read all relevant sheets from the Excel file
   (calculated values only — `data_only=True` if using Python/openpyxl —
   not the raw formulas).
3. Produce graphs #2–#4 (flow-rate calibration) — these are the most
   important and form the basis of the report's discussion.
4. Produce graph #5 (Pitot calibration).
5. Produce graphs #6–#8 (C vs. Re).
6. Produce graph #9 (Qreal=f(Qideal), slope=C) — save the three slopes that
   were calculated.
7. Produce graphs #1 and #10 (rotameter).
8. Consolidate all files into one output folder, with a short text report
   summarizing: which C values and slopes were obtained for each instrument,
   and which Re ranges were observed — so this can be used conveniently in
   the discussion/conclusions section of the report itself.

## 7. What not to do

- Do not change, "correct," or round values that come from the Excel file —
  if something looks unreasonable, report it and ask, don't silently fix it.
- Do not take any number (ΔP, Q, C, Re, etc.) from the reference report and
  present it as if it were ours — it belongs to a different experiment.
- Do not add a linear trendline "just to have one" on graphs #6–#8 — there,
  the points remain discrete, with no global regression.
- Do not skip error bars on any graph, even if the error looks small (see the
  inset requirement in section 4).
