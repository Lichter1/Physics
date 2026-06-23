# Flow-Rate lab — figure generation

Python scripts that build every graph required by the **"Flow Rate and Velocity
Measurement"** lab report directly from the Excel workbook
`מדידת_ספיקה_ומהירות_לוח_מדידה_1.xlsx` (one folder up).

The workbook is the **single source of truth**: every value and every error bar
is read from its calculated (green) cells. The scripts never invent numbers —
the only maths done in Python is the trendlines/slopes the report asks for, and
the error bar on the discharge coefficient *C* (the workbook has no `C_err`
column, so it is propagated from the `Q_real`/`Q_ideal` errors).

> **Current state:** the workbook's yellow **input** cells are still empty, so
> running the scripts today produces clearly-labelled *"Awaiting measurements"*
> placeholder images. Type your measured values into the yellow cells, save the
> file, re-run — and the real graphs (with error bars, trendlines and zoom
> insets) appear automatically. No code changes needed.

## What you type into Excel (the only inputs)

| Sheet | Cells | Meaning |
|---|---|---|
| Venturi / Diaphragm / Nozzle | `D4:D13` | Δh measured [m] |
| Venturi / Diaphragm / Nozzle | `I4:I13` | Q_real (rotameter) [L/min] |
| Pitot | `C4:C13` | Δh measured [m] |
| Pitot | `K4:K13` | Q rotameter [L/min] |
| Rotameter | `C4:D13` | reading + mass-flow (part 1, atmospheric) |
| Rotameter | `C17:E21` | abs-pressure + reading + mass-flow (part 2) |

Everything else (ΔP, Q_ideal, Re, all `_err` columns, the C-table) is computed
by Excel's own formulas.

## Uncertainties (what feeds the error bars)

All error bars are read from the workbook's `_err` columns; the underlying
constants live in the `Setup` sheet:

| Quantity | Source | Setup cell | Model |
|---|---|---|---|
| Δh manometer (ruler) | 0.0005 m | `C12` | constant → ΔP_err, Pitot √ΔP_err |
| ΔD diameter | 0.0001 m | `C13` | constant → Q_ideal_err, Re_err |
| Gauge pressure reading | 0.01 bar | `C15` | drives fig10 X error (√(P0/Pi)) |
| **Q_real** (LDN 1009 GAPL mass flow meter) | datasheet | `C33–C36` | **per-point, see below** |

**Q_real uncertainty** follows the LDN 1009 GAPL datasheet, combined in
quadrature (RSS):

```
ΔQ_real = √( (4%·Q)² + (0.5%·FS)² + (2%·Q)² ),   FS = 250 Nl/min
```

i.e. 4 % of reading ⊕ 0.5 % of range-end value (= 1.25 L/min) ⊕ 2 %
reproducibility. This replaced an earlier flat ±0.5 L/min constant — at full
flow the correct value is ≈ ±6.4 L/min (the old constant was ~13× too small).
The datasheet's ±2 °C *temperature* deviation is a temperature-reading spec and
is **not** included in the flow uncertainty.

This model is implemented as a live Excel formula (column `J` in
Venturi/Diaphragm/Nozzle, and the flow term of Pitot `M`) by
[update_excel_uncertainty.py](update_excel_uncertainty.py) — re-run it if you
change the datasheet numbers in `Setup` `C33:C36`.

## Setup

```bash
cd "Data analysis/experements/Flow_Rate/analysis"
pip install -r requirements.txt
```

## Run

```bash
python generate_all.py            # integrity check + all figures + summary
python make_calculations_doc.py   # regenerate calculations.md (formulas + worked examples)
```

…or generate a single figure:

```bash
python fig01_rotameter_calibration.py
python fig02_04_flow_calibration.py
python fig05_pitot_velocity.py
python fig06_08_C_vs_Re.py
python fig09_qreal_vs_qideal.py
python fig10_rotameter_pressure.py
python check_workbook.py        # just the integrity check
```

Outputs land in `figures/` as **both** `.png` (200 dpi) and `.pdf` (vector),
plus `figures/RESULTS_SUMMARY.txt` with the fitted slopes, C-values and Re
ranges for the discussion section. `make_calculations_doc.py` writes
[calculations.md](calculations.md) — every formula, every uncertainty, and a
worked numeric example pulled from the live data.

**Goodness of fit:** trendlines report **reduced chi-square** $\chi^2_\nu$
(not $R^2$), using the effective-variance method so both x- and y-errors count.
$\chi^2_\nu\approx1$ means the scatter matches the error bars; the purely
analytic curves (ideal $Q$, Pitot Bernoulli/1-7) give $\chi^2_\nu\approx0$.

## Files → graphs (numbering from the instructions)

| Script | Graph(s) | Manual § |
|---|---|---|
| `fig01_rotameter_calibration.py` | #1 Rotameter calibration (atmospheric) | §3.5.1 |
| `fig02_04_flow_calibration.py` | #2-4 Flow-rate calibration (Venturi/Diaphragm/Nozzle) | §4.a |
| `fig05_pitot_velocity.py` | #5 Pitot velocity calibration | §4.b |
| `fig06_08_C_vs_Re.py` | #6-8 Discharge coefficient C vs. Re | §4.c |
| `fig09_qreal_vs_qideal.py` | #9 Q_real = f(Q_ideal), slope = C | §4.d |
| `fig10_rotameter_pressure.py` | #10 Normalised flow vs. √(P0/Pi) | §4.f |

`flow_common.py` holds shared helpers (workbook reading, fits, styling, saving,
zoom-inset, placeholder). `check_workbook.py` is the integrity check.
`update_excel_uncertainty.py` rewrites the `Setup`/`J`/`M` cells for the LDN
1009 datasheet error model. `make_calculations_doc.py` regenerates
[calculations.md](calculations.md).

### How the graphs stay in sync with Excel

The scripts read Excel's **calculated** values. When the workbook is edited
programmatically, openpyxl drops Excel's cached results, so `flow_common`
transparently re-evaluates the workbook's **own formulas** via the optional
[`formulas`](https://pypi.org/project/formulas/) engine (never a Python
reimplementation of the physics). If `formulas` isn't installed, the scripts
fall back to whatever values Excel last cached — so open + save the workbook in
Excel/LibreOffice after editing it by hand, or keep `formulas` installed.

## Editing

Each `fig*.py` starts with a small **CONFIG** block — change column letters,
labels, colours, trendline mode, inset framing, etc. there. A few useful knobs:

- `fig01`: `THROUGH_ORIGIN` — force the calibration line through (0,0).
- `fig09`: `THROUGH_ORIGIN` — slope = C is the physical meaning (default `True`).
- `fig06_08`: `LITERATURE` — paste `(Re, C)` points from the Lecture-6 / standard
  chart to overlay a reference curve. Empty by default (measured points only).
- **`fig10`: `RATIO_MODE` and `P_REF_BAR`** — the workbook stores one pressure
  per point, but √(P0/Pi) needs a reference pressure too. **Confirm the exact
  definition against manual §4.f before quoting the slope.**

## Notes / open items to confirm with the report author

- **Graph #1** isn't explicitly in the manual's §4 list — decide whether it goes
  in the body or an appendix.
- **Graphs #6-8 literature curve** wasn't supplied in machine-readable form; by
  default only measured points are plotted and the caption says so.
- **Graph #10** √(P0/Pi) definition — see `RATIO_MODE` above.
- **Pitot flow error**: the Pitot `K` column is labelled "Q rotameter", but the
  LDN 1009 datasheet error has been applied to it (for consistency with
  Q_real). If the Pitot flow was actually read on the rotameter (a different
  instrument), tell me and the Pitot `M` formula should use that instrument's
  error instead.
- A pre-/post-edit copy of the workbook is kept in `../backups/`.
