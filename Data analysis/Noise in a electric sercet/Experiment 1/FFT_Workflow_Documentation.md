# White Noise Analysis Workflow Documentation

## Overview

This document describes the complete data-processing and analysis workflow for FFT (Fast Fourier Transform) instrument measurements of white noise in electrical circuits. The workflow consists of three Python scripts that progressively process FFT data to extract physical parameters, particularly the Boltzmann constant, from Johnson-Nyquist thermal noise measurements.

**Experiment Context:**
- Measurement of thermal (Johnson-Nyquist) noise in resistors
- FFT analyzer bandwidth: 250 Hz
- Temperature: 14.6°C (287.75 K)
- Frequency range: 0-100 kHz
- Multiple resistor values: 913 Ω to 68.3 kΩ

---

## Data Format

### Input Data
Each measurement file (e.g., `913 ohm.txt`, `1k ohm.txt`) contains tab-separated values:
```
frequency[Hz]    amplitude[dBm]
0.000000         48.780605
250.000000       -54.662156
500.000000       -93.603061
...
```

- **Frequency**: Sampled at 250 Hz intervals from 0 to ~100 kHz
- **Amplitude**: Power spectral density in dBm (decibels relative to 1 milliwatt)
- Each file contains approximately 400-450 data points

---

## Script 1: `noise_analysis.py`

### Purpose
Initial visualization and statistical analysis of raw FFT spectra across multiple resistors.

### Process

#### 1. Data Loading
- **Function**: `load_data(filepath)`
- Reads frequency and amplitude data from text files
- Parses tab-separated values, skipping empty lines
- Returns NumPy array of shape (N, 2) for N data points

#### 2. Resistance Extraction
- **Function**: `parse_resistance(filename)`
- Extracts resistance value from filename using regex
- Handles formats: "1k ohm", "2.4k ohm", "913 ohm", etc.
- Converts 'k' suffix to multiply by 1000
- Returns resistance in ohms

#### 3. Visualization

##### Combined Spectrum Plot
- Plots all resistance spectra on a single graph
- **Color mapping**: Logarithmic color gradient (blue → purple → red) based on resistance
  - Uses `LogNorm` for colorbar scale
  - Formula: `color_position = (log₁₀(R) - log₁₀(R_min)) / (log₁₀(R_max) - log₁₀(R_min))`
- **Main plot**: Full frequency range (0-100 kHz), y-axis in dBm
- **Inset plot**: Zoomed view of noise floor (-105 to -70 dBm)
- Output: `noise_spectra_raw.png`

##### Individual Spectrum Plots
- Separate plot for each resistor value
- Same color as in combined plot for consistency
- Fixed y-axis: -105 to -70 dBm (noise floor detail)
- Output: `noise_spectrum_<resistance>_ohm.png` (8 files)

#### 4. Statistical Summary
For each resistor:
- **Mean amplitude**: Average of all dBm values
- **Standard deviation**: Spread of dBm values
- **Min/Max**: Range of measured amplitudes

### Calculations
- No unit conversions at this stage
- Pure statistical measures in dB scale
- No uncertainty propagation

### Outputs
- 1 combined spectrum plot with inset zoom
- 8 individual spectrum plots (one per resistor)
- Console output with summary statistics

---

## Script 2: `noise_filter_analysis.py`

### Purpose
Filter out interference peaks and 1/f noise to isolate the white noise baseline, then perform statistical analysis with uncertainty quantification.

### Process

#### 1. Data Loading
- **Function**: `load_data(filepath)`
- Same as Script 1

#### 2. White Noise Filtering
- **Function**: `filter_white_noise(frequencies, amplitudes, method='iqr', threshold=3.0, neighbor_points=3)`

##### Filtering Algorithm

**Step 1: Convert to Linear Scale**
```
linear = 10^(amplitude_dB / 10)
```
Converts from dB (logarithmic) to linear power scale for statistical robustness.

**Step 2: Interquartile Range (IQR) Method**
```
Q1 = 25th percentile of linear values
Q3 = 75th percentile of linear values
IQR = Q3 - Q1
median = 50th percentile
upper_bound = median + threshold × IQR
```
- **Default threshold**: 2.5 IQR above median
- Points above upper_bound are marked as outliers (peaks/interference)
- Points below upper_bound are accepted as white noise baseline

**Step 3: Neighbor Point Rejection**
```
For each rejected point at index i:
    Reject points from (i - neighbor_points) to (i + neighbor_points)
```
- **Default**: 3 neighboring points on each side
- Removes "wings" of interference peaks
- Ensures clean separation between noise floor and peaks

**Alternative: Median Absolute Deviation (MAD)**
```
MAD = median(|linear - median|)
upper_bound = median + threshold × MAD
```
- Even more robust to outliers than IQR
- Used when data has extreme outliers

##### Filtering Statistics
- Typically retains 85-95% of data points
- Rejects 5-15% as interference peaks
- Percentage varies by resistor value and interference level

#### 3. Statistical Analysis

##### dB Scale Statistics
```
mean_dB = mean(accepted_amplitudes_dB)
std_dB = std(accepted_amplitudes_dB)
median_dB = median(accepted_amplitudes_dB)
```

##### Linear Scale Statistics
```
linear_accepted = 10^(accepted_amplitudes_dB / 10)
mean_linear = mean(linear_accepted)
std_linear = std(linear_accepted)
```

##### Gaussian Fit
- **Function**: `scipy.stats.norm.fit(amp_accepted)`
- Fits Gaussian distribution to filtered dB values
- Returns: μ (mean) and σ (standard deviation)
- **Assumptions**: White noise follows Gaussian distribution in dB scale

#### 4. Uncertainty Quantification

##### Confidence Intervals
```
±1σ interval: [μ - σ, μ + σ]     (68.3% confidence)
±2σ interval: [μ - 2σ, μ + 2σ]   (95.4% confidence)
```

These intervals represent the statistical uncertainty in the noise floor measurement.

#### 5. Visualization

##### Filtered Data Plot
- **Red crosses**: Rejected points (interference/peaks)
- **Green dots**: Accepted points (white noise baseline)
- **Main plot**: Full frequency range
- **Inset plot**: Zoomed view of middle frequency region
  - Window size: 1/6 of total frequency range
  - Shows detail of filtering at single-point resolution
- Output: `output/filtered_plots/filtered_<resistance>.png`

##### Distribution Plot
- **Histogram**: Probability density of accepted dB values (40 bins)
- **Red curve**: Fitted Gaussian distribution
- **Vertical lines**:
  - Red dashed: Mean (μ)
  - Orange dotted: ±1σ bounds
  - Yellow dotted: ±2σ bounds
- Output: `output/distribution_plots/distribution_<resistance>.png`

### Calculations and Formulas

#### dB to Linear Conversion
```
Power (linear) = 10^(Power_dBm / 10) mW
```
For noise measurements, we work with power (10^(dB/10)), not voltage (10^(dB/20)).

#### Gaussian Distribution
The filtered white noise amplitudes follow:
```
P(x) = (1 / (σ√(2π))) × exp(-(x - μ)² / (2σ²))
```
where x is amplitude in dB.

### Outputs
- 8 filtered spectrum plots (one per resistor)
- 8 distribution plots (one per resistor)
- Text file: `output/results/noise_analysis_results.txt`
  - Tabulated results for all resistors
  - Columns: Resistance | Mean V² (dB) | σ (dB) | ±2σ (95% CI)
- Console output with detailed statistics per resistor

---

## Script 3: `analyze_noise_vs_resistance.py`

### Purpose
Convert filtered noise data to physical units, perform linear regression to extract the Boltzmann constant from the Johnson-Nyquist relation, and validate results.

### Input
Reads `output/results/noise_analysis_results.txt` generated by Script 2.

### Measurement Parameters (Constants)

```python
PREAMPLIFIER_GAIN = 100              # G (e.g., SR552 × 100)
WINDOW_CORRECTION_FACTOR = 1.30      # C_win for Hamming window
SIGMA_MULTIPLIER = 2                 # For 95% confidence intervals
TEMPERATURE = 287.75                 # K (14.6°C)
DELTA_F = 250                        # Hz (bandwidth)
BOLTZMANN_EXPECTED = 1.380649e-23   # J/K (reference value)
```

### Process

#### 1. Data Parsing
- **Function**: `parse_results_file(filepath)`
- Reads tabulated results from Script 2
- Extracts: resistance (Ω), mean V² (dB), σ (dB)
- Handles both "Ω" and "kΩ" units

#### 2. Unit Conversion and Corrections

##### Step 2a: dB to Linear Conversion
```
V²_meas = 10^(V²_dB / 10)
```
Converts from dB to linear scale (V²).

##### Step 2b: Uncertainty Propagation (dB → Linear)
Using error propagation for exponential functions:
```
d(10^x)/dx = 10^x × ln(10)
```
where x = V²_dB / 10

Therefore:
```
σ_V²_meas = V²_meas × (ln(10) / 10) × σ_dB
```

**Function**: `propagate_db_uncertainty_to_linear(v2_db, sigma_db, gain, c_win)`

##### Step 2c: Preamplifier and Window Corrections
```
V²_corr = V²_meas / (G² × C_win)
```
where:
- **G = 100**: Preamplifier gain
- **C_win = 1.30**: Hamming window correction factor
- **G² × C_win = 13,000**: Total correction factor

The measured voltage is amplified by the preamplifier and modified by the windowing function. These corrections remove these systematic effects to obtain the true voltage at the resistor.

##### Step 2d: Uncertainty Propagation (Corrections)
```
σ_V²_corr = σ_V²_meas / (G² × C_win)
```

#### 3. Weighted Linear Regression

##### Model
Johnson-Nyquist thermal noise relation:
```
V²_corr = a × R + b
```
where:
- **a = 4kTΔf**: Slope, proportional to Boltzmann constant
- **b**: Intercept, represents effective internal resistance noise
- **R**: Resistance (Ω)
- **V²_corr**: Corrected noise power (V²)

##### Weighted Least Squares
- **Function**: `perform_weighted_linear_fit(x, y, sigma_y)`
- Weights: w_i = 1 / σ_i²
- Minimizes: χ² = Σ w_i (y_i - (ax_i + b))²

**Formulas**:
```
W = Σ w_i
W_x = Σ w_i × x_i
W_y = Σ w_i × y_i
W_xx = Σ w_i × x_i²
W_xy = Σ w_i × x_i × y_i

Δ = W × W_xx - W_x²

Slope (a) = (W × W_xy - W_x × W_y) / Δ
Intercept (b) = (W_xx × W_y - W_x × W_xy) / Δ

σ_a = √(W / Δ)
σ_b = √(W_xx / Δ)
```

##### Chi-Squared Goodness of Fit
```
χ² = Σ w_i (y_i - y_fit,i)²
χ²_reduced = χ² / (N - 2)
```
where N is the number of data points, and 2 is the number of fit parameters.

**Interpretation**:
- χ²_reduced ≈ 1: Good fit, uncertainties well-estimated
- χ²_reduced << 1: Overestimated uncertainties
- χ²_reduced >> 1: Underestimated uncertainties or model inadequacy

#### 4. Boltzmann Constant Calculation

##### Extraction from Slope
From Johnson-Nyquist relation:
```
V² = 4kTRΔf
```
Therefore:
```
k = a / (4TΔf)
```

- **Function**: `calculate_boltzmann_constant(a, a_err, T, delta_f)`
- **a**: Fitted slope (V²/Ω)
- **T = 287.75 K**: Temperature
- **Δf = 250 Hz**: Bandwidth

##### Uncertainty Propagation
```
σ_k = σ_a / (4TΔf)
```

Assumes T and Δf are exact (negligible uncertainty compared to measurement uncertainty).

#### 5. Internal Resistance Calculation

##### Physical Interpretation
The intercept b represents noise from effective internal resistance R₀:
```
b = 4kTΔf × R₀ = a × R₀
```
Therefore:
```
R₀ = b / a
```

##### Uncertainty Propagation (Quotient)
```
(σ_R₀ / R₀)² = (σ_a / a)² + (σ_b / b)²

σ_R₀ = R₀ × √((σ_a / a)² + (σ_b / b)²)
```

**Function**: `calculate_internal_resistance(a, b, a_err, b_err)`

#### 6. Statistical Validation

##### t-Test for Boltzmann Constant
Compares measured value with known value:
```
t = |k_measured - k_expected| / σ_k
```

**Interpretation**:
- t < 1: Excellent agreement (< 1σ)
- t < 2: Good agreement (< 2σ)
- t < 3: Acceptable agreement (< 3σ)
- t > 3: Significant discrepancy

**Function**: `perform_t_test(measured, measured_err, expected)`

##### Relative Error
```
ε_rel = |k_measured - k_expected| / k_expected × 100%
```

#### 7. Visualization
- **Scatter plot**: V²_corr vs. R with error bars (±2σ)
- **Fit line**: Linear regression result
- **Text box**: Fit parameters, derived quantities, corrections
- Output: `output/results/noise_vs_resistance_fit.png`

### Calculations Summary

#### Key Formulas

**1. Johnson-Nyquist Thermal Noise**
```
⟨V²⟩ = 4kTRΔf
```
- k: Boltzmann constant (1.380649 × 10⁻²³ J/K)
- T: Absolute temperature (K)
- R: Resistance (Ω)
- Δf: Measurement bandwidth (Hz)

**2. dB to Linear Conversion**
```
V²_linear = 10^(V²_dB / 10)
```

**3. Systematic Corrections**
```
V²_true = V²_measured / (G² × C_win)
```
- G: Preamplifier gain
- C_win: Window correction factor (1.30 for Hamming)

**4. Error Propagation (Exponential)**
```
σ_f(x) = |df/dx| × σ_x

For f(x) = 10^(x/10):
σ_V² = V² × (ln(10)/10) × σ_dB
```

**5. Error Propagation (Quotient)**
```
For R = A/B:
(σ_R/R)² = (σ_A/A)² + (σ_B/B)²
```

### Outputs
- Plot: `output/results/noise_vs_resistance_fit.png`
  - Scatter plot with error bars
  - Linear fit line
  - Fit parameters and derived quantities
- Console output:
  - Correction parameters
  - Linear fit results (slope, intercept, χ²)
  - Boltzmann constant with uncertainty
  - t-test results
  - Internal resistance with uncertainty

---

## Uncertainty Analysis

### Types of Uncertainties Considered

#### 1. Statistical Uncertainty (Type A)
- **Source**: Random fluctuations in noise measurements
- **Quantification**: Standard deviation (σ) from Gaussian fit
- **Propagation**: Through all calculation steps
- **Scripts 2 & 3**

#### 2. Systematic Uncertainties (Type B)
- **Preamplifier gain**: G = 100 (assumed exact)
- **Window correction**: C_win = 1.30 for Hamming window (theoretical value)
- **Temperature**: T = 287.75 K (assumed exact from measurement)
- **Bandwidth**: Δf = 250 Hz (FFT bin width, exact by definition)

**Note**: Script 3 applies systematic corrections but does not propagate their uncertainties. This could be a future enhancement.

### Uncertainty Propagation Chain

```
Raw FFT Data (dBm)
    ↓
[Script 2: Statistical filtering]
    ↓
Mean ± σ (dB) for white noise baseline
    ↓
[Script 3: Convert to linear, σ propagated via calculus]
    ↓
V²_measured ± σ_V²_measured (linear scale)
    ↓
[Script 3: Apply corrections, σ scaled]
    ↓
V²_corrected ± σ_V²_corrected (V²)
    ↓
[Script 3: Weighted linear fit]
    ↓
Slope a ± σ_a, Intercept b ± σ_b
    ↓
[Script 3: Extract physical parameters]
    ↓
k_B ± σ_k (J/K), R₀ ± σ_R₀ (Ω)
```

### Confidence Intervals
- **1σ (68.3% CI)**: Calculated directly from fit
- **2σ (95.4% CI)**: Multiplied by `SIGMA_MULTIPLIER = 2`
- Reported in final results for practical interpretation

---

## Workflow Execution

### Sequential Execution
1. Run `noise_analysis.py`:
   - Produces overview plots
   - Provides initial data exploration
   - No output files needed by subsequent scripts

2. Run `noise_filter_analysis.py`:
   - Filters white noise from interference
   - Produces `output/results/noise_analysis_results.txt`
   - **Required** for Script 3

3. Run `analyze_noise_vs_resistance.py`:
   - Reads results from Script 2
   - Performs physical parameter extraction
   - Produces final analysis

### Dependencies
```
noise_analysis.py (independent)

noise_filter_analysis.py (independent)
    ↓
    noise_analysis_results.txt
    ↓
analyze_noise_vs_resistance.py
```

### Input/Output Summary

| Script | Input | Output |
|--------|-------|--------|
| `noise_analysis.py` | `experiment 1 bandwidth 250Hz 14.6[C]/*.txt` | 9 PNG plots, console statistics |
| `noise_filter_analysis.py` | `experiment 1 bandwidth 250Hz 14.6[C]/*.txt` | 16 PNG plots, `noise_analysis_results.txt` |
| `analyze_noise_vs_resistance.py` | `output/results/noise_analysis_results.txt` | 1 PNG plot, console results |

---

## Example Results

### From Script 2 (noise_filter_analysis.py)

```
Resistance   | Mean V² (dB) |   σ (dB)  | ±2σ (95% CI)
------------ | ------------ | --------- | ------------
913 Ω        | -100.6205    | 0.6858    | ±1.3715
1.0 kΩ       | -100.1019    | 0.7588    | ±1.5176
2.4 kΩ       | -97.7779     | 0.3231    | ±0.6463
5.0 kΩ       | -95.0794     | 0.2105    | ±0.4211
22.0 kΩ      | -88.7809     | 0.2132    | ±0.4264
47.5 kΩ      | -85.2898     | 0.1898    | ±0.3795
68.3 kΩ      | -83.7577     | 0.1918    | ±0.3835
```

**Observations**:
- Noise amplitude increases with resistance (less negative dB)
- Lower uncertainty for higher resistances (better signal-to-noise)
- Typical uncertainty: 0.2-0.8 dB (1σ)

### From Script 3 (analyze_noise_vs_resistance.py)

**Example output**:
```
LINEAR FIT RESULTS: V² = a·R + b (2σ uncertainties)
Slope (a):      [value] ± [error] V²/Ω
Intercept (b):  [value] ± [error] V²
χ²/dof:         [value]

Boltzmann constant (k):
  Measured:  k = [value] ± [error] J/K  (2σ)
  Expected:  k = 1.3806e-23 J/K
  Relative error: [value] %
  t-statistic: [value]
  Agreement: [description]

Effective internal resistance (R₀):
  R₀ = [value] ± [error] Ω  (2σ)
```

---

## Technical Notes

### Filtering Methodology
The IQR-based filtering with neighbor rejection is chosen because:
1. **Robust**: IQR is insensitive to outliers
2. **Physical**: Interference peaks are additive, creating high-amplitude outliers
3. **Conservative**: Neighbor rejection ensures clean baseline

### Window Correction Factor
The Hamming window used in FFT analysis distorts power estimates. The correction factor C_win = 1.30 accounts for:
- Amplitude reduction at window edges
- Effective noise bandwidth change
- Theoretical value for Hamming window

### Why dB Scale?
1. **Dynamic range**: Noise spans orders of magnitude
2. **Gaussian approximation**: White noise is Gaussian in dB
3. **Standard**: FFT analyzers output in dB

### Why Convert to Linear for Fitting?
1. **Physical model**: V² ∝ R is linear in linear scale
2. **Error propagation**: Standard weighted least squares applies
3. **Parameter extraction**: Direct calculation of k_B from slope

---

## Limitations and Assumptions

### Assumptions
1. **White noise**: Constant power spectral density across frequency
2. **Gaussian statistics**: Noise follows normal distribution
3. **Linear model**: V² = aR + b is valid
4. **Exact corrections**: G, C_win, T, Δf have negligible uncertainty
5. **Thermal equilibrium**: Resistors at constant temperature

### Known Limitations
1. **Interference rejection**: Some genuine noise may be removed
2. **1/f noise**: Not explicitly modeled, but filtered out
3. **Systematic uncertainties**: Not fully propagated
4. **Temperature uniformity**: Single temperature value assumed

### Potential Improvements
1. Propagate systematic uncertainties (G, C_win, T)
2. Model 1/f noise explicitly
3. Use frequency-dependent analysis
4. Include correlations in uncertainty propagation
5. Perform Monte Carlo uncertainty analysis

---

## Physical Context

### Johnson-Nyquist Noise
Thermal noise arises from random thermal motion of charge carriers in resistors:
```
⟨V²⟩ = 4kTRΔf
```

This fundamental relation connects:
- **Thermodynamics**: Temperature T, Boltzmann constant k
- **Electronics**: Resistance R, voltage V
- **Information theory**: Bandwidth Δf

### Experimental Goal
By measuring noise voltage as a function of resistance, extract the Boltzmann constant k_B, validating:
1. Johnson-Nyquist theory
2. Measurement calibration
3. Analysis methodology

### Applications
- Calibration of noise measurement systems
- Validation of theoretical noise models
- Educational demonstration of statistical mechanics
- Precision metrology

---

## File Locations

### Scripts
- `/Data analysis/Noise in a electric sercet/noise_analysis.py`
- `/Data analysis/Noise in a electric sercet/noise_filter_analysis.py`
- `/Data analysis/Noise in a electric sercet/analyze_noise_vs_resistance.py`

### Data
- `/Data analysis/Noise in a electric sercet/experiment 1 bandwidth 250Hz 14.6[C]/*.txt`

### Outputs
- Plots: `/Data analysis/Noise in a electric sercet/output/{filtered_plots,distribution_plots,results}/`
- Results: `/Data analysis/Noise in a electric sercet/output/results/noise_analysis_results.txt`
- Final plot: `/Data analysis/Noise in a electric sercet/output/results/noise_vs_resistance_fit.png`

---

## References

### Theoretical Background
- **Johnson-Nyquist Noise**: Fundamental thermal noise in resistors
- **FFT Analysis**: Frequency-domain representation of time-domain signals
- **Hamming Window**: Spectral leakage reduction in FFT
- **Weighted Least Squares**: Optimal linear regression with heteroscedastic errors

### Statistical Methods
- **Interquartile Range (IQR)**: Robust outlier detection
- **Gaussian Distribution**: Model for white noise statistics
- **Chi-squared Test**: Goodness of fit assessment
- **t-Test**: Hypothesis testing for parameter validation
- **Error Propagation**: Uncertainty quantification through calculations

---

## Revision History

| Date | Version | Changes |
|------|---------|---------|
| 2026-01-13 | 1.0 | Initial documentation |

---

**Document prepared for**: Physics data analysis project
**Experiment**: White noise measurement in electrical circuits
**Instrumentation**: FFT analyzer, SR552 preamplifier (G=100)
**Analysis**: Python 3 with NumPy, SciPy, Matplotlib
