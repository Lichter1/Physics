#!/usr/bin/env python3
"""
Measurement Parameter Verification Script

This script helps you determine and verify the correct measurement parameters
for your Johnson-Nyquist thermal noise experiment.

It provides diagnostic calculations and asks you to verify instrument settings.
"""

import numpy as np
from pathlib import Path

# Colors for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_header(text):
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'='*70}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{text:^70}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'='*70}{Colors.ENDC}\n")

def print_section(text):
    print(f"\n{Colors.BOLD}{Colors.OKBLUE}{text}{Colors.ENDC}")
    print(f"{Colors.OKBLUE}{'-'*70}{Colors.ENDC}")

def print_warning(text):
    print(f"{Colors.WARNING}⚠️  {text}{Colors.ENDC}")

def print_ok(text):
    print(f"{Colors.OKGREEN}✓ {text}{Colors.ENDC}")

def print_question(text):
    print(f"{Colors.OKCYAN}❓ {text}{Colors.ENDC}")


def verify_temperature():
    """Verify temperature measurement"""
    print_section("1. TEMPERATURE VERIFICATION")

    temp_celsius = 14.6  # From folder name
    temp_kelvin = temp_celsius + 273.15

    print(f"From folder name: {temp_celsius}°C = {temp_kelvin} K")
    print()
    print_question("Was this the actual room temperature during measurement?")
    print_question("Did you measure it with a calibrated thermometer?")
    print_question("Did the temperature stay constant during the experiment?")
    print()
    print(f"Temperature uncertainty recommendations:")
    print(f"  - If measured with calibrated thermometer: ±0.5 K")
    print(f"  - If room temperature assumed: ±2 K")
    print(f"  - If not controlled: ±5 K")
    print()
    print(f"Impact on k_B: Δk/k = ΔT/T = ")
    for dt in [0.5, 2, 5]:
        print(f"  ±{dt} K → ±{dt/temp_kelvin*100:.2f}%")


def verify_preamplifier():
    """Verify preamplifier specifications"""
    print_section("2. PREAMPLIFIER VERIFICATION")

    print("Common models and specifications:")
    print()
    print("SR552 (Stanford Research Systems):")
    print("  - Voltage gain: ×10, ×100, ×1000")
    print("  - Input impedance: 100 MΩ || 25 pF")
    print("  - Frequency range: DC to 1 MHz")
    print()
    print("SR560 (Stanford Research Systems):")
    print("  - Voltage gain: 1 to 50,000 (adjustable)")
    print("  - Input impedance: 100 MΩ || 25 pF")
    print("  - Frequency range: DC to 1 MHz")
    print()

    print_question("What is your preamplifier model?")
    print_question("What gain setting did you use?")
    print_question("Is this VOLTAGE gain (V_out/V_in) or POWER gain?")
    print_warning("Most preamps specify VOLTAGE gain!")
    print()

    print("Gain verification:")
    gain_voltage = 100  # assumed
    gain_db = 20 * np.log10(gain_voltage)
    gain_power = gain_voltage ** 2
    gain_power_db = 10 * np.log10(gain_power)

    print(f"  If voltage gain = {gain_voltage}:")
    print(f"    → In dB (voltage): {gain_db:.1f} dBV")
    print(f"    → Power gain: {gain_power} (or {gain_power_db:.1f} dB)")
    print()

    print_question("What is the input impedance of your preamplifier?")
    print("  Common values: 100 MΩ (high-impedance), 1 MΩ, 50 Ω (matched)")
    print()
    print("If input impedance is HIGH (>1 MΩ):")
    print("  ✓ Minimal loading of source")
    print("  ✓ But voltage divider at analyzer's 50Ω input matters!")
    print("  → Need IMPEDANCE MATCHING correction")
    print()
    print("If input impedance is 50 Ω:")
    print("  ✓ Matched to analyzer")
    print("  ✗ But creates voltage divider with source resistance!")
    print("  → Need different impedance correction")


def verify_fft_analyzer():
    """Verify FFT analyzer settings"""
    print_section("3. FFT ANALYZER / SPECTRUM ANALYZER VERIFICATION")

    print_question("What is your FFT analyzer model?")
    print("  Common models: Agilent/Keysight, Rohde & Schwarz, Tektronix")
    print()

    print("Bandwidth (Δf) clarification:")
    print()
    print("  Resolution Bandwidth (RBW):")
    print("    - The frequency bin width of your FFT")
    print("    - From your data: 250 Hz (0, 250, 500, ... Hz)")
    print()
    print("  Equivalent Noise Bandwidth (ENBW):")
    print("    - The effective noise bandwidth after windowing")
    print("    - ENBW = RBW × correction factor")
    print()
    print("  Window type affects ENBW:")
    print("    - Rectangular: ENBW = RBW")
    print("    - Hamming: ENBW ≈ 1.36 × RBW")
    print("    - Hann: ENBW ≈ 1.50 × RBW")
    print("    - Blackman-Harris: ENBW ≈ 1.97 × RBW")
    print()
    print_question("What window function did you use?")
    print_question("Should you use RBW or ENBW in the formula V² = 4kTRΔf?")
    print()
    print_warning("Answer: Use ENBW for noise measurements!")
    print()

    print("For Hamming window:")
    rbw = 250
    enbw = 1.36 * rbw
    print(f"  RBW = {rbw} Hz")
    print(f"  ENBW ≈ {enbw:.1f} Hz")
    print()
    print(f"If you use RBW instead of ENBW, your k_B will be off by {enbw/rbw:.2f}×")


def verify_window_correction():
    """Verify window correction factor"""
    print_section("4. WINDOW CORRECTION FACTOR (C_win)")

    print("Window correction factor accounts for:")
    print("  1. Power spectral density distortion by windowing")
    print("  2. Depends on how your analyzer processes the FFT")
    print()

    print("Common definitions:")
    print()
    print("  Definition A: Coherent Power Gain")
    print("    - C_win = 1 / (Σw_i/N)²")
    print("    - For Hamming: C_win ≈ 1.59")
    print()
    print("  Definition B: Equivalent Noise Bandwidth factor")
    print("    - C_win = ENBW / RBW")
    print("    - For Hamming: C_win ≈ 1.36")
    print()
    print("  Definition C: Amplitude correction")
    print("    - C_win = sqrt(Σw_i²/N) / (Σw_i/N)")
    print("    - For Hamming: C_win ≈ 1.30")
    print()

    print_warning("The value 1.30 in your script suggests Definition C")
    print()
    print_question("Check your FFT analyzer manual for the exact definition!")
    print_question("Does it already apply window correction to the output?")
    print()
    print("Some analyzers apply correction automatically:")
    print("  ✓ If yes: Set C_win = 1.0 in your script")
    print("  ✗ If no: Use the appropriate C_win value")


def verify_analyzer_input_impedance():
    """Verify analyzer input impedance"""
    print_section("5. ANALYZER INPUT IMPEDANCE")

    print("Standard spectrum analyzer input impedance: 50 Ω")
    print()
    print("Your measurement chain:")
    print()
    print("  [Resistor R] → [Preamp (high Z)] → [Analyzer (50 Ω)]")
    print()
    print("If preamp input impedance is HIGH (>1 MΩ):")
    print("  - Preamp doesn't load the resistor")
    print("  - But analyzer's 50 Ω loads the preamp OUTPUT")
    print("  - Creates voltage divider: V_meas = V_preamp × 50/(50+0)")
    print("  - Actually, preamp output impedance matters!")
    print()
    print_question("What is the OUTPUT impedance of your preamplifier?")
    print("  - Common: 50 Ω (matched) or low (<10 Ω)")
    print()
    print("If preamp output impedance is LOW (<10 Ω):")
    print("  ✓ Drives the 50 Ω analyzer without loading")
    print("  ✓ No correction needed at this stage")
    print()
    print("If preamp output impedance is 50 Ω:")
    print("  - Forms voltage divider with analyzer input")
    print("  - V_analyzer = V_preamp × 50/(50+50) = V_preamp/2")
    print("  → Need to correct by factor of 2 in voltage (4 in power)")


def verify_impedance_matching():
    """Verify impedance matching scenario"""
    print_section("6. IMPEDANCE MATCHING CORRECTION")

    print("Question: Where does the 50 Ω come into play?")
    print()
    print("Scenario 1: Direct measurement (no preamp)")
    print("  [Resistor R] → [Analyzer (50 Ω)]")
    print()
    print("  - Voltage divider: V_meas = V_oc × 50/(R+50)")
    print("  - For R=1kΩ: only 4.8% of voltage measured!")
    print("  - Must correct: V_oc = V_meas × (R+50)/50")
    print()

    print("Scenario 2: With high-impedance preamp")
    print("  [Resistor R] → [Preamp (1MΩ)] → [Analyzer (50 Ω)]")
    print()
    print("  At resistor:")
    print("    - Preamp has high input Z → minimal loading")
    print("    - V_preamp_in ≈ V_oc (no voltage division)")
    print()
    print("  At preamp output:")
    print("    - Depends on preamp output impedance")
    print("    - If low output Z: drives analyzer without loss")
    print()
    print("  In FFT analyzer:")
    print("    - Signal is measured across 50 Ω internal impedance")
    print("    - dBm reading assumes 50 Ω load")
    print("    - Conversion: P_mW → V² uses R=50Ω")
    print()

    print_ok("For Scenario 2 (high-Z preamp with low output Z):")
    print("  1. Convert dBm to V² using 50Ω: V² = P_watts × 50")
    print("  2. Remove preamp gain: V²_true = V²_meas / G²")
    print("  3. Remove window correction: V²_true /= C_win")
    print("  4. NO additional impedance correction needed!")
    print()
    print_warning("Only apply impedance correction if measurement setup differs!")


def verify_data_consistency():
    """Verify data makes physical sense"""
    print_section("7. DATA CONSISTENCY CHECK")

    script_dir = Path(__file__).parent
    results_file = script_dir / 'output/results/noise_analysis_results.txt'

    if not results_file.exists():
        print_warning(f"Results file not found: {results_file}")
        return

    print(f"Reading data from: {results_file}")
    print()

    # Parse data (simplified)
    resistances = []
    v2_db = []

    with open(results_file, 'r') as f:
        for line in f:
            if 'Ω' in line and '|' in line:
                parts = line.split('|')
                r_str = parts[0].strip().replace('Ω', '').strip()
                if 'k' in r_str:
                    r = float(r_str.replace('k', '')) * 1000
                else:
                    r = float(r_str)
                db = float(parts[1].strip())
                resistances.append(r)
                v2_db.append(db)

    if len(resistances) == 0:
        print_warning("No data found in results file")
        return

    resistances = np.array(resistances)
    v2_db = np.array(v2_db)

    print(f"Number of data points: {len(resistances)}")
    print(f"Resistance range: {resistances.min():.0f} - {resistances.max():.0f} Ω")
    print(f"V² (dBm) range: {v2_db.min():.2f} - {v2_db.max():.2f} dBm")
    print()

    print("Checking: Does V² increase with R? (Johnson-Nyquist: V² ∝ R)")
    # Check correlation
    from scipy import stats as sp_stats
    correlation, p_value = sp_stats.pearsonr(resistances, v2_db)
    print(f"  Correlation coefficient: {correlation:.4f}")
    if correlation > 0.9:
        print_ok("Strong positive correlation - good!")
    elif correlation > 0.5:
        print_warning("Moderate correlation - check for issues")
    else:
        print_warning("Weak correlation - something is wrong!")
    print()

    print("Converting to linear scale and checking slope:")
    # Convert to V² in watts (incorrect way, as in original script)
    v2_linear_wrong = 10 ** (v2_db / 10)
    # Correct way
    v2_linear_correct = 10 ** (v2_db / 10) * 0.05  # mW to W, then × 50Ω

    # Rough slope estimate (no corrections)
    slope_wrong = (v2_linear_wrong[-1] - v2_linear_wrong[0]) / (resistances[-1] - resistances[0])
    slope_correct = (v2_linear_correct[-1] - v2_linear_correct[0]) / (resistances[-1] - resistances[0])

    # Expected slope (ignoring gain and window corrections)
    T = 287.75
    delta_f = 250
    k_B = 1.380649e-23
    expected_slope = 4 * k_B * T * delta_f
    gain = 100
    c_win = 1.30
    expected_slope_measured = expected_slope * gain**2 * c_win

    print(f"  Expected slope (after gain & window): {expected_slope_measured:.4e} V²/Ω")
    print(f"  Rough measured slope (old conversion): {slope_wrong:.4e} V²/Ω")
    print(f"  Rough measured slope (new conversion): {slope_correct:.4e} V²/Ω")
    print(f"  Ratio (old/expected): {slope_wrong/expected_slope_measured:.2f}×")
    print(f"  Ratio (new/expected): {slope_correct/expected_slope_measured:.2f}×")
    print()

    if 0.5 < slope_correct/expected_slope_measured < 2:
        print_ok("Slope is within factor of 2 - reasonable!")
    else:
        print_warning(f"Slope is off by {slope_correct/expected_slope_measured:.1f}× - check parameters!")


def provide_recommendations():
    """Provide summary and recommendations"""
    print_section("8. RECOMMENDATIONS")

    print("To get accurate results, you need to:")
    print()
    print("1. Verify temperature measurement")
    print("   → Use calibrated thermometer")
    print("   → Record temperature at start and end")
    print()
    print("2. Verify preamplifier specifications")
    print("   → Check manual for exact gain")
    print("   → Confirm it's voltage gain (not power gain)")
    print("   → Note input and output impedances")
    print()
    print("3. Verify FFT analyzer settings")
    print("   → Confirm window function used")
    print("   → Check if ENBW or RBW is reported")
    print("   → Verify window correction factor")
    print()
    print("4. Determine if impedance correction is needed")
    print("   → High-Z preamp + low output Z: NO correction")
    print("   → Direct measurement to 50Ω: YES correction")
    print()
    print("5. Run the corrected analysis script")
    print("   → Start with APPLY_IMPEDANCE_CORRECTION = False")
    print("   → Check if t-statistic improves")
    print("   → Try with correction if still off")
    print()

    print(f"{Colors.BOLD}Next steps:{Colors.ENDC}")
    print("  1. Review all parameters above")
    print("  2. Edit the CORRECTED script with verified values")
    print("  3. Run: python3 analyze_noise_vs_resistance_CORRECTED.py")
    print("  4. Compare t-statistic before and after")


def main():
    print_header("MEASUREMENT PARAMETER VERIFICATION")

    print("This script helps you verify all measurement parameters")
    print("for your Johnson-Nyquist thermal noise experiment.")
    print()
    print("Please review each section carefully and answer the questions.")

    verify_temperature()
    verify_preamplifier()
    verify_fft_analyzer()
    verify_window_correction()
    verify_analyzer_input_impedance()
    verify_impedance_matching()
    verify_data_consistency()
    provide_recommendations()

    print_header("VERIFICATION COMPLETE")
    print("Review the information above and update your analysis script accordingly.")
    print()


if __name__ == '__main__':
    main()
