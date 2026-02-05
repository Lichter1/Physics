#!/usr/bin/env python3
# Multi-MCP9600 Thermocouple Temperature Logger
# Supports multiple MCP9600 sensors on I2C bus with automatic detection
# Usage: python3 MCP9600_MultiTemp_V1.py [--stream_enable] [--testname=NAME]

import time
import sys
import types
import os
import csv
import argparse
from datetime import datetime

# ---- USER CONFIGURABLE DEFAULTS ----
DEFAULT_TESTNAME = None  # Change this to set a default test name (e.g., "Thermal_Test")
I2C_BUS_NUMBER = 6  # I2C bus number (adjust for your Raspberry Pi)
DEBUG = True  # Set to False to disable debug output

# Sensor addresses to scan (MCP9600 default addresses)
MCP9600_ADDRESSES = [0x67, 0x66, 0x65, 0x64, 0x60]

# Sensor configuration with descriptive names (customize as needed)
SENSOR_NAMES = {
    0x67: "MCP9600_0x67_Sensor1",
    0x66: "MCP9600_0x66_Sensor2",
    0x65: "MCP9600_0x65_Sensor3",
    0x64: "MCP9600_0x64_Sensor4",
    0x60: "MCP9600_0x60_Sensor5",
}

# Logging configuration
LOG_DIR = "/media/ssd"  # Change to your preferred log directory
SAMPLE_PERIOD = 0.1  # seconds between samples

# ---- Command line argument parsing ----
parser = argparse.ArgumentParser(description='MCP9600 multi-sensor temperature logger')
parser.add_argument('--stream_enable', action='store_true',
                    help='Enable live streaming/display of sensor data (default: logging only)')
parser.add_argument('--testname', type=str, default=DEFAULT_TESTNAME,
                    help='Test name to include in CSV filename (default: None)')
parser.add_argument('--bus', type=int, default=I2C_BUS_NUMBER,
                    help=f'I2C bus number (default: {I2C_BUS_NUMBER})')
parser.add_argument('--sample_rate', type=float, default=SAMPLE_PERIOD,
                    help=f'Sample period in seconds (default: {SAMPLE_PERIOD})')
args = parser.parse_args()

# Global flags
STREAM_MODE = args.stream_enable
TEST_NAME = args.testname
I2C_BUS_NUMBER = args.bus
SAMPLE_PERIOD = args.sample_rate

# ---- HARDENING: stub 'digitalio' BEFORE importing adafruit libraries ----
digitalio_stub = types.ModuleType("digitalio")


class _NoopDigitalInOut:
    def __init__(self, *args, **kwargs):
        pass


digitalio_stub.DigitalInOut = _NoopDigitalInOut
sys.modules["digitalio"] = digitalio_stub
# ----------------------------------------------------------------------

# Import I2C library
try:
    from adafruit_extended_bus import ExtendedI2C as I2C
except ImportError:
    print("adafruit_extended_bus not installed.")
    print("Install with: pip3 install adafruit-extended-bus")
    sys.exit(1)

# Import MCP9600 library
try:
    import adafruit_mcp9600
    HAS_MCP9600 = True
except ImportError:
    HAS_MCP9600 = False
    print("adafruit_mcp9600 not installed.")
    print("Install with: pip3 install adafruit-circuitpython-mcp9600")
    sys.exit(1)

# ---------------- SENSOR DETECTION ----------------
print(f"MCP9600 Multi-Sensor Temperature Logger")
print("=" * 50)
print(f"Scanning I2C bus {I2C_BUS_NUMBER} for MCP9600 sensors...\n")

# Initialize I2C
try:
    i2c = I2C(I2C_BUS_NUMBER)
except Exception as e:
    print(f"Failed to initialize I2C bus {I2C_BUS_NUMBER}: {e}")
    print(f"Try running: i2cdetect -y {I2C_BUS_NUMBER}")
    sys.exit(1)

detected_sensors = []

for addr in MCP9600_ADDRESSES:
    sensor_name = SENSOR_NAMES.get(addr, f"MCP9600_0x{addr:02X}")
    print(f"Trying {sensor_name} @ 0x{addr:02X}...", end=" ")

    try:
        # Initialize MCP9600
        mcp = adafruit_mcp9600.MCP9600(i2c, address=addr)
        print("OK")

        # Read initial values to verify communication
        if DEBUG:
            print(f"  Initial reading: Hot={mcp.temperature:.2f}C, "
                  f"Cold={mcp.ambient_temperature:.2f}C, "
                  f"Delta={mcp.delta_temperature:.2f}C")

        detected_sensors.append({
            'sensor': mcp,
            'address': addr,
            'name': sensor_name,
        })

    except Exception as e:
        print(f"Not found")
        if DEBUG:
            print(f"    Error: {e}")

if not detected_sensors:
    print(f"\nNo MCP9600 sensors detected on I2C bus {I2C_BUS_NUMBER}!")
    print(f"Run 'i2cdetect -y {I2C_BUS_NUMBER}' to verify hardware connections")
    sys.exit(1)

print(f"\nTotal sensors detected: {len(detected_sensors)}")
for sensor in detected_sensors:
    print(f"  - {sensor['name']}")
print()

# ---------------- CREATE LOG DIRECTORY & CSV FILE ----------------
if not os.path.exists(LOG_DIR):
    try:
        os.makedirs(LOG_DIR)
        print(f"Created log directory: {LOG_DIR}")
    except Exception as e:
        print(f"Warning: Could not create {LOG_DIR}: {e}")
        LOG_DIR = os.path.expanduser("~/sensor_logs")
        os.makedirs(LOG_DIR, exist_ok=True)
        print(f"Using fallback directory: {LOG_DIR}")

# Generate timestamped filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
if TEST_NAME:
    csv_filename = os.path.join(LOG_DIR, f"{TEST_NAME}_MCP9600_temp_{timestamp}.csv")
else:
    csv_filename = os.path.join(LOG_DIR, f"MCP9600_temp_{timestamp}.csv")

# Initialize CSV file with headers
csv_file = open(csv_filename, 'w', newline='')
csv_writer = csv.writer(csv_file)

# Create headers
headers = ['timestamp', 'elapsed_time_s']
for sensor in detected_sensors:
    headers.extend([
        f'{sensor["name"]}_hot_junction_C',
        f'{sensor["name"]}_cold_junction_C',
        f'{sensor["name"]}_delta_C'
    ])
csv_writer.writerow(headers)
print(f"Logging to: {csv_filename}")

# ---------------- MAIN LOGGING LOOP ----------------
print("\n" + "=" * 50)
if STREAM_MODE:
    print("STREAMING MODE ENABLED")
else:
    print("LOGGING MODE (no display)")
print("=" * 50)
print("Press Ctrl+C to stop\n")

sample_count = 0
t0 = time.time()

try:
    while True:
        current_time = datetime.now()
        elapsed = time.time() - t0

        # Prepare row data
        row = [current_time.isoformat(), f"{elapsed:.3f}"]

        # Read all sensors
        readings = {}
        for sensor_info in detected_sensors:
            mcp = sensor_info['sensor']
            name = sensor_info['name']

            try:
                hot_temp = mcp.temperature  # Hot junction (thermocouple) temperature
                cold_temp = mcp.ambient_temperature  # Cold junction (ambient) temperature
                delta_temp = mcp.delta_temperature  # Temperature difference

                readings[name] = {
                    'hot': hot_temp,
                    'cold': cold_temp,
                    'delta': delta_temp
                }

                row.extend([f"{hot_temp:.3f}", f"{cold_temp:.3f}", f"{delta_temp:.3f}"])

            except Exception as e:
                if DEBUG:
                    print(f"Warning: Error reading {name}: {e}")
                readings[name] = {'hot': None, 'cold': None, 'delta': None}
                row.extend(['', '', ''])

        # Write to CSV
        csv_writer.writerow(row)
        csv_file.flush()
        sample_count += 1

        # Display output in stream mode
        if STREAM_MODE:
            # Clear screen and show readings
            sys.stdout.write("\x1b[2J\x1b[H")  # Clear terminal
            print(f"MCP9600 Temperature Logger - t={elapsed:.1f}s - Samples: {sample_count}")
            print("=" * 60)
            print(f"{'Sensor':<30} {'Hot(C)':<10} {'Cold(C)':<10} {'Delta(C)':<10}")
            print("-" * 60)

            for sensor_info in detected_sensors:
                name = sensor_info['name']
                if name in readings and readings[name]['hot'] is not None:
                    r = readings[name]
                    # Color code based on temperature (if terminal supports it)
                    hot_str = f"{r['hot']:>8.2f}"
                    cold_str = f"{r['cold']:>8.2f}"
                    delta_str = f"{r['delta']:>8.2f}"
                    print(f"{name:<30} {hot_str:<10} {cold_str:<10} {delta_str:<10}")
                else:
                    print(f"{name:<30} {'ERROR':<10} {'ERROR':<10} {'ERROR':<10}")

            print("-" * 60)
            print(f"\nLogging to: {os.path.basename(csv_filename)}")
            print("Press Ctrl+C to stop")

        # Non-stream mode: periodic status update
        elif sample_count % 100 == 0:
            print(f"[{current_time.strftime('%H:%M:%S')}] {sample_count} samples logged")

        time.sleep(SAMPLE_PERIOD)

except KeyboardInterrupt:
    print("\n\n" + "=" * 50)
    print("STOPPING LOGGER...")
    print("=" * 50)

# Cleanup
csv_file.close()
elapsed_total = time.time() - t0
rate = sample_count / elapsed_total if elapsed_total > 0 else 0

print(f"\nLogging complete!")
print(f"  Duration: {elapsed_total:.1f} seconds")
print(f"  Samples: {sample_count:,}")
print(f"  Rate: {rate:.1f} samples/sec")
print(f"\nData saved to:")
print(f"  {csv_filename}")
print("\nDone.")
