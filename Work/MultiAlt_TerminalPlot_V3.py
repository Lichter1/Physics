#!/usr/bin/env python3
# Live terminal plots (ASCII) for 4 sensors: 2x BMP581 (0x47, 0x46) and 2x BMP390 (0x77, 0x76) on /dev/i2c-6
# CSV logging only with descriptive sensor naming
# Usage: python3 script.py [--stream_enable] [--testname=NAME]

import time, sys, types, shutil, select, csv, os, argparse
from collections import deque
from datetime import datetime

# ---- USER CONFIGURABLE DEFAULTS ----
DEFAULT_TESTNAME = None  # Change this to set a default test name (e.g., "Flight_Test")

# ---- Command line argument parsing ----
parser = argparse.ArgumentParser(description='Drone sensor data logger with optional live display')
parser.add_argument('--stream_enable', action='store_true',
                    help='Enable live streaming/display of sensor data (default: logging only)')
parser.add_argument('--testname', type=str, default=DEFAULT_TESTNAME,
                    help='Test name to include in CSV filename (default: None)')
args = parser.parse_args()

# Global flags
STREAM_MODE = args.stream_enable
TEST_NAME = args.testname

# ---- HARDENING: stub 'digitalio' BEFORE importing adafruit_bmp3xx ----
digitalio_stub = types.ModuleType("digitalio")


class _NoopDigitalInOut:
    def __init__(self, *args, **kwargs): pass


digitalio_stub.DigitalInOut = _NoopDigitalInOut
sys.modules["digitalio"] = digitalio_stub
# ----------------------------------------------------------------------

# Import asciichartpy only if streaming is enabled
if STREAM_MODE:
    try:
        import asciichartpy as ac
    except ImportError:
        print("asciichartpy not installed. Run: pip3 install asciichartpy")
        print("Or run without --stream_enable for logging-only mode")
        sys.exit(1)

from adafruit_extended_bus import ExtendedI2C as I2C
import bmp581  # pip: circuitpython-bmp581
from adafruit_bmp3xx import BMP3XX_I2C  # pip: adafruit-circuitpython-bmp3xx

# ---------------- Configuration ----------------
I2C_BUS_NUMBER = 6
SAMPLE_PERIOD = 0.01  # seconds between updates (actual delay ~0.02s due to ~0.01s script processing time)
ALT_HEIGHT = 10  # chart height (rows) for altitude
PRES_HEIGHT = 10  # chart height (rows) for pressure

# Sensor configuration with descriptive names
SENSORS_CONFIG = {
    'BMP581_0x47_back_BackDrone': {
        'type': 'BMP581',
        'address': 0x47,
        'location': 'back_BackDrone'
    },
    'BMP581_0x46_front_BackDrone': {
        'type': 'BMP581',
        'address': 0x46,
        'location': 'front_BackDrone'
    },
    'BMP390_0x77_Heat_sink': {
        'type': 'BMP390',
        'address': 0x77,
        'location': 'Heat_sink'
    },
    'BMP390_0x76_Orange_cube': {
        'type': 'BMP390',
        'address': 0x76,
        'location': 'Orange_cube'
    }
}

# Logging configuration
LOG_DIR = os.path.expanduser("~/Omri/sensor_logs")
CSV_FILE = None
csv_writer = None
sample_count = 0


# ------------------------------------------------

def rel_altitude_m(p_hPa, p0_hPa):
    """ISA hypsometric approximation"""
    return 44330.0 * ((p0_hPa / p_hPa) ** 0.1903 - 1.0)


def term_clear():
    """Clear terminal screen"""
    sys.stdout.write("\x1b[2J\x1b[H")  # clear + home
    sys.stdout.flush()


def init_logging():
    """Initialize CSV logging with descriptive headers"""
    global CSV_FILE, csv_writer

    # Create logs directory if it doesn't exist
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    # Generate timestamped filename with optional test name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if TEST_NAME:
        csv_filename = os.path.join(LOG_DIR, f"{TEST_NAME}_drone_sensor_data_{timestamp}.csv")
    else:
        csv_filename = os.path.join(LOG_DIR, f"drone_sensor_data_{timestamp}.csv")

    # Initialize CSV file with descriptive headers
    CSV_FILE = open(csv_filename, 'w', newline='')
    csv_writer = csv.writer(CSV_FILE)

    # Create headers based on sensor configuration
    headers = ['timestamp', 'elapsed_time_s']
    for sensor_name in SENSORS_CONFIG.keys():
        headers.extend([
            f'{sensor_name}_pressure_hPa',
            f'{sensor_name}_temp_C',
            f'{sensor_name}_alt_m'
        ])

    csv_writer.writerow(headers)
    print(f"Logging to: {csv_filename}")
    return csv_filename


def log_data(timestamp, elapsed, data):
    """Log data to CSV file"""
    global csv_writer, sample_count

    # Prepare row data
    row = [timestamp.isoformat(), elapsed]
    for sensor_name in SENSORS_CONFIG.keys():
        if sensor_name in data:
            row.extend([
                data[sensor_name]['pressure'],
                data[sensor_name]['temp'],
                data[sensor_name]['alt']
            ])
        else:
            row.extend([None, None, None])  # Handle missing sensors

    csv_writer.writerow(row)
    CSV_FILE.flush()
    sample_count += 1


def cleanup_logging():
    """Clean up logging resources"""
    global CSV_FILE
    if CSV_FILE:
        CSV_FILE.close()


def window_len():
    """Calculate appropriate buffer window length based on terminal width"""
    if not STREAM_MODE:
        return 1  # Minimal buffer when not streaming
    cols = shutil.get_terminal_size((100, 30)).columns
    # leave space for labels; keep a reasonable max
    return max(20, min(cols - 15, 120))


# Open shared I2C bus
i2c = I2C(I2C_BUS_NUMBER)

# Initialize sensors with descriptive names
sensors = {}
baselines = {}
print("Initializing sensors...")

for sensor_name, config in SENSORS_CONFIG.items():
    try:
        if config['type'] == 'BMP581':
            sensors[sensor_name] = bmp581.BMP581(i2c, address=config['address'])
        elif config['type'] == 'BMP390':
            sensors[sensor_name] = BMP3XX_I2C(i2c, address=config['address'])

        baselines[sensor_name] = None
        print(f"✓ {sensor_name} initialized")

    except Exception as e:
        print(f"✗ Failed to initialize {sensor_name}: {e}")
        sys.exit(1)

# Initialize logging
csv_file = init_logging()

# Initialize data buffers (only needed in stream mode)
if STREAM_MODE:
    win = window_len()
    buffers = {
        'alt': {sensor_name: deque(maxlen=win) for sensor_name in SENSORS_CONFIG.keys()},
        'pres': {sensor_name: deque(maxlen=win) for sensor_name in SENSORS_CONFIG.keys()}
    }
    term_clear()
    if TEST_NAME:
        print(f"Starting stream mode for test: {TEST_NAME}...")
    print("(press 'z' + Enter to re-zero alt baselines; Ctrl+C to exit)\n")
else:
    if TEST_NAME:
        print(f"Starting logging mode for test: {TEST_NAME}")
    else:
        print("Starting logging mode...")
    print(f"CSV logging to {os.path.basename(csv_file)}")
    print("Press Ctrl+C to stop")

t0 = time.time()

try:
    while True:
        # Handle dynamic terminal width (resize-safe) - only in stream mode
        if STREAM_MODE:
            wl = window_len()
            if wl != list(buffers['alt'].values())[0].maxlen:
                for sensor_type in ['alt', 'pres']:
                    for sensor_name in buffers[sensor_type]:
                        buffers[sensor_type][sensor_name] = deque(buffers[sensor_type][sensor_name], maxlen=wl)

        # Read all sensors
        current_data = {}
        timestamp = datetime.now()

        for sensor_name, sensor in sensors.items():
            try:
                # Read sensor data based on type
                if SENSORS_CONFIG[sensor_name]['type'] == 'BMP581':
                    pressure_hPa = sensor.pressure * 10.0  # Convert to hPa
                    temp_C = sensor.temperature
                elif SENSORS_CONFIG[sensor_name]['type'] == 'BMP390':
                    pressure_hPa = sensor.pressure  # Already in hPa
                    temp_C = sensor.temperature

                # Initialize baseline (first sample)
                if baselines[sensor_name] is None:
                    baselines[sensor_name] = pressure_hPa

                # Compute relative altitude
                alt_m = rel_altitude_m(pressure_hPa, baselines[sensor_name])

                # Store current readings
                current_data[sensor_name] = {
                    'pressure': pressure_hPa,
                    'temp': temp_C,
                    'alt': alt_m
                }

                # Update buffers (only in stream mode)
                if STREAM_MODE:
                    buffers['alt'][sensor_name].append(alt_m)
                    buffers['pres'][sensor_name].append(pressure_hPa)

            except Exception as e:
                print(f"Error reading {sensor_name}: {e}")
                current_data[sensor_name] = {'pressure': None, 'temp': None, 'alt': None}

        # Log data
        elapsed = time.time() - t0
        log_data(timestamp, elapsed, current_data)

        # Handle user input and display (only in stream mode)
        if STREAM_MODE:
            # Non-blocking keystroke: 'z' + Enter to re-zero
            rezero = False
            if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                cmd = sys.stdin.readline().strip().lower()
                if cmd == 'z':
                    rezero = True

            if rezero:
                for sensor_name in sensors.keys():
                    if sensor_name in current_data and current_data[sensor_name]['pressure'] is not None:
                        baselines[sensor_name] = current_data[sensor_name]['pressure']

                for sensor_type in ['alt', 'pres']:
                    for sensor_name in buffers[sensor_type]:
                        buffers[sensor_type][sensor_name].clear()

            # Render charts
            term_clear()
            print("4-Sensor Drone Setup â€” /dev/i2c-%d â€” t=%.1fs" % (I2C_BUS_NUMBER, elapsed))
            print("CSV Log: %s" % os.path.basename(csv_file))
            print("Tip: resize terminal to change plot width; 'z' + Enter re-zero\n")

            # ALTITUDE PANEL
            print("Altitude (m, relative to first sample)")
            alt_data = []
            for sensor_name in SENSORS_CONFIG.keys():
                alt_data.append(list(buffers['alt'][sensor_name]))

            try:
                alt_chart = ac.plot(alt_data, {'height': ALT_HEIGHT})
            except Exception:
                # Fallback: overlay individual plots
                alt_chart = ""
                for i, data in enumerate(alt_data):
                    if data:
                        alt_chart += ac.plot(data, {'height': ALT_HEIGHT}) + "\n"
            print(alt_chart)

            # Altitude stats
            for sensor_name in SENSORS_CONFIG.keys():
                if buffers['alt'][sensor_name]:
                    current_alt = buffers['alt'][sensor_name][-1]
                    min_alt = min(buffers['alt'][sensor_name])
                    max_alt = max(buffers['alt'][sensor_name])
                    print(f"  {sensor_name}: {current_alt:+7.2f}m (min:{min_alt:+7.2f} max:{max_alt:+7.2f})")

            print()  # Extra line for spacing

            # PRESSURE PANEL
            print("Pressure (hPa)")
            pres_data = []
            for sensor_name in SENSORS_CONFIG.keys():
                pres_data.append(list(buffers['pres'][sensor_name]))

            try:
                pres_chart = ac.plot(pres_data, {'height': PRES_HEIGHT})
            except Exception:
                pres_chart = ""
                for data in pres_data:
                    if data:
                        pres_chart += ac.plot(data, {'height': PRES_HEIGHT}) + "\n"
            print(pres_chart)

            # Pressure stats
            for sensor_name in SENSORS_CONFIG.keys():
                if buffers['pres'][sensor_name]:
                    current_pres = buffers['pres'][sensor_name][-1]
                    min_pres = min(buffers['pres'][sensor_name])
                    max_pres = max(buffers['pres'][sensor_name])
                    print(f"  {sensor_name}: {current_pres:7.1f} hPa (min:{min_pres:7.1f} max:{max_pres:7.1f})")

            # Temperature status
            print("\nTemperatures:")
            for sensor_name in SENSORS_CONFIG.keys():
                if sensor_name in current_data and current_data[sensor_name]['temp'] is not None:
                    temp = current_data[sensor_name]['temp']
                    print(f"  {sensor_name}: {temp:.2f}°C")

            print(f"\nSamples logged: {sample_count} | Press Ctrl+C to stop")
        #else:
           #  Non-stream mode: print periodic status
            #if sample_count % 100 == 0 or sample_count < 5:  # Print every 100 samples or first 5
             #   print(
              #      f"[{timestamp.strftime('%H:%M:%S')}] Logged {sample_count} samples to {os.path.basename(csv_file)}")

        time.sleep(SAMPLE_PERIOD)

except KeyboardInterrupt:
    print("\nStopping...")
    cleanup_logging()
    print("Data saved to:")
    print(f"  CSV: {csv_file}")
    print("Stopped.")