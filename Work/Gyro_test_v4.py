#!/usr/bin/env python3
# Multi-IMU High-Speed Vibration Logger
# Supports ICM20649 and LSM6DSO32 sensors with automatic detection

import time
import sys
import types
from datetime import datetime
import os
import gzip

# ---------------- CONFIGURATION ----------------
I2C_BUS_NUMBER = 6
DEBUG = True  # Set to False to disable screen output
TARGET_HZ = 0  # Target sample rate (set to 0 for maximum speed)

# Log directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = "/media/ssd"

# IMU sensor definitions - scans all these addresses
IMU_DEFS = [
    {"name": "ICM20649", "address": 0x68, "type": "icm"},
    {"name": "ICM20649", "address": 0x69, "type": "icm"},
    {"name": "LSM6DSO32", "address": 0x6A, "type": "lsm"},
    {"name": "LSM6DSO32", "address": 0x6B, "type": "lsm"},
]

# ---- HARDENING: stub modules to avoid GPIO detection ----
digitalio_stub = types.ModuleType("digitalio")

class _NoopDigitalInOut:
    def __init__(self, *args, **kwargs): pass

digitalio_stub.DigitalInOut = _NoopDigitalInOut
sys.modules["digitalio"] = digitalio_stub
# ---------------------------------------------------------

from adafruit_extended_bus import ExtendedI2C as I2C

# Import both sensor libraries
try:
    import adafruit_icm20x
    HAS_ICM = True
except ImportError:
    HAS_ICM = False
    print("⚠ adafruit_icm20x not installed - ICM20649 support disabled")

try:
    import adafruit_lsm6ds
    from adafruit_lsm6ds.lsm6dso32 import LSM6DSO32
    HAS_LSM = True
except ImportError:
    HAS_LSM = False
    print("⚠ adafruit_lsm6ds not installed - LSM6DSO32 support disabled")

if not HAS_ICM and not HAS_LSM:
    print("❌ No IMU libraries available!")
    print("Install with: python3.8 -m pip install --user adafruit-circuitpython-icm20x adafruit-circuitpython-lsm6ds")
    sys.exit(1)

# ---------------- SENSOR DETECTION ----------------
print(f"Scanning I2C bus {I2C_BUS_NUMBER} for IMU sensors...\n")

# Initialize I2C
i2c = I2C(I2C_BUS_NUMBER)

detected_sensors = []

for imu_def in IMU_DEFS:
    addr = imu_def["address"]
    imu_type = imu_def["type"]
    name = imu_def["name"]
    
    print(f"Trying {name} @ 0x{addr:02X}...", end=" ")
    
    try:
        if imu_type == "icm" and HAS_ICM:
            # Initialize ICM20649
            imu = adafruit_icm20x.ICM20649(i2c, address=addr)
            print("✓ ICM20649 detected")
            
            # Configure ICM20649 with maximum ranges for vibration analysis
            print(f"  Configuring...", end=" ")
            try:
                # ICM20649 supports: ±4g, ±8g, ±16g, ±30g
                # Use ±30g for high vibration environments
                imu.accelerometer_range = adafruit_icm20x.AccelRange.RANGE_30G
                print(f"Accel:±30G", end=" ")

                # ICM20649 supports: ±500, ±1000, ±2000, ±4000 dps
                # Use ±4000 dps for maximum range
                imu.gyro_range = adafruit_icm20x.GyroRange.RANGE_4000_DPS
                print(f"Gyro:±4000dps", end=" ")

                print("✓")
            except Exception as e:
                print(f"⚠ Config warning: {e}")
            
            detected_sensors.append({
                'imu': imu,
                'address': addr,
                'name': f'{name}_0x{addr:02X}',
                'type': 'icm'
            })
            
        elif imu_type == "lsm" and HAS_LSM:
            # Initialize LSM6DSO32
            imu = LSM6DSO32(i2c, address=addr)
            print("✓ LSM6DSO32 detected")
            
            # Configure LSM6DSO32
            print(f"  Configuring...", end=" ")
            try:
                # Set to maximum ranges for vibration analysis
                imu.accelerometer_range = adafruit_lsm6ds.AccelRange.RANGE_32G
                imu.gyro_range = adafruit_lsm6ds.GyroRange.RANGE_2000_DPS
                print(f"Accel:±32G Gyro:±2000dps ✓")
            except Exception as e:
                print(f"⚠ Config warning: {e}")
            
            detected_sensors.append({
                'imu': imu,
                'address': addr,
                'name': f'{name}_0x{addr:02X}',
                'type': 'lsm'
            })
        else:
            print(f"✗ Library not available")
            
    except Exception as e:
        print(f"✗ Not found")
        if DEBUG:
            print(f"    Error: {e}")

if not detected_sensors:
    print("\n❌ No IMU sensors detected!")
    print(f"Run 'i2cdetect -y {I2C_BUS_NUMBER}' to verify hardware")
    sys.exit(1)

print(f"\n✓ Total sensors detected: {len(detected_sensors)}")
for sensor in detected_sensors:
    print(f"  - {sensor['name']} ({sensor['type'].upper()})")
print()

# ---------------- CREATE LOG DIRECTORY & CSV FILES ----------------
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
    print(f"Created log directory: {LOG_DIR}")

timestamp = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
csv_files = {}

for sensor in detected_sensors:
    filename = os.path.join(LOG_DIR, f"{timestamp}_{sensor['name']}.csv.gz")
    f = gzip.open(filename, 'wt', compresslevel=4)
    f.write("timestamp_epoch,ax_ms2,ay_ms2,az_ms2,gx_rads,gy_rads,gz_rads\n")
    csv_files[sensor['address']] = {
        'file': f,
        'filename': filename
    }
    print(f"Logging {sensor['name']} → {os.path.basename(filename)}")

print("\n" + "="*60)
print("HIGH-SPEED LOGGING STARTED")
print("="*60)
if DEBUG:
    print("Debug output enabled - showing latest readings")
print("Press Ctrl+C to stop\n")

# ---------------- HIGH-SPEED LOGGING LOOP ----------------
sample_counts = {sensor['address']: 0 for sensor in detected_sensors}
start_time = time.time()
last_debug_time = time.time()

# Pre-calculate timing
if TARGET_HZ > 0:
    sample_interval = 1.0 / TARGET_HZ
else:
    sample_interval = 0  # No delay - maximum speed

# Use monotonic time for more accurate timing
next_sample_time = time.monotonic()

# Buffer for batch writing (reduces I/O overhead)
BUFFER_SIZE = 100  # Write to disk every N samples
buffers = {sensor['address']: [] for sensor in detected_sensors}

try:
    while True:
        loop_start = time.monotonic()

        # Use time.time() for timestamp (more precise than datetime for high-speed)
        current_time = time.time()

        for sensor in detected_sensors:
            imu = sensor['imu']
            addr = sensor['address']

            try:
                # Read acceleration and gyro data
                accel = imu.acceleration
                gyro = imu.gyro
                ax, ay, az = accel
                gx, gy, gz = gyro

                # Buffer the data instead of writing immediately
                # Reduced precision: 4 decimals for accel (m/s²), 5 for gyro (rad/s)
                buffers[addr].append(f"{current_time:.6f},{ax:.4f},{ay:.4f},{az:.4f},{gx:.5f},{gy:.5f},{gz:.5f}\n")
                sample_counts[addr] += 1

                # Store for debug display
                sensor['last_accel'] = accel
                sensor['last_gyro'] = gyro

            except Exception as e:
                if DEBUG:
                    print(f"⚠ Error reading {sensor['name']}: {e}")

        # Flush buffers when they reach BUFFER_SIZE
        for addr in buffers:
            if len(buffers[addr]) >= BUFFER_SIZE:
                csv_files[addr]['file'].write(''.join(buffers[addr]))
                buffers[addr] = []

        # Debug output every 1 second (reduced frequency for speed)
        if DEBUG and (loop_start - last_debug_time) > 1.0:
            elapsed = time.time() - start_time
            print(f"\n--- t={elapsed:.1f}s ---")

            for sensor in detected_sensors:
                if 'last_accel' in sensor and 'last_gyro' in sensor:
                    ax, ay, az = sensor['last_accel']
                    gx, gy, gz = sensor['last_gyro']
                    count = sample_counts[sensor['address']]
                    rate = count / elapsed if elapsed > 0 else 0
                    print(f"{sensor['name']:20s}: {count:6d} samples ({rate:5.1f} Hz) | "
                          f"A=({ax:+7.2f},{ay:+7.2f},{az:+7.2f})")

            last_debug_time = loop_start

        # Precise timing control
        if sample_interval > 0:
            next_sample_time += sample_interval
            sleep_time = next_sample_time - time.monotonic()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                # We're falling behind - reset timing
                next_sample_time = time.monotonic()
        
except KeyboardInterrupt:
    print("\n\n" + "="*60)
    print("STOPPING LOGGER...")
    print("="*60)

    # Flush any remaining buffered data
    for addr in buffers:
        if buffers[addr]:
            csv_files[addr]['file'].write(''.join(buffers[addr]))

    # Close all CSV files (flush ensures gzip stream is finalized properly)
    for addr in csv_files:
        csv_files[addr]['file'].flush()
        csv_files[addr]['file'].close()
    
    # Print summary
    elapsed = time.time() - start_time
    print(f"\nLogging duration: {elapsed:.1f} seconds")
    print("\nSample counts:")
    total_samples = 0
    for sensor in detected_sensors:
        count = sample_counts[sensor['address']]
        rate = count / elapsed if elapsed > 0 else 0
        total_samples += count
        print(f"  {sensor['name']:20s}: {count:6,} samples ({rate:6.1f} samples/sec)")
    
    avg_rate = total_samples / elapsed if elapsed > 0 else 0
    print(f"\n  Total: {total_samples:,} samples ({avg_rate:.1f} samples/sec)")
    
    print("\nData saved to:")
    for addr in csv_files:
        print(f"  {csv_files[addr]['filename']}")
    
    print("\n✓ Logging stopped cleanly")
