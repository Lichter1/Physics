#!/usr/bin/env python3
# Multi-IMU High-Speed Vibration Logger
# Supports ICM20649 and LSM6DSO32 sensors with automatic detection

import time
import sys
import types
from datetime import datetime
import os

# ---------------- CONFIGURATION ----------------
I2C_BUS_NUMBER = 6
DEBUG = True  # Set to False to disable screen output
TARGET_HZ = 500  # Target sample rate for polling mode (set to 0 for maximum speed)

# FIFO Configuration (for ICM20649 only)
USE_FIFO = True  # Enable hardware FIFO for higher sample rates (~1000+ Hz)
FIFO_ODR = 1125  # Output Data Rate in Hz (max 1125 for accel)

# Log directory (vibration_logs subfolder where this script is located)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(SCRIPT_DIR, "vibration_logs")

# ---------------- ICM20649 FIFO REGISTERS ----------------
# Register bank select
ICM_REG_BANK_SEL = 0x7F

# Bank 0 registers
ICM_USER_CTRL = 0x03
ICM_FIFO_EN_2 = 0x67
ICM_FIFO_RST = 0x68
ICM_FIFO_MODE = 0x69
ICM_FIFO_COUNT_H = 0x70
ICM_FIFO_COUNT_L = 0x71
ICM_FIFO_R_W = 0x72

# Bank 2 registers (sample rate dividers)
ICM_ACCEL_SMPLRT_DIV_1 = 0x10
ICM_ACCEL_SMPLRT_DIV_2 = 0x11
ICM_GYRO_SMPLRT_DIV = 0x00

# Bit masks
ICM_BIT_FIFO_EN = 0x40
ICM_BIT_ACCEL_FIFO_EN = 0x10
ICM_BITS_GYRO_FIFO_EN = 0x0E  # All 3 gyro axes

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

# ---------------- FIFO HELPER FUNCTIONS ----------------
def icm_write_reg(i2c, addr, reg, value):
    """Write single byte to ICM20649 register"""
    while not i2c.try_lock():
        pass
    try:
        i2c.writeto(addr, bytes([reg, value]))
    finally:
        i2c.unlock()

def icm_read_reg(i2c, addr, reg, length=1):
    """Read bytes from ICM20649 register"""
    while not i2c.try_lock():
        pass
    try:
        result = bytearray(length)
        i2c.writeto_then_readfrom(addr, bytes([reg]), result)
        return result
    finally:
        i2c.unlock()

def icm_set_bank(i2c, addr, bank):
    """Switch register bank (0-3)"""
    icm_write_reg(i2c, addr, ICM_REG_BANK_SEL, bank << 4)

def setup_fifo(i2c, addr, odr=1125):
    """Enable FIFO with specified output data rate"""
    print(f"    Setting up FIFO (ODR={odr}Hz)...", end=" ")

    # Bank 0 - Reset and configure FIFO
    icm_set_bank(i2c, addr, 0)

    # Reset FIFO
    icm_write_reg(i2c, addr, ICM_FIFO_RST, 0x1F)
    time.sleep(0.01)
    icm_write_reg(i2c, addr, ICM_FIFO_RST, 0x00)

    # Set FIFO mode (0x00 = stream mode, continues when full)
    icm_write_reg(i2c, addr, ICM_FIFO_MODE, 0x00)

    # Enable accel and gyro in FIFO
    icm_write_reg(i2c, addr, ICM_FIFO_EN_2,
                  ICM_BIT_ACCEL_FIFO_EN | ICM_BITS_GYRO_FIFO_EN)

    # Bank 2 - Set sample rate
    icm_set_bank(i2c, addr, 2)
    # Sample rate divider: ODR = 1125 / (1 + div)
    div = max(0, (1125 // odr) - 1)
    icm_write_reg(i2c, addr, ICM_ACCEL_SMPLRT_DIV_1, (div >> 8) & 0xFF)
    icm_write_reg(i2c, addr, ICM_ACCEL_SMPLRT_DIV_2, div & 0xFF)
    icm_write_reg(i2c, addr, ICM_GYRO_SMPLRT_DIV, div & 0xFF)

    # Back to Bank 0
    icm_set_bank(i2c, addr, 0)

    # Enable FIFO (MUST be last, after sensor is awake!)
    icm_write_reg(i2c, addr, ICM_USER_CTRL, ICM_BIT_FIFO_EN)

    actual_odr = 1125 / (1 + div)
    print(f"✓ (actual ODR: {actual_odr:.0f} Hz)")
    return actual_odr

def read_fifo(i2c, addr, accel_scale, gyro_scale):
    """Read all samples from FIFO, return list of (ax,ay,az,gx,gy,gz) tuples"""
    # Read FIFO count
    count_data = icm_read_reg(i2c, addr, ICM_FIFO_COUNT_H, 2)
    fifo_count = (count_data[0] << 8) | count_data[1]

    if fifo_count < 12:
        return []

    # Calculate complete samples (12 bytes each: 6 accel + 6 gyro)
    num_samples = fifo_count // 12
    bytes_to_read = num_samples * 12

    # Limit read size to prevent I2C issues (max ~500 bytes at a time)
    if bytes_to_read > 504:
        bytes_to_read = 504
        num_samples = 42

    # Burst read FIFO data
    data = icm_read_reg(i2c, addr, ICM_FIFO_R_W, bytes_to_read)

    # Parse samples
    samples = []
    for i in range(0, bytes_to_read, 12):
        # Accel: bytes 0-5 (big-endian signed 16-bit)
        ax_raw = int.from_bytes(data[i:i+2], 'big', signed=True)
        ay_raw = int.from_bytes(data[i+2:i+4], 'big', signed=True)
        az_raw = int.from_bytes(data[i+4:i+6], 'big', signed=True)

        # Gyro: bytes 6-11
        gx_raw = int.from_bytes(data[i+6:i+8], 'big', signed=True)
        gy_raw = int.from_bytes(data[i+8:i+10], 'big', signed=True)
        gz_raw = int.from_bytes(data[i+10:i+12], 'big', signed=True)

        # Convert to physical units
        ax = ax_raw * accel_scale
        ay = ay_raw * accel_scale
        az = az_raw * accel_scale
        gx = gx_raw * gyro_scale
        gy = gy_raw * gyro_scale
        gz = gz_raw * gyro_scale

        samples.append((ax, ay, az, gx, gy, gz))

    return samples

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

            # Scale factors for ±30g and ±4000dps
            # Accel: 30g / 32768 * 9.81 = m/s² per LSB
            # Gyro: 4000dps / 32768 * π/180 = rad/s per LSB
            accel_scale = (30.0 / 32768.0) * 9.81
            gyro_scale = (4000.0 / 32768.0) * (3.14159265 / 180.0)

            sensor_info = {
                'imu': imu,
                'address': addr,
                'name': f'{name}_0x{addr:02X}',
                'type': 'icm',
                'accel_scale': accel_scale,
                'gyro_scale': gyro_scale,
                'actual_odr': FIFO_ODR
            }

            # Setup FIFO if enabled
            if USE_FIFO:
                try:
                    sensor_info['actual_odr'] = setup_fifo(i2c, addr, FIFO_ODR)
                    sensor_info['use_fifo'] = True
                except Exception as e:
                    print(f"    ⚠ FIFO setup failed: {e} - using polling mode")
                    sensor_info['use_fifo'] = False
            else:
                sensor_info['use_fifo'] = False

            detected_sensors.append(sensor_info)
            
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
                'type': 'lsm',
                'use_fifo': False  # LSM uses polling mode
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
    filename = os.path.join(LOG_DIR, f"{timestamp}_{sensor['name']}.csv")
    f = open(filename, 'w')
    f.write("timestamp_epoch,ax_ms2,ay_ms2,az_ms2,gx_rads,gy_rads,gz_rads\n")
    csv_files[sensor['address']] = {
        'file': f,
        'filename': filename
    }
    print(f"Logging {sensor['name']} → {os.path.basename(filename)}")

# Check if any sensor uses FIFO
any_fifo = any(sensor.get('use_fifo', False) for sensor in detected_sensors)

print("\n" + "="*60)
print("HIGH-SPEED LOGGING STARTED")
if any_fifo:
    print(f"Mode: FIFO (hardware buffered) at {FIFO_ODR} Hz")
else:
    print(f"Mode: Polling at {TARGET_HZ} Hz target")
print("="*60)
if DEBUG:
    print("Debug output enabled - showing latest readings")
print("Press Ctrl+C to stop\n")

# ---------------- HIGH-SPEED LOGGING LOOP ----------------
sample_counts = {sensor['address']: 0 for sensor in detected_sensors}
start_time = time.time()
last_debug_time = time.time()

# Pre-calculate timing for polling mode
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
        read_time = time.time()

        for sensor in detected_sensors:
            addr = sensor['address']

            try:
                # Check if this sensor uses FIFO
                if sensor.get('use_fifo', False):
                    # FIFO mode: burst read all buffered samples
                    samples = read_fifo(i2c, addr,
                                       sensor['accel_scale'],
                                       sensor['gyro_scale'])

                    if samples:
                        # Calculate timestamps for each sample based on ODR
                        sample_interval_fifo = 1.0 / sensor['actual_odr']
                        num_samples = len(samples)

                        for idx, (ax, ay, az, gx, gy, gz) in enumerate(samples):
                            # Timestamp: read_time - time_ago for this sample
                            t = read_time - (num_samples - 1 - idx) * sample_interval_fifo
                            buffers[addr].append(
                                f"{t:.6f},{ax},{ay},{az},{gx},{gy},{gz}\n"
                            )
                            sample_counts[addr] += 1

                        # Store last sample for debug display
                        ax, ay, az, gx, gy, gz = samples[-1]
                        sensor['last_accel'] = (ax, ay, az)
                        sensor['last_gyro'] = (gx, gy, gz)

                else:
                    # Polling mode: read single sample via library
                    imu = sensor['imu']
                    accel = imu.acceleration
                    gyro = imu.gyro
                    ax, ay, az = accel
                    gx, gy, gz = gyro

                    buffers[addr].append(
                        f"{read_time:.6f},{ax},{ay},{az},{gx},{gy},{gz}\n"
                    )
                    sample_counts[addr] += 1

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

        # Debug output every 1 second
        if DEBUG and (loop_start - last_debug_time) > 1.0:
            elapsed = time.time() - start_time
            print(f"\n--- t={elapsed:.1f}s ---")

            for sensor in detected_sensors:
                if 'last_accel' in sensor and 'last_gyro' in sensor:
                    ax, ay, az = sensor['last_accel']
                    gx, gy, gz = sensor['last_gyro']
                    count = sample_counts[sensor['address']]
                    rate = count / elapsed if elapsed > 0 else 0
                    mode = "FIFO" if sensor.get('use_fifo', False) else "POLL"
                    print(f"{sensor['name']:20s} [{mode}]: {count:7d} samples ({rate:6.1f} Hz) | "
                          f"A=({ax:+7.2f},{ay:+7.2f},{az:+7.2f})")

            last_debug_time = loop_start

        # Timing control
        if any_fifo:
            # FIFO mode: sleep to let buffer fill (read every ~20ms)
            time.sleep(0.02)
        elif sample_interval > 0:
            # Polling mode: precise timing
            next_sample_time += sample_interval
            sleep_time = next_sample_time - time.monotonic()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_sample_time = time.monotonic()
        
except KeyboardInterrupt:
    print("\n\n" + "="*60)
    print("STOPPING LOGGER...")
    print("="*60)

    # Flush any remaining buffered data
    for addr in buffers:
        if buffers[addr]:
            csv_files[addr]['file'].write(''.join(buffers[addr]))

    # Close all CSV files
    for addr in csv_files:
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