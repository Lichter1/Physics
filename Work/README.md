# Gyro Vibration Logger - Setup Manual

Multi-IMU high-speed vibration logger for drone accelerometer and gyroscope data collection. Supports ICM20649 and LSM6DSO32 sensors with automatic detection.

---

## Step-by-Step Installation Guide

### Step 1: Install Python 3.8

```bash
sudo apt install python3.8
```

Verify installation:
```bash
python3.8 --version
```

---

### Step 2: Install Required Python Packages

Install the required libraries for the IMU sensors:

```bash
python3.8 -m pip install --user adafruit-circuitpython-icm20x
python3.8 -m pip install --user adafruit-circuitpython-lsm6ds
python3.8 -m pip install --user adafruit-extended-bus
```

---

### Step 3: Create the Script Directory

```bash
mkdir /home/admin/projects/perceptoApps/scripts/Accelerometer
```

---

### Step 4: Clone the Script

Copy only the `Gyro_test_v4.py` script to the directory:

```bash
cp Gyro_test_v4.py /home/admin/projects/perceptoApps/scripts/Accelerometer/
```

Or if cloning from a repository, copy only this file to the target directory.

---

### Step 5: Verify the Script Works

Navigate to the script directory and run a test:

```bash
cd /home/admin/projects/perceptoApps/scripts/Accelerometer
python3.8 Gyro_test_v4.py
```

You should see output showing:
- I2C bus scanning for IMU sensors
- Detected sensors with their addresses
- Sample rate and accelerometer readings (if DEBUG is enabled)

Press `Ctrl+C` to stop the test.

---

### Step 6: Set Up Supervisor Service

Copy the supervisor configuration file:

```bash
cp gyro.conf /home/admin/projects/perceptoApps/supervisor/conf.d/
```

---

### Step 7: Restart the Drone

Restart the drone to apply the supervisor configuration changes.

---

### Step 8: Verify Supervisor Status

Check that the gyro service is registered:

```bash
sudo supervisorctl status
```

You should see the service in the list:
```
gyro                             STOPPED
```

---

### Step 9: Control the Service

**Start the service:**
```bash
sudo supervisorctl start gyro
```

**Stop the service:**
```bash
sudo supervisorctl stop gyro
```

**Check service status:**
```bash
sudo supervisorctl status gyro
```

---

### Step 10: View Log Files

Log files are saved to `/media/ssd/`. To find your log files:

```bash
ls -ltr /media/ssd/
```

Files are named with timestamp and sensor info. Example:
```
-rw-r--r--  1 admin admin  7010389 Jan 18 10:34 2026_01_18__10_33_06_ICM20649_0x68.csv
```

The CSV files contain columns:
- `timestamp_epoch` - Unix timestamp with microsecond precision
- `ax_ms2, ay_ms2, az_ms2` - Accelerometer data (m/s²)
- `gx_rads, gy_rads, gz_rads` - Gyroscope data (rad/s)

---

## Script Description

This script is a high-speed vibration data logger designed for drone IMU sensors. It:

1. **Auto-detects IMU sensors** on the I2C bus (ICM20649 and LSM6DSO32)
2. **Configures sensors** for maximum range (±30G accel, ±4000 dps gyro for ICM20649)
3. **Logs data** at maximum possible speed to CSV files
4. **Buffers writes** for optimal I/O performance (100 samples per write)
5. **Displays real-time stats** when DEBUG mode is enabled

---

## Configurable Parameters

Edit these values at the top of `Gyro_test_v4.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `I2C_BUS_NUMBER` | `6` | I2C bus number to scan for sensors. Change if your sensors are on a different bus. |
| `DEBUG` | `True` | Set to `False` to disable screen output. Recommended for production use to maximize performance. |
| `TARGET_HZ` | `0` | Target sample rate in Hz. Set to `0` for maximum speed (no delay between samples). Set to a specific value (e.g., `1000`) to limit the sample rate. |
| `LOG_DIR` | `/media/ssd` | Directory where CSV log files are saved. |
| `BUFFER_SIZE` | `100` | Number of samples to buffer before writing to disk. Higher values reduce I/O overhead but increase memory usage. |

### Supported Sensors

The script scans for these sensors:

| Sensor | I2C Addresses | Accel Range | Gyro Range |
|--------|---------------|-------------|------------|
| ICM20649 | 0x68, 0x69 | ±30G | ±4000 dps |
| LSM6DSO32 | 0x6A, 0x6B | ±32G | ±2000 dps |

### Example Configuration Changes

**Change I2C bus to bus 1:**
```python
I2C_BUS_NUMBER = 1
```

**Disable debug output for production:**
```python
DEBUG = False
```

**Limit sample rate to 500 Hz:**
```python
TARGET_HZ = 500
```

**Change log directory:**
```python
LOG_DIR = "/home/admin/logs"
```

---

## Troubleshooting

**No sensors detected:**
```bash
i2cdetect -y 6
```
This will show which I2C addresses have devices. Adjust `I2C_BUS_NUMBER` if needed.


**View supervisor logs:**
```bash
cat /var/log/supervisor/gyro.err.log
cat /var/log/supervisor/gyro.out.log
```
