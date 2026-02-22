import smbus2
import time

BUS  = 6
ADDR = 0x67

def read_reg(bus, reg):
    data = bus.read_i2c_block_data(ADDR, reg, 2)
    raw = (data[0] << 8) | data[1]
    if raw >= 0x8000:   # convert to signed
        raw -= 0x10000
    return raw * 0.0625

bus = smbus2.SMBus(BUS)

while True:
    try:
        hot   = read_reg(bus, 0x00)  # thermocouple (hot junction)
        cold  = read_reg(bus, 0x02)  # ambient (cold junction)
        delta = read_reg(bus, 0x01)  # delta
        print(f"Hot: {hot:.2f}C  Cold: {cold:.2f}C  Delta: {delta:.2f}C")
    except Exception as e:
        print(f"Error: {e}")
    time.sleep(1)
