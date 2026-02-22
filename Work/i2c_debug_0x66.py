import smbus2

bus = smbus2.SMBus(6)

print("Plain read:")
print(hex(bus.read_byte(0x66)))

print("Register 0x00:")
print(hex(bus.read_byte_data(0x66, 0x00)))

bus.close()
