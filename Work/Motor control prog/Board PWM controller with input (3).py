from pymavlink import mavutil
import time

# --- Input Mode Selection ---
mode = input("Choose mode: [1] Constant PWM | [2] Stepping PWM | [3] Custom Step Sequence: ")

step_sequence = []

if mode == '1':
    try:
        time_delay = int(input("Enter duration for motor operation (in seconds): "))
        pwm = int(input("Enter constant PWM value for motors (e.g., 1530): "))
        if not (1000 <= pwm <= 2000):
            print("Warning: PWM values typically range from 1000 to 2000.")
    except ValueError:
        print("Invalid input. Please enter numbers.")
        exit(1)
    use_stepping = False

elif mode == '2':
    try:
        pwm_start = int(input("Enter starting PWM value (e.g., 1400): "))
        pwm_end = int(input("Enter ending PWM value (e.g., 1800): "))
        pwm_rate = float(input("Enter rate of change (PWM units per second): "))
        if pwm_start == pwm_end or pwm_rate <= 0:
            print("Start and end must differ, and rate must be positive.")
            exit(1)
        time_delay = abs(pwm_end - pwm_start) / pwm_rate
        print(f"Estimated total time: {round(time_delay)} seconds")
    except ValueError:
        print("Invalid input. Please enter numbers.")
        exit(1)
    use_stepping = True

elif mode == '3':
    try:
        step_count = int(input("How many PWM steps do you want to define? "))
        for i in range(step_count):
            pwm_val = int(input(f"Step {i+1}: Enter PWM value: "))
            duration = int(input(f"Step {i+1}: Enter duration (in seconds): "))
            step_sequence.append((pwm_val, duration))
        loop_forever = input("Do you want to loop this sequence endlessly? [y/N]: ").strip().lower() == 'y'
    except ValueError:
        print("Invalid input. Please enter numbers.")
        exit(1)
else:
    print("Invalid mode selected.")
    exit(1)

# --- Connect to MAVLink ---
ip = "tcp:127.0.0.1:8888"
print(f"Connecting to: {ip}")
PX = mavutil.mavlink_connection(ip, 115220, autoreconnect=True)

PX.wait_heartbeat()
print("Heartbeat from system (system %u component %u)" % (PX.target_system, PX.target_component))

# --- Disable Motors ---
motor_dis = 0
for i in range(1, 5):
    PX.mav.param_set_send(PX.target_system, PX.target_component,
                          bytes(f"SERVO{i}_FUNCTION", "utf-8"),
                          motor_dis, mavutil.mavlink.MAV_PARAM_TYPE_INT32)

# --- Main Operation ---
start_time = time.time()
time_now = 0
last_pwm = pwm_start if mode == '2' else None

try:
    if mode == '3':
        loop_count = 0
        while True:
            loop_count += 1
            print(f"\n--- Starting Step Sequence Loop #{loop_count} ---")
            for idx, (pwm_val, duration) in enumerate(step_sequence):
                print(f"[Step {idx+1}] Set PWM to {pwm_val} for {duration} seconds")
                for i in range(1, 5):
                    PX.mav.command_long_send(PX.target_system, PX.target_component,
                                             mavutil.mavlink.MAV_CMD_DO_SET_SERVO, 0, i, pwm_val, 0, 0, 0, 0, 0)
                step_start = time.time()
                while (time.time() - step_start) < duration:
                    elapsed = int(time.time() - step_start)
                    print(f"  Time: {elapsed} / {duration} sec", end="\r")
                    time.sleep(0.5)
            if not loop_forever:
                break

    else:
        start_time = time.time()
        while time_now < time_delay:
            elapsed_time = time.time() - start_time
            if int(elapsed_time) > time_now:
                time_now = int(elapsed_time)
                print(f"Time: {time_now} / {round(time_delay)} seconds")

                if mode == '2':
                    direction = 1 if pwm_end > pwm_start else -1
                    next_pwm = pwm_start + direction * int(pwm_rate * time_now)
                    next_pwm = min(max(next_pwm, min(pwm_start, pwm_end)), max(pwm_start, pwm_end))
                    if next_pwm != last_pwm:
                        print(f"Setting PWM: {next_pwm}")
                        for i in range(1, 5):
                            PX.mav.command_long_send(PX.target_system, PX.target_component,
                                                     mavutil.mavlink.MAV_CMD_DO_SET_SERVO, 0, i, next_pwm, 0, 0, 0, 0, 0)
                        last_pwm = next_pwm

                elif mode == '1' and time_now == 1:
                    print(f"Setting constant PWM: {pwm}")
                    for i in range(1, 5):
                        PX.mav.command_long_send(PX.target_system, PX.target_component,
                                                 mavutil.mavlink.MAV_CMD_DO_SET_SERVO, 0, i, pwm, 0, 0, 0, 0, 0)

            time.sleep(0.05)

except KeyboardInterrupt:
    print("\n[!] Emergency stop triggered by user!")

# --- Stop Motors ---
print("\nStopping motors...")
pwm_stop = 1000
for i in range(1, 5):
    PX.mav.command_long_send(PX.target_system, PX.target_component,
                             mavutil.mavlink.MAV_CMD_DO_SET_SERVO, 0, i, pwm_stop, 0, 0, 0, 0, 0)

print("Drone stop: " + ip)
