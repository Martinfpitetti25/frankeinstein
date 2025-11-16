import time
from machine import Pin, ADC
from servo import Servo
from picozero import Button
import random
import sys
import select

# Set up the switches and potentiometers
enable = Pin(6, Pin.IN, Pin.PULL_UP)
mode = Pin(7, Pin.IN, Pin.PULL_UP)
blink_pin = Pin(9, Pin.IN, Pin.PULL_UP)
led = Pin(25, Pin.OUT)
UD = ADC(26)
trim = ADC(27)
LR = ADC(28)

# Motion tracking parameters
deadzone_eye = 25
deadzone_neck = 20
neck_timer_ms = 1200
neck_trigger_time = 0
neck_flag = False
neck_x_map = 1.25
neck_y_map = 1
Kp = 0.03
last_update = time.ticks_ms()
error_x = 0
error_y = 0

# Define servos
servos = {
    "LR": Servo(pin_id=10),
    "UD": Servo(pin_id=11),
    "TL": Servo(pin_id=12),
    "BaseX": Servo(pin_id=13),
    "TR": Servo(pin_id=14),
    "BaseY": Servo(pin_id=15),
}

# Min, Max, target
servo_limits = {
    "LR": (40, 140),   
    "UD": (40, 140),
    "TL": (90, 170),
    "BaseX": (10, 170),
    "TR": (90, 10),
    "BaseY": (40, 140),
}

servo_targets = {
    "LR": 90,
    "UD": 90,
    "BaseX": 90,
    "BaseY": 90,
    "BXTarget":90,
    "BYTarget":90,
}

# Set all servos to central position for assembly
def calibrate():
    for name, servo in servos.items():
        servo.write(90)
              
        
def blink():
    servos["TL"].write(90)
    servos["TR"].write(90)

def lid_sync():
    ud_min, ud_max = servo_limits["UD"]
    tl_min, tl_max = servo_limits["TL"]
    tr_min, tr_max = servo_limits["TR"]

    # Normalize UD position to a 0-1 range
    ud_progress = (servo_targets["UD"] - ud_min) / (ud_max - ud_min)  # 0 (min) → 1 (max)
    
    # Find target positions
    tl_target = tl_max - ((tl_max - tl_min)*(0.5*(1-ud_progress)))-10
    tr_target = tr_max + ((tr_min - tr_max)*(0.5*(1-ud_progress)))+10
    
    servos["TL"].write(tl_target)
    servos["TR"].write(tr_target)
    
def flash():
    led.value(not led.value())
    time.sleep(0.2)
    led.value(not led.value())
    time.sleep(0.2)
    led.value(not led.value())
    time.sleep(0.2)
    led.value(not led.value())
    time.sleep(0.2)
    

    
def move_target(servo, error):
    if abs(error) <= deadzone_eye:
        return
    if error < 0:
        error = error - deadzone_eye
    elif error < 0:
        error = error + deadzone_eye
    
    step = int(Kp * error)
    if step == 0:
        step = 1 if error > 0 else -1
    
    new_target = servo_targets[servo] + step
    new_target = max(servo_limits[servo][0], min(servo_limits[servo][1], new_target))
    servo_targets[servo] = new_target
    
    servos[servo].write(new_target)
    
def neck_target(LR, UD):
    servo_targets["BXTarget"] = LR
    servo_targets["BYTarget"] = 90 - ((90 - UD) * 0.6) 
    


def neck_smooth_move(speed_deg_per_s=60):
    """Smoothly move BaseX and BaseY toward their targets at a fixed speed in deg/s."""
    global last_update
    now = time.ticks_ms()
    dt = time.ticks_diff(now, last_update)
    last_update = now
    
    step_size = (speed_deg_per_s * dt) / 1000.0
    
    # BaseX
    dx = servo_targets["BXTarget"] - servo_targets["BaseX"]
    if abs(dx) <= step_size:
        servo_targets["BaseX"] = servo_targets["BXTarget"]
    else:
        servo_targets["BaseX"] += step_size if dx > 0 else -step_size
    
    # BaseY
    dy = servo_targets["BYTarget"] - servo_targets["BaseY"]
    if abs(dy) <= step_size:
        servo_targets["BaseY"] = servo_targets["BYTarget"]
    else:
        servo_targets["BaseY"] += step_size if dy > 0 else -step_size
        
    servos["BaseX"].write(int(servo_targets["BaseX"]))
    servos["BaseY"].write(int(servo_targets["BaseY"]))
   



while True:
    mode_state = not mode.value()
    enable_state = not enable.value()
    if mode_state == 1: # Enter calibration mode when switch is in hold position
        calibrate()
        time.sleep_ms(500)
    else:
        if enable_state == 0: # Auto mode
            latest_line = None
            while sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                latest_line = sys.stdin.readline()
                
            if latest_line:
                x_str, y_str = latest_line.strip().split(',')
                error_x = int(x_str)
                error_y = int(y_str)
                
                move_target("LR", error_x*-1)
                move_target("UD", error_y)
                
                if random.randint(0,60) < 1:
                    blink()
                    time.sleep_ms(60)
                    lid_sync()
            time.sleep_ms(0)
            lid_sync()
            if abs(servo_targets["UD"] - 90) >= deadzone_neck or abs(servo_targets["LR"] - 90) >= deadzone_neck:
                if not neck_flag:
                    neck_trigger_time = time.ticks_ms()
                    neck_flag = True
                if neck_flag and (time.ticks_ms() - neck_trigger_time) >= neck_timer_ms:
                    neck_target(servo_targets["LR"], servo_targets["UD"])
                    neck_flag = False
            neck_smooth_move()                            
                

            
