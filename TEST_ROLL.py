#!/usr/bin/env python3
import time
from adafruit_servokit import ServoKit

# Pines en el PCA9685
S1, S2 = 15, 12

# Límites (compartidos). Si fueran distintos, mirá la variante B.
MIN_A, MAX_A = 70, 180
MID_A = (MIN_A + MAX_A) // 2

kit = ServoKit(channels=16)

# Config común (ajustá si tu servo necesita distinto rango/pulsos)
for ch in (S1, S2):
    s = kit.servo[ch]
    s.actuation_range = 180
    s.set_pulse_width_range(500, 2500)

def set_pair(a1):
    """Mueve S1 a a1 y S2 al espejo proporcional inverso."""
    a2 = (MIN_A + MAX_A) - a1           # --- clave: espejo directo ---
    kit.servo[S1].angle = a1
    kit.servo[S2].angle = a2
    print("Servo 1 a", a1, "y Servo 2 a", a2)

try:
    while True:
        for a in (MIN_A, MID_A, MAX_A, MID_A):
            set_pair(a)
            time.sleep(1.0)
        
except KeyboardInterrupt:
    # Posición de salida “segura”
    set_pair(MID_A)
    time.sleep(0.3)
