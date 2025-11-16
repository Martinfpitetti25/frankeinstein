#!/usr/bin/env python3
# test_boca.py - Movimiento continuo de servos entre 30° y 150°
import time
from adafruit_servokit import ServoKit

# === CONFIGURACIÓN ===
SERVO_PIN_1 = 12  # Pin 6 del PCA9685
SERVO_PIN_2 = 15 # Pin 7 del PCA9685
SERVO_PIN_3 = 14  # Pin 8 del PCA9685

# Límites de movimiento
ANGLE_MIN = 130
ANGLE_MAX = 180
ANGLE_MID = (ANGLE_MIN + ANGLE_MAX) // 2
ANGLE_FINAL = 180

ANGLE_MIN_PITCH = 75
ANGLE_MAX_PITCH= 120
ANGLE_MID_PITCH= (ANGLE_MIN_PITCH + ANGLE_MAX_PITCH) // 2
ANGLE_FINAL_PITCH= 75

TIEMPO = 0.6

# Inicializar ServoKit
print("🔧 Inicializando PCA9685...")
kit = ServoKit(channels=16)

# Configurar servos
print(f"⚙️  Configurando servos en pines {SERVO_PIN_1} ")
kit.servo[SERVO_PIN_1].actuation_range = 200
kit.servo[SERVO_PIN_1].set_pulse_width_range(500, 2500)
kit.servo[SERVO_PIN_2].actuation_range = 200
kit.servo[SERVO_PIN_2].set_pulse_width_range(500, 2500)
kit.servo[SERVO_PIN_3].actuation_range = 200
kit.servo[SERVO_PIN_3].set_pulse_width_range(500, 2500)

print(f"✅ Servos configurados")
print(f"\n🎬 Iniciando movimiento continuo entre {ANGLE_MIN}° y {ANGLE_MAX}°...\n")
print(f"\n🎬 Iniciando movimiento continuo entre {ANGLE_MIN_PITCH}° y {ANGLE_MAX_PITCH}°...\n")
time.sleep(2)
try:
    while True:
        # Movimiento a mínimo 
        print(f"📍 Moviendo a {ANGLE_MIN}°...")
        kit.servo[SERVO_PIN_1].angle = ANGLE_MIN
        kit.servo[SERVO_PIN_2].angle = ANGLE_MIN
        kit.servo[SERVO_PIN_3].angle = ANGLE_MIN_PITCH
        time.sleep(TIEMPO)
        
        
        # Movimiento a máximo
        print(f"📍 Moviendo a {ANGLE_MAX}°...")
        kit.servo[SERVO_PIN_1].angle = ANGLE_MAX
        kit.servo[SERVO_PIN_2].angle = ANGLE_MAX
        kit.servo[SERVO_PIN_3].angle = ANGLE_MAX_PITCH

       
        time.sleep(TIEMPO)
        
        
        
        
        
        print("🔄 Repitiendo secuencia...\n")

        

except KeyboardInterrupt:
    print("\n\n⚠️ Interrumpido por usuario")
    print("🔄 Centrando servos en 90° antes de salir...")
    kit.servo[SERVO_PIN_1].angle = ANGLE_FINAL
    kit.servo[SERVO_PIN_2].angle = ANGLE_FINAL
    kit.servo[SERVO_PIN_3].angle = ANGLE_FINAL_PITCH
    time.sleep(0.5)
    print("✅ Finalizado")
