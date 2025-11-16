#!/usr/bin/env python3
# test_boca.py - Movimiento continuo de servos entre 30° y 150°
import time
from adafruit_servokit import ServoKit

# === CONFIGURACIÓN ===
SERVO_PIN_1 = 13  # Pin 13 del PCA9685


# Límites de movimiento
ANGLE_MIN = 100
ANGLE_MAX = 170
ANGLE_FINAL = 90
TIEMPO = 0.5
# Inicializar ServoKit
print("🔧 Inicializando PCA9685...")
kit = ServoKit(channels=16)

# Configurar servos
print(f"⚙️  Configurando servos en pines {SERVO_PIN_1} ")
kit.servo[SERVO_PIN_1].actuation_range = 200
kit.servo[SERVO_PIN_1].set_pulse_width_range(500, 2500)


print(f"✅ Servos configurados")
print(f"\n🎬 Iniciando movimiento continuo entre {ANGLE_MIN}° y {ANGLE_MAX}°...\n")

try:
    while True:
        # Movimiento a mínimo (30°)
        print(f"📍 Moviendo a {ANGLE_MIN}°...")
        kit.servo[SERVO_PIN_1].angle = ANGLE_MIN

        time.sleep(TIEMPO)

        # Movimiento a máximo (150°)
        print(f"📍 Moviendo a {ANGLE_MAX}°...")
        kit.servo[SERVO_PIN_1].angle = ANGLE_MAX
       

        time.sleep(TIEMPO)
        
        print("🔄 Repitiendo secuencia...\n")


except KeyboardInterrupt:
    print("\n\n⚠️ Interrumpido por usuario")
    print("🔄 Centrando servos en 90° antes de salir...")
    kit.servo[SERVO_PIN_1].angle = ANGLE_FINAL
    time.sleep(TIEMPO)
    print("✅ Finalizado")
