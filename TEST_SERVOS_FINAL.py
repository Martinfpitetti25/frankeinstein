#!/usr/bin/env python3
# test_boca.py - Movimiento continuo de servos entre 30° y 150°
import time
from adafruit_servokit import ServoKit

# === CONFIGURACIÓN ===
SERVO_PIN_1 = 13   # Pin 12 del PCA9685


# Límites de movimiento

ANGLE_MIN = 90
ANGLE_MAX = 180
ANGLE_MIN_PARPADO_ABAJO = 60
ANGLE_MAX_PARPADO_ABAJO = 130
ANGLE_MIN_PARPADO_ARRIBA = 60
ANGLE_MAX_PARPADO_ARRIBA = 130
ANGLE_MIN_OJO_IZQUIERDO_VERTICAL = 70
ANGLE_MAX_OJO_IZQUIERDO_VERTICAL = 110
ANGLE_MIN_OJO_DERECHO_VERTICAL = 70 #NO ANDUVO, VALORES ESTIMADOS
ANGLE_MAX_OJO_DERECHO_VERTICAL = 110 #NO ANDUVO, VALORES ESTIMADOS
ANGLE_MIN_OJO_IZQUIERDO_HORIZONTAL = 15
ANGLE_MAX_OJO_IZQUIERDO_HORIZONTAL = 165
ANGLE_MIN_OJO_DERECHO_HORIZONTAL = 15
ANGLE_MAX_OJO_DERECHO_HORIZONTAL = 165

ANGLE_MID = (ANGLE_MIN + ANGLE_MAX) // 2
ANGLE_FINAL = 135
TIEMPO = 1.5
# Inicializar ServoKit
print("🔧 Inicializando PCA9685...")
kit = ServoKit(channels=16)

# Configurar servos
print(f"⚙️  Configurando servos en pines {SERVO_PIN_1} ")
kit.servo[SERVO_PIN_1].actuation_range = 180
kit.servo[SERVO_PIN_1].set_pulse_width_range(500, 2500)


print(f"✅ Servos configurados")
print(f"\n🎬 Iniciando movimiento continuo entre {ANGLE_MIN}° y {ANGLE_MAX}°...\n")

try:
    while True:
        # Movimiento a mínimo (30°)
        print(f"📍 Moviendo a {ANGLE_MIN}°...")
        kit.servo[SERVO_PIN_1].angle = ANGLE_MIN

        time.sleep(TIEMPO)

        # Movimiento al centro (90°)
        print(f"📍 Moviendo a {ANGLE_MID}° (centro)...")
        kit.servo[SERVO_PIN_1].angle = ANGLE_MID

        time.sleep(TIEMPO)

        # Movimiento a máximo (150°)
        print(f"📍 Moviendo a {ANGLE_MAX}°...")
        kit.servo[SERVO_PIN_1].angle = ANGLE_MAX

        time.sleep(TIEMPO)
        
        # Volver al centro
        print(f"📍 Volviendo al centro ({ANGLE_MID}°)...")
        kit.servo[SERVO_PIN_1].angle = ANGLE_MID

        time.sleep(TIEMPO)

        print("🔄 Repitiendo secuencia...\n")

except KeyboardInterrupt:
    print("\n\n⚠️ Interrumpido por usuario")
    print("🔄 Centrando servos en 90° antes de salir...")
    kit.servo[SERVO_PIN_1].angle = ANGLE_FINAL  
    time.sleep(0.5)
    print("✅ Finalizado")


