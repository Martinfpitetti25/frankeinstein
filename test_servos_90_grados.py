#!/usr/bin/env python3
"""
test_servos_90_grados.py - Poner todos los servos del PCA9685 en 90 grados
"""

import time
import logging
from adafruit_servokit import ServoKit

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    print("=" * 50)
    print("🤖 CONFIGURAR TODOS LOS SERVOS EN 90 GRADOS")
    print("=" * 50)
    
    try:
        # Inicializar ServoKit
        print("Inicializando PCA9685...")
        kit = ServoKit(channels=16, address=0x40)
        print("✅ PCA9685 inicializado correctamente")
        
        # Configurar todos los canales en 90 grados
        print("\n📐 Configurando todos los servos en 90 grados...")
        
        for channel in range(16):  # PCA9685 tiene 16 canales (0-15)
            try:
                print(f"Canal {channel:2d}: 90° ", end="")
                kit.servo[channel].angle = 90
                print("✅")
                time.sleep(0.1)  # Pequeña pausa entre servos
                
            except Exception as e:
                print(f"❌ Error: {e}")
        
        print("\n🎯 Todos los servos configurados en 90 grados")
        print("\n📊 Estado de los servos:")
        print("   Canal  0 (YAW Horizontal): 90°")
        print("   Canal  1 (PITCH Vertical): 90°")
        print("   Canal  5 (BOCA): 90°")
        print("   Canal 12 (ROLL Izquierdo): 90°")
        print("   Canal 13 (YAW Principal): 90°")
        print("   Canal 14 (PITCH Principal): 90°")
        print("   Canal 15 (ROLL Derecho): 90°")
        print("   Canales 2,3,4,6,7,8,9,10,11: 90°")
        
        print("\n✨ ¡Proceso completado!")
        
    except Exception as e:
        print(f"\n❌ Error inicializando PCA9685: {e}")
        print("Verifica:")
        print("• Conexión I2C")
        print("• Alimentación 5V")
        print("• Dirección 0x40")

if __name__ == "__main__":
    main()