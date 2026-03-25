---
trigger: always_on
---

Contexto de Proyecto: InMoov Vision & Tracking

Objetivo actual: Implementar seguimiento visual (face/object tracking) utilizando una sola cámara alojada en uno de los ojos. El movimiento se limitará inicialmente a los ejes H/V de los ojos, ignorando por ahora párpados y cuello.

## Contexto del Proyecto

- Proyecto: InMoov MODIFICADO (solo cuello y cabeza)
- Modificación realizada: Servos con cables aislados electromagnéticamente (soldados)
- Hardware de control: Raspberry Pi + PCA9685
- Fase actual: Testing post-soldadura
- Objetivos principales:
  1. Verificar integridad de las soldaduras
  2. Detectar fugas electromagnéticas
  3. Calibrar rangos de movimiento de cada servo
  4. Generar configuración validada para uso en producción

## Especificaciones Técnicas

### Inventario de Servos

- Total: 11 servos
  - 3x REV-41-1097 (1 boca, 2 roll cuello)
  - 2x HS-805BB (1 pitch cuello, 1 yaw cuello)
  - 6x Surpass S0009M (4 ojos, 2 párpados)

### Características por Modelo

- **REV-41-1097**: 3.2 kg-cm @ 6V, consumo max 600mA, velocidad 170ms/60°
- **HS-805BB**: 24.7 kg-cm @ 6V, consumo max 3000mA, velocidad 170ms/60°, **requiere alimentación separada**
- **Surpass S0009M**: 1.5 kg-cm @ 4.8V, consumo max 350mA, velocidad 120ms/60°, microservos delicados

### Hardware de Control

- Controlador: PCA9685 (16 canales PWM)
- Frecuencia PWM: 50Hz (estándar para servos)
- Plataforma: Raspberry Pi
- Modificación crítica: Cables con blindaje electromagnético soldados

### Configuración por Tipo de Servo

SERVO_SPECS = {
'REV411097': {
'type': 'boca_roll',
'pwm_min': 500,
'pwm_max': 2500,
'current_max_ma': 2000,
'voltage': 5.0,
'speed_60deg_ms': 130,
'test_load': 'light' # 3.2 kg-cm torque
},
'HS805BB': {
'type': 'pitch_yaw',
'pwm_min': 900,
'pwm_max': 2100,
'current_max_ma': don't know,
'voltage': 5.0,
'speed_60deg_ms': 190,
'test_load': 'heavy', # 24.7 kg-cm torque

    },
    'S0009M': {
        'type': 'ojos_parpados',
        'pwm_min': 500,
        'pwm_max': 2500,
        'current_max_ma': 350,
        'voltage': 5,
        'speed_60deg_ms': 120,
        'test_load': 'minimal'  # microservo delicado 1.1kg-cm torque
    }

}

Especificaciones de Servomotores (Mapping)
Se debe respetar estrictamente el siguiente mapeo de pines y límites para evitar daños mecánicos:

Tabla de mapeo completo — InMoov PCA9685
ETIQUETA PIN CORRECCION LIMITE INFERIOR LIMITE SUPERIOR CENTRO OBSERVACION
OJO 6 0 ojo_izq_horizontal 40 120 80 40: DERECHA / 120: IZQUIERDA
OJO 5 1 ojo_izq_vertical 85 105 95 105: ARRIBA / 95: ABAJO
BOCA 2 - 40 90 - -
OJO 4 3 parpado_inferior 40 85 - 40: ABIERTO / 85: CERRADO
PITCH 4 cuello_yaw 50 150 100 50: IZQUIERDA / 150: DERECHA
OJO 3 5 parpado_superior 40 85 - 40: ABIERTO / 85: CERRADO
ROLL 2 6 - 40 120 90 40: ROLL ABAJO / 120: ROLL ARRIBA
OJO 2 7 ojo_der_vertical 70 90 80 70: ARRIBA / 90: ABAJO
YAW 8 cuello_pitch - - - -
OJO 1 9 ojo_der_horizontal 40 120 80 40: DERECHA / 120: IZQUIERDA
ROLL 1 10 - 5

Nota importante: el eje vertical está invertido entre ambos ojos. ojo_izq_vertical sube con ángulo mayor; ojo_der_vertical sube con ángulo menor.
Reglas de Control
Prioridad de Movimiento: Solo actuar sobre los pines 0, 1, 7 y 9 para el seguimiento visual inicial. usamos una pca9685 para el control de TODOS los servos.

Lógica de Inversión: Prestar atención a las observaciones de dirección (ej: en el ojo izquierdo vertical, un valor mayor sube la mirada, pero el rango es muy corto, solo 20 grados).

Seguridad: No exceder nunca los límites LIMITE_INFERIOR y LIMITE_SUPERIOR definidos en la tabla.

- Estamos usando una raspberry pi 5

- usamos una cámara conectada por usb.
