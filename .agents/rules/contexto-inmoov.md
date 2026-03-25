---
trigger: always_on
---

Contexto de Proyecto: InMoov Vision & Tracking

Objetivo actual: Implementar seguimiento visual (face/object tracking) utilizando una sola cámara alojada en uno de los ojos. El movimiento se limitará inicialmente a los ejes H/V de los ojos, ignorando por ahora párpados y cuello.

Especificaciones de Servomotores (Mapping)
Se debe respetar estrictamente el siguiente mapeo de pines y límites para evitar daños mecánicos:

Tabla de mapeo completo — InMoov PCA9685
Etiqueta Pin Función Límite Inf. Límite Sup. Centro Observaciones
OJO 6 0 ojo_izq_horizontal 40 120 80 40=mirar a su derecha · 120=mirar izquierda
OJO 5 1 ojo_izq_vertical 85 105 95 105=arriba · 95=abajo
BOCA 2 — 40 90 — —
OJO 4 3 parpado_inferior 40 85 — 40=abierto · 85=cerrado
PITCH 4 cuello_yaw 50 150 100 50=mirar izquierda · 150=mirar derecha
OJO 3 5 parpado_superior 40 85 — 40=abierto · 85=cerrado
ROLL 2 6 — 40 120 90 40=roll abajo · 120=roll arriba
OJO 2 7 ojo_der_vertical 70 90 80 70=arriba · 90=abajo
YAW 8 cuello_pitch — — — —
OJO 1 9 ojo_der_horizontal 40 120 80 40=mirar a su derecha · 120=mirar izquierda
ROLL 1 10 — 5 80 45 40=roll abajo · 120=roll arriba

Nota importante: el eje vertical está invertido entre ambos ojos. ojo_izq_vertical sube con ángulo mayor; ojo_der_vertical sube con ángulo menor.
Reglas de Control
Prioridad de Movimiento: Solo actuar sobre los pines 0, 1, 7 y 9 para el seguimiento visual inicial. usamos una pca9685 para el control de TODOS los servos.

Lógica de Inversión: Prestar atención a las observaciones de dirección (ej: en el ojo izquierdo vertical, un valor mayor sube la mirada, pero el rango es muy corto, solo 20 grados).

Seguridad: No exceder nunca los límites LIMITE_INFERIOR y LIMITE_SUPERIOR definidos en la tabla.

- Estamos usando una raspberry pi 5

- usamos una cámara conectada por usb.
