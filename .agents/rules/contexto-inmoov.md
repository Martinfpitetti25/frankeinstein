---
trigger: always_on
---

Contexto de Proyecto: InMoov Vision & Tracking

Objetivo actual: Implementar seguimiento visual (face/object tracking) utilizando una sola cámara alojada en uno de los ojos. El movimiento se limitará inicialmente a los ejes H/V de los ojos, ignorando por ahora párpados y cuello.

Especificaciones de Servomotores (Mapping)
Se debe respetar estrictamente el siguiente mapeo de pines y límites para evitar daños mecánicos:

Componente,Pin,Eje,Mín,Máx,Centro,Notas
Ojo Izquierdo,0,Horizontal,40,120,80,"40: Der (suya), 120: Izq"
Ojo Izquierdo,1,Vertical,85,105,95,"105: Arriba, 95: Abajo"
Ojo Derecho,9,Horizontal,40,120,80,"40: Der (suya), 120: Izq"
Ojo Derecho,7,Vertical,70,90,80,"70: Arriba, 90: Abajo"

Reglas de Control
Prioridad de Movimiento: Solo actuar sobre los pines 0, 1, 7 y 9 para el seguimiento visual inicial. usamos una pca9685 para el control de TODOS los servos.

Lógica de Inversión: Prestar atención a las observaciones de dirección (ej: en el ojo izquierdo vertical, un valor mayor sube la mirada, pero el rango es muy corto, solo 20 grados).

Seguridad: No exceder nunca los límites LIMITE_INFERIOR y LIMITE_SUPERIOR definidos en la tabla.

- Estamos usando una raspberry pi 5

- usamos una cámara conectada por usb.
