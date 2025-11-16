#!/usr/bin/env python3
"""
InMoov Face Tracker - Adaptado del código de Will Cogley
Sistema de seguimiento facial para InMoov con PCA9685

Hardware:
- Raspberry Pi 5 (8GB)
- PCA9685 (I2C servo controller)
- USB Camera (montada en la cabeza)
- 10 servos: 4 cabeza + 6 ojos

Autor: Adaptado para InMoov
Basado en: Will Cogley's face tracking system
"""

import cv2
import mediapipe as mp
import time
import numpy as np
from adafruit_servokit import ServoKit
import logging

# ============================================================
# CONFIGURACIÓN DE LOGGING
# ============================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================
# CONFIGURACIÓN DEL HARDWARE
# ============================================================

class InMoovFaceTracker:
    """Sistema de seguimiento facial para InMoov"""
    
    def __init__(self):
        """Inicializar el sistema de seguimiento"""
        
        # Inicializar PCA9685
        try:
            self.kit = ServoKit(channels=16)
            
            # Aplicar márgenes de seguridad PWM a todos los canales
            PWM_MIN_SAFE = 650
            PWM_MAX_SAFE = 2000
            for channel in range(16):
                try:
                    self.kit.servo[channel].set_pulse_width_range(PWM_MIN_SAFE, PWM_MAX_SAFE)
                except:
                    pass
            
            logger.info("✓ PCA9685 inicializado")
            logger.info("✓ Márgenes de seguridad PWM: %d-%dμs" % (PWM_MIN_SAFE, PWM_MAX_SAFE))
        except Exception as e:
            logger.error("❌ Error inicializando PCA9685: %s" % str(e))
            raise
        
        # ============================================================
        # MAPEO DE SERVOS - CONFIGURACIÓN INMOOV
        # ============================================================
        # NOTA: Cambiar estos números según tu conexión al PCA9685
        
        # CABEZA (4 servos)
        self.SERVO_ROLL_LEFT = 0   # Roll izquierda (usar mínimamente)
        self.SERVO_ROLL_RIGHT = 1  # Roll derecha (usar mínimamente)
        self.SERVO_PITCH = 2       # Pitch cabeza (arriba/abajo)
        self.SERVO_JAW = 3         # Mandíbula (no usado en tracking)
        
        # PÁRPADOS (2 servos - cada uno controla ambos lados)
        self.SERVO_EYELID_UPPER = 4  # Párpados superiores (ambos)
        self.SERVO_EYELID_LOWER = 5  # Párpados inferiores (ambos)
        
        # OJOS - MOVIMIENTO (4 servos - independientes)
        self.SERVO_EYE_LEFT_H = 6    # Ojo izquierdo horizontal (izq/der)
        self.SERVO_EYE_RIGHT_H = 7   # Ojo derecho horizontal (izq/der)
        self.SERVO_EYE_LEFT_V = 8    # Ojo izquierdo vertical (arriba/abajo)
        self.SERVO_EYE_RIGHT_V = 9   # Ojo derecho vertical (arriba/abajo)
        
        # ============================================================
        # POSICIONES DE CENTRO (CALIBRACIÓN)
        # ============================================================
        # NOTA: Ajustar estos valores según la posición neutral de tu robot
        
        # Cabeza
        self.CENTER_ROLL_LEFT = 90
        self.CENTER_ROLL_RIGHT = 90
        self.CENTER_PITCH = 90
        self.CENTER_JAW = 90
        
        # Párpados (abiertos en posición neutral)
        self.CENTER_EYELID_UPPER = 90  # Párpados superiores abiertos
        self.CENTER_EYELID_LOWER = 90  # Párpados inferiores abiertos
        
        # Ojos (mirando al frente)
        self.CENTER_EYE_LEFT_H = 90    # Ojo izquierdo horizontal centro
        self.CENTER_EYE_RIGHT_H = 90   # Ojo derecho horizontal centro
        self.CENTER_EYE_LEFT_V = 90    # Ojo izquierdo vertical centro
        self.CENTER_EYE_RIGHT_V = 90   # Ojo derecho vertical centro
        
        # ============================================================
        # LÍMITES DE MOVIMIENTO (AJUSTAR SEGÚN TU ROBOT)
        # ============================================================
        
        # OJOS - Límite de seguridad ±20° desde 90°
        # Solo los ojos tienen esta restricción conservadora
        self.EYE_H_MIN = 70   # 90° - 20° (horizontal)
        self.EYE_H_MAX = 110  # 90° + 20° (horizontal)
        self.EYE_V_MIN = 70   # 90° - 20° (vertical)
        self.EYE_V_MAX = 110  # 90° + 20° (vertical)
        
        # CABEZA - Límites según configuración del proyecto
        # Estos mantienen los límites originales del proyecto
        self.PITCH_MIN = 60   # Pitch cabeza arriba
        self.PITCH_MAX = 120  # Pitch cabeza abajo
        
        # Roll de cabeza (usar mínimamente según configuración del proyecto)
        self.ROLL_MIN = 75
        self.ROLL_MAX = 105
        
        # ============================================================
        # PARÁMETROS DE CONTROL
        # ============================================================
        
        # Ganancia proporcional (ajustar para suavidad vs velocidad)
        self.KP_EYES = 0.15      # Ganancia para ojos (más rápidos)
        self.KP_HEAD = 0.08      # Ganancia para cabeza (más lento)
        
        # Deadband (zona muerta) en píxeles
        self.DEADBAND_X = 40     # Horizontal
        self.DEADBAND_Y = 30     # Vertical
        
        # Umbral para activar movimiento de cabeza (% del rango de ojos)
        self.HEAD_THRESHOLD = 0.70  # Si ojos están al 70% del límite, mover cabeza
        
        # Suavizado exponencial
        self.SMOOTH_ALPHA = 0.3  # 0.1=muy suave, 1.0=sin suavizado
        
        # ============================================================
        # ESTADO ACTUAL
        # ============================================================
        
        # Posiciones actuales de servos
        self.current_eye_left_h = self.CENTER_EYE_LEFT_H
        self.current_eye_right_h = self.CENTER_EYE_RIGHT_H
        self.current_eye_left_v = self.CENTER_EYE_LEFT_V
        self.current_eye_right_v = self.CENTER_EYE_RIGHT_V
        self.current_pitch = self.CENTER_PITCH
        self.current_roll_left = self.CENTER_ROLL_LEFT
        self.current_roll_right = self.CENTER_ROLL_RIGHT
        
        # ============================================================
        # MEDIAPIPE
        # ============================================================
        
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.5
        )
        
        # ============================================================
        # CÁMARA
        # ============================================================
        
        self.cap = None
        self.frame_width = 640
        self.frame_height = 480
        self.center_x = self.frame_width // 2
        self.center_y = self.frame_height // 2
        
        logger.info("✓ InMoov Face Tracker inicializado")
    
    def start_camera(self, camera_index=0):
        """Iniciar cámara"""
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        
        if not self.cap.isOpened():
            logger.error("❌ No se pudo abrir la cámara")
            return False
        
        # Obtener resolución real
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.center_x = self.frame_width // 2
        self.center_y = self.frame_height // 2
        
        logger.info("✓ Cámara iniciada: %dx%d" % (self.frame_width, self.frame_height))
        return True
    
    def move_to_center(self):
        """Mover todos los servos a posición central"""
        logger.info("🎯 Moviendo a posición central...")
        
        try:
            # Ojos - Movimiento
            self.set_servo_safe(self.SERVO_EYE_LEFT_H, self.CENTER_EYE_LEFT_H)
            self.set_servo_safe(self.SERVO_EYE_RIGHT_H, self.CENTER_EYE_RIGHT_H)
            self.set_servo_safe(self.SERVO_EYE_LEFT_V, self.CENTER_EYE_LEFT_V)
            self.set_servo_safe(self.SERVO_EYE_RIGHT_V, self.CENTER_EYE_RIGHT_V)
            
            # Párpados (abiertos)
            self.set_servo_safe(self.SERVO_EYELID_UPPER, self.CENTER_EYELID_UPPER)
            self.set_servo_safe(self.SERVO_EYELID_LOWER, self.CENTER_EYELID_LOWER)
            
            # Cabeza
            self.set_servo_safe(self.SERVO_PITCH, self.CENTER_PITCH)
            self.set_servo_safe(self.SERVO_ROLL_LEFT, self.CENTER_ROLL_LEFT)
            self.set_servo_safe(self.SERVO_ROLL_RIGHT, self.CENTER_ROLL_RIGHT)
            
            # Actualizar estado
            self.current_eye_left_h = self.CENTER_EYE_LEFT_H
            self.current_eye_right_h = self.CENTER_EYE_RIGHT_H
            self.current_eye_left_v = self.CENTER_EYE_LEFT_V
            self.current_eye_right_v = self.CENTER_EYE_RIGHT_V
            self.current_pitch = self.CENTER_PITCH
            self.current_roll_left = self.CENTER_ROLL_LEFT
            self.current_roll_right = self.CENTER_ROLL_RIGHT
            
            time.sleep(0.5)
            logger.info("✓ Centrado completado (ojos + párpados + cabeza)")
            
        except Exception as e:
            logger.error("❌ Error al centrar: %s" % str(e))
    
    def set_servo_safe(self, channel, angle):
        """
        Establecer ángulo de servo con manejo de errores
        
        Los límites específicos se aplican en los métodos de cálculo de cada servo
        """
        try:
            # Clamp 0-180 (límite físico de servos)
            angle = max(0, min(180, angle))
            
            self.kit.servo[channel].angle = angle
        except Exception as e:
            logger.warning("⚠️ Error en servo %d: %s" % (channel, str(e)))
    
    def clamp(self, value, min_val, max_val):
        """Limitar valor entre mínimo y máximo"""
        return max(min_val, min(max_val, value))
    
    def calculate_eye_position(self, error_x, error_y):
        """
        Calcular nuevas posiciones de ojos basado en error
        
        Retorna: (eye_left_h, eye_right_h, eye_left_v, eye_right_v)
        """
        
        # Aplicar deadband
        if abs(error_x) < self.DEADBAND_X:
            error_x = 0
        if abs(error_y) < self.DEADBAND_Y:
            error_y = 0
        
        # Calcular movimiento horizontal
        # error_x negativo = cara a la derecha → ojos van a la derecha (aumentar ángulo)
        # error_x positivo = cara a la izquierda → ojos van a la izquierda (disminuir ángulo)
        delta_h = -error_x * self.KP_EYES
        
        new_eye_left_h = self.current_eye_left_h + delta_h
        new_eye_right_h = self.current_eye_right_h + delta_h
        
        # Aplicar límites
        new_eye_left_h = self.clamp(new_eye_left_h, self.EYE_H_MIN, self.EYE_H_MAX)
        new_eye_right_h = self.clamp(new_eye_right_h, self.EYE_H_MIN, self.EYE_H_MAX)
        
        # Calcular movimiento vertical
        # error_y positivo = cara arriba → ojos van arriba (disminuir ángulo)
        # error_y negativo = cara abajo → ojos van abajo (aumentar ángulo)
        delta_v = -error_y * self.KP_EYES
        
        new_eye_left_v = self.current_eye_left_v + delta_v
        new_eye_right_v = self.current_eye_right_v + delta_v
        
        # Aplicar límites
        new_eye_left_v = self.clamp(new_eye_left_v, self.EYE_V_MIN, self.EYE_V_MAX)
        new_eye_right_v = self.clamp(new_eye_right_v, self.EYE_V_MIN, self.EYE_V_MAX)
        
        # Suavizado exponencial
        new_eye_left_h = self.SMOOTH_ALPHA * new_eye_left_h + (1 - self.SMOOTH_ALPHA) * self.current_eye_left_h
        new_eye_right_h = self.SMOOTH_ALPHA * new_eye_right_h + (1 - self.SMOOTH_ALPHA) * self.current_eye_right_h
        new_eye_left_v = self.SMOOTH_ALPHA * new_eye_left_v + (1 - self.SMOOTH_ALPHA) * self.current_eye_left_v
        new_eye_right_v = self.SMOOTH_ALPHA * new_eye_right_v + (1 - self.SMOOTH_ALPHA) * self.current_eye_right_v
        
        return new_eye_left_h, new_eye_right_h, new_eye_left_v, new_eye_right_v
    
    def calculate_head_compensation(self):
        """
        Calcular si la cabeza debe moverse para compensar
        cuando los ojos están cerca de sus límites
        
        Retorna: (new_pitch, compensate)
        """
        
        # Calcular qué tan lejos están los ojos del centro (normalizado 0-1)
        eye_v_range = self.EYE_V_MAX - self.EYE_V_MIN
        eye_v_center = (self.EYE_V_MAX + self.EYE_V_MIN) / 2.0
        
        eye_avg_v = (self.current_eye_left_v + self.current_eye_right_v) / 2.0
        eye_offset_v = abs(eye_avg_v - eye_v_center) / (eye_v_range / 2.0)
        
        # Si los ojos están más allá del umbral, compensar con cabeza
        if eye_offset_v > self.HEAD_THRESHOLD:
            # Calcular dirección de compensación
            if eye_avg_v < eye_v_center:
                # Ojos mirando arriba → cabeza sube
                delta_pitch = -(eye_avg_v - eye_v_center) * self.KP_HEAD
            else:
                # Ojos mirando abajo → cabeza baja
                delta_pitch = (eye_avg_v - eye_v_center) * self.KP_HEAD
            
            new_pitch = self.current_pitch + delta_pitch
            new_pitch = self.clamp(new_pitch, self.PITCH_MIN, self.PITCH_MAX)
            
            # Suavizado
            new_pitch = self.SMOOTH_ALPHA * new_pitch + (1 - self.SMOOTH_ALPHA) * self.current_pitch
            
            return new_pitch, True
        
        return self.current_pitch, False
    
    def update_servos(self, eye_left_h, eye_right_h, eye_left_v, eye_right_v, pitch):
        """Actualizar posiciones de servos"""
        
        # Actualizar ojos
        self.set_servo_safe(self.SERVO_EYE_LEFT_H, int(eye_left_h))
        self.set_servo_safe(self.SERVO_EYE_RIGHT_H, int(eye_right_h))
        self.set_servo_safe(self.SERVO_EYE_LEFT_V, int(eye_left_v))
        self.set_servo_safe(self.SERVO_EYE_RIGHT_V, int(eye_right_v))
        
        # Actualizar cabeza (pitch)
        self.set_servo_safe(self.SERVO_PITCH, int(pitch))
        
        # Guardar estado actual
        self.current_eye_left_h = eye_left_h
        self.current_eye_right_h = eye_right_h
        self.current_eye_left_v = eye_left_v
        self.current_eye_right_v = eye_right_v
        self.current_pitch = pitch
    
    def run(self, show_video=True):
        """
        Ejecutar el loop de seguimiento facial
        
        Args:
            show_video: Mostrar ventana de video (False para headless)
        """
        
        if not self.start_camera():
            return
        
        # Centrar servos
        self.move_to_center()
        time.sleep(1)
        
        logger.info("🔄 Iniciando seguimiento facial...")
        logger.info("   Presiona 'q' o ESC para salir")
        
        frame_count = 0
        fps_start = time.time()
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    logger.error("❌ Error al leer frame")
                    break
                
                frame_count += 1
                
                # Mostrar FPS cada 30 frames
                if frame_count % 30 == 0:
                    elapsed = time.time() - fps_start
                    fps = 30 / elapsed if elapsed > 0 else 0
                    logger.info("📹 FPS: %.1f | Frames: %d" % (fps, frame_count))
                    fps_start = time.time()
                
                # Convertir a RGB para MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.face_detection.process(rgb_frame)
                
                error_x = None
                error_y = None
                
                if results.detections:
                    # Tomar la primera detección (cara principal)
                    detection = results.detections[0]
                    
                    if show_video:
                        self.mp_drawing.draw_detection(frame, detection)
                    
                    # Obtener centro de la cara
                    bbox = detection.location_data.relative_bounding_box
                    face_x = int((bbox.xmin + bbox.width / 2) * self.frame_width)
                    face_y = int((bbox.ymin + bbox.height / 2) * self.frame_height)
                    
                    # Calcular error desde el centro
                    error_x = self.center_x - face_x
                    error_y = self.center_y - face_y
                    
                    # Calcular nuevas posiciones de ojos
                    eye_left_h, eye_right_h, eye_left_v, eye_right_v = self.calculate_eye_position(error_x, error_y)
                    
                    # Calcular compensación de cabeza si es necesario
                    new_pitch, compensated = self.calculate_head_compensation()
                    
                    # Actualizar servos
                    self.update_servos(eye_left_h, eye_right_h, eye_left_v, eye_right_v, new_pitch)
                    
                    # Debug cada 10 frames
                    if frame_count % 10 == 0:
                        comp_str = " [HEAD COMP]" if compensated else ""
                        logger.info("👤 CARA | Error: X=%+4d Y=%+4d px | Eyes: H=%.0f° V=%.0f°%s" % 
                                  (error_x, error_y, eye_left_h, eye_left_v, comp_str))
                    
                    if show_video:
                        cv2.circle(frame, (face_x, face_y), 5, (0, 0, 255), -1)
                
                if show_video:
                    # Dibujar cruz central
                    cv2.drawMarker(frame, (self.center_x, self.center_y), 
                                 (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
                    
                    # Mostrar info
                    if error_x is not None and error_y is not None:
                        text = "Error X: %d px, Y: %d px" % (error_x, error_y)
                        cv2.putText(frame, text, (10, self.frame_height - 20),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # Mostrar posiciones actuales
                    pos_text = "Eyes H:%.0f V:%.0f | Head:%.0f" % (
                        self.current_eye_left_h, self.current_eye_left_v, self.current_pitch)
                    cv2.putText(frame, pos_text, (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    
                    cv2.imshow('InMoov Face Tracker', frame)
                    
                    if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
                        break
                else:
                    # Modo headless: solo detectar tecla Ctrl+C
                    time.sleep(0.01)
        
        except KeyboardInterrupt:
            logger.info("\n⚠️ Interrumpido por usuario")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Limpiar recursos"""
        logger.info("\n🔄 Limpiando recursos...")
        
        # Centrar servos
        self.move_to_center()
        
        # Liberar cámara
        if self.cap:
            self.cap.release()
        
        if cv2.getWindowImageRect('InMoov Face Tracker') != (-1, -1, -1, -1):
            cv2.destroyAllWindows()
        
        # Cerrar MediaPipe
        self.face_detection.close()
        
        logger.info("✅ Recursos liberados")


# ============================================================
# MAIN
# ============================================================

def main():
    """Función principal"""
    
    print("\n" + "="*60)
    print("🤖 InMoov Face Tracker")
    print("="*60)
    print("\nAdaptado del código de Will Cogley para InMoov")
    print("Hardware: RPi5 + PCA9685 + USB Camera")
    print("\n🛡️  LÍMITE DE SEGURIDAD - OJOS:")
    print("   - Servos de ojos limitados a: 70° - 110°")
    print("   - Movimiento de ojos: ±20° desde el centro (90°)")
    print("   - Servos de cabeza: Límites según configuración del proyecto")
    print("\n⚠️  IMPORTANTE:")
    print("   - Asegúrate de tener los servos conectados correctamente")
    print("   - Los canales del PCA9685 deben estar configurados")
    print("   - La cámara USB debe estar conectada")
    print("\n")
    
    # Preguntar modo de visualización
    response = input("¿Mostrar video en pantalla? (s/n) [s]: ").strip().lower()
    show_video = response != 'n'
    
    # Crear tracker
    try:
        tracker = InMoovFaceTracker()
        
        # Ejecutar
        tracker.run(show_video=show_video)
        
    except Exception as e:
        logger.error("❌ Error fatal: %s" % str(e))
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

