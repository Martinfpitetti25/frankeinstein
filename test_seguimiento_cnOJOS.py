#!/usr/bin/env python3
"""
Sistema de seguimiento facial COMPLETO: OJOS + CUELLO
- Tracking con OJOS primero (±20° rápido)
- Compensación con CUELLO cuando ojos llegan al límite
- MediaPipe Face Detection
- Búsqueda inteligente dual
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")

import cv2
import time
import mediapipe as mp
from adafruit_servokit import ServoKit
import signal
import sys

kit = ServoKit(channels=16)

# ============================================================
# CONFIGURACIÓN DE SERVOS - CANALES PCA9685
# ============================================================

# CUELLO (del TEST_SEGUIMIENTO_FINAL.PY)
SERVO_ROLL_LEFT  = 12  # Hombro izquierdo
SERVO_ROLL_RIGHT = 15  # Hombro derecho
SERVO_PITCH      = 14  # Control de pitch (vertical)
SERVO_YAW        = 13  # Control de yaw (horizontal)

# OJOS - PÁRPADOS
SERVO_EYELID_UPPER = 10  # Párpado superior (ambos)
SERVO_EYELID_LOWER = 8   # Párpado inferior (ambos)

# OJOS - MOVIMIENTO
SERVO_EYE_LEFT_V  = 7   # Ojo izquierdo vertical
SERVO_EYE_LEFT_H  = 6   # Ojo izquierdo horizontal
SERVO_EYE_RIGHT_H = 9   # Ojo derecho horizontal
SERVO_EYE_RIGHT_V = 11  # Ojo derecho vertical

# ============================================================
# CONFIGURACIÓN DE SERVOS - CON MÁRGENES DE SEGURIDAD
# ============================================================

# Pulse width con márgenes de seguridad (evita extremos del PCA9685)
# Estándar: 500-2500 microsegundos
# Seguro: 600-2400 microsegundos (margen de 100μs en cada extremo)
PWM_MIN_SAFE = 650   # Mínimo seguro (en vez de 500)
PWM_MAX_SAFE = 2000  # Máximo seguro (en vez de 2500)

# Configurar servos de CUELLO
for pin in [SERVO_ROLL_LEFT, SERVO_ROLL_RIGHT, SERVO_PITCH]:
    kit.servo[pin].actuation_range = 200
    kit.servo[pin].set_pulse_width_range(PWM_MIN_SAFE, PWM_MAX_SAFE)

kit.servo[SERVO_YAW].actuation_range = 180
kit.servo[SERVO_YAW].set_pulse_width_range(PWM_MIN_SAFE, PWM_MAX_SAFE)

# Configurar servos de OJOS (con márgenes de seguridad)
for pin in [SERVO_EYE_LEFT_H, SERVO_EYE_RIGHT_H, SERVO_EYE_LEFT_V, SERVO_EYE_RIGHT_V]:
    kit.servo[pin].actuation_range = 180
    kit.servo[pin].set_pulse_width_range(PWM_MIN_SAFE, PWM_MAX_SAFE)

# Configurar servos de PÁRPADOS (si necesitan configuración diferente)
for pin in [SERVO_EYELID_UPPER, SERVO_EYELID_LOWER]:
    kit.servo[pin].actuation_range = 180
    kit.servo[pin].set_pulse_width_range(PWM_MIN_SAFE, PWM_MAX_SAFE)

# ============================================================
# LÍMITES DE MOVIMIENTO - VALORES REALES CALIBRADOS
# ============================================================

# PÁRPADOS - Límites calibrados
ANGLE_MIN_PARPADO_ABAJO = 60
ANGLE_MAX_PARPADO_ABAJO = 130
ANGLE_MIN_PARPADO_ARRIBA = 60
ANGLE_MAX_PARPADO_ARRIBA = 130

# OJOS VERTICALES - Límites calibrados
ANGLE_MIN_OJO_IZQUIERDO_VERTICAL = 70
ANGLE_MAX_OJO_IZQUIERDO_VERTICAL = 110
ANGLE_MIN_OJO_DERECHO_VERTICAL = 70   # NO ANDUVO, VALORES ESTIMADOS
ANGLE_MAX_OJO_DERECHO_VERTICAL = 110  # NO ANDUVO, VALORES ESTIMADOS

# OJOS HORIZONTALES - Límites calibrados
ANGLE_MIN_OJO_IZQUIERDO_HORIZONTAL = 15
ANGLE_MAX_OJO_IZQUIERDO_HORIZONTAL = 165
ANGLE_MIN_OJO_DERECHO_HORIZONTAL = 15
ANGLE_MAX_OJO_DERECHO_HORIZONTAL = 165

# CUELLO - Límites del proyecto original
PITCH_MIN, PITCH_MAX = 60, 180
PITCH_CENTER = 120
ROLL_MIN, ROLL_MAX = 130, 180
ROLL_CENTER = 155
YAW_MIN, YAW_MAX = 90, 180
YAW_CENTER = 135

# Centros de ojos (calculados como punto medio)
EYE_LEFT_H_CENTER = (ANGLE_MIN_OJO_IZQUIERDO_HORIZONTAL + ANGLE_MAX_OJO_IZQUIERDO_HORIZONTAL) // 2  # 90
EYE_RIGHT_H_CENTER = (ANGLE_MIN_OJO_DERECHO_HORIZONTAL + ANGLE_MAX_OJO_DERECHO_HORIZONTAL) // 2      # 90
EYE_LEFT_V_CENTER = (ANGLE_MIN_OJO_IZQUIERDO_VERTICAL + ANGLE_MAX_OJO_IZQUIERDO_VERTICAL) // 2       # 90
EYE_RIGHT_V_CENTER = (ANGLE_MIN_OJO_DERECHO_VERTICAL + ANGLE_MAX_OJO_DERECHO_VERTICAL) // 2          # 90

# Párpados
EYELID_UPPER_OPEN = ANGLE_MIN_PARPADO_ARRIBA
EYELID_UPPER_CLOSED = ANGLE_MAX_PARPADO_ARRIBA
EYELID_LOWER_OPEN = ANGLE_MIN_PARPADO_ABAJO
EYELID_LOWER_CLOSED = ANGLE_MAX_PARPADO_ABAJO

# Variables de compatibilidad con código existente (usando promedio de ambos ojos)
EYE_H_MIN = min(ANGLE_MIN_OJO_IZQUIERDO_HORIZONTAL, ANGLE_MIN_OJO_DERECHO_HORIZONTAL)  # 15
EYE_H_MAX = max(ANGLE_MAX_OJO_IZQUIERDO_HORIZONTAL, ANGLE_MAX_OJO_DERECHO_HORIZONTAL)  # 165
EYE_V_MIN = min(ANGLE_MIN_OJO_IZQUIERDO_VERTICAL, ANGLE_MIN_OJO_DERECHO_VERTICAL)      # 70
EYE_V_MAX = max(ANGLE_MAX_OJO_IZQUIERDO_VERTICAL, ANGLE_MAX_OJO_DERECHO_VERTICAL)      # 110
EYE_CENTER_H = (EYE_H_MIN + EYE_H_MAX) // 2  # 90
EYE_CENTER_V = (EYE_V_MIN + EYE_V_MAX) // 2  # 90

# ============================================================
# CONFIGURACIÓN DEL SISTEMA
# ============================================================

CAM_INDEX = 0
HEADLESS_MODE = True

# PID OJOS (rápido y preciso)
KP_EYE = 0.15
KI_EYE = 0.01
KD_EYE = 0.0
SMOOTH_ALPHA_EYE = 0.3
DEADBAND_EYE_X = 40
DEADBAND_EYE_Y = 30

# PID CUELLO VERTICAL (del proyecto original)
INVERT_PITCH = -1
KP_PITCH = 20.0
KI_PITCH = 0.05
KD_PITCH = 1.0
SMOOTH_ALPHA_PITCH = 0.5
DEADBAND_NECK_Y = 15

# PID CUELLO HORIZONTAL (del proyecto original)
INVERT_YAW = -1
KP_YAW = 7.0
KI_YAW = 0.01
KD_YAW = 0.0
SMOOTH_ALPHA_YAW = 0.25
DEADBAND_NECK_X = 40

# Común
I_CLAMP = 30.0

# Umbral para activar movimiento de cuello (70% del rango de ojos)
NECK_THRESHOLD = 0.70

# Búsqueda cuando pierde el rostro
LOST_AFTER_MS = 300
SEARCH_RATE_DPS_EYE = 15.0  # Ojos buscan más rápido
SEARCH_RATE_DPS_NECK_Y = 30.0  # Cuello vertical
SEARCH_RATE_DPS_NECK_X = 25.0  # Cuello horizontal
RETURN_CENTER_AFTER_MS = 8000

# ============================================================
# INICIALIZACIÓN
# ============================================================

print("=" * 60)
print("🤖 SEGUIMIENTO FACIAL COMPLETO: OJOS + CUELLO")
print("=" * 60)

print("\n🎥 Iniciando cámara...")
cap = cv2.VideoCapture(CAM_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("❌ No se pudo abrir la cámara")
    sys.exit(1)

print("🧠 Inicializando MediaPipe Face Detection...")
mp_fd = mp.solutions.face_detection
fd = mp_fd.FaceDetection(model_selection=0, min_detection_confidence=0.4)

print("📍 Posicionando servos en centro...")
# Cuello
kit.servo[SERVO_YAW].angle = YAW_CENTER
kit.servo[SERVO_PITCH].angle = PITCH_CENTER
kit.servo[SERVO_ROLL_LEFT].angle = ROLL_CENTER
kit.servo[SERVO_ROLL_RIGHT].angle = ROLL_CENTER
# Ojos - Con centros específicos calibrados
kit.servo[SERVO_EYE_LEFT_H].angle = EYE_LEFT_H_CENTER
kit.servo[SERVO_EYE_RIGHT_H].angle = EYE_RIGHT_H_CENTER
kit.servo[SERVO_EYE_LEFT_V].angle = EYE_LEFT_V_CENTER
kit.servo[SERVO_EYE_RIGHT_V].angle = EYE_RIGHT_V_CENTER
# Párpados abiertos
kit.servo[SERVO_EYELID_UPPER].angle = EYELID_UPPER_OPEN
kit.servo[SERVO_EYELID_LOWER].angle = EYELID_LOWER_OPEN
time.sleep(0.5)

# ============================================================
# ESTADO DEL SISTEMA
# ============================================================

# Estado OJOS - Inicializar con centros específicos
last_eye_left_h = float(EYE_LEFT_H_CENTER)
last_eye_right_h = float(EYE_RIGHT_H_CENTER)
last_eye_left_v = float(EYE_LEFT_V_CENTER)
last_eye_right_v = float(EYE_RIGHT_V_CENTER)

# PID OJOS
last_err_eye_x = 0.0
sum_err_eye_x = 0.0
last_err_eye_y = 0.0
sum_err_eye_y = 0.0

# Estado CUELLO
last_yaw = float(YAW_CENTER)
last_pitch = float(PITCH_CENTER)

# PID CUELLO
last_err_yaw = 0.0
sum_err_yaw = 0.0
last_err_pitch = 0.0
sum_err_pitch = 0.0

# General
last_time = time.time()
last_seen_ms = time.time() * 1000.0
last_seen_dir_x = 0
last_seen_dir_y = 0
is_centered = False
returning_to_center = False
frame_count = 0
fps_start = time.time()

# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v

def signal_handler(sig, frame):
    print("\n\n⚠️ Ctrl+C detectado - Deteniendo...")
    cleanup_and_exit()

def cleanup_and_exit():
    print("🔄 Volviendo servos a posición de centro...")
    try:
        # Cuello
        kit.servo[SERVO_YAW].angle = YAW_CENTER
        kit.servo[SERVO_PITCH].angle = PITCH_CENTER
        kit.servo[SERVO_ROLL_LEFT].angle = ROLL_CENTER
        kit.servo[SERVO_ROLL_RIGHT].angle = ROLL_CENTER
        # Ojos - Con centros específicos
        kit.servo[SERVO_EYE_LEFT_H].angle = EYE_LEFT_H_CENTER
        kit.servo[SERVO_EYE_RIGHT_H].angle = EYE_RIGHT_H_CENTER
        kit.servo[SERVO_EYE_LEFT_V].angle = EYE_LEFT_V_CENTER
        kit.servo[SERVO_EYE_RIGHT_V].angle = EYE_RIGHT_V_CENTER
        time.sleep(0.3)
    except:
        pass
    cap.release()
    if not HEADLESS_MODE:
        cv2.destroyAllWindows()
    print("✅ Sistema finalizado")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

print("\n📊 Configuración:")
print(f"   • Cámara: /dev/video{CAM_INDEX} @ 640x480")
print(f"   • PÁRPADOS: Superior({ANGLE_MIN_PARPADO_ARRIBA}°-{ANGLE_MAX_PARPADO_ARRIBA}°) Inferior({ANGLE_MIN_PARPADO_ABAJO}°-{ANGLE_MAX_PARPADO_ABAJO}°)")
print(f"   • OJO IZQ: H({ANGLE_MIN_OJO_IZQUIERDO_HORIZONTAL}°-{ANGLE_MAX_OJO_IZQUIERDO_HORIZONTAL}°) V({ANGLE_MIN_OJO_IZQUIERDO_VERTICAL}°-{ANGLE_MAX_OJO_IZQUIERDO_VERTICAL}°)")
print(f"   • OJO DER: H({ANGLE_MIN_OJO_DERECHO_HORIZONTAL}°-{ANGLE_MAX_OJO_DERECHO_HORIZONTAL}°) V({ANGLE_MIN_OJO_DERECHO_VERTICAL}°-{ANGLE_MAX_OJO_DERECHO_VERTICAL}°) [ESTIMADO]")
print(f"   • YAW (cuello H): Pin {SERVO_YAW} ({YAW_MIN}°-{YAW_MAX}°)")
print(f"   • PITCH (cuello V): Pin {SERVO_PITCH} ({PITCH_MIN}°-{PITCH_MAX}°)")
print(f"   • Jerarquía: Ojos primero → Cuello si ojos > 70%")
print(f"\n🎯 TRACKING COMPLETO - Ojos + Cuello")

# ============================================================
# INICIALIZACIÓN DE SERVOS
# ============================================================

print("🔧 Inicializando servos en posiciones calibradas...")
try:
    # Cuello a centro
    kit.servo[SERVO_YAW].angle = YAW_CENTER
    kit.servo[SERVO_PITCH].angle = PITCH_CENTER
    kit.servo[SERVO_ROLL_LEFT].angle = ROLL_CENTER
    kit.servo[SERVO_ROLL_RIGHT].angle = ROLL_CENTER
    
    # Ojos a sus centros específicos (calibrados)
    kit.servo[SERVO_EYE_LEFT_H].angle = EYE_LEFT_H_CENTER
    kit.servo[SERVO_EYE_RIGHT_H].angle = EYE_RIGHT_H_CENTER
    kit.servo[SERVO_EYE_LEFT_V].angle = EYE_LEFT_V_CENTER
    kit.servo[SERVO_EYE_RIGHT_V].angle = EYE_RIGHT_V_CENTER
    
    # Párpados abiertos
    kit.servo[SERVO_EYELID_UPPER].angle = EYELID_UPPER_OPEN
    kit.servo[SERVO_EYELID_LOWER].angle = EYELID_LOWER_OPEN
    
    print(f"✅ Servos inicializados:")
    print(f"   • Ojo Izq: H={EYE_LEFT_H_CENTER}° V={EYE_LEFT_V_CENTER}°")
    print(f"   • Ojo Der: H={EYE_RIGHT_H_CENTER}° V={EYE_RIGHT_V_CENTER}°")
    print(f"   • Cuello: YAW={YAW_CENTER}° PITCH={PITCH_CENTER}°")
    
    time.sleep(1.0)  # Dar tiempo para posicionarse
    
except Exception as e:
    print(f"⚠️ Error inicializando servos: {e}")

print("\n🔄 Procesando frames...\n")

# ============================================================
# LOOP PRINCIPAL
# ============================================================

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_count += 1
        
        if frame_count % 30 == 0:
            elapsed = time.time() - fps_start
            fps = 30 / elapsed if elapsed > 0 else 0
            print(f"📹 FPS: {fps:.1f}")
            fps_start = time.time()

        now = time.time()
        dt = max(1e-3, now - last_time)
        last_time = now
        now_ms = now * 1000.0

        h, w = frame.shape[:2]
        cx_frame = w // 2
        cy_frame = h // 2

        # Pre-procesamiento
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        frame_eq = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        rgb = cv2.cvtColor(frame_eq, cv2.COLOR_BGR2RGB)
        results = fd.process(rgb)
        have_face = bool(results.detections)

        if have_face:
            # ========== ROSTRO DETECTADO ==========
            is_centered = False
            returning_to_center = False
            
            detection = max(results.detections, key=lambda d: d.score[0])
            bbox = detection.location_data.relative_bounding_box
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            bw = int(bbox.width * w)
            bh = int(bbox.height * h)
            face_cx = x + bw // 2
            face_cy = y + bh // 2
            
            if not HEADLESS_MODE:
                cv2.rectangle(frame, (x, y), (x+bw, y+bh), (0, 255, 0), 2)
                cv2.circle(frame, (face_cx, face_cy), 6, (0, 0, 255), -1)

            # ========== PASO 1: CONTROL DE OJOS (PRIORITARIO) ==========
            
            # Error horizontal de ojos
            err_px_eye_x = cx_frame - face_cx
            if abs(err_px_eye_x) <= DEADBAND_EYE_X:
                err_px_eye_x = 0.0
            err_eye_x = err_px_eye_x / (w / 2.0)
            
            sum_err_eye_x += err_eye_x * dt
            sum_err_eye_x = clamp(sum_err_eye_x, -I_CLAMP, I_CLAMP)
            
            der_eye_x = (err_eye_x - last_err_eye_x) / dt
            last_err_eye_x = err_eye_x
            
            pid_out_eye_x = KP_EYE * err_eye_x + KI_EYE * sum_err_eye_x + KD_EYE * der_eye_x
            
            # Actualizar posición horizontal de ojos
            desired_eye_h = last_eye_left_h - pid_out_eye_x * (w / 2.0)
            desired_eye_h = clamp(desired_eye_h, EYE_H_MIN, EYE_H_MAX)
            smooth_eye_h = SMOOTH_ALPHA_EYE * desired_eye_h + (1 - SMOOTH_ALPHA_EYE) * last_eye_left_h
            last_eye_left_h = smooth_eye_h
            last_eye_right_h = smooth_eye_h  # Ambos ojos se mueven igual
            
            # Error vertical de ojos
            err_px_eye_y = cy_frame - face_cy
            if abs(err_px_eye_y) <= DEADBAND_EYE_Y:
                err_px_eye_y = 0.0
            err_eye_y = err_px_eye_y / (h / 2.0)
            
            sum_err_eye_y += err_eye_y * dt
            sum_err_eye_y = clamp(sum_err_eye_y, -I_CLAMP, I_CLAMP)
            
            der_eye_y = (err_eye_y - last_err_eye_y) / dt
            last_err_eye_y = err_eye_y
            
            pid_out_eye_y = KP_EYE * err_eye_y + KI_EYE * sum_err_eye_y + KD_EYE * der_eye_y
            
            # Actualizar posición vertical de ojos
            desired_eye_v = last_eye_left_v - pid_out_eye_y * (h / 2.0)
            desired_eye_v = clamp(desired_eye_v, EYE_V_MIN, EYE_V_MAX)
            smooth_eye_v = SMOOTH_ALPHA_EYE * desired_eye_v + (1 - SMOOTH_ALPHA_EYE) * last_eye_left_v
            last_eye_left_v = smooth_eye_v
            last_eye_right_v = smooth_eye_v  # Ambos ojos se mueven igual
            
            # Aplicar movimiento de ojos CON LÍMITES ESPECÍFICOS
            # Ojo izquierdo horizontal
            eye_left_h_angle = int(round(clamp(smooth_eye_h, 
                                             ANGLE_MIN_OJO_IZQUIERDO_HORIZONTAL, 
                                             ANGLE_MAX_OJO_IZQUIERDO_HORIZONTAL)))
            
            # Ojo derecho horizontal  
            eye_right_h_angle = int(round(clamp(smooth_eye_h, 
                                              ANGLE_MIN_OJO_DERECHO_HORIZONTAL, 
                                              ANGLE_MAX_OJO_DERECHO_HORIZONTAL)))
            
            # Ojo izquierdo vertical
            eye_left_v_angle = int(round(clamp(smooth_eye_v, 
                                             ANGLE_MIN_OJO_IZQUIERDO_VERTICAL, 
                                             ANGLE_MAX_OJO_IZQUIERDO_VERTICAL)))
            
            # Ojo derecho vertical
            eye_right_v_angle = int(round(clamp(smooth_eye_v, 
                                              ANGLE_MIN_OJO_DERECHO_VERTICAL, 
                                              ANGLE_MAX_OJO_DERECHO_VERTICAL)))
            
            # Aplicar movimientos respetando límites individuales
            kit.servo[SERVO_EYE_LEFT_H].angle = eye_left_h_angle
            kit.servo[SERVO_EYE_RIGHT_H].angle = eye_right_h_angle
            kit.servo[SERVO_EYE_LEFT_V].angle = eye_left_v_angle
            kit.servo[SERVO_EYE_RIGHT_V].angle = eye_right_v_angle
            
            # ========== PASO 2: CALCULAR SI NECESITA COMPENSACIÓN DE CUELLO ==========
            
            # Calcular qué tan lejos están los ojos del centro (normalizado 0-1)
            eye_h_range = EYE_H_MAX - EYE_H_MIN
            eye_h_center = (EYE_H_MAX + EYE_H_MIN) / 2.0
            eye_h_offset = abs(smooth_eye_h - eye_h_center) / (eye_h_range / 2.0)
            
            eye_v_range = EYE_V_MAX - EYE_V_MIN
            eye_v_center = (EYE_V_MAX + EYE_V_MIN) / 2.0
            eye_v_offset = abs(smooth_eye_v - eye_v_center) / (eye_v_range / 2.0)
            
            # Flags de compensación
            compensate_yaw = eye_h_offset > NECK_THRESHOLD
            compensate_pitch = eye_v_offset > NECK_THRESHOLD
            
            # ========== PASO 3: CONTROL DE CUELLO (SOLO SI ES NECESARIO) ==========
            
            if compensate_yaw:
                # Control horizontal del cuello (YAW)
                err_px_neck_x = (face_cx - cx_frame) * INVERT_YAW
                if abs(err_px_neck_x) <= DEADBAND_NECK_X:
                    err_px_neck_x = 0.0
                err_neck_x = err_px_neck_x / (w / 2.0)
                
                sum_err_yaw += err_neck_x * dt
                sum_err_yaw = clamp(sum_err_yaw, -I_CLAMP, I_CLAMP)
                
                der_x = (err_neck_x - last_err_yaw) / dt
                last_err_yaw = err_neck_x
                
                pid_out_yaw = KP_YAW * err_neck_x + KI_YAW * sum_err_yaw + KD_YAW * der_x
                desired_yaw = clamp(last_yaw + pid_out_yaw, YAW_MIN, YAW_MAX)
                
                smooth_yaw = SMOOTH_ALPHA_YAW * desired_yaw + (1 - SMOOTH_ALPHA_YAW) * last_yaw
                last_yaw = smooth_yaw
                kit.servo[SERVO_YAW].angle = int(round(smooth_yaw))
            
            if compensate_pitch:
                # Control vertical del cuello (PITCH)
                err_px_neck_y = (face_cy - cy_frame) * INVERT_PITCH
                if abs(err_px_neck_y) <= DEADBAND_NECK_Y:
                    err_px_neck_y = 0.0
                err_neck_y = err_px_neck_y / (h / 2.0)
                
                sum_err_pitch += err_neck_y * dt
                sum_err_pitch = clamp(sum_err_pitch, -I_CLAMP, I_CLAMP)
                
                der_y = (err_neck_y - last_err_pitch) / dt
                last_err_pitch = err_neck_y
                
                pid_out_pitch = KP_PITCH * err_neck_y + KI_PITCH * sum_err_pitch + KD_PITCH * der_y
                desired_pitch = clamp(last_pitch + pid_out_pitch, PITCH_MIN, PITCH_MAX)
                
                smooth_pitch = SMOOTH_ALPHA_PITCH * desired_pitch + (1 - SMOOTH_ALPHA_PITCH) * last_pitch
                last_pitch = smooth_pitch
                
                pitch_angle = int(round(smooth_pitch))
                kit.servo[SERVO_PITCH].angle = pitch_angle
                
                # ROLL (inverso al pitch)
                roll_angle = int(ROLL_MAX - ((pitch_angle - PITCH_MIN) / (PITCH_MAX - PITCH_MIN)) * (ROLL_MAX - ROLL_MIN))
                kit.servo[SERVO_ROLL_LEFT].angle = roll_angle
                kit.servo[SERVO_ROLL_RIGHT].angle = roll_angle

            last_seen_ms = now_ms
            last_seen_dir_x = -1 if (face_cx < cx_frame) else +1
            last_seen_dir_y = -1 if (face_cy < cy_frame) else +1

            if frame_count % 10 == 0:
                comp_text = ""
                if compensate_yaw:
                    comp_text += " [YAW]"
                if compensate_pitch:
                    comp_text += " [PITCH]"
                
                print(f"👤 ROSTRO | Ojos: H={eye_left_h_angle}°/{eye_right_h_angle}° V={eye_left_v_angle}°/{eye_right_v_angle}° | "
                      f"Cuello: YAW={int(last_yaw)}° PITCH={int(last_pitch)}°{comp_text}")

            if not HEADLESS_MODE:
                cv2.putText(frame, f"Eyes:L{eye_left_h_angle}/{eye_left_v_angle} R{eye_right_h_angle}/{eye_right_v_angle} Neck:{int(last_yaw)}/{int(last_pitch)}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        else:
            # ========== SIN ROSTRO DETECTADO ==========
            time_without_face = now_ms - last_seen_ms
            
            # RETORNO A CENTRO (tras 8 segundos sin detección)
            if time_without_face > RETURN_CENTER_AFTER_MS and not is_centered:
                if not returning_to_center:
                    print(f"\n⏺️  SIN ROSTRO POR {int(time_without_face/1000)}s - VOLVIENDO A CENTRO")
                    returning_to_center = True
                
                # Centrar ojos con sus centros específicos
                diff_eye_left_h = EYE_LEFT_H_CENTER - last_eye_left_h
                diff_eye_left_v = EYE_LEFT_V_CENTER - last_eye_left_v
                diff_eye_right_h = EYE_RIGHT_H_CENTER - last_eye_right_h
                diff_eye_right_v = EYE_RIGHT_V_CENTER - last_eye_right_v
                
                # Centrar cuello
                diff_yaw = YAW_CENTER - last_yaw
                diff_pitch = PITCH_CENTER - last_pitch
                
                if (abs(diff_eye_left_h) > 1 or abs(diff_eye_left_v) > 1 or 
                    abs(diff_eye_right_h) > 1 or abs(diff_eye_right_v) > 1 or 
                    abs(diff_yaw) > 1 or abs(diff_pitch) > 1):
                    
                    # Mover cada ojo hacia su centro específico
                    last_eye_left_h += diff_eye_left_h * 0.15
                    last_eye_left_v += diff_eye_left_v * 0.15
                    last_eye_right_h += diff_eye_right_h * 0.15
                    last_eye_right_v += diff_eye_right_v * 0.15
                    
                    # Aplicar límites individuales y mover servos
                    eye_left_h_angle = int(round(clamp(last_eye_left_h, 
                                                     ANGLE_MIN_OJO_IZQUIERDO_HORIZONTAL, 
                                                     ANGLE_MAX_OJO_IZQUIERDO_HORIZONTAL)))
                    eye_right_h_angle = int(round(clamp(last_eye_right_h, 
                                                      ANGLE_MIN_OJO_DERECHO_HORIZONTAL, 
                                                      ANGLE_MAX_OJO_DERECHO_HORIZONTAL)))
                    eye_left_v_angle = int(round(clamp(last_eye_left_v, 
                                                     ANGLE_MIN_OJO_IZQUIERDO_VERTICAL, 
                                                     ANGLE_MAX_OJO_IZQUIERDO_VERTICAL)))
                    eye_right_v_angle = int(round(clamp(last_eye_right_v, 
                                                      ANGLE_MIN_OJO_DERECHO_VERTICAL, 
                                                      ANGLE_MAX_OJO_DERECHO_VERTICAL)))
                    
                    kit.servo[SERVO_EYE_LEFT_H].angle = eye_left_h_angle
                    kit.servo[SERVO_EYE_RIGHT_H].angle = eye_right_h_angle
                    kit.servo[SERVO_EYE_LEFT_V].angle = eye_left_v_angle
                    kit.servo[SERVO_EYE_RIGHT_V].angle = eye_right_v_angle
                    
                    # Mover cuello
                    last_yaw += diff_yaw * 0.15
                    last_pitch += diff_pitch * 0.15
                    
                    last_yaw = clamp(last_yaw, YAW_MIN, YAW_MAX)
                    last_pitch = clamp(last_pitch, PITCH_MIN, PITCH_MAX)
                    
                    yaw_angle = int(round(last_yaw))
                    pitch_angle = int(round(last_pitch))
                    
                    kit.servo[SERVO_YAW].angle = yaw_angle
                    kit.servo[SERVO_PITCH].angle = pitch_angle
                    
                    roll_angle = int(ROLL_MAX - ((pitch_angle - PITCH_MIN) / (PITCH_MAX - PITCH_MIN)) * (ROLL_MAX - ROLL_MIN))
                    kit.servo[SERVO_ROLL_LEFT].angle = roll_angle
                    kit.servo[SERVO_ROLL_RIGHT].angle = roll_angle
                    
                    if frame_count % 15 == 0:
                        print(f"↩️  Centrando... Ojos:L{eye_left_h_angle}/{eye_left_v_angle} R{eye_right_h_angle}/{eye_right_v_angle} Cuello:{yaw_angle}/{pitch_angle}")
                else:
                    # Ya centrado
                    if not is_centered:
                        # Centrar todo con posiciones específicas
                        kit.servo[SERVO_EYE_LEFT_H].angle = EYE_LEFT_H_CENTER
                        kit.servo[SERVO_EYE_RIGHT_H].angle = EYE_RIGHT_H_CENTER
                        kit.servo[SERVO_EYE_LEFT_V].angle = EYE_LEFT_V_CENTER
                        kit.servo[SERVO_EYE_RIGHT_V].angle = EYE_RIGHT_V_CENTER
                        kit.servo[SERVO_YAW].angle = YAW_CENTER
                        kit.servo[SERVO_PITCH].angle = PITCH_CENTER
                        kit.servo[SERVO_ROLL_LEFT].angle = ROLL_CENTER
                        kit.servo[SERVO_ROLL_RIGHT].angle = ROLL_CENTER
                        
                        last_eye_left_h = float(EYE_LEFT_H_CENTER)
                        last_eye_right_h = float(EYE_RIGHT_H_CENTER)
                        last_eye_left_v = float(EYE_LEFT_V_CENTER)
                        last_eye_right_v = float(EYE_RIGHT_V_CENTER)
                        last_yaw = float(YAW_CENTER)
                        last_pitch = float(PITCH_CENTER)
                        
                        is_centered = True
                        returning_to_center = False
                        last_seen_dir_x = 0
                        last_seen_dir_y = 0
                        print(f"✓ CENTRADO (Todo en posición neutral)\n")
            
            # MODO BÚSQUEDA (entre 300ms y 8s sin rostro)
            elif time_without_face > LOST_AFTER_MS and (last_seen_dir_x != 0 or last_seen_dir_y != 0) and not is_centered:
                # Búsqueda con ojos primero, luego cuello
                if last_seen_dir_x != 0:
                    # Ojos horizontal
                    last_eye_left_h += last_seen_dir_x * SEARCH_RATE_DPS_EYE * dt
                    last_eye_left_h = clamp(last_eye_left_h, EYE_H_MIN, EYE_H_MAX)
                    last_eye_right_h = last_eye_left_h
                    
                    kit.servo[SERVO_EYE_LEFT_H].angle = int(round(last_eye_left_h))
                    kit.servo[SERVO_EYE_RIGHT_H].angle = int(round(last_eye_right_h))
                    
                    # Cuello horizontal
                    search_dir_x = last_seen_dir_x * INVERT_YAW
                    last_yaw += search_dir_x * SEARCH_RATE_DPS_NECK_X * dt
                    last_yaw = clamp(last_yaw, YAW_MIN, YAW_MAX)
                    kit.servo[SERVO_YAW].angle = int(round(last_yaw))
                
                if last_seen_dir_y != 0:
                    # Ojos vertical
                    last_eye_left_v += last_seen_dir_y * SEARCH_RATE_DPS_EYE * dt
                    last_eye_left_v = clamp(last_eye_left_v, EYE_V_MIN, EYE_V_MAX)
                    last_eye_right_v = last_eye_left_v
                    
                    kit.servo[SERVO_EYE_LEFT_V].angle = int(round(last_eye_left_v))
                    kit.servo[SERVO_EYE_RIGHT_V].angle = int(round(last_eye_right_v))
                    
                    # Cuello vertical
                    search_dir_y = last_seen_dir_y * INVERT_PITCH
                    last_pitch += search_dir_y * SEARCH_RATE_DPS_NECK_Y * dt
                    last_pitch = clamp(last_pitch, PITCH_MIN, PITCH_MAX)
                    
                    pitch_angle = int(round(last_pitch))
                    kit.servo[SERVO_PITCH].angle = pitch_angle
                    
                    roll_angle = int(ROLL_MAX - ((pitch_angle - PITCH_MIN) / (PITCH_MAX - PITCH_MIN)) * (ROLL_MAX - ROLL_MIN))
                    kit.servo[SERVO_ROLL_LEFT].angle = roll_angle
                    kit.servo[SERVO_ROLL_RIGHT].angle = roll_angle
                
                if frame_count % 30 == 0:
                    dir_x_text = "IZQ" if last_seen_dir_x < 0 else "DER" if last_seen_dir_x > 0 else ""
                    dir_y_text = "ARRIBA" if last_seen_dir_y < 0 else "ABAJO" if last_seen_dir_y > 0 else ""
                    print(f"🔍 BUSCANDO {dir_x_text} {dir_y_text}... | Ojos+Cuello | {int(time_without_face/1000)}s")
            
            else:
                # Pérdida corta: mantener posición
                if frame_count % 60 == 0 and time_without_face > 50:
                    print(f"⏱️  ESPERANDO... Sin rostro: {int(time_without_face/1000)}s")

        if not HEADLESS_MODE:
            cv2.line(frame, (cx_frame, 0), (cx_frame, h), (255, 0, 0), 1)
            cv2.line(frame, (0, cy_frame), (w, cy_frame), (255, 0, 0), 1)
            cv2.imshow("Seguimiento: Ojos + Cuello", frame)
        
        if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
            break

except KeyboardInterrupt:
    print("\n⚠️ Interrumpido")

finally:
    cleanup_and_exit()

