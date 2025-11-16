#!/usr/bin/env python3
"""
Sistema de seguimiento facial DUAL (X + Y)
- Tracking horizontal (YAW) + vertical (PITCH con ROLL)
- Enfoque en CARA con MediaPipe
- Búsqueda inteligente en ambos ejes
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

# Márgenes de seguridad PWM globales
PWM_MIN_SAFE = 650
PWM_MAX_SAFE = 2000

# Pines del PCA9685
SERVO_ROLL_LEFT  = 12  # Hombro izquierdo
SERVO_ROLL_RIGHT = 15  # Hombro derecho
SERVO_PITCH      = 14  # Control de pitch (vertical)
SERVO_YAW        = 13  # Control de yaw (horizontal)

# Configurar servos con márgenes de seguridad
for pin in [SERVO_ROLL_LEFT, SERVO_ROLL_RIGHT, SERVO_PITCH]:
    kit.servo[pin].actuation_range = 200
    kit.servo[pin].set_pulse_width_range(PWM_MIN_SAFE, PWM_MAX_SAFE)

# Servo YAW (horizontal) - configuración con márgenes de seguridad
kit.servo[SERVO_YAW].actuation_range = 180
kit.servo[SERVO_YAW].set_pulse_width_range(PWM_MIN_SAFE, PWM_MAX_SAFE)

# LÍMITES DE MOVIMIENTO
# Vertical (PITCH - Pin 14)
PITCH_MIN, PITCH_MAX = 90, 180
PITCH_CENTER = 120
# Roll (Pins 12 y 15)
ROLL_MIN, ROLL_MAX = 130, 180
ROLL_CENTER = 155
# Horizontal (YAW - Pin 13)
YAW_MIN, YAW_MAX = 90, 160
YAW_CENTER = 135

CAM_INDEX = 0
HEADLESS_MODE = True

# PID VERTICAL (mantener valores que funcionan bien)
INVERT_PITCH = -1
KP_PITCH = 20.0
KI_PITCH = 0.05
KD_PITCH = 1.0
SMOOTH_ALPHA_PITCH = 0.5
DEADBAND_PX_Y = 15

# PID HORIZONTAL (de test_seguir_rostro.py)
INVERT_YAW = -1
KP_YAW = 7.0
KI_YAW = 0.01
KD_YAW = 0.0
SMOOTH_ALPHA_YAW = 0.25
DEADBAND_PX_X = 40

# Configuración común
I_CLAMP = 30.0

# Búsqueda cuando pierde el rostro
LOST_AFTER_MS = 300
SEARCH_RATE_DPS_Y = 30.0  # Vertical
SEARCH_RATE_DPS_X = 25.0  # Horizontal
RETURN_CENTER_AFTER_MS = 8000

print("=" * 60)
print("🤖 SEGUIMIENTO FACIAL DUAL (Horizontal + Vertical)")
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
kit.servo[SERVO_YAW].angle = YAW_CENTER
kit.servo[SERVO_PITCH].angle = PITCH_CENTER
kit.servo[SERVO_ROLL_LEFT].angle = ROLL_CENTER
kit.servo[SERVO_ROLL_RIGHT].angle = ROLL_CENTER
time.sleep(0.5)

# Estado del sistema
last_yaw = float(YAW_CENTER)
last_pitch = float(PITCH_CENTER)

# PID para YAW (horizontal)
last_err_yaw = 0.0
sum_err_yaw = 0.0

# PID para PITCH (vertical)
last_err_pitch = 0.0
sum_err_pitch = 0.0

last_time = time.time()
last_seen_ms = time.time() * 1000.0
last_seen_dir_x = 0  # -1 = izquierda, +1 = derecha
last_seen_dir_y = 0  # -1 = arriba, +1 = abajo
is_centered = False
returning_to_center = False
frame_count = 0
fps_start = time.time()

def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v

def signal_handler(sig, frame):
    print("\n\n⚠️ Ctrl+C detectado - Deteniendo...")
    cleanup_and_exit()

def cleanup_and_exit():
    print("🔄 Volviendo servos a posición de centro...")
    try:
        kit.servo[SERVO_YAW].angle = YAW_CENTER
        kit.servo[SERVO_PITCH].angle = PITCH_CENTER
        kit.servo[SERVO_ROLL_LEFT].angle = ROLL_CENTER
        kit.servo[SERVO_ROLL_RIGHT].angle = ROLL_CENTER
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
print(f"   • YAW (horizontal): Pin {SERVO_YAW} ({YAW_MIN}°-{YAW_MAX}°, centro:{YAW_CENTER}°)")
print(f"   • PITCH (vertical): Pin {SERVO_PITCH} ({PITCH_MIN}°-{PITCH_MAX}°, centro:{PITCH_CENTER}°)")
print(f"   • ROLL: Pins {SERVO_ROLL_LEFT}, {SERVO_ROLL_RIGHT}")
print(f"   • PID Vertical: KP={KP_PITCH}, KI={KI_PITCH}, KD={KD_PITCH}")
print(f"   • PID Horizontal: KP={KP_YAW}, KI={KI_YAW}, KD={KD_YAW}")
print(f"   • Búsqueda: V={SEARCH_RATE_DPS_Y}°/s, H={SEARCH_RATE_DPS_X}°/s")
print(f"\n🎯 TRACKING DUAL (X + Y) - FUSIONADO")
print("\n🔄 Procesando frames...\n")

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

            # ========== CONTROL HORIZONTAL (YAW) ==========
            err_px_x = (face_cx - cx_frame) * INVERT_YAW
            if abs(err_px_x) <= DEADBAND_PX_X:
                err_px_x = 0.0
            err_x = err_px_x / (w / 2.0)

            sum_err_yaw += err_x * dt
            sum_err_yaw = clamp(sum_err_yaw, -I_CLAMP, I_CLAMP)
            
            der_x = (err_x - last_err_yaw) / dt
            last_err_yaw = err_x
            
            pid_out_yaw = KP_YAW * err_x + KI_YAW * sum_err_yaw + KD_YAW * der_x
            desired_yaw = clamp(last_yaw + pid_out_yaw, YAW_MIN, YAW_MAX)
            
            smooth_yaw = SMOOTH_ALPHA_YAW * desired_yaw + (1 - SMOOTH_ALPHA_YAW) * last_yaw
            last_yaw = smooth_yaw

            # ========== CONTROL VERTICAL (PITCH) ==========
            err_px_y = (face_cy - cy_frame) * INVERT_PITCH
            if abs(err_px_y) <= DEADBAND_PX_Y:
                err_px_y = 0.0
            err_y = err_px_y / (h / 2.0)

            sum_err_pitch += err_y * dt
            sum_err_pitch = clamp(sum_err_pitch, -I_CLAMP, I_CLAMP)
            
            der_y = (err_y - last_err_pitch) / dt
            last_err_pitch = err_y
            
            pid_out_pitch = KP_PITCH * err_y + KI_PITCH * sum_err_pitch + KD_PITCH * der_y
            desired_pitch = clamp(last_pitch + pid_out_pitch, PITCH_MIN, PITCH_MAX)
            
            smooth_pitch = SMOOTH_ALPHA_PITCH * desired_pitch + (1 - SMOOTH_ALPHA_PITCH) * last_pitch
            last_pitch = smooth_pitch

            # ========== ACTUALIZAR SERVOS ==========
            yaw_angle = int(round(smooth_yaw))
            pitch_angle = int(round(smooth_pitch))
            
            # YAW (horizontal)
            kit.servo[SERVO_YAW].angle = yaw_angle
            
            # PITCH (vertical)
            kit.servo[SERVO_PITCH].angle = pitch_angle
            
            # ROLL (inverso al pitch)
            roll_angle = int(ROLL_MAX - ((pitch_angle - PITCH_MIN) / (PITCH_MAX - PITCH_MIN)) * (ROLL_MAX - ROLL_MIN))
            kit.servo[SERVO_ROLL_LEFT].angle = roll_angle
            kit.servo[SERVO_ROLL_RIGHT].angle = roll_angle

            last_seen_ms = now_ms
            # Guardar direcciones
            last_seen_dir_x = -1 if (face_cx < cx_frame) else +1
            last_seen_dir_y = -1 if (face_cy < cy_frame) else +1

            if frame_count % 10 == 0:
                print(f"👤 ROSTRO | Pos:({face_cx},{face_cy}) | "
                      f"Err:({err_px_x:+3.0f},{err_px_y:+3.0f})px | "
                      f"YAW:{yaw_angle}° PITCH:{pitch_angle}° ROLL:{roll_angle}°")

            if not HEADLESS_MODE:
                cv2.putText(frame, f"YAW:{yaw_angle} PITCH:{pitch_angle} ROLL:{roll_angle}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        else:
            # ========== SIN ROSTRO DETECTADO ==========
            time_without_face = now_ms - last_seen_ms
            
            # RETORNO A CENTRO (tras 8 segundos sin detección)
            if time_without_face > RETURN_CENTER_AFTER_MS and not is_centered:
                if not returning_to_center:
                    print(f"\n⏺️  SIN ROSTRO POR {int(time_without_face/1000)}s - VOLVIENDO A CENTRO")
                    returning_to_center = True
                
                diff_yaw = YAW_CENTER - last_yaw
                diff_pitch = PITCH_CENTER - last_pitch
                
                if abs(diff_yaw) > 1 or abs(diff_pitch) > 1:
                    step_yaw = diff_yaw * 0.15
                    step_pitch = diff_pitch * 0.15
                    
                    last_yaw += step_yaw
                    last_pitch += step_pitch
                    
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
                        print(f"↩️  Centrando... YAW:{yaw_angle}° PITCH:{pitch_angle}°")
                else:
                    # Ya centrado
                    if not is_centered:
                        kit.servo[SERVO_YAW].angle = YAW_CENTER
                        kit.servo[SERVO_PITCH].angle = PITCH_CENTER
                        kit.servo[SERVO_ROLL_LEFT].angle = ROLL_CENTER
                        kit.servo[SERVO_ROLL_RIGHT].angle = ROLL_CENTER
                        last_yaw = float(YAW_CENTER)
                        last_pitch = float(PITCH_CENTER)
                        is_centered = True
                        returning_to_center = False
                        last_seen_dir_x = 0
                        last_seen_dir_y = 0
                        print(f"✓ CENTRADO (YAW:{YAW_CENTER}° PITCH:{PITCH_CENTER}°)\n")
            
            # MODO BÚSQUEDA (entre 300ms y 8s sin rostro)
            elif time_without_face > LOST_AFTER_MS and (last_seen_dir_x != 0 or last_seen_dir_y != 0) and not is_centered:
                # Búsqueda prioritaria en horizontal, luego vertical
                if last_seen_dir_x != 0:
                    search_dir_x = last_seen_dir_x * INVERT_YAW
                    last_yaw += search_dir_x * SEARCH_RATE_DPS_X * dt
                    last_yaw = clamp(last_yaw, YAW_MIN, YAW_MAX)
                    kit.servo[SERVO_YAW].angle = int(round(last_yaw))
                
                if last_seen_dir_y != 0:
                    search_dir_y = last_seen_dir_y * INVERT_PITCH
                    last_pitch += search_dir_y * SEARCH_RATE_DPS_Y * dt
                    last_pitch = clamp(last_pitch, PITCH_MIN, PITCH_MAX)
                    
                    pitch_angle = int(round(last_pitch))
                    kit.servo[SERVO_PITCH].angle = pitch_angle
                    
                    roll_angle = int(ROLL_MAX - ((pitch_angle - PITCH_MIN) / (PITCH_MAX - PITCH_MIN)) * (ROLL_MAX - ROLL_MIN))
                    kit.servo[SERVO_ROLL_LEFT].angle = roll_angle
                    kit.servo[SERVO_ROLL_RIGHT].angle = roll_angle
                
                if frame_count % 30 == 0:
                    dir_x_text = "IZQ" if last_seen_dir_x < 0 else "DER" if last_seen_dir_x > 0 else ""
                    dir_y_text = "ARRIBA" if last_seen_dir_y < 0 else "ABAJO" if last_seen_dir_y > 0 else ""
                    print(f"🔍 BUSCANDO {dir_x_text} {dir_y_text}... | YAW:{int(last_yaw)}° PITCH:{int(last_pitch)}° | Sin rostro: {int(time_without_face/1000)}s")
            
            else:
                # Pérdida corta: mantener posición
                if frame_count % 60 == 0 and time_without_face > 50:
                    print(f"⏱️  ESPERANDO... Sin rostro: {int(time_without_face/1000)}s")

        if not HEADLESS_MODE:
            # Líneas de referencia
            cv2.line(frame, (cx_frame, 0), (cx_frame, h), (255, 0, 0), 1)  # Vertical
            cv2.line(frame, (0, cy_frame), (w, cy_frame), (255, 0, 0), 1)  # Horizontal
            cv2.imshow("Seguimiento Facial Dual (X+Y)", frame)
        
        if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
            break

except KeyboardInterrupt:
    print("\n⚠️ Interrumpido")
    kit.servo[14].angle = 80

finally:
    cleanup_and_exit()
