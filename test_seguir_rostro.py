# follow_face_servokit_mediapipe.py — PID simple + búsqueda (solo horizontal)
# Adaptado de ESP32 UDP a ServoKit directo - Pin 13 del PCA9685
import cv2
import time
import mediapipe as mp
from adafruit_servokit import ServoKit

# === SERVO (PCA9685) ===
kit = ServoKit(channels=16)
SERVO_PAN = 13  # Pin 13 del PCA9685 para movimiento horizontal

# Configuración del servo
kit.servo[SERVO_PAN].actuation_range = 180
kit.servo[SERVO_PAN].set_pulse_width_range(500, 2500)

# === CÁMARA ===
CAM_INDEX = 0  # GENERAL WEBCAM USB en /dev/video0

# === MODO HEADLESS (sin ventana) ===
HEADLESS_MODE = True  # True = sin GUI, False = con ventana de visualización

# === LÍMITES DEL SERVO ===
ANGLE_MIN, ANGLE_MAX = 90, 180

# === PID (seguimiento) ===
INVERT_DIR    = -1     # si "esquiva", cambiá a +1 o -1
KP            = 7.0
KI            = 0.01
KD            = 0.0
I_CLAMP       = 30.0   # anti-windup

# === Suavizado ===
SMOOTH_ALPHA  = 0.25   # 0.2=suave, 0.5=rápido
DEADBAND_PX   = 40

# === BÚSQUEDA (cuando te pierde) ===
LOST_AFTER_MS   = 300          # entra a buscar si pasa este tiempo sin cara
SEARCH_RATE_DPS = 25.0         # °/s de giro en búsqueda
SEARCH_FLIP     = 1            # 0 = normal, 1 = invierte solo la búsqueda

# === RETORNO A CENTRO ===
RETURN_CENTER_AFTER_MS = 8000  # Vuelve a 90° si no detecta nada en 8 segundos
CENTER_ANGLE = 90              # Ángulo de reposo

# --- Setup ---
print("🎥 Iniciando cámara...")
cap = cv2.VideoCapture(CAM_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("❌ No se pudo abrir la cámara")
    exit(1)

print("🤖 Inicializando MediaPipe Face Detection...")
mp_fd = mp.solutions.face_detection
fd = mp_fd.FaceDetection(model_selection=0, min_detection_confidence=0.3)  # Bajado de 0.5 a 0.3

# Centrar servo
print("📍 Centrando servo...")
kit.servo[SERVO_PAN].angle = 90

# Estado
last_time = time.time()
last_ang  = 90.0
last_err  = 0.0
sum_err   = 0.0

last_seen_ms  = time.time() * 1000.0  # Inicializar con tiempo actual
last_seen_dir = 0      # -1 izq, +1 der

# Estado de retorno a centro
is_centered = False    # NO comienza centrado (aunque físicamente sí lo está)
returning_to_center = False

# Debug FPS
frame_count = 0
fps_start = time.time()

def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v

print("✅ Iniciando seguimiento facial...")
print("Presiona 'q' o ESC para salir" if not HEADLESS_MODE else "Presiona Ctrl+C para salir")
print("\n📊 Estado del sistema:")
print(f"   Cámara: {CAM_INDEX}")
print(f"   Servo: Pin {SERVO_PAN}")
print(f"   Modo headless: {HEADLESS_MODE}")
print(f"   Resolución: 640x480")
print("\n🔄 Procesando frames...\n")

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            print("❌ Error al leer frame de la cámara")
            break

        frame_count += 1
        
        # Mostrar FPS cada 30 frames
        if frame_count % 30 == 0:
            elapsed = time.time() - fps_start
            fps = 30 / elapsed
            print(f"📹 FPS: {fps:.1f} | Frames procesados: {frame_count}")
            fps_start = time.time()

        now = time.time()
        dt  = max(1e-3, now - last_time)
        last_time = now
        now_ms = now * 1000.0

        # Pre-proceso
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        frame_eq = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        h, w = frame.shape[:2]
        cx_frame = w // 2

        rgb = cv2.cvtColor(frame_eq, cv2.COLOR_BGR2RGB)
        res = fd.process(rgb)
        have_face = bool(res.detections)

        if have_face:
            # ===== CARA DETECTADA =====
            is_centered = False
            returning_to_center = False
            
            # Cara principal
            det = max(res.detections, key=lambda d: d.score[0])
            box = det.location_data.relative_bounding_box
            x, y = int(box.xmin * w), int(box.ymin * h)
            bw, bh = int(box.width * w), int(box.height * h)
            cx = x + bw // 2

            # Dibujos (solo si no es headless)
            if not HEADLESS_MODE:
                cv2.rectangle(frame, (x, y), (x+bw, y+bh), (0,255,0), 2)
                cv2.circle(frame, (cx, y + bh//2), 4, (0,0,255), -1)

            # --- PID sobre error normalizado ---
            err_px = (cx - cx_frame) * INVERT_DIR
            if abs(err_px) <= DEADBAND_PX:
                err_px = 0.0
            err = (err_px / (w / 2.0))  # -1..1

            # Integrador con anti-windup
            sum_err += err * dt
            sum_err = clamp(sum_err, -I_CLAMP, I_CLAMP)

            # Derivada
            der = (err - last_err) / dt
            last_err = err

            # Salida PID en "grados"
            pid_out = KP*err + KI*sum_err + KD*der

            desired = clamp(last_ang + pid_out, ANGLE_MIN, ANGLE_MAX)

            # Suavizado exponencial
            smooth_ang = SMOOTH_ALPHA * desired + (1 - SMOOTH_ALPHA) * last_ang
            last_ang = smooth_ang

            # Actualizar servo
            servo_angle = int(round(smooth_ang))
            kit.servo[SERVO_PAN].angle = servo_angle

            # Estado de última vista
            last_seen_ms  = now_ms
            last_seen_dir = -1 if (cx < cx_frame) else +1

            # Debug en consola (solo cada 10 frames para no saturar)
            if frame_count % 10 == 0:
                print(f"👤 CARA DETECTADA | X:{cx}/{w} | Error:{err_px:+4.0f}px | PID:{pid_out:+5.1f} | Servo:{servo_angle}°")

            if not HEADLESS_MODE:
                cv2.putText(frame, f"PID:{pid_out:+.2f} ANG={int(smooth_ang)}",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        else:
            # ===== SIN CARA DETECTADA =====
            time_without_face = now_ms - last_seen_ms
            
            # Si pasaron más de 8 segundos sin detectar cara, volver a centro
            if time_without_face > RETURN_CENTER_AFTER_MS and not is_centered:
                if not returning_to_center:
                    print(f"\n⏺️  SIN DETECCIÓN POR {int(time_without_face/1000)}s - VOLVIENDO A CENTRO (90°)")
                    returning_to_center = True
                
                # Movimiento suave hacia el centro
                diff = CENTER_ANGLE - last_ang
                if abs(diff) > 1:  # Si no está en el centro
                    step = diff * 0.1  # Movimiento suave
                    last_ang += step
                    last_ang = clamp(last_ang, ANGLE_MIN, ANGLE_MAX)
                    
                    servo_angle = int(round(last_ang))
                    kit.servo[SERVO_PAN].angle = servo_angle
                    
                    if frame_count % 15 == 0:  # Mostrar cada medio segundo aprox
                        print(f"↩️  Centrando... | Actual:{servo_angle}° → Objetivo:90°")
                else:
                    # Ya está centrado
                    if not is_centered:
                        kit.servo[SERVO_PAN].angle = CENTER_ANGLE
                        last_ang = CENTER_ANGLE
                        is_centered = True
                        returning_to_center = False
                        last_seen_dir = 0  # Resetear dirección para que no busque más
                        print(f"✓ CENTRADO EN 90° - Esperando detección...\n")
            
            # ===== MODO BÚSQUEDA (si aún no pasaron 8 segundos) =====
            elif time_without_face > LOST_AFTER_MS and last_seen_dir != 0 and not is_centered:
                flip = -1 if SEARCH_FLIP == 1 else 1
                search_dir = last_seen_dir * flip
                last_ang += search_dir * SEARCH_RATE_DPS * dt
                last_ang = clamp(last_ang, ANGLE_MIN, ANGLE_MAX)
                
                servo_angle = int(round(last_ang))
                kit.servo[SERVO_PAN].angle = servo_angle
                
                if frame_count % 30 == 0:  # Mostrar cada segundo aprox
                    print(f"🔍 BUSCANDO... | Dir:{search_dir:+d} | Servo:{servo_angle}° | Sin cara: {int(time_without_face/1000)}s")
            
            else:
                # Pérdida corta: quedate quieto
                if frame_count % 60 == 0 and time_without_face > 50:
                    print(f"⏱️  ESPERANDO... | Sin cara por {int(time_without_face/1000)}s")

        # Visualización (solo si no es headless)
        if not HEADLESS_MODE:
            # Línea centro
            cv2.line(frame, (cx_frame, 0), (cx_frame, h), (255,0,0), 1)
            cv2.putText(frame, f"Servo: PIN {SERVO_PAN} = {int(last_ang)}°",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

            cv2.imshow("Seguimiento Facial - ServoKit + MediaPipe", frame)
        
        # Detectar teclas (funciona en ambos modos)
        if (cv2.waitKey(1) & 0xFF) in (ord('q'), 27):
            break

except KeyboardInterrupt:
    print("\n⚠️ Interrumpido por usuario")

finally:
    print("\n🔄 Centrando servo...")
    kit.servo[SERVO_PAN].angle = 90
    time.sleep(0.5)
    
    cap.release()
    if not HEADLESS_MODE:
        cv2.destroyAllWindows()
    
    print("✅ Finalizado")
