#!/usr/bin/env python3
# fusion_ojos_cabeza_parpadeo_pca9685.py
# Raspberry Pi + PCA9685 (Adafruit ServoKit) + MediaPipe
# Fusión de:
#  - CABEZA: tu "Sistema de seguimiento facial DUAL (X + Y)" (SIN CAMBIOS de lógica/ganancias)
#  - OJOS : tu "seguimiento_ojos_pca9685_completo.py" (SIN CAMBIOS de lógica/ganancias)
#  - PÁRPADOS: tu "parpadeo_humano_pca9685.py" integrado como tarea no bloqueante
# IMPORTANTE: TODOS los servos están NUMERADOS por el PCA9685.

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")

import cv2
import time
import math
import random
import mediapipe as mp
from adafruit_servokit import ServoKit
import signal
import sys

# =========================================================
# PCA9685 (UNO SOLO PARA TODO)
# =========================================================
kit = ServoKit(channels=16)  # servos numerados por PCA9685

# =========================================================
# PINES (PCA9685)
# =========================================================
# Cabeza
SERVO_ROLL_LEFT  = 12  # Hombro izquierdo
SERVO_ROLL_RIGHT = 15  # Hombro derecho
SERVO_PITCH      = 14  # Control de pitch (vertical)
SERVO_YAW        = 13  # Control de yaw (horizontal)

# Ojos
PIN_OJO_IZQ_H = 10
PIN_OJO_IZQ_V = 8
PIN_OJO_DER_H = 11
PIN_OJO_DER_V = 9

# Párpados
PIN_PARPADO_ABAJO  = 6
PIN_PARPADO_ARRIBA = 7

# =========================================================
# CONFIGURACIÓN SERVOS
# =========================================================
# Cabeza (exacto a tu script)
for pin in [SERVO_ROLL_LEFT, SERVO_ROLL_RIGHT, SERVO_PITCH]:
    kit.servo[pin].actuation_range = 200
    kit.servo[pin].set_pulse_width_range(500, 2500)
kit.servo[SERVO_YAW].actuation_range = 180
kit.servo[SERVO_YAW].set_pulse_width_range(500, 2500)

# Ojos (exacto a tu script)
for p in [PIN_OJO_IZQ_H, PIN_OJO_IZQ_V, PIN_OJO_DER_H, PIN_OJO_DER_V]:
    kit.servo[p].actuation_range = 180
    kit.servo[p].set_pulse_width_range(500, 2500)

# Párpados
for p in [PIN_PARPADO_ABAJO, PIN_PARPADO_ARRIBA]:
    kit.servo[p].actuation_range = 180
    kit.servo[p].set_pulse_width_range(500, 2500)

# =========================================================
# LÍMITES / CENTROS
# =========================================================
# Cabeza (exacto)
PITCH_MIN, PITCH_MAX = 60, 180
PITCH_CENTER = 120
ROLL_MIN, ROLL_MAX = 130, 180
ROLL_CENTER = 155
YAW_MIN, YAW_MAX = 90, 180
YAW_CENTER = 135

# Ojos (exacto)
H_MIN, H_MAX = 15, 165
H_CENTER = 90
V_MIN_IZQ, V_MAX_IZQ = 90, 120
V_CENTER_IZQ = 90
V_MIN_DER, V_MAX_DER = 70, 110
V_CENTER_DER = 90

# Párpados (tus valores)
ANGLE_MIN_PARPADO_ABAJO  = 60   # abierto
ANGLE_MAX_PARPADO_ABAJO  = 130  # cerrado
ANGLE_MIN_PARPADO_ARRIBA = 90 # abierto
ANGLE_MAX_PARPADO_ARRIBA = 140  # cerrado

# =========================================================
# CÁMARA
# =========================================================
CAM_INDEX = 0
HEADLESS_MODE = True
FRAME_W, FRAME_H = 640, 480

# =========================================================
# OFFSET de “MIRA” (cámara en la frente) — OJOS
# =========================================================
USE_OFFSET_FRAC = True
OFFSET_X_FRAC = 0.00
OFFSET_Y_FRAC = 0.00   # + abajo / - arriba
OFFSET_X_PX = 0
OFFSET_Y_PX = 80

# =========================================================
# CONTROL OJOS (PID + ABS + anti-overshoot) — EXACTO
# =========================================================
INVERT_H = -1
INVERT_V_IZQ = -1
INVERT_V_DER = +1

H_ABS_GAIN_DEG = 65.0

KP_H, KI_H, KD_H = 9.0, 0.0, 0.25
KP_V, KI_V, KD_V = 9.0, 0.0, 0.10

SMOOTH_ALPHA_H = 0.48
SMOOTH_ALPHA_V = 0.35

DEADBAND_PX_X_EYES = 20
DEADBAND_PX_Y_EYES = 22
I_CLAMP_EYES = 25.0

MAX_STEP_DEG_H = 3.0
MAX_STEP_DEG_V = 2.0

ERR_LP_ALPHA = 0.28

RETURN_CENTER_AFTER_MS_EYES = 6000

# =========================================================
# CONTROL CABEZA (PID clásico) — EXACTO
# =========================================================
INVERT_PITCH = -1
KP_PITCH = 20.0
KI_PITCH = 0.05
KD_PITCH = 1.0
SMOOTH_ALPHA_PITCH = 0.5
DEADBAND_PX_Y_HEAD = 15

INVERT_YAW = -1
KP_YAW = 7.0
KI_YAW = 0.01
KD_YAW = 0.0
SMOOTH_ALPHA_YAW = 0.25
DEADBAND_PX_X_HEAD = 40

I_CLAMP_HEAD = 30.0

# Búsqueda cabeza
LOST_AFTER_MS = 300
SEARCH_RATE_DPS_Y = 30.0
SEARCH_RATE_DPS_X = 25.0
RETURN_CENTER_AFTER_MS_HEAD = 8000

# =========================================================
# PARPADEO — parámetros (de tu script)
# =========================================================
SUPERIOR_ESCALA_CIERRE = 1.00
INFERIOR_ESCALA_CIERRE = 1.00

DUR_MS_CIERRE_SUP = (45, 70)
DUR_MS_APERT_SUP  = (90, 130)
DUR_MS_CIERRE_INF = (55, 85)
DUR_MS_APERT_INF  = (110, 160)

INTERVALO_PARPADEO_S = (2.0, 6.0)
PROB_DOBLE_PARPADEO  = 0.30
PAUSA_ENTRE_DOBLE_S  = (0.09, 0.18)
HOLD_CERRADO_S       = (0.02, 0.06)

# =========================================================
# HELPERS
# =========================================================
def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v

def mix(a, b, w=0.6):
    return w * a + (1.0 - w) * b

def limit_step(current, target, max_step):
    if target > current:
        return min(target, current + max_step)
    else:
        return max(target, current - max_step)

def near_gain(e, e0=0.25, min_gain=0.35):
    a = abs(e)
    if a >= e0: return 1.0
    return min_gain + (1.0 - min_gain) * (a / e0)

def ease_in_out(t: float) -> float:
    # 3t^2 - 2t^3  (suave)
    return 3 * t**2 - 2 * t**3

def eyelid_open_angles():
    sup_open = ANGLE_MIN_PARPADO_ARRIBA
    inf_open = ANGLE_MIN_PARPADO_ABAJO
    return sup_open, inf_open

def eyelid_closed_angles():
    sup_open, inf_open = eyelid_open_angles()
    sup_close = sup_open + SUPERIOR_ESCALA_CIERRE * (ANGLE_MAX_PARPADO_ARRIBA - ANGLE_MIN_PARPADO_ARRIBA)
    inf_close = inf_open + INFERIOR_ESCALA_CIERRE * (ANGLE_MAX_PARPADO_ABAJO  - ANGLE_MIN_PARPADO_ABAJO)
    return sup_close, inf_close

def center_eyes():
    kit.servo[PIN_OJO_IZQ_H].angle = H_CENTER
    kit.servo[PIN_OJO_DER_H].angle = H_CENTER
    kit.servo[PIN_OJO_IZQ_V].angle = V_CENTER_IZQ
    kit.servo[PIN_OJO_DER_V].angle = V_CENTER_DER
    time.sleep(0.2)

def center_head():
    kit.servo[SERVO_YAW].angle   = YAW_CENTER
    kit.servo[SERVO_PITCH].angle = PITCH_CENTER
    kit.servo[SERVO_ROLL_LEFT].angle  = ROLL_CENTER
    kit.servo[SERVO_ROLL_RIGHT].angle = ROLL_CENTER
    time.sleep(0.2)

def center_eyelids_open():
    sup_open, inf_open = eyelid_open_angles()
    kit.servo[PIN_PARPADO_ARRIBA].angle = sup_open
    kit.servo[PIN_PARPADO_ABAJO].angle  = inf_open

def center_all():
    center_eyes()
    center_head()
    center_eyelids_open()

# =========================================================
# ESTADOS OJOS
# =========================================================
last_h_izq = float(H_CENTER); last_v_izq = float(V_CENTER_IZQ)
last_h_der = float(H_CENTER); last_v_der = float(V_CENTER_DER)
last_err_h_e = 0.0; sum_err_h_e = 0.0
last_err_v_e = 0.0; sum_err_v_e = 0.0
err_x_f = 0.0; err_y_f = 0.0
last_seen_ms_eyes = time.time() * 1000.0
is_centered_eyes = False

# =========================================================
# ESTADOS CABEZA
# =========================================================
last_yaw = float(YAW_CENTER)
last_pitch = float(PITCH_CENTER)
last_err_yaw = 0.0; sum_err_yaw = 0.0
last_err_pitch = 0.0; sum_err_pitch = 0.0
last_seen_ms_head = time.time() * 1000.0
last_seen_dir_x = 0
last_seen_dir_y = 0
is_centered_head = False
returning_to_center_head = False

# =========================================================
# ESTADO PARPADEO (no bloqueante)
# =========================================================
blink_phase = 'idle'    # 'idle' | 'closing' | 'hold' | 'opening'
phase_t0 = 0.0
next_blink_time = time.time() + random.uniform(*INTERVALO_PARPADEO_S)
blink_queue = 0         # 0/1/2 (para doble parpadeo)

# Duraciones por párpado (se sortean al iniciar cada blink)
dur_close_sup = dur_open_sup = 0.0
dur_close_inf = dur_open_inf = 0.0
hold_dur = 0.0

def schedule_next_blink(now):
    global next_blink_time, blink_queue
    if random.random() < PROB_DOBLE_PARPADEO:
        blink_queue = 2
    else:
        blink_queue = 1
    next_blink_time = now  # disparar ya

def start_blink(now):
    global blink_phase, phase_t0
    global dur_close_sup, dur_open_sup, dur_close_inf, dur_open_inf, hold_dur
    blink_phase = 'closing'
    phase_t0 = now
    dur_close_sup = random.randint(*DUR_MS_CIERRE_SUP) / 1000.0
    dur_open_sup  = random.randint(*DUR_MS_APERT_SUP)  / 1000.0
    dur_close_inf = random.randint(*DUR_MS_CIERRE_INF) / 1000.0
    dur_open_inf  = random.randint(*DUR_MS_APERT_INF)  / 1000.0
    hold_dur      = random.uniform(*HOLD_CERRADO_S)

def update_blink(now):
    """Actualiza los párpados 1 frame. Devuelve None."""
    global blink_phase, phase_t0, next_blink_time, blink_queue
    if blink_phase == 'idle':
        if now >= next_blink_time:
            if blink_queue == 0:
                schedule_next_blink(now)
            if blink_queue > 0:
                start_blink(now)
        return

    sup_open, inf_open = eyelid_open_angles()
    sup_close, inf_close = eyelid_closed_angles()

    if blink_phase == 'closing':
        # Progresos independientes (sup/inf) con ease-in/out
        k_sup = clamp((now - phase_t0) / max(1e-3, dur_close_sup), 0.0, 1.0)
        k_inf = clamp((now - phase_t0) / max(1e-3, dur_close_inf), 0.0, 1.0)
        e_sup = ease_in_out(k_sup)
        e_inf = ease_in_out(k_inf)
        ang_sup = sup_open + (sup_close - sup_open) * e_sup
        ang_inf = inf_open + (inf_close - inf_open) * e_inf
        kit.servo[PIN_PARPADO_ARRIBA].angle = ang_sup
        kit.servo[PIN_PARPADO_ABAJO].angle  = ang_inf
        if k_sup >= 1.0 and k_inf >= 1.0:
            blink_phase = 'hold'
            phase_t0 = now
        return

    if blink_phase == 'hold':
        if (now - phase_t0) >= hold_dur:
            blink_phase = 'opening'
            phase_t0 = now
        return

    if blink_phase == 'opening':
        k_sup = clamp((now - phase_t0) / max(1e-3, dur_open_sup), 0.0, 1.0)
        k_inf = clamp((now - phase_t0) / max(1e-3, dur_open_inf), 0.0, 1.0)
        e_sup = ease_in_out(k_sup)
        e_inf = ease_in_out(k_inf)
        ang_sup = sup_close + (sup_open - sup_close) * e_sup
        ang_inf = inf_close + (inf_open - inf_close) * e_inf
        kit.servo[PIN_PARPADO_ARRIBA].angle = ang_sup
        kit.servo[PIN_PARPADO_ABAJO].angle  = ang_inf
        if k_sup >= 1.0 and k_inf >= 1.0:
            # Blink terminado
            blink_queue -= 1
            if blink_queue > 0:
                # Pausa corta entre doble parpadeo
                next_blink_time = now + random.uniform(*PAUSA_ENTRE_DOBLE_S)
            else:
                next_blink_time = now + random.uniform(*INTERVALO_PARPADEO_S)
            blink_phase = 'idle'
        return

# =========================================================
# SEÑALES / SALIDA
# =========================================================
def cleanup_and_exit(cap):
    print("🔄 Volviendo servos a posición de centro...")
    try:
        center_all()
        # Igual que tu salida de cabeza
        kit.servo[SERVO_PITCH].angle = 135
        time.sleep(0.25)
    except Exception:
        pass
    cap.release()
    if not HEADLESS_MODE:
        cv2.destroyAllWindows()
    print("✅ Sistema finalizado")
    sys.exit(0)

def signal_handler(sig, frame):
    print("\n\n⚠️ Ctrl+C detectado - Deteniendo...")
    cleanup_and_exit(cap)

signal.signal(signal.SIGINT, signal_handler)

# =========================================================
# INICIO / CÁMARA / MEDIAPIPE
# =========================================================
print("=" * 70)
print("🤖 FUSIÓN OJOS + CABEZA + PÁRPADOS (PCA9685) — Ojos primero, Cabeza después; parpadeo no bloqueante")
print("=" * 70)

print("\n🎥 Iniciando cámara...")
cap = cv2.VideoCapture(CAM_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
if not cap.isOpened():
    print("❌ No se pudo abrir la cámara")
    sys.exit(1)

print("🧠 Inicializando MediaPipe Face Detection...")
mp_fd = mp.solutions.face_detection
fd = mp_fd.FaceDetection(model_selection=0, min_detection_confidence=0.45)

print("📍 Posicionando servos en centro...")
center_all()

print("\n📊 Configuración CABEZA: YAW pin", SERVO_YAW, "| PITCH pin", SERVO_PITCH, "| ROLL pins", SERVO_ROLL_LEFT, "/", SERVO_ROLL_RIGHT)
print("📊 Configuración OJOS: IZQ(H=10,V=8)  DER(H=11,V=9)")
print("📊 PÁRPADOS: ARRIBA(pin 7) / ABAJO(pin 6)")

print("\n🔄 Procesando frames...\n")

# =========================================================
# LOOP PRINCIPAL (OJOS → CABEZA → PÁRPADOS)
# =========================================================
frame_count = 0
fps_start = time.time()
last_time = time.time()

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

        # Target offset (ojos)
        if USE_OFFSET_FRAC:
            tx = cx_frame + int(OFFSET_X_FRAC * w)
            ty = cy_frame + int(OFFSET_Y_FRAC * h)   # +abajo
        else:
            tx = cx_frame + int(OFFSET_X_PX)
            ty = cy_frame + int(OFFSET_Y_PX)

        # Pre-proceso
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        frame_eq = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        rgb = cv2.cvtColor(frame_eq, cv2.COLOR_BGR2RGB)
        results = fd.process(rgb)
        have_face = bool(results.detections)

        if have_face:
            # Detección única
            det = max(results.detections, key=lambda d: d.score[0])
            bbox = det.location_data.relative_bounding_box
            x = int(bbox.xmin * w); y = int(bbox.ymin * h)
            bw = int(bbox.width * w); bh = int(bbox.height * h)
            face_cx = x + bw // 2; face_cy = y + bh // 2

            # ===================== 1) OJOS (PRIMERO) =====================
            err_px_x_e = (face_cx - tx)
            if abs(err_px_x_e) <= DEADBAND_PX_X_EYES: err_px_x_e = 0.0
            err_x_e = err_px_x_e / (w / 2.0)

            err_px_y_e = (face_cy - ty)
            if abs(err_px_y_e) <= DEADBAND_PX_Y_EYES: err_px_y_e = 0.0
            err_y_e = err_px_y_e / (h / 2.0)

            # filtro low-pass
            err_x_f = (1 - ERR_LP_ALPHA) * err_x_f + ERR_LP_ALPHA * err_x_e
            err_y_f = (1 - ERR_LP_ALPHA) * err_y_f + ERR_LP_ALPHA * err_y_e

            # ganancia adaptativa
            g_h = near_gain(err_x_f, e0=0.25, min_gain=0.35)
            g_v = near_gain(err_y_f, e0=0.25, min_gain=0.40)

            # PID ojos
            sum_err_h_e = clamp(sum_err_h_e + err_x_f * dt, -I_CLAMP_EYES, I_CLAMP_EYES)
            der_h_e = (err_x_f - last_err_h_e) / dt; last_err_h_e = err_x_f
            pid_h = (KP_H * err_x_f + KI_H * sum_err_h_e + KD_H * der_h_e) * g_h

            sum_err_v_e = clamp(sum_err_v_e + err_y_f * dt, -I_CLAMP_EYES, I_CLAMP_EYES)
            der_v_e = (err_y_f - last_err_v_e) / dt; last_err_v_e = err_y_f
            pid_v = (KP_V * err_y_f + KI_V * sum_err_v_e + KD_V * der_v_e) * g_v

            # mix ABS + INC
            abs_h = clamp(H_CENTER + INVERT_H * (H_ABS_GAIN_DEG * g_h) * err_x_f, H_MIN, H_MAX)
            inc_h_izq = clamp(last_h_izq + INVERT_H * pid_h, H_MIN, H_MAX)
            inc_h_der = clamp(last_h_der + INVERT_H * pid_h, H_MIN, H_MAX)
            w_abs = 0.6 * near_gain(err_x_f, e0=0.35, min_gain=0.25)
            desired_h_izq = mix(abs_h, inc_h_izq, w=w_abs)
            desired_h_der = mix(abs_h, inc_h_der, w=w_abs)

            desired_v_izq = clamp(last_v_izq + INVERT_V_IZQ * pid_v, V_MIN_IZQ, V_MAX_IZQ)
            desired_v_der = clamp(last_v_der + INVERT_V_DER * pid_v, V_MIN_DER, V_MAX_DER)

            # suavizado + slew-rate
            smooth_h_izq = SMOOTH_ALPHA_H * desired_h_izq + (1 - SMOOTH_ALPHA_H) * last_h_izq
            smooth_h_der = SMOOTH_ALPHA_H * desired_h_der + (1 - SMOOTH_ALPHA_H) * last_h_der
            smooth_v_izq = SMOOTH_ALPHA_V * desired_v_izq + (1 - SMOOTH_ALPHA_V) * last_v_izq
            smooth_v_der = SMOOTH_ALPHA_V * desired_v_der + (1 - SMOOTH_ALPHA_V) * last_v_der

            smooth_h_izq = limit_step(last_h_izq, smooth_h_izq, MAX_STEP_DEG_H)
            smooth_h_der = limit_step(last_h_der, smooth_h_der, MAX_STEP_DEG_H)
            smooth_v_izq = limit_step(last_v_izq, smooth_v_izq, MAX_STEP_DEG_V)
            smooth_v_der = limit_step(last_v_der, smooth_v_der, MAX_STEP_DEG_V)

            # ESCRITURA OJOS
            kit.servo[PIN_OJO_IZQ_H].angle = int(round(smooth_h_izq))
            kit.servo[PIN_OJO_DER_H].angle = int(round(smooth_h_der))
            kit.servo[PIN_OJO_IZQ_V].angle = int(round(smooth_v_izq))
            kit.servo[PIN_OJO_DER_V].angle = int(round(smooth_v_der))

            # actualizar estados ojos
            last_h_izq, last_h_der = smooth_h_izq, smooth_h_der
            last_v_izq, last_v_der = smooth_v_izq, smooth_v_der
            last_seen_ms_eyes = now_ms

            # ===================== 2) CABEZA (DESPUÉS) =====================
            is_centered_head = False
            returning_to_center_head = False

            # YAW
            err_px_x_head = (face_cx - cx_frame) * INVERT_YAW
            if abs(err_px_x_head) <= DEADBAND_PX_X_HEAD:
                err_px_x_head = 0.0
            err_x_head = err_px_x_head / (w / 2.0)

            sum_err_yaw += err_x_head * dt
            sum_err_yaw = clamp(sum_err_yaw, -I_CLAMP_HEAD, I_CLAMP_HEAD)
            der_x = (err_x_head - last_err_yaw) / dt
            last_err_yaw = err_x_head
            pid_out_yaw = KP_YAW * err_x_head + KI_YAW * sum_err_yaw + KD_YAW * der_x
            desired_yaw = clamp(last_yaw + pid_out_yaw, YAW_MIN, YAW_MAX)
            smooth_yaw = SMOOTH_ALPHA_YAW * desired_yaw + (1 - SMOOTH_ALPHA_YAW) * last_yaw
            last_yaw = smooth_yaw
            kit.servo[SERVO_YAW].angle = int(round(smooth_yaw))

            # PITCH
            err_px_y_head = (face_cy - cy_frame) * INVERT_PITCH
            if abs(err_px_y_head) <= DEADBAND_PX_Y_HEAD:
                err_px_y_head = 0.0
            err_y_head = err_px_y_head / (h / 2.0)

            sum_err_pitch += err_y_head * dt
            sum_err_pitch = clamp(sum_err_pitch, -I_CLAMP_HEAD, I_CLAMP_HEAD)
            der_y = (err_y_head - last_err_pitch) / dt
            last_err_pitch = err_y_head
            pid_out_pitch = KP_PITCH * err_y_head + KI_PITCH * sum_err_pitch + KD_PITCH * der_y
            desired_pitch = clamp(last_pitch + pid_out_pitch, PITCH_MIN, PITCH_MAX)
            smooth_pitch = SMOOTH_ALPHA_PITCH * desired_pitch + (1 - SMOOTH_ALPHA_PITCH) * last_pitch
            last_pitch = smooth_pitch
            kit.servo[SERVO_PITCH].angle = int(round(smooth_pitch))

            # ROLL inverso al pitch
            roll_angle = int(ROLL_MAX - ((int(round(smooth_pitch)) - PITCH_MIN) / (PITCH_MAX - PITCH_MIN)) * (ROLL_MAX - ROLL_MIN))
            kit.servo[SERVO_ROLL_LEFT].angle = roll_angle
            kit.servo[SERVO_ROLL_RIGHT].angle = roll_angle

            last_seen_ms_head = now_ms
            last_seen_dir_x = -1 if (face_cx < cx_frame) else +1
            last_seen_dir_y = -1 if (face_cy < cy_frame) else +1

        else:
            # ================= OJOS: retorno a centro (PRIMERO) =================
            time_without_e = now_ms - last_seen_ms_eyes
            if time_without_e > RETURN_CENTER_AFTER_MS_EYES and not is_centered_eyes:
                step = 0.15
                last_h_izq = clamp(last_h_izq + (H_CENTER - last_h_izq) * step, H_MIN, H_MAX)
                last_h_der = clamp(last_h_der + (H_CENTER - last_h_der) * step, H_MIN, H_MAX)
                last_v_izq = clamp(last_v_izq + (V_CENTER_IZQ - last_v_izq) * step, V_MIN_IZQ, V_MAX_IZQ)
                last_v_der = clamp(last_v_der + (V_CENTER_DER - last_v_der) * step, V_MIN_DER, V_MAX_DER)

                kit.servo[PIN_OJO_IZQ_H].angle = int(round(last_h_izq))
                kit.servo[PIN_OJO_DER_H].angle = int(round(last_h_der))
                kit.servo[PIN_OJO_IZQ_V].angle = int(round(last_v_izq))
                kit.servo[PIN_OJO_DER_V].angle = int(round(last_v_der))

                if (abs(H_CENTER - last_h_izq) < 1 and abs(H_CENTER - last_h_der) < 1 and
                    abs(V_CENTER_IZQ - last_v_izq) < 1 and abs(V_CENTER_DER - last_v_der) < 1):
                    center_eyes()
                    last_h_izq = float(H_CENTER); last_h_der = float(H_CENTER)
                    last_v_izq = float(V_CENTER_IZQ); last_v_der = float(V_CENTER_DER)
                    is_centered_eyes = True

            # ================= CABEZA: tu lógica “sin rostro” (DESPUÉS) =================
            time_without_head = now_ms - last_seen_ms_head

            if time_without_head > RETURN_CENTER_AFTER_MS_HEAD and not is_centered_head:
                if not returning_to_center_head:
                    print(f"\n⏺️  SIN ROSTRO POR {int(time_without_head/1000)}s - VOLVIENDO A CENTRO")
                    returning_to_center_head = True

                diff_yaw = YAW_CENTER - last_yaw
                diff_pitch = PITCH_CENTER - last_pitch

                if abs(diff_yaw) > 1 or abs(diff_pitch) > 1:
                    step_yaw = diff_yaw * 0.15
                    step_pitch = diff_pitch * 0.15

                    last_yaw += step_yaw
                    last_pitch += step_pitch

                    last_yaw = clamp(last_yaw, YAW_MIN, YAW_MAX)
                    last_pitch = clamp(last_pitch, PITCH_MIN, PITCH_MAX)

                    kit.servo[SERVO_YAW].angle = int(round(last_yaw))
                    kit.servo[SERVO_PITCH].angle = int(round(last_pitch))

                    roll_angle = int(ROLL_MAX - ((int(round(last_pitch)) - PITCH_MIN) / (PITCH_MAX - PITCH_MIN)) * (ROLL_MAX - ROLL_MIN))
                    kit.servo[SERVO_ROLL_LEFT].angle = roll_angle
                    kit.servo[SERVO_ROLL_RIGHT].angle = roll_angle
                else:
                    if not is_centered_head:
                        center_head()
                        last_yaw = float(YAW_CENTER)
                        last_pitch = float(PITCH_CENTER)
                        is_centered_head = True
                        returning_to_center_head = False
                        last_seen_dir_x = 0
                        last_seen_dir_y = 0

            elif time_without_head > LOST_AFTER_MS and (last_seen_dir_x != 0 or last_seen_dir_y != 0) and not is_centered_head:
                if last_seen_dir_x != 0:
                    search_dir_x = last_seen_dir_x * INVERT_YAW
                    last_yaw += search_dir_x * SEARCH_RATE_DPS_X * dt
                    last_yaw = clamp(last_yaw, YAW_MIN, YAW_MAX)
                    kit.servo[SERVO_YAW].angle = int(round(last_yaw))
                if last_seen_dir_y != 0:
                    search_dir_y = last_seen_dir_y * INVERT_PITCH
                    last_pitch += search_dir_y * SEARCH_RATE_DPS_Y * dt
                    last_pitch = clamp(last_pitch, PITCH_MIN, PITCH_MAX)
                    kit.servo[SERVO_PITCH].angle = int(round(last_pitch))
                    roll_angle = int(ROLL_MAX - ((int(round(last_pitch)) - PITCH_MIN) / (PITCH_MAX - PITCH_MIN)) * (ROLL_MAX - ROLL_MIN))
                    kit.servo[SERVO_ROLL_LEFT].angle = roll_angle
                    kit.servo[SERVO_ROLL_RIGHT].angle = roll_angle

        # ===================== 3) PÁRPADOS (NO BLOQUEANTE) =====================
        update_blink(now)

        # UI opcional
        if not HEADLESS_MODE:
            cv2.line(frame, (cx_frame, 0), (cx_frame, h), (255, 0, 0), 1)
            cv2.line(frame, (0, cy_frame), (w, cy_frame), (255, 0, 0), 1)
            cv2.circle(frame, (tx, ty), 6, (255, 0, 255), -1)
            cv2.imshow("Fusión Ojos + Cabeza + Párpados", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                cleanup_and_exit(cap)

except KeyboardInterrupt:
    print("\n⚠️ Interrumpido")

finally:
    cleanup_and_exit(cap)
