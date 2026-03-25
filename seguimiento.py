#!/usr/bin/env python3
"""
seguimiento.py — Seguimiento facial con ojos · InMoov
------------------------------------------------------
Una sola cámara mueve ambos ojos de forma independiente.
Solo actúa sobre pines 0, 1, 7 y 9 (ojos, sin párpados ni cuello).

Mapping (según calibración del proyecto):
  Pin 0 → Ojo Izq Horizontal  40–120  centro 80  (40=Der, 120=Izq)
  Pin 1 → Ojo Izq Vertical    85–105  centro 95  (105=Arriba)
  Pin 9 → Ojo Der Horizontal  40–120  centro 80  (40=Der, 120=Izq)
  Pin 7 → Ojo Der Vertical    70–90   centro 80  (70=Arriba)

Nota: el eje vertical está INVERTIDO entre ambos ojos
      (IzqV sube con ángulo mayor; DerV sube con ángulo menor).
"""

import cv2
import time
import urllib.request
import pathlib
import json
import os
import random
from adafruit_servokit import ServoKit
import signal
import sys

# ──────────────────────────────────────────────
# HARDWARE
# ──────────────────────────────────────────────
kit = ServoKit(channels=16)

# Pines
PIN_LH = 0   # Ojo Izquierdo Horizontal
PIN_LV = 1   # Ojo Izquierdo Vertical
PIN_RH = 9   # Ojo Derecho Horizontal
PIN_RV = 7   # Ojo Derecho Vertical

PIN_PARPADO_INF = 3   # Párpado inferior (40=abierto, 85=cerrado)
PIN_PARPADO_SUP = 5   # Párpado superior (40=abierto, 85=cerrado)
PIN_PITCH       = 4   # Cuello Yaw (50=Mirar Izq, 150=Mirar Der)
PARPADO_ABIERTO = 40
PARPADO_CERRADO = 95

# Límites calibrados: dict con lo(mín), hi(máx), mid(centro)
LH = dict(lo=40,  hi=130, mid=90)  # Izq Horizontal (80=Der, 140=Izq)
LV = dict(lo=80,  hi=100, mid=90)  # Izq Vertical   (105=arriba, 85=abajo)
RH = dict(lo=40,  hi=130, mid=90)  # Der Horizontal (80=Der, 140=Izq)
RV = dict(lo=80,  hi=100,  mid=90) # Der Vertical   (70=arriba, 90=abajo) ← INVERTIDO
PITCH = dict(lo=50, hi=150, mid=100) # Cuello Yaw (50=Mirar Izq, 150=Mirar Der)

# ──────────────────────────────────────────────
# PARÁMETROS
# ──────────────────────────────────────────────
CAM_INDEX   = 0
HEADLESS    = False     # False → muestra ventana de video

# ── Configuración persistente ──
CONFIG_FILE = pathlib.Path(__file__).parent / "config.json"
cfg = {
    "KP": 0.80,
    "KI": 0.01,
    "SMOOTH": 0.15,
    "OFFSET_X": 0       # Compensa si físicamente los ojos no miran al centro
}

if CONFIG_FILE.exists():
    try:
        with open(CONFIG_FILE, "r") as f:
            cfg.update(json.load(f))
    except: pass

KP          = cfg["KP"]
KI          = cfg["KI"]
SMOOTH      = cfg["SMOOTH"]
OFFSET_X    = cfg["OFFSET_X"]

DEADBAND_X  = 30        # píxeles de zona muerta horizontal
DEADBAND_Y  = 25        # píxeles de zona muerta vertical
I_CLAMP     = 20.0      # Límite del integrador

LOST_MS     = 400       # ms sin cara → iniciar búsqueda
SEARCH_DPS  = 18.0      # velocidad búsqueda (°/s)
RETURN_MS   = 4000      # ms sin cara → volver al centro

# ──────────────────────────────────────────────
# UTILIDADES
# ──────────────────────────────────────────────
def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def init_servos():
    for pin in (PIN_LH, PIN_LV, PIN_RH, PIN_RV, PIN_PARPADO_INF, PIN_PARPADO_SUP, PIN_PITCH):
        kit.servo[pin].actuation_range = 180
        # Margen de seguridad para InMoov
        kit.servo[pin].set_pulse_width_range(600, 2350)
    # Párpados abiertos permanentemente (necesario para las cámaras)
    kit.servo[PIN_PARPADO_INF].angle = PARPADO_ABIERTO
    kit.servo[PIN_PARPADO_SUP].angle = PARPADO_ABIERTO
    center_all()
    time.sleep(0.5)


def center_all():
    kit.servo[PIN_LH].angle = LH["mid"]
    kit.servo[PIN_LV].angle = LV["mid"]
    kit.servo[PIN_RH].angle = RH["mid"]
    kit.servo[PIN_RV].angle = RV["mid"]
    kit.servo[PIN_PITCH].angle = PITCH["mid"]


def apply_eyes(lh, lv, rh, rv):
    kit.servo[PIN_LH].angle = int(round(clamp(lh, LH["lo"], LH["hi"])))
    kit.servo[PIN_LV].angle = int(round(clamp(lv, LV["lo"], LV["hi"])))
    kit.servo[PIN_RH].angle = int(round(clamp(rh, RH["lo"], RH["hi"])))
    kit.servo[PIN_RV].angle = int(round(clamp(rv, RV["lo"], RV["hi"])))


def cleanup():
    print("\n🔄 Centrando y saliendo...")
    try:
        center_all()
        time.sleep(0.3)
    except Exception:
        pass
    cap.release()
    sys.exit(0)


signal.signal(signal.SIGINT, lambda s, f: cleanup())

# ──────────────────────────────────────────────
# INICIO
# ──────────────────────────────────────────────
print("🤖 InMoov — Seguimiento ocular")

cap = cv2.VideoCapture(CAM_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
if not cap.isOpened():
    print("❌ No se pudo abrir la cámara")
    sys.exit(1)

_MODEL = pathlib.Path(__file__).parent / "models" / "yunet.onnx"
_URL   = ("https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
          "models/face_detection_yunet/face_detection_yunet_2023mar.onnx")

if not _MODEL.exists() or _MODEL.stat().st_size < 100000:
    print("⬇️  Descargando modelo YuNet (puede tardar unos segundos)...")
    _MODEL.parent.mkdir(exist_ok=True)
    if _MODEL.exists():
        _MODEL.unlink()  # Borra el archivo corrupto
    urllib.request.urlretrieve(_URL, _MODEL)
    print("✅ Modelo guardado en models/yunet.onnx")

fd = cv2.FaceDetectorYN.create(str(_MODEL), "", (640, 480),
                               score_threshold=0.6, nms_threshold=0.3)

init_servos()

print(f"✅ Listo | cam /dev/video{CAM_INDEX}")
print(f"   Izq → H(pin {PIN_LH}): {LH['lo']}–{LH['hi']} ctr {LH['mid']} | "
      f"V(pin {PIN_LV}): {LV['lo']}–{LV['hi']} ctr {LV['mid']}")
print(f"   Der → H(pin {PIN_RH}): {RH['lo']}–{RH['hi']} ctr {RH['mid']} | "
      f"V(pin {PIN_RV}): {RV['lo']}–{RV['hi']} ctr {RV['mid']}")

# ──────────────────────────────────────────────
# ESTADO
# ──────────────────────────────────────────────
lh = float(LH["mid"])
lv = float(LV["mid"])
rh = float(RH["mid"])
rv = float(RV["mid"])
pitch_ang = float(PITCH["mid"])

sum_ex = sum_ey = 0.0

last_time   = time.time()
last_seen   = time.time() * 1000.0
dir_x = dir_y = 0
centered    = False
returning   = False
frames      = 0
fps_t       = time.time()
tracked_face = None # Memoria de la cara actual
face_first_seen_time = 0.0  # Temporizador de retardo para el cuello

# Estado del parpadeo asíncrono
blink_phase     = "IDLE"
next_blink_time = time.time() + random.uniform(2.0, 8.0)
blink_state_end = 0
blinks_to_do    = 0

if not HEADLESS:
    cv2.namedWindow("Seguimiento ocular")
    # OpenCV trackbars solo admiten enteros:
    # KP(0-200) -> 0.0 a 2.0
    # KI(0-100) -> 0.0 a 0.1
    # SMOOTH(0-100) -> 0.0 a 1.0
    # OFFSET_X(0-80) -> -40 a +40 grados (para compensar el servo físico)
    cv2.createTrackbar("KP (0-2.0)", "Seguimiento ocular", int(KP * 100), 200, lambda x: None)
    cv2.createTrackbar("KI (0-0.1)", "Seguimiento ocular", int(KI * 1000), 100, lambda x: None)
    cv2.createTrackbar("SMOOTH /100", "Seguimiento ocular", int(SMOOTH * 100), 100, lambda x: None)
    cv2.createTrackbar("OFFSET_X+-40", "Seguimiento ocular", int(OFFSET_X + 40), 80, lambda x: None)

# ── LOOP PRINCIPAL ──
try:
    while True:
        if not HEADLESS:
            new_KP = cv2.getTrackbarPos("KP (0-2.0)", "Seguimiento ocular") / 100.0
            new_KI = cv2.getTrackbarPos("KI (0-0.1)", "Seguimiento ocular") / 1000.0
            new_SM = cv2.getTrackbarPos("SMOOTH /100", "Seguimiento ocular") / 100.0
            new_OX = cv2.getTrackbarPos("OFFSET_X+-40", "Seguimiento ocular") - 40
            
            # Autoguardado si algo cambió
            if new_KP != KP or new_KI != KI or new_SM != SMOOTH or new_OX != OFFSET_X:
                KP, KI, SMOOTH, OFFSET_X = new_KP, new_KI, new_SM, new_OX
                with open(CONFIG_FILE, "w") as f:
                    json.dump({"KP": KP, "KI": KI, "SMOOTH": SMOOTH, "OFFSET_X": OFFSET_X}, f, indent=2)

        ok, frame = cap.read()
        if not ok:
            break

        # Compensar rotación física (cámara rotada a la derecha → corregir a la izquierda)
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        frames += 1
        if frames % 30 == 0:
            fps = 30.0 / max(time.time() - fps_t, 1e-6)
            fps_t = time.time()
            print(f"📹 FPS: {fps:.1f}")

        now    = time.time()
        dt     = max(1e-3, now - last_time)
        last_time = now
        now_ms = now * 1000.0
        
        # ── LÓGICA DE PARPADEO (ASÍNCRONO O NO BLOQUEANTE) ──
        if blink_phase == "IDLE":
            if now > next_blink_time:
                blinks_to_do = random.randint(1, 2)
                kit.servo[PIN_PARPADO_INF].angle = PARPADO_CERRADO
                kit.servo[PIN_PARPADO_SUP].angle = PARPADO_CERRADO
                blink_phase = "CLOSED"
                blink_state_end = now + 0.15  # tiempo de ojos cerrados (150ms)
        
        elif blink_phase == "CLOSED":
            if now > blink_state_end:
                kit.servo[PIN_PARPADO_INF].angle = PARPADO_ABIERTO
                kit.servo[PIN_PARPADO_SUP].angle = PARPADO_ABIERTO
                blinks_to_do -= 1
                if blinks_to_do > 0:
                    blink_phase = "OPEN_WAIT"
                    blink_state_end = now + 0.30  # 300ms entre parpadeos dobles
                else:
                    blink_phase = "IDLE"
                    next_blink_time = now + random.uniform(2.0, 10.0)
        
        elif blink_phase == "OPEN_WAIT":
            if now > blink_state_end:
                kit.servo[PIN_PARPADO_INF].angle = PARPADO_CERRADO
                kit.servo[PIN_PARPADO_SUP].angle = PARPADO_CERRADO
                blink_phase = "CLOSED"
                blink_state_end = now + 0.15
        # ────────────────────────────────────────────────────

        # Dimensiones actualizadas post-rotación
        h, w   = frame.shape[:2]
        cx, cy = w // 2, h // 2
        
        # YuNet necesita saber en tiempo real el tamaño para encontrar las caras
        fd.setInputSize((w, h))

        _, faces = fd.detect(frame)

        if faces is not None:
            centered = returning = False

            # LÓGICA DE FIJACIÓN DE OBJETIVO (Target Lock)
            # Evita saltar de una cara a otra eligiendo siempre la más cercana a la última pos conocida.
            best = None
            if tracked_face is not None:
                min_dist = float('inf')
                for f in faces:
                    fx_t = f[0] + f[2] // 2
                    fy_t = f[1] + f[3] // 2
                    dist = (fx_t - tracked_face[0])**2 + (fy_t - tracked_face[1])**2
                    if dist < min_dist:
                        min_dist = dist
                        best = f
            else:
                # Si recién entra, pesca el rostro más prominente/cercano (según score)
                best = faces[faces[:, 14].argmax()]
                if tracked_face is None:
                    face_first_seen_time = now  # Iniciar cronómetro de seguimiento continuo

            bx, by    = int(best[0]), int(best[1])
            bw, bh    = int(best[2]), int(best[3])
            fx, fy    = bx + bw // 2, by + bh // 2
            
            # Anotar esta cara para seguir persiguiendo a ésta la próxima vez
            tracked_face = (fx, fy)

            if not HEADLESS:
                cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), (0, 255, 0), 2)
                cv2.circle(frame, (fx, fy), 5, (0, 0, 255), -1)

            # Error en píxeles → zona muerta → normalización [-1, +1]
            epx = cx - fx
            epy = cy - fy
            if abs(epx) <= DEADBAND_X: epx = 0.0
            if abs(epy) <= DEADBAND_Y: epy = 0.0
            ex = epx / (w / 2.0)
            ey = epy / (h / 2.0)

            # PID (P + I)
            sum_ex = clamp(sum_ex + ex * dt, -I_CLAMP, I_CLAMP)
            sum_ey = clamp(sum_ey + ey * dt, -I_CLAMP, I_CLAMP)
            pid_x  = KP * ex + KI * sum_ex
            pid_y  = KP * ey + KI * sum_ey

            # ── Mapeo a ángulos ───────────────────────────────────────
            # Horizontal: ambos ojos igual (40=der, 120=izq)
            # OFFSET_X corre todo el bloque de límites físicamente de ser necesario
            half_h = (LH["hi"] - LH["lo"]) / 2.0   # 40°
            t_lh   = LH["mid"] + OFFSET_X + pid_x * half_h
            t_rh   = RH["mid"] + OFFSET_X + pid_x * half_h

            # Vertical: ejes OPUESTOS entre ojos
            #   ey>0 → cara arriba → ojos suben
            #   Izq: sube con ángulo mayor  → +pid_y
            #   Der: sube con ángulo menor  → -pid_y
            half_lv = (LV["hi"] - LV["lo"]) / 2.0  # 10°
            half_rv = (RV["hi"] - RV["lo"]) / 2.0  # 10°
            t_lv    = LV["mid"] + pid_y * half_lv
            t_rv    = RV["mid"] - pid_y * half_rv

            # Suavizado EMA
            lh = SMOOTH * lh + (1 - SMOOTH) * t_lh
            lv = SMOOTH * lv + (1 - SMOOTH) * t_lv
            rh = SMOOTH * rh + (1 - SMOOTH) * t_rh
            # Cuello PITCH Yaw progresivo y despacio hacia la cara:
            # (Invertido) +ex suma grados (va hacia 150/Derecha si la cara está a la izquierda, o 50/Izq si la cara está a la der).
            # Solo la cabeza empieza a moverse si hemos estado viendo una cara por > 2.0s seguidos.
            if (now - face_first_seen_time) > 2.0:
                pitch_ang = clamp(pitch_ang + (ex * 18.0 * dt), PITCH["lo"], PITCH["hi"])
                kit.servo[PIN_PITCH].angle = int(pitch_ang)

            apply_eyes(lh, lv, rh, rv)

            last_seen = now_ms
            dir_x = -1 if fx < cx else 1
            dir_y = -1 if fy < cy else 1

            if not HEADLESS:
                pass # Ya no pintamos nada en el frame para evitar saturar la imagen

            if frames % 10 == 0:
                print(f"👤 IzqH={int(lh)}° IzqV={int(lv)}° | DerH={int(rh)}° DerV={int(rv)}°")

        else:
            dt_lost = now_ms - last_seen

            # ── Retorno al centro (> 4 s sin cara) ───────────────────
            if dt_lost > LOST_MS * 1.5:
                # Pérdida real de fijación (se cambia de persona si aparece otra)
                tracked_face = None
                face_first_seen_time = 0.0
                
            if dt_lost > RETURN_MS and not centered:
                if not returning:
                    print("⏺️  Sin rostro → volviendo al centro...")
                    returning = True

                dlh = LH["mid"] - lh;  dlv = LV["mid"] - lv
                drh = RH["mid"] - rh;  drv = RV["mid"] - rv
                dpitch = PITCH["mid"] - pitch_ang

                if max(abs(dlh), abs(dlv), abs(drh), abs(drv), abs(dpitch)) > 1:
                    lh += dlh * 0.15;  lv += dlv * 0.15
                    rh += drh * 0.15;  rv += drv * 0.15
                    pitch_ang += dpitch * 0.10  # cuello vuelve al centro un poco mas lento
                    kit.servo[PIN_PITCH].angle = int(pitch_ang)
                    apply_eyes(lh, lv, rh, rv)
                else:
                    center_all()
                    lh, lv = float(LH["mid"]), float(LV["mid"])
                    rh, rv = float(RH["mid"]), float(RV["mid"])
                    pitch_ang = float(PITCH["mid"])
                    sum_ex = sum_ey = 0.0
                    dir_x = dir_y = 0
                    centered = True;  returning = False
                    print("✓ Centrado\n")

            # ── Búsqueda (entre 400 ms y 8 s sin cara) ───────────────
            elif dt_lost > LOST_MS and (dir_x or dir_y) and not centered:
                if dir_x:
                    # Ambos ojos misma dirección: 40=der, 120=izq → dir_x>0 fue a der → bajar ángulo
                    lh = clamp(lh - dir_x * SEARCH_DPS * dt, LH["lo"], LH["hi"])
                    rh = clamp(rh - dir_x * SEARCH_DPS * dt, RH["lo"], RH["hi"])
                if dir_y:
                    # Izq: arriba=mayor ángulo; Der: arriba=menor ángulo → opuestos
                    lv = clamp(lv - dir_y * SEARCH_DPS * dt, LV["lo"], LV["hi"])
                    rv = clamp(rv + dir_y * SEARCH_DPS * dt, RV["lo"], RV["hi"])

                apply_eyes(lh, lv, rh, rv)

                if frames % 30 == 0:
                    lado = "IZQ" if dir_x < 0 else "DER" if dir_x > 0 else ""
                    alto = "ARRIBA" if dir_y < 0 else "ABAJO" if dir_y > 0 else ""
                    print(f"🔍 Buscando {lado} {alto}... {int(dt_lost/1000)}s")

            # ── Pérdida corta: mantener posición ─────────────────────
            elif frames % 60 == 0 and dt_lost > 50:
                print(f"⏱️  Esperando... {int(dt_lost/1000)}s")

        if not HEADLESS:
            cv2.line(frame, (cx, 0), (cx, h), (255, 0, 0), 1)
            cv2.line(frame, (0, cy), (w, cy), (255, 0, 0), 1)
            cv2.imshow("Seguimiento ocular", frame)
            if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                break

except KeyboardInterrupt:
    pass
finally:
    cleanup()
