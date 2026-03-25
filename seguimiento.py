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
PARPADO_ABIERTO = 40

# Límites calibrados: dict con lo(mín), hi(máx), mid(centro)
LH = dict(lo=40,  hi=120, mid=80)   # Izq Horizontal
LV = dict(lo=85,  hi=105, mid=95)   # Izq Vertical  (105=arriba, 85=abajo)
RH = dict(lo=40,  hi=120, mid=80)   # Der Horizontal
RV = dict(lo=70,  hi=90,  mid=80)   # Der Vertical  (70=arriba, 90=abajo) ← INVERTIDO

# ──────────────────────────────────────────────
# PARÁMETROS
# ──────────────────────────────────────────────
CAM_INDEX   = 0
HEADLESS    = False     # False → muestra ventana de video

KP          = 0.18      # Ganancia proporcional
KI          = 0.01      # Ganancia integral
SMOOTH      = 0.35      # Suavizado EMA (0=rígido, 1=sin movimiento)
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
    for pin in (PIN_LH, PIN_LV, PIN_RH, PIN_RV, PIN_PARPADO_INF, PIN_PARPADO_SUP):
        kit.servo[pin].actuation_range = 180
        kit.servo[pin].set_pulse_width_range(650, 2000)
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
_URL   = ("https://raw.githubusercontent.com/opencv/opencv_zoo/main/"
          "models/face_detection_yunet/face_detection_yunet_2023mar.onnx")
if not _MODEL.exists():
    print("⬇️  Descargando modelo YuNet (primera vez)...")
    _MODEL.parent.mkdir(exist_ok=True)
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

sum_ex = sum_ey = 0.0

last_time   = time.time()
last_seen   = time.time() * 1000.0
dir_x = dir_y = 0
centered    = False
returning   = False
frames      = 0
fps_t       = time.time()

# ──────────────────────────────────────────────
# LOOP PRINCIPAL
# ──────────────────────────────────────────────
try:
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frames += 1
        if frames % 30 == 0:
            fps = 30.0 / max(time.time() - fps_t, 1e-6)
            fps_t = time.time()
            print(f"📹 FPS: {fps:.1f}")

        now    = time.time()
        dt     = max(1e-3, now - last_time)
        last_time = now
        now_ms = now * 1000.0
        h, w   = frame.shape[:2]
        cx, cy = w // 2, h // 2

        _, faces = fd.detect(frame)

        if faces is not None:
            centered = returning = False

            best      = faces[faces[:, 14].argmax()]
            bx, by    = int(best[0]), int(best[1])
            bw, bh    = int(best[2]), int(best[3])
            fx, fy    = bx + bw // 2, by + bh // 2

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
            #   ex>0 → cara a la izquierda → ojos izquierda → ángulo sube
            half_h = (LH["hi"] - LH["lo"]) / 2.0   # 40°
            t_lh   = LH["mid"] + pid_x * half_h
            t_rh   = RH["mid"] + pid_x * half_h

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
            rv = SMOOTH * rv + (1 - SMOOTH) * t_rv

            apply_eyes(lh, lv, rh, rv)

            last_seen = now_ms
            dir_x = -1 if fx < cx else 1
            dir_y = -1 if fy < cy else 1

            if not HEADLESS:
                cv2.putText(frame, f"Izq H{int(lh)} V{int(lv)}  Der H{int(rh)} V{int(rv)}",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            if frames % 10 == 0:
                print(f"👤 IzqH={int(lh)}° IzqV={int(lv)}° | DerH={int(rh)}° DerV={int(rv)}°")

        else:
            dt_lost = now_ms - last_seen

            # ── Retorno al centro (> 4 s sin cara) ───────────────────
            if dt_lost > RETURN_MS and not centered:
                if not returning:
                    print("⏺️  Sin rostro → volviendo al centro...")
                    returning = True

                dlh = LH["mid"] - lh;  dlv = LV["mid"] - lv
                drh = RH["mid"] - rh;  drv = RV["mid"] - rv

                if max(abs(dlh), abs(dlv), abs(drh), abs(drv)) > 1:
                    lh += dlh * 0.15;  lv += dlv * 0.15
                    rh += drh * 0.15;  rv += drv * 0.15
                    apply_eyes(lh, lv, rh, rv)
                else:
                    center_all()
                    lh, lv = float(LH["mid"]), float(LV["mid"])
                    rh, rv = float(RH["mid"]), float(RV["mid"])
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
