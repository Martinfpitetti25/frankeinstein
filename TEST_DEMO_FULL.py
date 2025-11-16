#!/usr/bin/env python3
"""
TEST_DEMO_FULL.py - Demostración completa del robot
Secuencia:
1. Presentación con movimiento de servos (simulando habla)
2. Volver al origen y describir la escena con YOLO
3. Seguimiento facial durante 30 segundos
4. Volver al origen
"""

import sys
import os
import time
import cv2
import numpy as np
from adafruit_servokit import ServoKit
from ultralytics import YOLO
import mediapipe as mp
import logging
import threading

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Intentar importar pyttsx3 para texto a voz
try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    logger.warning("pyttsx3 no disponible, se mostrará solo texto")

# ============================================================
# CONFIGURACIÓN DE SERVOS
# ============================================================
SERVO_YAW = 13       # Horizontal (cuello)
SERVO_PITCH = 14     # Vertical
SERVO_ROLL_LEFT = 12
SERVO_ROLL_RIGHT = 15
SERVO_BOCA = 5       # Boca

# Posiciones de centro
CENTER_YAW = 135
CENTER_PITCH = 120
CENTER_ROLL = 155
CENTER_BOCA = 50

# ============================================================
# INICIALIZACIÓN
# ============================================================

print("=" * 60)
print("🤖 DEMOSTRACIÓN COMPLETA DEL ROBOT")
print("=" * 60)
print()

# Inicializar ServoKit
try:
    kit = ServoKit(channels=16)
    logger.info("✓ ServoKit inicializado")
except Exception as e:
    logger.error(f"❌ Error inicializando servos: {e}")
    sys.exit(1)

# Inicializar TTS si está disponible
engine = None
if TTS_AVAILABLE:
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)  # Velocidad
        engine.setProperty('volume', 0.9)  # Volumen
        # Intentar usar voz en español
        voices = engine.getProperty('voices')
        for voice in voices:
            if 'spanish' in voice.name.lower() or 'español' in voice.name.lower():
                engine.setProperty('voice', voice.id)
                break
        logger.info("✓ TTS inicializado")
    except Exception as e:
        logger.warning(f"⚠️ Error inicializando TTS: {e}")
        engine = None

# Inicializar YOLO
try:
    yolo_model = YOLO("yolov8n.pt")
    logger.info("✓ YOLO cargado")
except Exception as e:
    logger.error(f"❌ Error cargando YOLO: {e}")
    yolo_model = None

# Inicializar MediaPipe para detección de rostros
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.4
)
logger.info("✓ MediaPipe Face Detection inicializado")

# ============================================================
# FUNCIONES DE UTILIDAD
# ============================================================

def traducir_objeto(obj_name):
    """Traduce nombres de objetos de YOLO del inglés al español"""
    traducciones = {
        'person': 'persona',
        'bicycle': 'bicicleta',
        'car': 'automóvil',
        'motorcycle': 'motocicleta',
        'airplane': 'avión',
        'bus': 'autobús',
        'train': 'tren',
        'truck': 'camión',
        'boat': 'bote',
        'traffic light': 'semáforo',
        'fire hydrant': 'hidrante',
        'stop sign': 'señal de alto',
        'parking meter': 'parquímetro',
        'bench': 'banca',
        'bird': 'pájaro',
        'cat': 'gato',
        'dog': 'perro',
        'horse': 'caballo',
        'sheep': 'oveja',
        'cow': 'vaca',
        'elephant': 'elefante',
        'bear': 'oso',
        'zebra': 'cebra',
        'giraffe': 'jirafa',
        'backpack': 'mochila',
        'umbrella': 'paraguas',
        'handbag': 'bolso',
        'tie': 'corbata',
        'suitcase': 'maleta',
        'frisbee': 'frisbee',
        'skis': 'esquís',
        'snowboard': 'tabla de nieve',
        'sports ball': 'balón',
        'kite': 'cometa',
        'baseball bat': 'bate de béisbol',
        'baseball glove': 'guante de béisbol',
        'skateboard': 'patineta',
        'surfboard': 'tabla de surf',
        'tennis racket': 'raqueta de tenis',
        'bottle': 'botella',
        'wine glass': 'copa',
        'cup': 'taza',
        'fork': 'tenedor',
        'knife': 'cuchillo',
        'spoon': 'cuchara',
        'bowl': 'tazón',
        'banana': 'plátano',
        'apple': 'manzana',
        'sandwich': 'sándwich',
        'orange': 'naranja',
        'broccoli': 'brócoli',
        'carrot': 'zanahoria',
        'hot dog': 'hot dog',
        'pizza': 'pizza',
        'donut': 'dona',
        'cake': 'pastel',
        'chair': 'silla',
        'couch': 'sofá',
        'potted plant': 'planta',
        'bed': 'cama',
        'dining table': 'mesa',
        'toilet': 'inodoro',
        'tv': 'televisor',
        'laptop': 'computadora portátil',
        'mouse': 'ratón',
        'remote': 'control remoto',
        'keyboard': 'teclado',
        'cell phone': 'teléfono celular',
        'microwave': 'microondas',
        'oven': 'horno',
        'toaster': 'tostadora',
        'sink': 'lavabo',
        'refrigerator': 'refrigerador',
        'book': 'libro',
        'clock': 'reloj',
        'vase': 'florero',
        'scissors': 'tijeras',
        'teddy bear': 'oso de peluche',
        'hair drier': 'secadora de pelo',
        'toothbrush': 'cepillo de dientes'
    }
    
    return traducciones.get(obj_name.lower(), obj_name)

def speak(text):
    """Reproduce texto a voz y lo muestra en consola"""
    print(f"🗣️  {text}")
    if engine:
        try:
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            logger.warning(f"Error en TTS: {e}")

def speak_with_movement(text, duration=None):
    """Habla mientras mueve los servos (en paralelo usando threading)"""
    print(f"🗣️  {text}")
    
    if engine:
        # Crear thread para el audio
        def audio_thread():
            try:
                engine.say(text)
                engine.runAndWait()
            except Exception as e:
                logger.warning(f"Error en TTS: {e}")
        
        # Iniciar audio en segundo plano
        audio = threading.Thread(target=audio_thread)
        audio.start()
        
        # Mientras tanto, mover servos
        if duration is None:
            duration = len(text) * 0.08  # Aproximación: 0.08 seg por carácter
        
        simulate_talking(duration=duration)
        
        # Esperar a que termine el audio
        audio.join()
    else:
        # Sin audio, solo simular habla
        if duration is None:
            duration = len(text) * 0.08
        simulate_talking(duration=duration)
    
    time.sleep(0.5)

def set_servo_safe(channel, angle):
    """Establece ángulo de servo con manejo de errores"""
    try:
        kit.servo[channel].angle = angle
    except Exception as e:
        logger.warning(f"Error moviendo servo {channel}: {e}")

def move_to_center():
    """Mueve todos los servos al centro"""
    print("🎯 Volviendo al origen...")
    set_servo_safe(SERVO_YAW, CENTER_YAW)
    time.sleep(0.1)
    set_servo_safe(SERVO_PITCH, CENTER_PITCH)
    time.sleep(0.1)
    set_servo_safe(SERVO_ROLL_LEFT, CENTER_ROLL)
    set_servo_safe(SERVO_ROLL_RIGHT, CENTER_ROLL)
    time.sleep(0.1)
    set_servo_safe(SERVO_BOCA, CENTER_BOCA)
    time.sleep(0.3)

def simulate_talking(duration=3.0, speed=0.15):
    """Simula hablar moviendo servos y boca"""
    start_time = time.time()
    boca_open = False
    
    while time.time() - start_time < duration:
        # Mover boca
        if boca_open:
            set_servo_safe(SERVO_BOCA, 50)  # Cerrada
        else:
            set_servo_safe(SERVO_BOCA, 75)  # Abierta
        boca_open = not boca_open
        
        # Pequeños movimientos de cabeza
        yaw_offset = np.random.randint(-5, 6)
        pitch_offset = np.random.randint(-3, 4)
        set_servo_safe(SERVO_YAW, CENTER_YAW + yaw_offset)
        set_servo_safe(SERVO_PITCH, CENTER_PITCH + pitch_offset)
        
        time.sleep(speed)
    
    # Volver a posición neutral
    set_servo_safe(SERVO_BOCA, CENTER_BOCA)
    set_servo_safe(SERVO_YAW, CENTER_YAW)
    set_servo_safe(SERVO_PITCH, CENTER_PITCH)

# ============================================================
# FASE 1: PRESENTACIÓN
# ============================================================

def fase_presentacion():
    """Presentación del robot con movimientos"""
    print()
    print("=" * 60)
    print("📢 FASE 1: PRESENTACIÓN")
    print("=" * 60)
    print()
    
    move_to_center()
    time.sleep(1)
    
    # Texto de presentación
    texto = "Hola! Soy un asistente robótico creado por Javier Agustín y Francisco. Muchas gracias por venir y espero que se diviertan."
    
    # Hablar con movimientos (en paralelo, sin entrecortar)
    speak_with_movement(texto, duration=10.0)
    
    time.sleep(1)

# ============================================================
# FASE 2: DESCRIPCIÓN DE ESCENA CON YOLO
# ============================================================

def fase_descripcion_yolo():
    """Describe la escena usando YOLO"""
    print()
    print("=" * 60)
    print("👁️  FASE 2: DESCRIPCIÓN DE LA ESCENA")
    print("=" * 60)
    print()
    
    move_to_center()
    time.sleep(1)
    
    if not yolo_model:
        speak("Lo siento, no tengo disponible la visión artificial.")
        return
    
    speak("Voy a observar el entorno.")
    time.sleep(0.5)
    
    # Abrir cámara
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        speak("No puedo abrir la cámara.")
        return
    
    # Capturar frame
    time.sleep(1)  # Dar tiempo a la cámara
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        speak("No pude capturar la imagen.")
        return
    
    # Detectar objetos
    results = yolo_model(frame, conf=0.5, verbose=False)
    
    # Contar objetos detectados
    detections = {}
    for r in results:
        for box in r.boxes:
            class_id = int(box.cls[0])
            class_name = yolo_model.names[class_id]
            if class_name in detections:
                detections[class_name] += 1
            else:
                detections[class_name] = 1
    
    # Describir lo que ve de forma natural
    if len(detections) == 0:
        speak("Mmm, no distingo objetos específicos en este momento, aunque puedo ver el entorno.")
    else:
        # Construir descripción natural
        total_objects = sum(detections.values())
        
        # Introducción variada según cantidad
        if total_objects == 1:
            speak("Veo un objeto frente a mí.")
        elif total_objects <= 3:
            speak(f"Puedo ver {total_objects} objetos aquí.")
        else:
            speak(f"Observo varios objetos, en total {total_objects}.")
        
        time.sleep(0.8)
        
        # Describir objetos de forma más natural
        items = list(detections.items())[:5]  # Máximo 5 objetos
        
        for i, (obj, count) in enumerate(items):
            # Traducir algunos nombres comunes al español
            obj_es = traducir_objeto(obj)
            
            # Variación en la forma de describir
            if i == 0:
                if count == 1:
                    speak(f"Veo un {obj_es}")
                else:
                    speak(f"Hay {count} {obj_es}s")
            elif i == len(items) - 1 and len(items) > 1:
                if count == 1:
                    speak(f"y también un {obj_es}")
                else:
                    speak(f"y también {count} {obj_es}s")
            else:
                if count == 1:
                    speak(f"un {obj_es}")
                else:
                    speak(f"{count} {obj_es}s")
            
            time.sleep(0.4)
    
    time.sleep(1)

# ============================================================
# FASE 3: SEGUIMIENTO FACIAL
# ============================================================

def fase_seguimiento():
    """Seguimiento facial durante 30 segundos"""
    print()
    print("=" * 60)
    print("👤 FASE 3: SEGUIMIENTO FACIAL")
    print("=" * 60)
    print()
    
    move_to_center()
    time.sleep(1)
    
    speak("Ahora seguiré durante 30 segundos a todos los que estén aquí.")
    time.sleep(1)
    
    # Abrir cámara
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        speak("No puedo abrir la cámara.")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    start_time = time.time()
    duration = 30.0
    
    # Variables de control
    current_yaw = CENTER_YAW
    current_pitch = CENTER_PITCH
    
    print("🔄 Iniciando seguimiento...")
    
    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            break
        
        h, w = frame.shape[:2]
        cx_frame = w // 2
        cy_frame = h // 2
        
        # Convertir a RGB para MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_frame)
        
        if results.detections:
            # Tomar el primer rostro detectado
            detection = results.detections[0]
            bbox = detection.location_data.relative_bounding_box
            
            # Calcular centro del rostro
            cx_face = int((bbox.xmin + bbox.width / 2) * w)
            cy_face = int((bbox.ymin + bbox.height / 2) * h)
            
            # Calcular errores
            error_x = cx_face - cx_frame
            error_y = cy_face - cy_frame
            
            # Control proporcional simple
            if abs(error_x) > 40:  # Deadband horizontal
                current_yaw -= error_x * 0.05
                current_yaw = np.clip(current_yaw, 90, 180)
            
            if abs(error_y) > 15:  # Deadband vertical
                current_pitch += error_y * 0.08
                current_pitch = np.clip(current_pitch, 60, 180)
            
            # Aplicar movimientos
            set_servo_safe(SERVO_YAW, int(current_yaw))
            set_servo_safe(SERVO_PITCH, int(current_pitch))
            
            # Calcular roll
            roll_angle = 130 + ((current_pitch - 60) / (180 - 60)) * (180 - 130)
            set_servo_safe(SERVO_ROLL_LEFT, int(roll_angle))
            set_servo_safe(SERVO_ROLL_RIGHT, int(roll_angle))
        
        time.sleep(0.05)
    
    cap.release()
    speak("Seguimiento completado.")
    time.sleep(0.5)

# ============================================================
# FASE 4: DESPEDIDA
# ============================================================

def fase_despedida():
    """Despedida y vuelta al origen"""
    print()
    print("=" * 60)
    print("👋 FASE 4: DESPEDIDA")
    print("=" * 60)
    print()
    
    move_to_center()
    time.sleep(1)
    
    speak("Muchas gracias por su atención. Hasta pronto!")
    time.sleep(1)
    
    # Pequeño gesto de despedida
    for _ in range(2):
        set_servo_safe(SERVO_YAW, 120)
        time.sleep(0.3)
        set_servo_safe(SERVO_YAW, 150)
        time.sleep(0.3)
    
    move_to_center()

# ============================================================
# MAIN
# ============================================================

def main():
    """Función principal - ejecuta toda la demo"""
    try:
        # Posición inicial
        move_to_center()
        time.sleep(2)
        
        # Ejecutar fases
        fase_presentacion()
        fase_descripcion_yolo()
        fase_seguimiento()
        fase_despedida()
        
        print()
        print("=" * 60)
        print("✅ DEMOSTRACIÓN COMPLETADA")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n⚠️  Interrupción detectada")
    except Exception as e:
        logger.error(f"❌ Error en demostración: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Volver al centro y cerrar recursos
        print("\n🔄 Limpiando recursos...")
        move_to_center()
        face_detection.close()
        if engine:
            try:
                engine.stop()
            except:
                pass
        print("✅ Recursos liberados")

if __name__ == "__main__":
    main()
