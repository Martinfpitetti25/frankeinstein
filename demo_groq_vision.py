#!/usr/bin/env python3
"""
Demo de Groq con Visión + YOLO
"""
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from services.groq_service import GroqService
from services.camera_service import CameraService

print("="*70)
print("🤖 DEMO: Groq + Visión YOLO")
print("="*70)
print("\nEste demo muestra cómo Groq puede 'ver' y describir escenas")
print("usando las detecciones de YOLO.\n")

# Inicializar servicios
print("Inicializando servicios...")
groq_service = GroqService()
camera_service = CameraService()

# Verificar Groq
if not groq_service.is_available():
    print("❌ Error: Groq no está configurado")
    print("   Ejecuta: python test_groq.py")
    sys.exit(1)
print("✅ Groq disponible")

# Iniciar cámara
if not camera_service.start_camera():
    print("❌ Error: No se pudo iniciar la cámara")
    sys.exit(1)
print("✅ Cámara iniciada")

# Cargar YOLO
if not camera_service.load_yolo_model("yolov8n.pt"):
    print("❌ Error: No se pudo cargar YOLO")
    camera_service.stop_camera()
    sys.exit(1)
print("✅ YOLO cargado")

print("\n" + "-"*70)
print("Capturando y analizando la escena...")
print("-"*70 + "\n")

# Capturar frames para obtener detecciones estables
for i in range(15):
    camera_service.get_frame_with_detection()
    time.sleep(0.1)

# Obtener resumen de detecciones
vision_context = camera_service.get_detection_summary()
detections = camera_service.get_current_detections()

print("📷 DETECCIONES DE YOLO:")
print(f"   {vision_context}\n")

if detections:
    print("   Objetos detectados:")
    for i, det in enumerate(detections, 1):
        conf_percent = det['confidence'] * 100
        print(f"   {i}. {det['class']} ({conf_percent:.1f}% confianza)")

print("\n" + "-"*70)
print("🤖 GROQ DESCRIBE LA ESCENA:")
print("-"*70 + "\n")

# Preguntar a Groq sobre la escena
question = "Describe detalladamente lo que ves en esta imagen"
print(f"Pregunta: {question}\n")
print("Groq está pensando...\n")

response = groq_service.send_message(question, vision_context=vision_context)

print(f"Groq responde:\n{response}")

print("\n" + "-"*70)
print("🎨 PRUEBA CON OTRA PREGUNTA:")
print("-"*70 + "\n")

question2 = "¿Qué puedo hacer con los objetos que ves?"
print(f"Pregunta: {question2}\n")

response2 = groq_service.send_message(question2, vision_context=vision_context)

print(f"Groq responde:\n{response2}")

# Cleanup
camera_service.stop_camera()

print("\n" + "="*70)
print("✅ DEMO COMPLETADO")
print("="*70)
print("\n💡 En la aplicación principal:")
print("   1. python src/main.py")
print("   2. Selecciona 'Groq' del menú")
print("   3. Pregunta: '¿Qué ves en la imagen?'")
print("   4. Groq describirá automáticamente lo que YOLO detectó")
print("\n🎯 Palabras clave para activar visión:")
print("   - 'qué ves', 'describe', 'imagen', 'cámara', 'objetos'")
print()
