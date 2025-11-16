#!/usr/bin/env python3
"""
Demo script to showcase the vision + chat integration
This script demonstrates how the assistant can "see" objects through the camera
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from services.camera_service import CameraService
from services.chat_service import OllamaService
import time


def demo_vision_chat():
    """Interactive demo of vision + chat integration"""
    
    print("\n" + "="*70)
    print("🤖 DEMO: Integración Visión + Chat")
    print("="*70)
    print("\nEste demo muestra cómo el asistente puede 'ver' objetos detectados")
    print("por la cámara y responder preguntas sobre ellos.\n")
    
    # Initialize services
    print("Inicializando servicios...")
    camera_service = CameraService()
    ollama_service = OllamaService()
    
    # Start camera
    if not camera_service.start_camera():
        print("❌ Error: No se pudo iniciar la cámara")
        return
    print("✅ Cámara iniciada")
    
    # Load YOLO
    if not camera_service.load_yolo_model("yolov8n.pt"):
        print("❌ Error: No se pudo cargar el modelo YOLO")
        camera_service.stop_camera()
        return
    print("✅ Modelo YOLO cargado")
    
    # Check Ollama
    if not ollama_service.is_available():
        print("❌ Error: Ollama no está ejecutándose")
        print("   Ejecuta: ollama serve")
        camera_service.stop_camera()
        return
    print("✅ Ollama disponible")
    
    print("\n" + "-"*70)
    print("Capturando y analizando la escena...")
    print("-"*70 + "\n")
    
    # Capture frames
    for i in range(15):
        camera_service.get_frame_with_detection()
        time.sleep(0.1)
    
    # Show detections
    summary = camera_service.get_detection_summary()
    detections = camera_service.get_current_detections()
    
    print("📷 DETECCIONES DE LA CÁMARA:")
    print(f"   {summary}\n")
    
    if detections:
        print("   Objetos detectados con confianza:")
        for i, det in enumerate(detections, 1):
            conf_percent = det['confidence'] * 100
            print(f"   {i}. {det['class']} ({conf_percent:.1f}% confianza)")
    
    print("\n" + "-"*70)
    print("EJEMPLOS DE PREGUNTAS (prueba estas en la aplicación):")
    print("-"*70)
    
    example_questions = [
        "¿Qué objetos ves?",
        "Describe lo que hay en la imagen",
        "¿Cuántos objetos detectas?",
        "¿Hay alguna laptop en la cámara?",
        "¿Qué puedo hacer con los objetos que ves?"
    ]
    
    for i, q in enumerate(example_questions, 1):
        print(f"{i}. {q}")
    
    print("\n" + "-"*70)
    print("DEMO INTERACTIVO - Probemos una pregunta:")
    print("-"*70 + "\n")
    
    test_question = "¿Qué objetos ves en la imagen?"
    print(f"Usuario: {test_question}")
    print(f"Sistema: 📷 {summary}")
    print("\nAsistente está pensando...\n")
    
    response = ollama_service.send_message(test_question, vision_context=summary)
    
    print(f"Asistente: {response}")
    
    print("\n" + "-"*70)
    print("Comparación: Pregunta SIN contexto de visión:")
    print("-"*70 + "\n")
    
    normal_question = "Hola, ¿cómo estás?"
    print(f"Usuario: {normal_question}")
    
    response = ollama_service.send_message(normal_question)
    
    print(f"Asistente: {response}")
    
    # Cleanup
    camera_service.stop_camera()
    
    print("\n" + "="*70)
    print("✅ DEMO COMPLETADO")
    print("="*70)
    print("\nPara usar esta funcionalidad en la aplicación principal:")
    print("1. Ejecuta: python src/main.py")
    print("2. Haz preguntas que incluyan palabras como:")
    print("   - 'qué ves', 'describe', 'imagen', 'cámara', 'objetos'")
    print("3. El sistema automáticamente agregará contexto visual a tu pregunta")
    print("\nRevisa VISION_EXAMPLES.md para más ejemplos y consejos.\n")


if __name__ == "__main__":
    try:
        demo_vision_chat()
    except KeyboardInterrupt:
        print("\n\n❌ Demo interrumpido por el usuario")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error en el demo: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
