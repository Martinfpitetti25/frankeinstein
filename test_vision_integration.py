#!/usr/bin/env python3
"""
Test script for vision + chat integration
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from services.camera_service import CameraService
from services.chat_service import OllamaService
import time


def test_vision_integration():
    """Test the vision + chat integration"""
    
    print("=" * 60)
    print("Testing Vision + Chat Integration")
    print("=" * 60)
    
    # Initialize services
    print("\n1. Initializing Camera Service...")
    camera_service = CameraService()
    
    # Start camera
    if not camera_service.start_camera():
        print("❌ Failed to start camera")
        return False
    print("✅ Camera started")
    
    # Load YOLO model
    print("\n2. Loading YOLO model...")
    if not camera_service.load_yolo_model("yolov8n.pt"):
        print("❌ Failed to load YOLO model")
        camera_service.stop_camera()
        return False
    print("✅ YOLO model loaded")
    
    # Initialize Ollama
    print("\n3. Checking Ollama service...")
    ollama_service = OllamaService()
    if not ollama_service.is_available():
        print("❌ Ollama is not running")
        camera_service.stop_camera()
        return False
    print("✅ Ollama is available")
    
    # Capture some frames to get detections
    print("\n4. Capturing frames and detecting objects...")
    print("   (capturing 10 frames to get stable detections)")
    for i in range(10):
        success, frame, detections = camera_service.get_frame_with_detection()
        if success:
            print(f"   Frame {i+1}/10 - Detected {len(detections)} objects")
        time.sleep(0.1)
    
    # Get detection summary
    print("\n5. Current Detection Summary:")
    summary = camera_service.get_detection_summary()
    print(f"   {summary}")
    
    detections = camera_service.get_current_detections()
    if detections:
        print(f"\n   Detailed detections:")
        for i, det in enumerate(detections, 1):
            print(f"   {i}. {det['class']} (confidence: {det['confidence']:.2f})")
    
    # Test vision context with chat
    print("\n6. Testing Chat with Vision Context:")
    print("   Query: '¿Qué objetos ves en la imagen?'")
    
    # Simulate vision-enabled query
    message = "¿Qué objetos ves en la imagen?"
    vision_context = camera_service.get_detection_summary()
    
    print(f"\n   Vision Context: {vision_context}")
    print("   Sending to Ollama...")
    
    response = ollama_service.send_message(message, vision_context=vision_context)
    print(f"\n   Response:\n   {response}")
    
    # Test without vision context
    print("\n7. Testing Chat WITHOUT Vision Context:")
    print("   Query: 'Hola, ¿cómo estás?'")
    
    response = ollama_service.send_message("Hola, ¿cómo estás?")
    print(f"\n   Response:\n   {response}")
    
    # Cleanup
    print("\n8. Cleaning up...")
    camera_service.stop_camera()
    print("✅ Camera stopped")
    
    print("\n" + "=" * 60)
    print("✅ Vision + Chat Integration Test Complete!")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    try:
        success = test_vision_integration()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n❌ Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
