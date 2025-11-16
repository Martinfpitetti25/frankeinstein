"""
Test camera detection and YOLO functionality
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from services import CameraService

def test_camera():
    """Test camera detection"""
    print("🔍 Testing Camera Service")
    print("="*50)
    
    camera_service = CameraService()
    
    # Test 1: Find camera
    print("\n1. Searching for cameras...")
    camera_index = camera_service.find_camera()
    
    if camera_index is not None:
        print(f"   ✅ Camera found at index: {camera_index}")
    else:
        print("   ❌ No camera found")
        print("   Note: Make sure you have a webcam connected")
        return False
    
    # Test 2: Start camera
    print("\n2. Starting camera...")
    if camera_service.start_camera(camera_index):
        print("   ✅ Camera started successfully")
    else:
        print("   ❌ Failed to start camera")
        return False
    
    # Test 3: Read a frame
    print("\n3. Reading test frame...")
    ret, frame = camera_service.read_frame()
    if ret and frame is not None:
        print(f"   ✅ Frame captured: {frame.shape}")
    else:
        print("   ❌ Failed to capture frame")
        camera_service.stop_camera()
        return False
    
    # Test 4: Load YOLO model
    print("\n4. Loading YOLO model (this may take a moment)...")
    if camera_service.load_yolo_model("yolov8n.pt"):
        print("   ✅ YOLO model loaded successfully")
    else:
        print("   ❌ Failed to load YOLO model")
        camera_service.stop_camera()
        return False
    
    # Test 5: Run detection on frame
    print("\n5. Running YOLO detection on test frame...")
    annotated_frame, detections = camera_service.detect_objects(frame)
    print(f"   ✅ Detection completed")
    print(f"   📊 Objects detected: {len(detections)}")
    
    if detections:
        print("\n   Detected objects:")
        for i, det in enumerate(detections[:5], 1):
            print(f"      {i}. {det['class']} (confidence: {det['confidence']:.2f})")
        if len(detections) > 5:
            print(f"      ... and {len(detections) - 5} more")
    else:
        print("   No objects detected in the test frame")
    
    # Cleanup
    camera_service.stop_camera()
    print("\n" + "="*50)
    print("✅ All camera tests passed!")
    print("\nYou can now run the main application:")
    print("   python src/main.py")
    return True

if __name__ == "__main__":
    try:
        success = test_camera()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
