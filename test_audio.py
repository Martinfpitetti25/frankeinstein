"""
Test audio service - microphone and text-to-speech
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from services import AudioService
import time

def test_audio():
    """Test audio service"""
    print("🔊 Testing Audio Service")
    print("="*50)
    
    audio_service = AudioService()
    
    # Test 1: Check microphone availability
    print("\n1. Checking microphone availability...")
    if audio_service.is_microphone_available():
        print("   ✅ Microphone detected")
        mics = audio_service.get_microphone_list()
        print(f"   📋 Available microphones ({len(mics)}):")
        for i, mic in enumerate(mics, 1):
            print(f"      {i}. {mic}")
    else:
        print("   ❌ No microphone found")
        print("   Note: Make sure you have a microphone connected")
        return False
    
    # Test 2: Test microphone
    print("\n2. Testing microphone connection...")
    success, msg = audio_service.test_microphone()
    if success:
        print(f"   ✅ {msg}")
    else:
        print(f"   ❌ {msg}")
        return False
    
    # Test 3: Test speakers/TTS
    print("\n3. Testing text-to-speech...")
    print("   🔊 You should hear: 'Audio test'")
    success, msg = audio_service.test_speakers()
    if success:
        print(f"   ✅ {msg}")
    else:
        print(f"   ❌ {msg}")
        return False
    
    time.sleep(1)
    
    # Test 4: Speech recognition
    print("\n4. Testing speech recognition...")
    print("   🎤 Please say something in the next 5 seconds...")
    print("   (Try saying: 'Hello, this is a test')")
    
    success, text = audio_service.listen_once(timeout=5, phrase_time_limit=10)
    
    if success:
        print(f"   ✅ Recognized: '{text}'")
        
        # Test 5: Speak back what was recognized
        print("\n5. Speaking back what you said...")
        print(f"   🔊 You should hear: '{text}'")
        audio_service.speak(text, blocking=True)
        print("   ✅ Text-to-speech playback complete")
    else:
        print(f"   ⚠️  {text}")
        print("   Note: Speech recognition requires internet connection")
    
    print("\n" + "="*50)
    print("✅ Audio service test completed!")
    print("\nTips:")
    print("- Speak clearly and close to the microphone")
    print("- Ensure you have a good internet connection for speech recognition")
    print("- Adjust microphone volume if recognition fails")
    print("\nYou can now use voice features in the main application:")
    print("   python src/main.py")
    return True

if __name__ == "__main__":
    try:
        success = test_audio()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
