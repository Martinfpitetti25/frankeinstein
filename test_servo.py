#!/usr/bin/env python3
"""
Test script for servo motor control
Tests basic servo functionality before integrating into main application
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from services import ServoService
import time


def test_servo_basic():
    """Test basic servo movements"""
    print("=" * 50)
    print("SERVO MOTOR TEST")
    print("=" * 50)
    
    # Ask user which method to use
    print("\nSelect servo control method:")
    print("1. gpiozero (recommended for simple setup)")
    print("2. pigpio (most accurate, requires pigpiod daemon)")
    print("3. PCA9685 (for I2C servo controller)")
    
    choice = input("\nEnter choice (1-3) [default: 1]: ").strip() or "1"
    
    method_map = {
        "1": ServoService.METHOD_GPIOZERO,
        "2": ServoService.METHOD_PIGPIO,
        "3": ServoService.METHOD_PCA9685
    }
    
    method = method_map.get(choice, ServoService.METHOD_GPIOZERO)
    print(f"\n✓ Using method: {method}")
    
    # Create servo service
    servo = ServoService(method=method)
    
    # Get pin configuration
    print("\nServo pin configuration:")
    if method == ServoService.METHOD_PCA9685:
        h_pin = int(input("Horizontal servo channel [0]: ") or "0")
        v_pin = int(input("Vertical servo channel [1]: ") or "1")
    else:
        h_pin = int(input("Horizontal servo GPIO pin [17]: ") or "17")
        v_pin = int(input("Vertical servo GPIO pin [27]: ") or "27")
    
    # Initialize servos
    print(f"\n⏳ Initializing servos on pins/channels {h_pin}, {v_pin}...")
    
    if method == ServoService.METHOD_PIGPIO:
        print("\n⚠️  Note: pigpio requires the pigpiod daemon to be running.")
        print("   If not started, run: sudo pigpiod")
        input("\nPress Enter to continue...")
    
    if not servo.initialize(horizontal_pin=h_pin, vertical_pin=v_pin):
        print("❌ Failed to initialize servos!")
        print("\nTroubleshooting:")
        print("- Check that pins are correct")
        if method == ServoService.METHOD_PIGPIO:
            print("- Ensure pigpiod is running: sudo pigpiod")
        print("- Check servo power supply")
        print("- Verify GPIO permissions")
        return
    
    print("✓ Servos initialized successfully!")
    time.sleep(1)
    
    # Test sequence
    try:
        print("\n" + "=" * 50)
        print("TEST 1: Move to center position")
        print("=" * 50)
        servo.move_to_center()
        time.sleep(2)
        
        print("\n" + "=" * 50)
        print("TEST 2: Horizontal movement test")
        print("=" * 50)
        print("Moving horizontal servo from 0° to 180°...")
        servo.set_angle("horizontal", 0, smooth=True)
        time.sleep(1)
        servo.set_angle("horizontal", 180, smooth=True)
        time.sleep(1)
        servo.set_angle("horizontal", 90, smooth=True)
        print("✓ Horizontal test complete")
        time.sleep(2)
        
        print("\n" + "=" * 50)
        print("TEST 3: Vertical movement test")
        print("=" * 50)
        print("Moving vertical servo from 0° to 180°...")
        servo.set_angle("vertical", 0, smooth=True)
        time.sleep(1)
        servo.set_angle("vertical", 180, smooth=True)
        time.sleep(1)
        servo.set_angle("vertical", 90, smooth=True)
        print("✓ Vertical test complete")
        time.sleep(2)
        
        print("\n" + "=" * 50)
        print("TEST 4: Horizontal scan (180° sweep)")
        print("=" * 50)
        print("Scanning from left to right...")
        servo.scan_horizontal(start_angle=0, end_angle=180, step=5, delay=0.05)
        servo.move_to_center()
        print("✓ Horizontal scan complete")
        time.sleep(2)
        
        print("\n" + "=" * 50)
        print("TEST 5: Vertical scan (180° sweep)")
        print("=" * 50)
        print("Scanning from top to bottom...")
        servo.scan_vertical(start_angle=0, end_angle=180, step=5, delay=0.05)
        servo.move_to_center()
        print("✓ Vertical scan complete")
        time.sleep(2)
        
        print("\n" + "=" * 50)
        print("✓ ALL TESTS PASSED!")
        print("=" * 50)
        
        h_pos, v_pos = servo.get_position()
        print(f"\nFinal position: Horizontal={h_pos}°, Vertical={v_pos}°")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
    
    finally:
        print("\n🔧 Cleaning up...")
        servo.cleanup()
        print("✓ Cleanup complete")


if __name__ == "__main__":
    print("\n⚠️  IMPORTANT: Ensure servos are properly powered!")
    print("⚠️  Servos typically need external 5-6V power supply")
    print("⚠️  Do NOT power servos directly from Raspberry Pi 5V pins!\n")
    
    response = input("Have you connected servo power supply? (yes/no): ").strip().lower()
    if response != "yes":
        print("\n❌ Please connect power supply before testing.")
        print("Servo connections:")
        print("  - Signal wire → GPIO pin")
        print("  - Power wire (red) → External 5-6V power supply (+)")
        print("  - Ground wire (brown/black) → External power supply (-) AND Raspberry Pi GND")
        sys.exit(1)
    
    test_servo_basic()
