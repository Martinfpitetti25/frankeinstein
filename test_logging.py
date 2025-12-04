#!/usr/bin/env python3
"""
Test script for logging system
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from utils.logger import get_logger, get_log_files_info

# Test different loggers
logger_main = get_logger("main")
logger_camera = get_logger("services.camera_service")
logger_audio = get_logger("services.audio_service")

print("=" * 60)
print("Testing Frankeinstein AI Logging System")
print("=" * 60)

# Test different log levels
logger_main.debug("This is a DEBUG message (only in file)")
logger_main.info("This is an INFO message (console + file)")
logger_main.warning("This is a WARNING message")
logger_main.error("This is an ERROR message (also in errors.log)")

print("\n" + "=" * 60)
print("Testing service-specific loggers")
print("=" * 60)

logger_camera.debug("Camera: DEBUG level test (verbose)")
logger_camera.info("Camera: INFO level test")
logger_audio.info("Audio: INFO level test")

print("\n" + "=" * 60)
print("Testing exception logging")
print("=" * 60)

try:
    # Simulate an error
    result = 10 / 0
except Exception as e:
    logger_main.error("Error in calculation", exc_info=True)

print("\n" + "=" * 60)
print("Log Files Information")
print("=" * 60)

log_info = get_log_files_info()
for filename, info in log_info.items():
    print(f"  {filename}:")
    print(f"    Path: {info['path']}")
    print(f"    Size: {info['size_mb']} MB")

print("\n" + "=" * 60)
print("✅ Logging test completed!")
print("=" * 60)
print("\nCheck the following locations:")
print("  - Console output (INFO and above)")
print("  - logs/robot_ai.log (all messages)")
print("  - logs/errors.log (ERROR and above)")
