"""
Servo motor control service for robot head movement
Supports multiple control methods: GPIO PWM, pigpio, and PCA9685
"""
from typing import Optional, Tuple
import time
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.logger import get_logger

logger = get_logger(__name__)


class ServoService:
    """Service for controlling servo motors on robot head"""
    
    # Servo control methods
    METHOD_GPIOZERO = "gpiozero"
    METHOD_PIGPIO = "pigpio"
    METHOD_PCA9685 = "pca9685"
    
    # PWM Safety margins for PCA9685 (microseconds)
    # Protege el PCA9685 y los servos evitando extremos absolutos
    PWM_MIN_SAFE = 650   # Mínimo seguro (estándar: 500μs)
    PWM_MAX_SAFE = 2000  # Máximo seguro (estándar: 2500μs)
    
    def __init__(self, method: str = METHOD_GPIOZERO):
        """
        Initialize servo service
        
        Args:
            method: Control method to use ("gpiozero", "pigpio", or "pca9685")
        """
        self.method = method
        self.servo_horizontal = None  # Pan servo (left-right)
        self.servo_vertical = None    # Tilt servo (up-down)
        self.is_initialized = False
        
        # Current positions (in degrees, 0-180)
        self.current_horizontal = 90  # Center
        self.current_vertical = 90    # Center
        
        # Movement limits (can be adjusted based on physical constraints)
        self.horizontal_min = 0
        self.horizontal_max = 180
        self.vertical_min = 0
        self.vertical_max = 180
        
        logger.info(f"ServoService initialized with method: {method}")
    
    def initialize(self, horizontal_pin: int = 17, vertical_pin: int = 27, 
                   i2c_channel: int = 0, i2c_address: int = 0x40) -> bool:
        """
        Initialize servo motors
        
        Args:
            horizontal_pin: GPIO pin for horizontal servo (pan)
            vertical_pin: GPIO pin for vertical servo (tilt)
            i2c_channel: I2C channel for PCA9685 (if using)
            i2c_address: I2C address for PCA9685 (default 0x40)
            
        Returns:
            True if initialization successful
        """
        try:
            if self.method == self.METHOD_GPIOZERO:
                from gpiozero import Servo
                from gpiozero.pins.pigpio import PiGPIOFactory
                
                # Use pigpio pin factory for better servo control
                try:
                    factory = PiGPIOFactory()
                    self.servo_horizontal = Servo(horizontal_pin, pin_factory=factory)
                    self.servo_vertical = Servo(vertical_pin, pin_factory=factory)
                except Exception:
                    # Fallback to default factory
                    self.servo_horizontal = Servo(horizontal_pin)
                    self.servo_vertical = Servo(vertical_pin)
                
                logger.info(f"gpiozero servos initialized on pins {horizontal_pin}, {vertical_pin}")
                
            elif self.method == self.METHOD_PIGPIO:
                import pigpio
                
                self.pi = pigpio.pi()
                if not self.pi.connected:
                    logger.error("Failed to connect to pigpio daemon")
                    return False
                
                self.horizontal_pin = horizontal_pin
                self.vertical_pin = vertical_pin
                
                logger.info(f"pigpio servos initialized on pins {horizontal_pin}, {vertical_pin}")
                
            elif self.method == self.METHOD_PCA9685:
                from adafruit_servokit import ServoKit
                
                self.kit = ServoKit(channels=16, address=i2c_address)
                
                # Aplicar márgenes de seguridad PWM a TODOS los canales
                for channel in range(16):
                    try:
                        self.kit.servo[channel].set_pulse_width_range(
                            self.PWM_MIN_SAFE, 
                            self.PWM_MAX_SAFE
                        )
                    except Exception as e:
                        # Algunos canales pueden no tener servos, ignorar errores
                        pass
                
                self.servo_horizontal = self.kit.servo[horizontal_pin]  # Channel number
                self.servo_vertical = self.kit.servo[vertical_pin]      # Channel number
                
                logger.info(f"PCA9685 servos initialized successfully (channels H:{horizontal_pin}, V:{vertical_pin})")
                logger.debug(f"PWM safety margins: {self.PWM_MIN_SAFE}-{self.PWM_MAX_SAFE}μs")
            
            # Move to center position
            self.move_to_center()
            self.is_initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize servos: {e}")
            return False
    
    def set_angle(self, servo: str, angle: float, smooth: bool = False, 
                  steps: int = 10, delay: float = 0.02) -> bool:
        """
        Set servo angle
        
        Args:
            servo: "horizontal" or "vertical"
            angle: Target angle in degrees (0-180)
            smooth: If True, move smoothly in steps
            steps: Number of steps for smooth movement
            delay: Delay between steps in seconds
            
        Returns:
            True if successful
        """
        if not self.is_initialized:
            logger.warning("Servos not initialized")
            return False
        
        # Clamp angle to limits
        if servo == "horizontal":
            angle = max(self.horizontal_min, min(self.horizontal_max, angle))
            current = self.current_horizontal
        elif servo == "vertical":
            angle = max(self.vertical_min, min(self.vertical_max, angle))
            current = self.current_vertical
        else:
            logger.error(f"Invalid servo: {servo}")
            return False
        
        try:
            if smooth:
                # Smooth movement in steps
                step_size = (angle - current) / steps
                for i in range(steps + 1):
                    intermediate_angle = current + (step_size * i)
                    self._set_servo_angle(servo, intermediate_angle)
                    time.sleep(delay)
            else:
                # Direct movement
                self._set_servo_angle(servo, angle)
            
            # Update current position
            if servo == "horizontal":
                self.current_horizontal = angle
                logger.debug(f"Servo horizontal moved to {angle}°")
            else:
                self.current_vertical = angle
                logger.debug(f"Servo vertical moved to {angle}°")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to set servo angle: {e}", exc_info=True)
            return False
    
    def _set_servo_angle(self, servo: str, angle: float):
        """Internal method to set servo to specific angle"""
        if self.method == self.METHOD_GPIOZERO:
            # gpiozero uses -1 to 1 range
            value = (angle - 90) / 90.0  # Convert 0-180 to -1 to 1
            if servo == "horizontal":
                self.servo_horizontal.value = value
            else:
                self.servo_vertical.value = value
                
        elif self.method == self.METHOD_PIGPIO:
            # pigpio uses pulse width (500-2500 microseconds)
            pulse_width = 500 + (angle / 180.0) * 2000
            if servo == "horizontal":
                self.pi.set_servo_pulsewidth(self.horizontal_pin, pulse_width)
            else:
                self.pi.set_servo_pulsewidth(self.vertical_pin, pulse_width)
                
        elif self.method == self.METHOD_PCA9685:
            # PCA9685 uses 0-180 degrees directly
            if servo == "horizontal":
                self.servo_horizontal.angle = angle
            else:
                self.servo_vertical.angle = angle
    
    def move_to_center(self):
        """Move both servos to center position (90 degrees)"""
        self.set_angle("horizontal", 90)
        self.set_angle("vertical", 90)
        logger.info("Servos moved to center position")
    
    def scan_horizontal(self, start_angle: float = 0, end_angle: float = 180, 
                       step: float = 10, delay: float = 0.1):
        """
        Scan horizontally between two angles
        
        Args:
            start_angle: Starting angle
            end_angle: Ending angle
            step: Step size in degrees
            delay: Delay between steps
        """
        angle = start_angle
        while angle <= end_angle:
            self.set_angle("horizontal", angle)
            time.sleep(delay)
            angle += step
    
    def scan_vertical(self, start_angle: float = 0, end_angle: float = 180, 
                     step: float = 10, delay: float = 0.1):
        """
        Scan vertically between two angles
        
        Args:
            start_angle: Starting angle
            end_angle: Ending angle
            step: Step size in degrees
            delay: Delay between steps
        """
        angle = start_angle
        while angle <= end_angle:
            self.set_angle("vertical", angle)
            time.sleep(delay)
            angle += step
    
    def get_position(self) -> Tuple[float, float]:
        """
        Get current servo positions
        
        Returns:
            Tuple of (horizontal_angle, vertical_angle)
        """
        return (self.current_horizontal, self.current_vertical)
    
    def cleanup(self):
        """Cleanup servo resources"""
        try:
            if self.method == self.METHOD_GPIOZERO:
                if self.servo_horizontal:
                    self.servo_horizontal.close()
                if self.servo_vertical:
                    self.servo_vertical.close()
                    
            elif self.method == self.METHOD_PIGPIO:
                if hasattr(self, 'pi') and self.pi.connected:
                    self.pi.set_servo_pulsewidth(self.horizontal_pin, 0)
                    self.pi.set_servo_pulsewidth(self.vertical_pin, 0)
                    self.pi.stop()
            
            self.is_initialized = False
            logger.info("Servo service cleaned up")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
