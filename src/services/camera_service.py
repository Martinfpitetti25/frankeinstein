"""
Camera and YOLO detection service
"""
import cv2
import numpy as np
from ultralytics import YOLO
from typing import Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CameraService:
    """Service for managing webcam capture and YOLO detection"""
    
    def __init__(self):
        self.camera = None
        self.camera_index = None
        self.is_running = False
        self.yolo_model = None
        self.model_loaded = False
        self.last_detections = []  # Store last detections for vision integration
        
        # Configurable parameters
        self.confidence = 0.5
        self.target_width = 640
        self.target_height = 480
        self.yolo_enabled = True
        
    def find_camera(self, max_cameras=5) -> Optional[int]:
        """
        Find available camera by testing indices
        
        Args:
            max_cameras: Maximum number of camera indices to test
            
        Returns:
            Camera index if found, None otherwise
        """
        logger.info("Searching for available cameras...")
        
        for index in range(max_cameras):
            cap = cv2.VideoCapture(index)
            if cap.isOpened():
                ret, _ = cap.read()
                cap.release()
                if ret:
                    logger.info(f"Camera found at index {index}")
                    return index
        
        logger.warning("No camera found")
        return None
    
    def start_camera(self, camera_index: Optional[int] = None) -> bool:
        """
        Start camera capture
        
        Args:
            camera_index: Specific camera index to use. If None, will search for camera
            
        Returns:
            True if camera started successfully, False otherwise
        """
        if self.is_running:
            logger.warning("Camera is already running")
            return True
        
        # Find camera if index not provided
        if camera_index is None:
            camera_index = self.find_camera()
            if camera_index is None:
                logger.error("Could not find any camera")
                return False
        
        self.camera_index = camera_index
        self.camera = cv2.VideoCapture(camera_index)
        
        if not self.camera.isOpened():
            logger.error(f"Could not open camera at index {camera_index}")
            return False
        
        # Set camera properties for better performance
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.camera.set(cv2.CAP_PROP_FPS, 30)
        
        self.is_running = True
        logger.info(f"Camera started successfully at index {camera_index}")
        return True
    
    def stop_camera(self):
        """Stop camera capture"""
        if self.camera is not None:
            self.camera.release()
            self.camera = None
        self.is_running = False
        logger.info("Camera stopped")
    
    def load_yolo_model(self, model_name: str = "yolov8n.pt") -> bool:
        """
        Load YOLO model
        
        Args:
            model_name: Name of the YOLO model to load (yolov8n.pt, yolov8s.pt, etc.)
            
        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            logger.info(f"Loading YOLO model: {model_name}")
            self.yolo_model = YOLO(model_name)
            self.model_loaded = True
            logger.info("YOLO model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading YOLO model: {str(e)}")
            self.model_loaded = False
            return False
    
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a frame from the camera
        
        Returns:
            Tuple of (success, frame)
        """
        if not self.is_running or self.camera is None:
            return False, None
        
        ret, frame = self.camera.read()
        return ret, frame
    
    def detect_objects(self, frame: np.ndarray, confidence: float = 0.5) -> Tuple[np.ndarray, list]:
        """
        Run YOLO detection on a frame
        
        Args:
            frame: Input image frame
            confidence: Minimum confidence threshold for detections
            
        Returns:
            Tuple of (annotated frame, list of detections)
        """
        if not self.model_loaded or self.yolo_model is None:
            return frame, []
        
        try:
            # Run inference
            results = self.yolo_model(frame, conf=confidence, verbose=False)
            
            # Get annotated frame
            annotated_frame = results[0].plot()
            
            # Extract detection information
            detections = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    detection = {
                        'class': result.names[int(box.cls[0])],
                        'confidence': float(box.conf[0]),
                        'bbox': box.xyxy[0].tolist()
                    }
                    detections.append(detection)
            
            # Store detections for vision integration
            self.last_detections = detections
            
            return annotated_frame, detections
        
        except Exception as e:
            logger.error(f"Error during detection: {str(e)}")
            return frame, []
    
    def get_frame_with_detection(self, confidence: float = None) -> Tuple[bool, Optional[np.ndarray], list]:
        """
        Get a frame and run detection on it
        
        Args:
            confidence: Minimum confidence threshold for detections (uses self.confidence if None)
            
        Returns:
            Tuple of (success, annotated frame, detections)
        """
        ret, frame = self.read_frame()
        if not ret or frame is None:
            return False, None, []
        
        # Use instance confidence if not provided
        if confidence is None:
            confidence = self.confidence
        
        # Only run detection if YOLO is enabled and model is loaded
        if self.yolo_enabled and self.model_loaded:
            annotated_frame, detections = self.detect_objects(frame, confidence)
            return True, annotated_frame, detections
        else:
            return True, frame, []
    
    def is_camera_available(self) -> bool:
        """Check if camera is available"""
        return self.find_camera() is not None
    
    def get_current_detections(self) -> list:
        """
        Get the list of current object detections
        
        Returns:
            List of detection dictionaries with 'class', 'confidence', and 'bbox'
        """
        return self.last_detections.copy()
    
    def get_detection_summary(self) -> str:
        """
        Get a human-readable summary of current detections
        
        Returns:
            String describing detected objects
        """
        if not self.last_detections:
            return "No hay objetos detectados en la imagen actualmente."
        
        # Count objects by class
        object_counts = {}
        for detection in self.last_detections:
            obj_class = detection['class']
            object_counts[obj_class] = object_counts.get(obj_class, 0) + 1
        
        # Build summary with better formatting
        summary_parts = []
        for obj_class, count in sorted(object_counts.items()):
            # Map common objects to Spanish
            obj_name = obj_class
            if count == 1:
                summary_parts.append(f"{count} {obj_name}")
            else:
                # Handle plural
                if obj_name.endswith('s'):
                    summary_parts.append(f"{count} {obj_name}")
                else:
                    summary_parts.append(f"{count} {obj_name}s")
        
        # Format: "Veo: X objetos (1 chair, 2 cups)" to avoid safety filters
        total = len(self.last_detections)
        if total == 1:
            return f"Detección de cámara: Veo {total} objeto - {summary_parts[0]}."
        else:
            return f"Detección de cámara: Veo {total} objetos - {', '.join(summary_parts)}."
    
    def __del__(self):
        """Cleanup on deletion"""
        self.stop_camera()
