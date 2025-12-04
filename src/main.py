"""
AI Chat Assistant - Main Window
A PySide6 GUI application for chatting with ChatGPT and Ollama
"""
import sys
import os
import logging
import subprocess
import time
import re
import traceback
from pathlib import Path
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QLineEdit, QPushButton, QComboBox, QLabel, QMessageBox,
    QTabWidget, QSlider, QCheckBox, QGroupBox, QFormLayout, QSpinBox, QScrollArea, QFrame,
    QInputDialog
)
from PySide6.QtCore import Qt, QThread, Signal, Slot, QTimer, QEvent
from PySide6.QtGui import QFont, QTextCursor, QImage, QPixmap, QKeySequence, QShortcut
from dotenv import load_dotenv
import cv2
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))
from services import ChatGPTService, OllamaService, GroqService, CameraService, AudioService, ServoService

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NoScrollComboBox(QComboBox):
    """ComboBox that ignores mouse wheel events to prevent accidental changes while scrolling"""
    
    def wheelEvent(self, event):
        """Ignore wheel events when not focused"""
        if not self.hasFocus():
            event.ignore()
        else:
            super().wheelEvent(event)


class NoScrollSlider(QSlider):
    """Slider that ignores mouse wheel events to prevent accidental changes while scrolling"""
    
    def wheelEvent(self, event):
        """Ignore wheel events when not focused"""
        if not self.hasFocus():
            event.ignore()
        else:
            super().wheelEvent(event)


class NoScrollSpinBox(QSpinBox):
    """SpinBox that ignores mouse wheel events to prevent accidental changes while scrolling"""
    
    def wheelEvent(self, event):
        """Ignore wheel events when not focused"""
        if not self.hasFocus():
            event.ignore()
        else:
            super().wheelEvent(event)


class AudioWorker(QThread):
    """Worker thread for handling audio recognition without blocking UI"""
    
    audio_recognized = Signal(str)
    error_occurred = Signal(str)
    status_update = Signal(str)
    
    def __init__(self, audio_service, timeout=5):
        super().__init__()
        self.audio_service = audio_service
        self.timeout = timeout
    
    def run(self):
        """Listen to microphone and recognize speech"""
        try:
            self.status_update.emit("Listening...")
            success, text = self.audio_service.listen_once(timeout=self.timeout, phrase_time_limit=10)
            
            if success:
                self.audio_recognized.emit(text)
            else:
                self.error_occurred.emit(text)
        except Exception as e:
            self.error_occurred.emit(str(e))


class ChatWorker(QThread):
    """Worker thread for handling chat API calls without blocking UI"""
    
    response_ready = Signal(str)
    error_occurred = Signal(str)
    
    def __init__(self, service, message, conversation_history=None, vision_context=None):
        super().__init__()
        self.service = service
        self.message = message
        self.conversation_history = conversation_history or []
        self.vision_context = vision_context
    
    def run(self):
        """Execute the chat request in a separate thread"""
        try:
            response = self.service.send_message(
                self.message, 
                self.conversation_history,
                self.vision_context
            )
            self.response_ready.emit(response)
        except Exception as e:
            self.error_occurred.emit(str(e))


class VideoWorker(QThread):
    """Worker thread for handling video capture and YOLO detection"""
    
    frame_ready = Signal(QImage, list)  # Signal emits QImage and list of detections
    error_occurred = Signal(str)
    
    def __init__(self, camera_service, main_window):
        super().__init__()
        self.camera_service = camera_service
        self.main_window = main_window
        self.running = True
    
    def run(self):
        """Capture frames and run YOLO detection (optimized), render video only when preview is active"""
        frame_count = 0
        last_detection_time = time.time()
        cached_detections = []
        
        while self.running:
            try:
                frame_count += 1
                current_time = time.time()
                
                # Adaptive processing based on preview state
                if self.main_window.camera_preview_active:
                    # Preview ON: Process every frame for smooth video
                    success, frame, detections = self.camera_service.get_frame_with_detection()
                    process_delay = 33  # ~30 FPS
                else:
                    # Preview OFF: Process every 5th frame to save CPU (6 FPS for detections)
                    if frame_count % 5 != 0:
                        self.msleep(33)
                        continue
                    success, frame, detections = self.camera_service.get_frame_with_detection()
                    process_delay = 166  # ~6 FPS
                
                if success and frame is not None:
                    # Update detections if new ones found or cache is stale (>2s)
                    if detections or (current_time - last_detection_time > 2.0):
                        if detections:
                            cached_detections = detections
                            last_detection_time = current_time
                        
                        # Thread-safe update of shared detections
                        try:
                            self.main_window.detections_lock.acquire()
                            self.main_window.latest_detections = cached_detections
                        finally:
                            try:
                                self.main_window.detections_lock.release()
                            except:
                                pass
                    
                    # Only render video to UI if preview is active (saves CPU/RAM)
                    if self.main_window.camera_preview_active:
                        # Convert BGR to RGB (in-place when possible)
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        h, w, ch = rgb_frame.shape
                        bytes_per_line = ch * w
                        
                        # Convert to QImage (sharing data buffer to avoid copy)
                        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
                        self.frame_ready.emit(qt_image, cached_detections)
                    
                    self.msleep(process_delay)
                else:
                    self.msleep(100)  # Wait longer on error
                    
            except Exception as e:
                self.error_occurred.emit(str(e))
                self.msleep(100)
    
    def stop(self):
        """Stop the worker thread"""
        self.running = False


class ChatWindow(QMainWindow):
    """Main application window for the chat assistant"""
    
    def __init__(self):
        super().__init__()
        self.chatgpt_service = None
        self.ollama_service = None
        self.camera_service = None
        self.audio_service = None
        self.conversation_history = []
        self.current_worker = None
        self.video_worker = None
        self.audio_worker = None
        self.latest_detections = []
        # Protect shared detections with a lock to avoid race conditions
        import threading
        self.detections_lock = threading.Lock()
        self.voice_response_enabled = True  # Enable voice responses by default
        self.camera_preview_active = False  # Camera preview OFF by default to save resources
        self.total_messages = 0  # Message counter for chat tab
        
        # Configuration settings with default values
        self.listen_timeout = 5
        self.auto_vision_enabled = True
        
        self.init_services()
        self.init_prompts()
        self.init_ui()
        self.check_services_status()
        self.start_camera()
    
    def init_prompts(self):
        """Initialize predefined prompts"""
        self.prompts = {
            "Asistente General": "Eres un asistente de IA útil y amigable. Responde de manera clara, concisa y precisa a las preguntas del usuario.",
            
            "Compañero Natural": """Eres un compañero inteligente y observador. Tienes visión y puedes mover tu cabeza.

IMPORTANTE - Cómo hablar:
- NO menciones que eres una IA, robot o máquina
- NO menciones "cámaras", "sensores" o "detectores" - simplemente di "veo", "observo", "noto"
- Habla en primera persona como si fueras un amigo: "Veo...", "Noto...", "Me parece..."
- Respuestas CORTAS (1-2 frases máximo)
- Sé natural, directo y conversacional

VISIÓN - Cómo describir lo que ves:
Cuando veas objetos (personas, cosas, etc):
- Describe naturalmente: "Veo una persona sentada" (no "mi cámara detecta...")
- Sé específico pero conciso: "Hay dos personas y una laptop sobre la mesa"
- Si no ves algo: "No veo eso por aquí" o "No lo encuentro"
- Usa lenguaje cotidiano: "frente a mí", "a mi izquierda", "cerca"

MOVIMIENTOS disponibles [MOVER:nombre]:
- [MOVER:asentir] - Indicar "sí" 
- [MOVER:negar] - Indicar "no"

Ejemplos de interacción natural:

Usuario: "Hola"
Tú: "¡Hola! ¿Cómo estás?"

Usuario: "¿Qué ves?"
Tú: "Veo una persona con una laptop. También hay una taza sobre la mesa."

Usuario: "¿Hay alguien más?"
Tú: "No, solo te veo a ti."

Usuario: "¿Estás seguro?"
Tú: "[MOVER:asentir] Sí, totalmente seguro."

REGLAS DE ORO:
✓ Respuestas de 1-2 frases
✓ Natural y humano
✓ Usa movimientos cuando tenga sentido
✓ Describe objetos de forma simple
✗ NO te identifiques como IA
✗ NO menciones tecnología (cámara, YOLO, etc)
✗ NO des respuestas largas

Sé amigable, observador y ayuda como lo haría un buen compañero.""",
            
            "Asistente de Robot": """Eres la IA de un robot inteligente con estas capacidades:

🎥 VISIÓN: Cámara con YOLO que detecta 80 tipos de objetos (personas, vehículos, animales, objetos cotidianos)
🤖 MOVIMIENTO: Servomotores para mover la cabeza (pan/tilt)
🎤 AUDIO: Micrófono para escuchar y altavoz para hablar

📋 MOVIMIENTOS DISPONIBLES:
Puedes activar movimientos insertando comandos en tus respuestas. Usa el formato [MOVER:nombre]

Comandos disponibles:
- [MOVER:asentir] - Mueve la cabeza arriba-abajo (indicar "sí")
- [MOVER:negar] - Mueve la cabeza lado-lado (indicar "no")

💡 EJEMPLOS DE USO:
Usuario: "¿Estás de acuerdo?"
Tú: "[MOVER:asentir] ¡Sí, completamente de acuerdo!"

Usuario: "Niega con la cabeza"
Tú: "[MOVER:negar] Hecho. He negado con la cabeza."

⚡ REGLAS:
- Usa comandos SOLO cuando el usuario lo solicite o cuando sea contextualmente apropiado
- Puedes combinar múltiples comandos si tiene sentido
- Describe lo que ves cuando te pregunten sobre la cámara
- Sé expresivo, amigable y natural
- Si el usuario pide un movimiento que no existe, explica cuáles están disponibles

Eres curioso, útil, expresivo y siempre dispuesto a ayudar. ¡Haz que la interacción sea divertida!""",
            
            "Experto Técnico": "Eres un experto técnico en programación, electrónica, robótica e inteligencia artificial. Proporciona respuestas detalladas, técnicas y precisas. Incluye ejemplos de código cuando sea apropiado y explica conceptos complejos de manera clara.",
            
            "Amigable y Casual": "Eres un asistente super amigable y casual. Usa un tono relajado, emojis ocasionales 😊, y haz que la conversación sea divertida y entretenida. Sé cercano como un amigo que ayuda con cualquier cosa.",
            
            "Profesional Formal": "Eres un asistente profesional y formal. Utiliza lenguaje técnico apropiado, mantén un tono respetuoso y profesional, y proporciona respuestas bien estructuradas y fundamentadas.",
            
            "Custom 1": "",
            "Custom 2": "",
            "Custom 3": ""
        }
        self.current_prompt_name = "Compañero Natural"  # Default to the new natural prompt
        self.load_prompts_from_file()
    
    def setup_shortcuts(self):
        """Setup keyboard shortcuts for the application"""
        # Ctrl+Q: Emergency quit
        quit_shortcut = QShortcut(QKeySequence("Ctrl+Q"), self)
        quit_shortcut.activated.connect(self.emergency_close_shortcut)
        
        # Ctrl+Shift+Q: Force quit (even more aggressive)
        force_quit_shortcut = QShortcut(QKeySequence("Ctrl+Shift+Q"), self)
        force_quit_shortcut.activated.connect(self.force_quit)
        
        logger.info("Keyboard shortcuts configured:")
        logger.info("  - Ctrl+Q: Emergency close")
        logger.info("  - Ctrl+Shift+Q: Force quit (no cleanup)")
    
    def init_services(self):
        """Initialize chat, camera, and audio services"""
        load_dotenv()
        self.chatgpt_service = ChatGPTService()
        self.ollama_service = OllamaService()
        self.groq_service = GroqService()
        self.camera_service = CameraService()
        self.audio_service = AudioService()
        
        # Set initial system prompt for all LLM services
        if hasattr(self, 'prompts') and self.current_prompt_name in self.prompts:
            initial_prompt = self.prompts[self.current_prompt_name]
            self.chatgpt_service.system_prompt = initial_prompt
            self.ollama_service.system_prompt = initial_prompt
            self.groq_service.system_prompt = initial_prompt
        
        # Initialize servo service (PCA9685)
        self.servo_service = ServoService(method=ServoService.METHOD_PCA9685)
        try:
            # Initialize with PCA9685 channels 0 and 1
            if self.servo_service.initialize(horizontal_pin=0, vertical_pin=1, i2c_address=0x40):
                logger.info("✓ Servo service initialized successfully")
            else:
                logger.warning("⚠ Servo service initialization failed (servos may not be connected)")
        except Exception as e:
            logger.warning(f"⚠ Could not initialize servos: {e}")
            self.servo_service = None
    
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("Frankeinstein AI Assistant")
        self.setGeometry(100, 100, 1400, 800)
        
        # Apply modern stylesheet
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f7fa;
            }
            QWidget {
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QLabel {
                color: #2c3e50;
            }
            QGroupBox {
                border: 2px solid #e1e8ed;
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 10px;
                background-color: white;
                font-weight: 600;
                color: #2c3e50;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 8px;
                color: #34495e;
            }
        """)
        
        # Setup keyboard shortcuts
        self.setup_shortcuts()
        
        # Central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # Header section with modern title
        header_layout = QHBoxLayout()
        
        title_label = QLabel("🤖 Frankeinstein AI")
        title_font = QFont()
        title_font.setPointSize(18)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setStyleSheet("color: #2c3e50; letter-spacing: 1px;")
        header_layout.addWidget(title_label)
        
        header_layout.addStretch()
        
        main_layout.addLayout(header_layout)
        
        # Create tab widget with modern styling
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: 2px solid #e1e8ed;
                border-radius: 8px;
                background-color: white;
                padding: 10px;
            }
            QTabBar::tab {
                background-color: #ecf0f1;
                border: none;
                padding: 12px 24px;
                margin-right: 4px;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                color: #7f8c8d;
                font-weight: 600;
                font-size: 11pt;
            }
            QTabBar::tab:selected {
                background-color: white;
                color: #2c3e50;
            }
            QTabBar::tab:hover:!selected {
                background-color: #d5dbdb;
            }
        """)
        
        # Create Chat tab
        chat_tab = self.create_chat_tab()
        self.tab_widget.addTab(chat_tab, "💬 Chat")
        
        # Create Settings tab
        settings_tab = self.create_settings_tab()
        self.tab_widget.addTab(settings_tab, "⚙️ Configuración")
        
        # Create Demo tab
        demo_tab = self.create_demo_tab()
        self.tab_widget.addTab(demo_tab, "🎮 DEMO")
        
        main_layout.addWidget(self.tab_widget)
    
    def create_chat_tab(self):
        """Create the chat interface tab with camera feed"""
        chat_widget = QWidget()
        chat_main_layout = QVBoxLayout(chat_widget)
        chat_main_layout.setSpacing(15)
        chat_main_layout.setContentsMargins(15, 15, 15, 15)
        
        # ========== ENHANCED TOP BAR ==========
        top_bar = QWidget()
        top_bar.setStyleSheet("""
            QWidget {
                background-color: #ffffff;
                border-radius: 10px;
                padding: 10px;
            }
        """)
        top_bar_layout = QHBoxLayout(top_bar)
        top_bar_layout.setSpacing(15)
        top_bar_layout.setContentsMargins(15, 10, 15, 10)
        
        # Model selector with icon and better styling
        model_container = QWidget()
        model_layout = QHBoxLayout(model_container)
        model_layout.setSpacing(8)
        model_layout.setContentsMargins(0, 0, 0, 0)
        
        model_icon = QLabel("🤖")
        model_icon.setStyleSheet("font-size: 16pt;")
        model_layout.addWidget(model_icon)
        
        model_label = QLabel("Modelo:")
        model_label.setStyleSheet("font-weight: 600; color: #34495e; font-size: 10pt;")
        model_layout.addWidget(model_label)
        
        self.model_selector = NoScrollComboBox()
        self.model_selector.addItems(["Groq", "Ollama", "ChatGPT"])
        self.model_selector.setMinimumWidth(130)
        self.model_selector.setStyleSheet("""
            QComboBox {
                background-color: #ecf0f1;
                border: 2px solid #bdc3c7;
                border-radius: 6px;
                padding: 6px 12px;
                font-weight: 600;
                color: #2c3e50;
                font-size: 10pt;
            }
            QComboBox:hover {
                background-color: #d5dbdb;
                border-color: #95a5a6;
            }
            QComboBox::drop-down {
                border: none;
                padding-right: 10px;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid #7f8c8d;
                margin-right: 5px;
            }
        """)
        self.model_selector.currentTextChanged.connect(self.on_model_changed)
        model_layout.addWidget(self.model_selector)
        
        top_bar_layout.addWidget(model_container)
        
        # Status indicators
        top_bar_layout.addStretch()
        
        # AI Status indicator
        self.ai_status_indicator = QLabel("●")
        self.ai_status_indicator.setStyleSheet("color: #27ae60; font-size: 14pt;")
        self.ai_status_indicator.setToolTip("Estado de IA")
        top_bar_layout.addWidget(self.ai_status_indicator)
        
        self.status_label = QLabel("Listo")
        self.status_label.setStyleSheet("color: #27ae60; font-weight: 600; font-size: 10pt;")
        top_bar_layout.addWidget(self.status_label)
        
        # Separator
        separator = QLabel("|")
        separator.setStyleSheet("color: #bdc3c7; font-size: 12pt;")
        top_bar_layout.addWidget(separator)
        
        # Camera status with indicator
        self.camera_status_indicator = QLabel("●")
        self.camera_status_indicator.setStyleSheet("color: #f39c12; font-size: 14pt;")
        self.camera_status_indicator.setToolTip("Estado de cámara")
        top_bar_layout.addWidget(self.camera_status_indicator)
        
        self.camera_status_label = QLabel("Cámara: Iniciando...")
        self.camera_status_label.setStyleSheet("color: #7f8c8d; font-weight: 600; font-size: 10pt;")
        top_bar_layout.addWidget(self.camera_status_label)
        
        chat_main_layout.addWidget(top_bar)
        
        # ========== MAIN CONTENT: Two columns (Chat + Camera) ==========
        content_layout = QHBoxLayout()
        content_layout.setSpacing(15)
        
        # ========== LEFT COLUMN: Enhanced Chat Interface ==========
        chat_column = QWidget()
        chat_column.setStyleSheet("""
            QWidget {
                background-color: #ffffff;
                border-radius: 10px;
            }
        """)
        chat_layout = QVBoxLayout(chat_column)
        chat_layout.setSpacing(12)
        chat_layout.setContentsMargins(15, 15, 15, 15)
        
        # Chat header with icon
        chat_header = QHBoxLayout()
        chat_icon = QLabel("💬")
        chat_icon.setStyleSheet("font-size: 16pt;")
        chat_header.addWidget(chat_icon)
        
        chat_title = QLabel("Conversación")
        chat_title_font = QFont()
        chat_title_font.setPointSize(12)
        chat_title_font.setBold(True)
        chat_title.setFont(chat_title_font)
        chat_title.setStyleSheet("color: #2c3e50;")
        chat_header.addWidget(chat_title)
        
        chat_header.addStretch()
        
        # Conversation counter
        self.message_counter = QLabel("0 mensajes")
        self.message_counter.setStyleSheet("""
            background-color: #ecf0f1;
            color: #7f8c8d;
            padding: 4px 10px;
            border-radius: 10px;
            font-size: 9pt;
            font-weight: 600;
        """)
        chat_header.addWidget(self.message_counter)
        
        chat_layout.addLayout(chat_header)
        
        # Enhanced chat display area with scroll
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setFont(QFont("Segoe UI", 10))
        self.chat_display.setStyleSheet("""
            QTextEdit {
                background-color: #f8f9fa;
                border: 2px solid #e1e8ed;
                border-radius: 10px;
                padding: 15px;
                color: #2c3e50;
                selection-background-color: #3498db;
            }
            QScrollBar:vertical {
                background: #ecf0f1;
                width: 10px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical {
                background: #bdc3c7;
                border-radius: 5px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background: #95a5a6;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)
        chat_layout.addWidget(self.chat_display)
        
        # ========== ENHANCED INPUT AREA ==========
        input_container = QWidget()
        input_container.setStyleSheet("""
            QWidget {
                background-color: #f8f9fa;
                border-radius: 10px;
                padding: 8px;
            }
        """)
        input_main_layout = QVBoxLayout(input_container)
        input_main_layout.setSpacing(8)
        input_main_layout.setContentsMargins(8, 8, 8, 8)
        
        # Text input with send button
        text_input_layout = QHBoxLayout()
        text_input_layout.setSpacing(8)
        
        self.message_input = QLineEdit()
        self.message_input.setPlaceholderText("💭 Escribe tu mensaje aquí...")
        self.message_input.setFont(QFont("Segoe UI", 10))
        self.message_input.setMinimumHeight(42)
        self.message_input.setStyleSheet("""
            QLineEdit {
                padding: 12px 15px;
                border: 2px solid #d5dbdb;
                border-radius: 8px;
                background-color: #ffffff;
                color: #2c3e50;
                font-size: 10pt;
            }
            QLineEdit:focus {
                border: 2px solid #3498db;
                background-color: #fefefe;
            }
        """)
        self.message_input.returnPressed.connect(self.send_message)
        text_input_layout.addWidget(self.message_input)
        
        self.send_button = QPushButton("➤")
        self.send_button.setFont(QFont("Segoe UI", 14, QFont.Bold))
        self.send_button.setFixedSize(42, 42)
        self.send_button.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                border-radius: 21px;
                font-weight: 700;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #21618c;
                transform: scale(0.95);
            }
            QPushButton:disabled {
                background-color: #bdc3c7;
            }
        """)
        self.send_button.setToolTip("Enviar mensaje (Enter)")
        self.send_button.clicked.connect(self.send_message)
        text_input_layout.addWidget(self.send_button)
        
        input_main_layout.addLayout(text_input_layout)
        
        # Action buttons row
        actions_layout = QHBoxLayout()
        actions_layout.setSpacing(8)
        
        # Voice input button
        self.voice_button = QPushButton("🎤 Voz")
        self.voice_button.setFont(QFont("Segoe UI", 9))
        self.voice_button.setStyleSheet("""
            QPushButton {
                background-color: #9b59b6;
                color: white;
                padding: 8px 16px;
                border: none;
                border-radius: 6px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #8e44ad;
            }
            QPushButton:pressed {
                background-color: #7d3c98;
            }
            QPushButton:disabled {
                background-color: #bdc3c7;
            }
        """)
        self.voice_button.setToolTip("Entrada por voz")
        self.voice_button.clicked.connect(self.start_voice_input)
        actions_layout.addWidget(self.voice_button)
        
        # Voice response toggle button
        self.voice_response_button = QPushButton("🔊 Audio ON")
        self.voice_response_button.setFont(QFont("Segoe UI", 9))
        self.voice_response_button.setCheckable(True)
        self.voice_response_button.setChecked(True)
        self.voice_response_button.setStyleSheet("""
            QPushButton {
                color: white;
                padding: 8px 16px;
                border: none;
                border-radius: 6px;
                font-weight: 600;
            }
            QPushButton:checked {
                background-color: #27ae60;
            }
            QPushButton:!checked {
                background-color: #95a5a6;
            }
            QPushButton:hover:checked {
                background-color: #229954;
            }
            QPushButton:hover:!checked {
                background-color: #7f8c8d;
            }
        """)
        self.voice_response_button.setToolTip("Alternar respuesta por voz")
        self.voice_response_button.clicked.connect(self.toggle_voice_response)
        actions_layout.addWidget(self.voice_response_button)
        
        actions_layout.addStretch()
        
        self.clear_button = QPushButton("🗑️ Limpiar")
        self.clear_button.setFont(QFont("Segoe UI", 9))
        self.clear_button.setStyleSheet("""
            QPushButton {
                background-color: #ecf0f1;
                color: #7f8c8d;
                padding: 8px 16px;
                border: 2px solid #bdc3c7;
                border-radius: 6px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #e74c3c;
                color: white;
                border-color: #c0392b;
            }
        """)
        self.clear_button.setToolTip("Limpiar conversación")
        self.clear_button.clicked.connect(self.clear_chat)
        actions_layout.addWidget(self.clear_button)
        
        input_main_layout.addLayout(actions_layout)
        
        chat_layout.addWidget(input_container)
        
        # Initialize message counter
        self.total_messages = 0
        
        # Welcome message
        self.add_system_message(
            "¡Bienvenido a Frankeinstein AI! 🤖\n\n"
            "Selecciona un modelo y comienza a conversar.\n\n"
            "⌨️ Atajos de teclado:\n"
            "  • Enter - Enviar mensaje\n"
            "  • Ctrl+Q - Cerrar programa\n"
            "  • Ctrl+Shift+Q - Cierre de emergencia"
        )
        
        # Add chat column to content
        content_layout.addWidget(chat_column, stretch=3)
        
        # ========== RIGHT COLUMN: Enhanced Camera Feed ==========
        camera_column = QWidget()
        camera_column.setStyleSheet("""
            QWidget {
                background-color: #ffffff;
                border-radius: 10px;
            }
        """)
        camera_layout = QVBoxLayout(camera_column)
        camera_layout.setSpacing(12)
        camera_layout.setContentsMargins(15, 15, 15, 15)
        
        # Camera header
        camera_header = QHBoxLayout()
        camera_icon = QLabel("📹")
        camera_icon.setStyleSheet("font-size: 16pt;")
        camera_header.addWidget(camera_icon)
        
        camera_title = QLabel("Visión en Vivo")
        camera_title_font = QFont()
        camera_title_font.setPointSize(12)
        camera_title_font.setBold(True)
        camera_title.setFont(camera_title_font)
        camera_title.setStyleSheet("color: #2c3e50;")
        camera_header.addWidget(camera_title)
        
        camera_header.addStretch()
        
        # YOLO badge
        yolo_badge = QLabel("YOLO")
        yolo_badge.setStyleSheet("""
            background-color: #e67e22;
            color: white;
            padding: 4px 10px;
            border-radius: 10px;
            font-size: 8pt;
            font-weight: 700;
        """)
        yolo_badge.setToolTip("Detección de objetos YOLO activada")
        camera_header.addWidget(yolo_badge)
        
        camera_layout.addLayout(camera_header)
        
        # Video display with modern frame
        video_container = QWidget()
        video_container.setStyleSheet("""
            QWidget {
                background-color: #1c1c1c;
                border-radius: 10px;
            }
        """)
        video_container_layout = QVBoxLayout(video_container)
        video_container_layout.setContentsMargins(0, 0, 0, 0)
        
        self.video_label = QLabel()
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setMaximumSize(640, 480)
        self.video_label.setStyleSheet("""
            QLabel {
                background-color: #1c1c1c;
                border-radius: 10px;
                color: #7f8c8d;
                font-size: 12pt;
            }
        """)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setText("📷\n\nCámara desactivada\n\nPresiona el botón para activar")
        video_container_layout.addWidget(self.video_label)
        
        camera_layout.addWidget(video_container)
        
        # Camera controls
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(8)
        
        self.camera_toggle_btn = QPushButton("▶️ Activar Preview")
        self.camera_toggle_btn.setMinimumHeight(36)
        self.camera_toggle_btn.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                border: none;
                padding: 10px 16px;
                border-radius: 8px;
                font-weight: 600;
                font-size: 10pt;
            }
            QPushButton:hover {
                background-color: #229954;
            }
            QPushButton:pressed {
                background-color: #1e8449;
            }
        """)
        self.camera_toggle_btn.clicked.connect(self.toggle_camera_preview)
        self.camera_toggle_btn.setToolTip("Activar/Desactivar vista previa de cámara")
        controls_layout.addWidget(self.camera_toggle_btn)
        
        camera_layout.addLayout(controls_layout)
        
        # Detection info panel
        detection_panel = QWidget()
        detection_panel.setStyleSheet("""
            QWidget {
                background-color: #f8f9fa;
                border-radius: 8px;
                padding: 10px;
            }
        """)
        detection_layout = QVBoxLayout(detection_panel)
        detection_layout.setSpacing(5)
        detection_layout.setContentsMargins(10, 8, 10, 8)
        
        detection_header = QLabel("🎯 Detecciones")
        detection_header.setStyleSheet("font-weight: 600; color: #34495e; font-size: 9pt;")
        detection_layout.addWidget(detection_header)
        
        self.detection_label = QLabel("Ninguna detección")
        self.detection_label.setStyleSheet("""
            color: #7f8c8d;
            font-size: 9pt;
            padding: 4px 0;
        """)
        self.detection_label.setWordWrap(True)
        detection_layout.addWidget(self.detection_label)
        
        camera_layout.addWidget(detection_panel)
        
        camera_layout.addStretch()
        
        # Add camera column to content
        content_layout.addWidget(camera_column, stretch=2)
        
        # Add content layout to main layout
        chat_main_layout.addLayout(content_layout)
        
        return chat_widget
    
    def create_settings_tab(self):
        """Create the settings/configuration tab"""
        settings_widget = QWidget()
        settings_widget.setStyleSheet("""
            QWidget {
                background-color: #f5f7fa;
            }
        """)
        
        # Create scroll area for settings
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: #f5f7fa;
            }
            QScrollBar:vertical {
                background-color: #ecf0f1;
                width: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background-color: #95a5a6;
                border-radius: 6px;
                min-height: 30px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #7f8c8d;
            }
        """)
        
        # Container widget for scroll area
        container = QWidget()
        main_layout = QVBoxLayout(container)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(25, 25, 25, 25)
        
        # Title with modern styling
        title_label = QLabel("⚙️ Configuración")
        title_label.setStyleSheet("""
            QLabel {
                font-size: 22pt;
                font-weight: 700;
                color: #2c3e50;
                padding: 10px 0;
                border-bottom: 3px solid #3498db;
            }
        """)
        main_layout.addWidget(title_label)
        
        # ========== LLM CONFIGURATION ==========
        llm_group = QGroupBox("🤖 Modelos de IA")
        llm_group.setStyleSheet("""
            QGroupBox {
                font-weight: 600;
                font-size: 13pt;
                color: #2c3e50;
                background-color: white;
                border: 2px solid #e0e6ed;
                border-radius: 10px;
                margin-top: 12px;
                padding-top: 15px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 5px 10px;
            }
        """)
        llm_layout = QFormLayout()
        llm_layout.setSpacing(15)
        llm_layout.setContentsMargins(20, 20, 20, 20)
        
        # Ollama model selector
        self.ollama_model_combo = NoScrollComboBox()
        self.ollama_model_combo.addItems(["llama3.2:1b", "mistral:7b"])
        self.ollama_model_combo.setCurrentText("llama3.2:1b")
        self.ollama_model_combo.setStyleSheet("""
            QComboBox {
                padding: 8px 12px;
                border: 2px solid #e0e6ed;
                border-radius: 8px;
                background-color: #f8f9fa;
                font-size: 10pt;
            }
            QComboBox:hover {
                border-color: #3498db;
            }
            QComboBox::drop-down {
                border: none;
                width: 30px;
            }
        """)
        llm_layout.addRow("🦙 Modelo Ollama:", self.ollama_model_combo)
        
        # Groq model selector
        self.groq_model_combo = NoScrollComboBox()
        self.groq_model_combo.addItems([
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768",
            "gemma2-9b-it"
        ])
        self.groq_model_combo.setCurrentText("llama-3.3-70b-versatile")
        self.groq_model_combo.setStyleSheet("""
            QComboBox {
                padding: 8px 12px;
                border: 2px solid #e0e6ed;
                border-radius: 8px;
                background-color: #f8f9fa;
                font-size: 10pt;
            }
            QComboBox:hover {
                border-color: #3498db;
            }
            QComboBox::drop-down {
                border: none;
                width: 30px;
            }
        """)
        llm_layout.addRow("⚡ Modelo Groq:", self.groq_model_combo)
        
        # Groq API Key
        self.groq_api_key_input = QLineEdit()
        self.groq_api_key_input.setPlaceholderText("gsk_...")
        self.groq_api_key_input.setEchoMode(QLineEdit.Password)
        groq_api_layout = QHBoxLayout()
        groq_api_layout.addWidget(self.groq_api_key_input)
        show_groq_btn = QPushButton("👁")
        show_groq_btn.setMaximumWidth(40)
        show_groq_btn.setCheckable(True)
        show_groq_btn.toggled.connect(lambda checked: self.groq_api_key_input.setEchoMode(
            QLineEdit.Normal if checked else QLineEdit.Password
        ))
        groq_api_layout.addWidget(show_groq_btn)
        llm_layout.addRow("API Key Groq:", groq_api_layout)
        
        # ChatGPT API Key
        self.chatgpt_api_key_input = QLineEdit()
        self.chatgpt_api_key_input.setPlaceholderText("sk-proj-...")
        self.chatgpt_api_key_input.setEchoMode(QLineEdit.Password)
        chatgpt_api_layout = QHBoxLayout()
        chatgpt_api_layout.addWidget(self.chatgpt_api_key_input)
        show_chatgpt_btn = QPushButton("👁")
        show_chatgpt_btn.setMaximumWidth(40)
        show_chatgpt_btn.setCheckable(True)
        show_chatgpt_btn.toggled.connect(lambda checked: self.chatgpt_api_key_input.setEchoMode(
            QLineEdit.Normal if checked else QLineEdit.Password
        ))
        chatgpt_api_layout.addWidget(show_chatgpt_btn)
        llm_layout.addRow("API Key ChatGPT:", chatgpt_api_layout)
        
        # Save API Keys button
        save_keys_btn = QPushButton("💾 Guardar API Keys")
        save_keys_btn.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 8px;
                font-weight: 600;
                font-size: 11pt;
            }
            QPushButton:hover {
                background-color: #229954;
            }
            QPushButton:pressed {
                background-color: #1e8449;
            }
        """)
        save_keys_btn.clicked.connect(self.save_api_keys)
        llm_layout.addRow("", save_keys_btn)
        
        llm_group.setLayout(llm_layout)
        main_layout.addWidget(llm_group)
        
        # ========== CAMERA CONFIGURATION ==========
        camera_group = QGroupBox("📷 Cámara y Visión")
        camera_group.setStyleSheet("""
            QGroupBox {
                font-weight: 600;
                font-size: 13pt;
                color: #2c3e50;
                background-color: white;
                border: 2px solid #e0e6ed;
                border-radius: 10px;
                margin-top: 12px;
                padding-top: 15px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 5px 10px;
            }
        """)
        camera_layout = QFormLayout()
        camera_layout.setSpacing(15)
        camera_layout.setContentsMargins(20, 20, 20, 20)
        
        # YOLO confidence slider
        yolo_conf_layout = QHBoxLayout()
        self.yolo_confidence_slider = NoScrollSlider(Qt.Horizontal)
        self.yolo_confidence_slider.setMinimum(10)
        self.yolo_confidence_slider.setMaximum(90)
        self.yolo_confidence_slider.setValue(50)
        self.yolo_confidence_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 8px;
                background: #e0e6ed;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #e67e22;
                width: 20px;
                height: 20px;
                margin: -6px 0;
                border-radius: 10px;
            }
            QSlider::handle:horizontal:hover {
                background: #d35400;
            }
            QSlider::sub-page:horizontal {
                background: #e67e22;
                border-radius: 4px;
            }
        """)
        self.yolo_confidence_label = QLabel("0.50")
        self.yolo_confidence_label.setStyleSheet("""
            QLabel {
                font-weight: 600;
                font-size: 10pt;
                color: #2c3e50;
                min-width: 40px;
            }
        """)
        self.yolo_confidence_slider.valueChanged.connect(
            lambda v: self.yolo_confidence_label.setText(f"{v/100:.2f}")
        )
        yolo_conf_layout.addWidget(self.yolo_confidence_slider)
        yolo_conf_layout.addWidget(self.yolo_confidence_label)
        camera_layout.addRow("🎯 Confianza:", yolo_conf_layout)
        
        # Resolution selector
        self.resolution_combo = NoScrollComboBox()
        self.resolution_combo.addItems(["320x240", "640x480", "1280x720"])
        self.resolution_combo.setCurrentText("640x480")
        self.resolution_combo.setStyleSheet("""
            QComboBox {
                padding: 8px 12px;
                border: 2px solid #e0e6ed;
                border-radius: 8px;
                background-color: #f8f9fa;
                font-size: 10pt;
            }
            QComboBox:hover {
                border-color: #3498db;
            }
        """)
        camera_layout.addRow("📐 Resolución:", self.resolution_combo)
        
        # Enable YOLO checkbox
        self.enable_yolo_checkbox = QCheckBox("Habilitar detección YOLO")
        self.enable_yolo_checkbox.setChecked(True)
        self.enable_yolo_checkbox.setStyleSheet("""
            QCheckBox {
                font-size: 10pt;
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 20px;
                height: 20px;
                border-radius: 4px;
                border: 2px solid #e0e6ed;
            }
            QCheckBox::indicator:checked {
                background-color: #27ae60;
                border-color: #27ae60;
            }
        """)
        camera_layout.addRow("", self.enable_yolo_checkbox)
        
        camera_group.setLayout(camera_layout)
        main_layout.addWidget(camera_group)
        
        # ========== AUDIO CONFIGURATION ==========
        audio_group = QGroupBox("🎤 Audio y Voz")
        audio_group.setStyleSheet("""
            QGroupBox {
                font-weight: 600;
                font-size: 13pt;
                color: #2c3e50;
                background-color: white;
                border: 2px solid #e0e6ed;
                border-radius: 10px;
                margin-top: 12px;
                padding-top: 15px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 5px 10px;
            }
        """)
        audio_layout = QFormLayout()
        audio_layout.setSpacing(15)
        audio_layout.setContentsMargins(20, 20, 20, 20)
        
        # TTS Engine selector
        tts_engine_layout = QHBoxLayout()
        self.tts_engine_combo = NoScrollComboBox()
        self.tts_engine_combo.addItems(["pyttsx3 (Offline)", "Google TTS (Online)"])
        self.tts_engine_combo.setCurrentIndex(0)  # Default to pyttsx3
        self.tts_engine_combo.setStyleSheet("""
            QComboBox {
                padding: 8px 12px;
                border: 2px solid #e0e6ed;
                border-radius: 8px;
                background-color: #f8f9fa;
                font-size: 10pt;
            }
            QComboBox:hover {
                border-color: #3498db;
            }
        """)
        self.tts_engine_combo.currentIndexChanged.connect(self.on_tts_engine_changed)
        tts_engine_layout.addWidget(self.tts_engine_combo)
        
        # Status indicator for gTTS availability
        self.gtts_status_label = QLabel()
        if self.audio_service and self.audio_service.is_gtts_available():
            self.gtts_status_label.setText("✅")
            self.gtts_status_label.setStyleSheet("color: #27ae60; font-size: 14pt; font-weight: bold;")
            self.gtts_status_label.setToolTip("gTTS disponible")
        else:
            self.gtts_status_label.setText("❌")
            self.gtts_status_label.setStyleSheet("color: #e74c3c; font-size: 14pt; font-weight: bold;")
            self.gtts_status_label.setToolTip("gTTS no disponible")
            self.tts_engine_combo.setItemData(1, 0, Qt.UserRole - 1)  # Disable gTTS option
        tts_engine_layout.addWidget(self.gtts_status_label)
        
        audio_layout.addRow("🔊 Motor de voz:", tts_engine_layout)
        
        # Voice volume slider
        volume_layout = QHBoxLayout()
        self.voice_volume_slider = NoScrollSlider(Qt.Horizontal)
        self.voice_volume_slider.setMinimum(0)
        self.voice_volume_slider.setMaximum(100)
        self.voice_volume_slider.setValue(95)  # Optimized to 95%
        self.voice_volume_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 8px;
                background: #e0e6ed;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #3498db;
                width: 20px;
                height: 20px;
                margin: -6px 0;
                border-radius: 10px;
            }
            QSlider::handle:horizontal:hover {
                background: #2980b9;
            }
            QSlider::sub-page:horizontal {
                background: #3498db;
                border-radius: 4px;
            }
        """)
        self.voice_volume_label = QLabel("95%")
        self.voice_volume_label.setStyleSheet("""
            QLabel {
                font-weight: 600;
                font-size: 10pt;
                color: #2c3e50;
                min-width: 40px;
            }
        """)
        self.voice_volume_slider.valueChanged.connect(
            lambda v: self.voice_volume_label.setText(f"{v}%")
        )
        volume_layout.addWidget(self.voice_volume_slider)
        volume_layout.addWidget(self.voice_volume_label)
        audio_layout.addRow("🔉 Volumen:", volume_layout)
        
        # Voice speed slider (only for pyttsx3)
        speed_layout = QHBoxLayout()
        self.voice_speed_slider = NoScrollSlider(Qt.Horizontal)
        self.voice_speed_slider.setMinimum(50)
        self.voice_speed_slider.setMaximum(200)
        self.voice_speed_slider.setValue(130)  # Optimized to 130 (more natural)
        self.voice_speed_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 8px;
                background: #e0e6ed;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #9b59b6;
                width: 20px;
                height: 20px;
                margin: -6px 0;
                border-radius: 10px;
            }
            QSlider::handle:horizontal:hover {
                background: #8e44ad;
            }
            QSlider::sub-page:horizontal {
                background: #9b59b6;
                border-radius: 4px;
            }
        """)
        self.voice_speed_label = QLabel("130")
        self.voice_speed_label.setStyleSheet("""
            QLabel {
                font-weight: 600;
                font-size: 10pt;
                color: #2c3e50;
                min-width: 40px;
            }
        """)
        self.voice_speed_slider.valueChanged.connect(
            lambda v: self.voice_speed_label.setText(f"{v}")
        )
        speed_layout.addWidget(self.voice_speed_slider)
        speed_layout.addWidget(self.voice_speed_label)
        self.voice_speed_row_label = QLabel("⚡ Velocidad:")
        audio_layout.addRow(self.voice_speed_row_label, speed_layout)
        
        # Listen timeout
        self.listen_timeout_spin = NoScrollSpinBox()
        self.listen_timeout_spin.setMinimum(3)
        self.listen_timeout_spin.setMaximum(10)
        self.listen_timeout_spin.setValue(5)
        self.listen_timeout_spin.setSuffix(" segundos")
        self.listen_timeout_spin.setStyleSheet("""
            QSpinBox {
                padding: 8px 12px;
                border: 2px solid #e0e6ed;
                border-radius: 8px;
                background-color: #f8f9fa;
                font-size: 10pt;
            }
            QSpinBox:hover {
                border-color: #3498db;
            }
            QSpinBox::up-button, QSpinBox::down-button {
                border: none;
                width: 20px;
            }
        """)
        audio_layout.addRow("⏱️ Timeout escucha:", self.listen_timeout_spin)
        
        audio_group.setLayout(audio_layout)
        main_layout.addWidget(audio_group)
        
        # ========== VISION CONFIGURATION ==========
        vision_group = QGroupBox("👁 Configuración de Visión")
        vision_group.setStyleSheet("QGroupBox { font-weight: bold; font-size: 12pt; }")
        vision_layout = QFormLayout()
        vision_layout.setSpacing(10)
        
        # Auto-vision checkbox
        self.auto_vision_checkbox = QCheckBox("Detectar automáticamente palabras clave de visión")
        self.auto_vision_checkbox.setChecked(True)
        vision_layout.addRow("", self.auto_vision_checkbox)
        
        # Keywords info
        keywords_label = QLabel("Palabras clave: qué ves, describe, imagen, cámara, objetos, detecta, mira")
        keywords_label.setStyleSheet("color: #666; font-style: italic; font-size: 9pt;")
        keywords_label.setWordWrap(True)
        vision_layout.addRow("", keywords_label)
        
        vision_group.setLayout(vision_layout)
        main_layout.addWidget(vision_group)
        
        # ========== PROMPTS CONFIGURATION ==========
        prompts_group = QGroupBox("💬 Configuración de Prompts del Sistema")
        prompts_group.setStyleSheet("QGroupBox { font-weight: bold; font-size: 12pt; }")
        prompts_layout = QVBoxLayout()
        prompts_layout.setSpacing(10)
        
        # Info label
        info_label = QLabel("Define el comportamiento y personalidad de la IA mediante prompts del sistema.")
        info_label.setStyleSheet("color: #666; font-style: italic; font-size: 9pt;")
        info_label.setWordWrap(True)
        prompts_layout.addWidget(info_label)
        
        # Prompt selector
        selector_layout = QHBoxLayout()
        selector_label = QLabel("Prompt activo:")
        selector_layout.addWidget(selector_label)
        
        self.prompt_selector = QComboBox()
        self.prompt_selector.addItems([
            "Asistente General",
            "Asistente de Robot",
            "Experto Técnico",
            "Amigable y Casual",
            "Profesional Formal",
            "Custom 1",
            "Custom 2",
            "Custom 3"
        ])
        self.prompt_selector.currentTextChanged.connect(self.load_selected_prompt)
        selector_layout.addWidget(self.prompt_selector, 1)
        prompts_layout.addLayout(selector_layout)
        
        # Prompt editor
        self.prompt_editor = QTextEdit()
        self.prompt_editor.setPlaceholderText("Escribe el prompt del sistema aquí...")
        self.prompt_editor.setMinimumHeight(150)
        self.prompt_editor.setMaximumHeight(250)
        self.prompt_editor.setStyleSheet("""
            QTextEdit {
                border: 2px solid #ddd;
                border-radius: 5px;
                padding: 8px;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 10pt;
                background-color: #f9f9f9;
            }
            QTextEdit:focus {
                border: 2px solid #2196F3;
                background-color: white;
            }
        """)
        prompts_layout.addWidget(self.prompt_editor)
        
        # Character counter
        self.prompt_char_count = QLabel("0 caracteres")
        self.prompt_char_count.setStyleSheet("color: #666; font-size: 9pt;")
        self.prompt_editor.textChanged.connect(self.update_char_count)
        prompts_layout.addWidget(self.prompt_char_count)
        
        # Prompt action buttons
        prompt_buttons_layout = QHBoxLayout()
        
        save_prompt_btn = QPushButton("💾 Guardar Prompt")
        save_prompt_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                font-weight: bold;
                padding: 8px 15px;
                font-size: 10pt;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        save_prompt_btn.clicked.connect(self.save_current_prompt)
        prompt_buttons_layout.addWidget(save_prompt_btn)
        
        new_prompt_btn = QPushButton("➕ Nuevo Prompt")
        new_prompt_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 8px 15px;
                font-size: 10pt;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        new_prompt_btn.clicked.connect(self.create_new_prompt)
        prompt_buttons_layout.addWidget(new_prompt_btn)
        
        delete_prompt_btn = QPushButton("🗑 Eliminar Prompt")
        delete_prompt_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                font-weight: bold;
                padding: 8px 15px;
                font-size: 10pt;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
        """)
        delete_prompt_btn.clicked.connect(self.delete_current_prompt)
        prompt_buttons_layout.addWidget(delete_prompt_btn)
        
        prompt_buttons_layout.addStretch()
        prompts_layout.addLayout(prompt_buttons_layout)
        
        prompts_group.setLayout(prompts_layout)
        main_layout.addWidget(prompts_group)
        
        # ========== ACTION BUTTONS ==========
        buttons_layout = QHBoxLayout()
        buttons_layout.addStretch()
        
        # Apply button
        apply_btn = QPushButton("✓ Aplicar Cambios")
        apply_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 10px 20px;
                font-size: 11pt;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        apply_btn.clicked.connect(self.apply_settings)
        buttons_layout.addWidget(apply_btn)
        
        # Reset button
        reset_btn = QPushButton("↺ Restaurar Valores por Defecto")
        reset_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                font-weight: bold;
                padding: 10px 20px;
                font-size: 11pt;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
        """)
        reset_btn.clicked.connect(self.reset_settings)
        buttons_layout.addWidget(reset_btn)
        
        buttons_layout.addStretch()
        main_layout.addLayout(buttons_layout)
        
        main_layout.addStretch()
        
        # Set container as scroll area widget
        scroll.setWidget(container)
        
        # Create layout for settings_widget and add scroll area
        settings_layout = QVBoxLayout(settings_widget)
        settings_layout.setContentsMargins(0, 0, 0, 0)
        settings_layout.addWidget(scroll)
        
        # Load current settings from .env
        self.load_current_settings()
        
        # Load and set initial prompt
        self.load_selected_prompt(self.current_prompt_name)
        self.prompt_selector.setCurrentText(self.current_prompt_name)
        
        return settings_widget
    
    def load_prompts_from_file(self):
        """Load saved prompts from JSON file"""
        try:
            prompts_file = Path(__file__).parent.parent / "data" / "prompts.json"
            if prompts_file.exists():
                import json
                with open(prompts_file, 'r', encoding='utf-8') as f:
                    saved_prompts = json.load(f)
                    self.prompts.update(saved_prompts)
                    if "current_prompt" in saved_prompts:
                        self.current_prompt_name = saved_prompts["current_prompt"]
        except Exception as e:
            logger.error(f"Error loading prompts: {e}")
    
    def save_prompts_to_file(self):
        """Save prompts to JSON file"""
        try:
            import json
            data_dir = Path(__file__).parent.parent / "data"
            data_dir.mkdir(exist_ok=True)
            
            prompts_file = data_dir / "prompts.json"
            save_data = self.prompts.copy()
            save_data["current_prompt"] = self.current_prompt_name
            
            with open(prompts_file, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)
            
            logger.info("Prompts saved successfully")
        except Exception as e:
            logger.error(f"Error saving prompts: {e}")
    
    def load_selected_prompt(self, prompt_name):
        """Load the selected prompt into the editor"""
        if prompt_name in self.prompts:
            self.prompt_editor.setText(self.prompts[prompt_name])
            self.current_prompt_name = prompt_name
    
    def update_char_count(self):
        """Update character counter for prompt editor"""
        text = self.prompt_editor.toPlainText()
        count = len(text)
        self.prompt_char_count.setText(f"{count} caracteres")
    
    def save_current_prompt(self):
        """Save the current prompt"""
        prompt_name = self.prompt_selector.currentText()
        prompt_text = self.prompt_editor.toPlainText().strip()
        
        if not prompt_text:
            QMessageBox.warning(
                self,
                "Prompt Vacío",
                "El prompt no puede estar vacío."
            )
            return
        
        self.prompts[prompt_name] = prompt_text
        self.current_prompt_name = prompt_name
        self.save_prompts_to_file()
        
        # Update system prompt in active LLM service
        self.update_llm_system_prompt(prompt_text)
        
        QMessageBox.information(
            self,
            "Prompt Guardado",
            f"El prompt '{prompt_name}' se ha guardado correctamente.\n\n"
            f"Este prompt ahora se usará en las conversaciones con la IA."
        )
    
    def create_new_prompt(self):
        """Create a new custom prompt slot"""
        
        name, ok = QInputDialog.getText(
            self,
            "Nuevo Prompt",
            "Nombre del nuevo prompt:"
        )
        
        if ok and name:
            if name in self.prompts:
                QMessageBox.warning(
                    self,
                    "Nombre Duplicado",
                    f"Ya existe un prompt con el nombre '{name}'."
                )
                return
            
            self.prompts[name] = ""
            self.prompt_selector.addItem(name)
            self.prompt_selector.setCurrentText(name)
            self.save_prompts_to_file()
    
    def delete_current_prompt(self):
        """Delete the current custom prompt"""
        prompt_name = self.prompt_selector.currentText()
        
        # Prevent deleting predefined prompts
        predefined = ["Asistente General", "Asistente de Robot", "Experto Técnico", 
                     "Amigable y Casual", "Profesional Formal"]
        
        if prompt_name in predefined:
            QMessageBox.warning(
                self,
                "No se puede eliminar",
                "No puedes eliminar prompts predefinidos.\n\n"
                "Solo puedes eliminar prompts personalizados."
            )
            return
        
        reply = QMessageBox.question(
            self,
            "Confirmar Eliminación",
            f"¿Estás seguro de que quieres eliminar el prompt '{prompt_name}'?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            del self.prompts[prompt_name]
            index = self.prompt_selector.findText(prompt_name)
            if index >= 0:
                self.prompt_selector.removeItem(index)
            
            # Switch to default prompt
            self.prompt_selector.setCurrentText("Asistente de Robot")
            self.save_prompts_to_file()
            
            QMessageBox.information(
                self,
                "Prompt Eliminado",
                f"El prompt '{prompt_name}' ha sido eliminado."
            )
    
    def update_llm_system_prompt(self, prompt_text):
        """Update system prompt in LLM services"""
        try:
            if hasattr(self.chatgpt_service, 'system_prompt'):
                self.chatgpt_service.system_prompt = prompt_text
            if hasattr(self.ollama_service, 'system_prompt'):
                self.ollama_service.system_prompt = prompt_text
            if hasattr(self.groq_service, 'system_prompt'):
                self.groq_service.system_prompt = prompt_text
            logger.info(f"System prompt updated: {prompt_text[:50]}...")
        except Exception as e:
            logger.error(f"Error updating system prompt: {e}")
    
    def load_current_settings(self):
        """Load current settings from .env file"""
        try:
            # Load API keys
            groq_key = os.getenv("GROQ_API_KEY", "")
            chatgpt_key = os.getenv("OPENAI_API_KEY", "")
            
            if groq_key:
                self.groq_api_key_input.setText(groq_key)
            if chatgpt_key:
                self.chatgpt_api_key_input.setText(chatgpt_key)
                
        except Exception as e:
            print(f"Error loading settings: {e}")
    
    def save_api_keys(self):
        """Save API keys to .env file"""
        try:
            env_path = Path(__file__).parent.parent / ".env"
            
            # Read current .env content
            env_content = {}
            if env_path.exists():
                with open(env_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            key, value = line.split('=', 1)
                            env_content[key.strip()] = value.strip()
            
            # Update API keys
            groq_key = self.groq_api_key_input.text().strip()
            chatgpt_key = self.chatgpt_api_key_input.text().strip()
            
            if groq_key:
                env_content['GROQ_API_KEY'] = groq_key
            if chatgpt_key:
                env_content['OPENAI_API_KEY'] = chatgpt_key
            
            # Ensure OLLAMA_BASE_URL exists
            if 'OLLAMA_BASE_URL' not in env_content:
                env_content['OLLAMA_BASE_URL'] = 'http://localhost:11434'
            
            # Write back to .env
            with open(env_path, 'w') as f:
                for key, value in env_content.items():
                    f.write(f"{key}={value}\n")
            
            QMessageBox.information(
                self,
                "API Keys Guardadas",
                "Las API keys se han guardado correctamente.\n\n"
                "Reinicia la aplicación para que los cambios tomen efecto."
            )
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Error al guardar las API keys:\n{str(e)}"
            )
    
    def apply_settings(self):
        """Apply configuration changes with validation"""
        try:
            # Apply Ollama model
            ollama_model = self.ollama_model_combo.currentText()
            if hasattr(self.ollama_service, 'model'):
                self.ollama_service.model = ollama_model
            
            # Apply Groq model
            groq_model = self.groq_model_combo.currentText()
            if hasattr(self.groq_service, 'model'):
                self.groq_service.model = groq_model
            
            # Validate and apply YOLO confidence (0.1 - 0.9)
            yolo_confidence_raw = self.yolo_confidence_slider.value() / 100.0
            yolo_confidence = max(0.1, min(0.9, yolo_confidence_raw))
            
            # Warn if confidence is very low
            if yolo_confidence < 0.3:
                reply = QMessageBox.question(
                    self,
                    "⚠️ Confianza Baja",
                    f"Una confianza de {yolo_confidence:.2f} puede generar muchas detecciones falsas.\n\n"
                    "¿Continuar de todos modos?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )
                if reply == QMessageBox.No:
                    return
            
            if hasattr(self, 'camera_service') and self.camera_service:
                self.camera_service.confidence = yolo_confidence
            
            # Apply resolution
            resolution = self.resolution_combo.currentText()
            if hasattr(self, 'camera_service') and self.camera_service:
                try:
                    width, height = map(int, resolution.split('x'))
                    # Validate resolution is sane
                    if width < 160 or height < 120 or width > 1920 or height > 1080:
                        raise ValueError(f"Resolución fuera de rango: {width}x{height}")
                    # Store for next camera restart
                    self.camera_service.target_width = width
                    self.camera_service.target_height = height
                except ValueError as ve:
                    QMessageBox.warning(
                        self,
                        "Resolución Inválida",
                        f"La resolución '{resolution}' no es válida.\n{ve}"
                    )
                    return
            
            # Apply YOLO enable/disable
            yolo_enabled = self.enable_yolo_checkbox.isChecked()
            if hasattr(self, 'camera_service') and self.camera_service:
                self.camera_service.yolo_enabled = yolo_enabled
            
            # Validate and apply audio settings
            if hasattr(self, 'audio_service') and self.audio_service:
                volume_raw = self.voice_volume_slider.value() / 100.0
                volume = max(0.0, min(1.0, volume_raw))  # Clamp 0-1
                
                speed_raw = self.voice_speed_slider.value()
                speed = max(50, min(300, speed_raw))  # Clamp 50-300 WPM
                
                # Warn if speed is extreme
                if speed > 200:
                    reply = QMessageBox.question(
                        self,
                        "⚠️ Velocidad Alta",
                        f"Velocidad de {speed} WPM puede ser difícil de entender.\n\n"
                        "¿Continuar?",
                        QMessageBox.Yes | QMessageBox.No,
                        QMessageBox.No
                    )
                    if reply == QMessageBox.No:
                        return
                
                # Update TTS engine properties
                try:
                    if hasattr(self.audio_service, 'tts_engine') and self.audio_service.tts_engine:
                        self.audio_service.tts_engine.setProperty('volume', volume)
                        self.audio_service.tts_engine.setProperty('rate', speed)
                except Exception as e:
                    logger.warning(f"No se pudo actualizar propiedades de audio: {e}")
            
            # Store listen timeout for next voice input (validate 3-10 seconds)
            listen_timeout_raw = self.listen_timeout_spin.value()
            self.listen_timeout = max(3, min(10, listen_timeout_raw))
            
            # Store auto-vision setting
            self.auto_vision_enabled = self.auto_vision_checkbox.isChecked()
            
            QMessageBox.information(
                self,
                "✅ Configuración Aplicada",
                "Los cambios se han aplicado correctamente.\n\n"
                f"• Modelo Ollama: {ollama_model}\n"
                f"• Modelo Groq: {groq_model}\n"
                f"• Confianza YOLO: {yolo_confidence:.2f}\n"
                f"• Resolución: {resolution}\n"
                f"• YOLO: {'Habilitado' if yolo_enabled else 'Deshabilitado'}\n"
                f"• Volumen: {int(volume*100)}%\n"
                f"• Velocidad: {speed} WPM\n"
                f"• Timeout escucha: {self.listen_timeout}s\n"
                f"• Auto-visión: {'Habilitada' if self.auto_vision_enabled else 'Deshabilitada'}"
            )
            
        except Exception as e:
            logger.error(f"Error applying settings: {e}", exc_info=True)
            QMessageBox.critical(
                self,
                "❌ Error",
                f"Error al aplicar la configuración:\n{str(e)}"
            )
    
    def reset_settings(self):
        """Reset all settings to default values"""
        reply = QMessageBox.question(
            self,
            "Restaurar Valores por Defecto",
            "¿Estás seguro de que quieres restaurar todos los valores a sus configuraciones por defecto?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # Reset LLM settings
            self.ollama_model_combo.setCurrentText("llama3.2:1b")
            self.groq_model_combo.setCurrentText("llama-3.3-70b-versatile")
            
            # Reset camera settings
            self.yolo_confidence_slider.setValue(50)
            self.resolution_combo.setCurrentText("640x480")
            self.enable_yolo_checkbox.setChecked(True)
            
            # Reset audio settings
            self.voice_volume_slider.setValue(80)
            self.voice_speed_slider.setValue(150)
            self.listen_timeout_spin.setValue(5)
            
            # Reset vision settings
            self.auto_vision_checkbox.setChecked(True)
            
            QMessageBox.information(
                self,
                "Valores Restaurados",
                "✓ Todos los valores han sido restaurados a sus configuraciones por defecto.\n\n"
                "Haz clic en 'Aplicar Cambios' para confirmar."
            )
    
    def create_demo_tab(self):
        """Create the demo tab"""
        demo_widget = QWidget()
        demo_layout = QVBoxLayout(demo_widget)
        demo_layout.setSpacing(20)
        demo_layout.setContentsMargins(20, 20, 20, 20)
        
        # Title
        title_label = QLabel("🎮 DEMO")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title_label.setFont(title_font)
        demo_layout.addWidget(title_label)
        
        # Main horizontal layout for two sections
        sections_layout = QHBoxLayout()
        sections_layout.setSpacing(20)
        
        # ========== MOVIMIENTO SECTION ==========
        movimiento_group = QGroupBox("🚗 MOVIMIENTO")
        movimiento_group.setStyleSheet("QGroupBox { font-weight: bold; font-size: 12pt; }")
        movimiento_layout = QVBoxLayout()
        movimiento_layout.setSpacing(10)
        
        # Button: DEMO Completo
        btn_demo = QPushButton("🎬 DEMO Completo")
        btn_demo.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                font-weight: 600;
                padding: 18px;
                font-size: 12pt;
                border-radius: 10px;
                border: none;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #21618c;
            }
        """)
        btn_demo.clicked.connect(self.demo_movimiento)
        movimiento_layout.addWidget(btn_demo)
        
        # Separator
        separator1 = QLabel("")
        separator1.setStyleSheet("border-top: 2px solid #e1e8ed; margin: 10px 0;")
        movimiento_layout.addWidget(separator1)
        
        # Label: Movimientos Básicos
        predefined_label = QLabel("✨ Movimientos Básicos")
        predefined_label.setStyleSheet("""
            font-weight: 600;
            font-size: 11pt;
            margin-top: 8px;
            color: #34495e;
        """)
        movimiento_layout.addWidget(predefined_label)
        
        # Button: Asentir (Sí)
        btn_asentir = QPushButton("✅ Asentir (Sí)")
        btn_asentir.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                font-weight: 600;
                padding: 14px;
                font-size: 10pt;
                border-radius: 8px;
                border: none;
            }
            QPushButton:hover {
                background-color: #229954;
            }
            QPushButton:pressed {
                background-color: #1e8449;
            }
        """)
        btn_asentir.clicked.connect(self.movimiento_asentir)
        movimiento_layout.addWidget(btn_asentir)
        
        # Button: Negar (No)
        btn_negar = QPushButton("❌ Negar (No)")
        btn_negar.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                font-weight: 600;
                padding: 14px;
                font-size: 10pt;
                border-radius: 8px;
                border: none;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
            QPushButton:pressed {
                background-color: #a93226;
            }
        """)
        btn_negar.clicked.connect(self.movimiento_negar)
        movimiento_layout.addWidget(btn_negar)
        
        # Separator
        separator2 = QLabel("")
        separator2.setStyleSheet("border-top: 2px solid #e1e8ed; margin: 10px 0;")
        movimiento_layout.addWidget(separator2)
        
        # Button: Volver al Origen
        btn_volver_origen = QPushButton("🎯 Posición Inicial")
        btn_volver_origen.setStyleSheet("""
            QPushButton {
                background-color: #95a5a6;
                color: white;
                font-weight: 600;
                padding: 14px;
                font-size: 10pt;
                border-radius: 8px;
                border: none;
            }
            QPushButton:hover {
                background-color: #7f8c8d;
            }
            QPushButton:pressed {
                background-color: #5d6d7e;
            }
        """)
        btn_volver_origen.clicked.connect(self.volver_al_origen)
        movimiento_layout.addWidget(btn_volver_origen)
        
        movimiento_layout.addStretch()
        movimiento_group.setLayout(movimiento_layout)
        sections_layout.addWidget(movimiento_group)
        
        # ========== INTELIGENCIA ARTIFICIAL SECTION ==========
        ia_group = QGroupBox("🤖 INTELIGENCIA ARTIFICIAL")
        ia_group.setStyleSheet("QGroupBox { font-weight: bold; font-size: 12pt; }")
        ia_layout = QVBoxLayout()
        ia_layout.setSpacing(10)
        
        # Button: Buscar Objeto
        btn_buscar_objeto = QPushButton("🔍 Buscar Objeto")
        btn_buscar_objeto.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                font-weight: 600;
                padding: 18px;
                font-size: 12pt;
                border-radius: 10px;
                border: none;
            }
            QPushButton:hover {
                background-color: #229954;
            }
            QPushButton:pressed {
                background-color: #1e8449;
            }
        """)
        btn_buscar_objeto.clicked.connect(self.demo_buscar_objeto)
        ia_layout.addWidget(btn_buscar_objeto)
        
        # Button: Seguir Persona (Face Tracking Dual)
        btn_seguir_persona = QPushButton("👤 Seguimiento Facial")
        btn_seguir_persona.setStyleSheet("""
            QPushButton {
                background-color: #9b59b6;
                color: white;
                font-weight: 600;
                padding: 18px;
                font-size: 12pt;
                border-radius: 10px;
                border: none;
            }
            QPushButton:hover {
                background-color: #8e44ad;
            }
            QPushButton:pressed {
                background-color: #7d3c98;
            }
        """)
        btn_seguir_persona.clicked.connect(self.demo_seguir_persona)
        ia_layout.addWidget(btn_seguir_persona)
        
        # Button: Demo Completa IA
        btn_demo_completa = QPushButton("🎯 Demo Completa IA")
        btn_demo_completa.setStyleSheet("""
            QPushButton {
                background-color: #e67e22;
                color: white;
                font-weight: 600;
                padding: 18px;
                font-size: 12pt;
                border-radius: 10px;
                border: none;
            }
            QPushButton:hover {
                background-color: #d35400;
            }
            QPushButton:pressed {
                background-color: #ba4a00;
            }
        """)
        btn_demo_completa.clicked.connect(self.demo_completa_ia)
        ia_layout.addWidget(btn_demo_completa)
        
        # Separator
        separator3 = QLabel("")
        separator3.setStyleSheet("border-top: 2px solid #e1e8ed; margin: 15px 0;")
        ia_layout.addWidget(separator3)
        
        # Button: Cerrar Programa (Emergency)
        btn_cerrar_programa = QPushButton("⛔ CERRAR PROGRAMA")
        btn_cerrar_programa.setStyleSheet("""
            QPushButton {
                background-color: #c0392b;
                color: white;
                font-weight: 600;
                padding: 16px;
                font-size: 11pt;
                border-radius: 8px;
                border: 2px solid #a93226;
            }
            QPushButton:hover {
                background-color: #a93226;
            }
            QPushButton:pressed {
                background-color: #922b21;
            }
        """)
        btn_cerrar_programa.clicked.connect(self.emergency_close)
        ia_layout.addWidget(btn_cerrar_programa)
        
        ia_layout.addStretch()
        ia_group.setLayout(ia_layout)
        sections_layout.addWidget(ia_group)
        
        # Add sections to main layout
        demo_layout.addLayout(sections_layout)
        
        return demo_widget
    

    
    def demo_movimiento(self):
        """Execute full movement demo"""
        if not self.servo_service or not self.servo_service.is_initialized:
            QMessageBox.warning(
                self,
                "Servos No Disponibles",
                "⚠️ Los servomotores no están inicializados.\n\n"
                "Asegúrate de que:\n"
                "• El PCA9685 esté conectado al I2C\n"
                "• Los servos estén en los canales 0 y 1\n"
                "• La fuente de 5V esté conectada"
            )
            return
        
        try:
            import time
            
            # Show status
            self.add_system_message("🎬 Iniciando demostración completa de movimiento...")
            
            # Step 1: Center position
            self.add_system_message("1️⃣ Posición central...")
            self.servo_service.move_to_center()
            time.sleep(1)
            
            # Step 2: Horizontal scan
            self.add_system_message("2️⃣ Escaneo horizontal (izquierda → derecha)...")
            self.servo_service.scan_horizontal(0, 180, step=10, delay=0.1)
            time.sleep(0.5)
            
            # Step 3: Return to center
            self.servo_service.set_angle("horizontal", 90, smooth=True)
            time.sleep(0.5)
            
            # Step 4: Vertical scan
            self.add_system_message("3️⃣ Escaneo vertical (arriba → abajo)...")
            self.servo_service.scan_vertical(0, 180, step=10, delay=0.1)
            time.sleep(0.5)
            
            # Step 5: Return to center
            self.add_system_message("4️⃣ Retornando a posición central...")
            self.servo_service.move_to_center()
            time.sleep(1)
            
            # Step 6: Figure-8 pattern
            self.add_system_message("5️⃣ Patrón de movimiento combinado...")
            angles = [(45, 45), (135, 45), (135, 135), (45, 135), (90, 90)]
            for h, v in angles:
                self.servo_service.set_angle("horizontal", h, smooth=True, steps=5)
                self.servo_service.set_angle("vertical", v, smooth=True, steps=5)
                time.sleep(0.3)
            
            self.add_system_message("✓ ¡Demo de movimiento completada!")
            logger.info("Full movement demo completed")
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Error ejecutando demo de movimiento:\n{str(e)}"
            )
            logger.error(f"Error in movement demo: {e}")
    

    
    def movimiento_asentir(self):
        """Predefined movement: Nod yes - Ejecuta TEST_ASENTIR.py"""
        try:
            import subprocess
            import os
            
            self.add_system_message("✅ Ejecutando movimiento de asentir...")
            
            # Ruta al archivo TEST_ASENTIR.py
            script_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "TEST_ASENTIR.py")
            
            # Ruta al intérprete de Python del entorno virtual
            python_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "env", "bin", "python")
            
            # Ejecutar el script en background
            subprocess.Popen(
                [python_path, script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            self.add_system_message("✓ Movimiento de asentir iniciado")
            
        except Exception as e:
            logger.error(f"Error in asentir: {e}")
            self.add_system_message(f"❌ Error al ejecutar asentir: {e}")
    
    def movimiento_negar(self):
        """Predefined movement: Shake no - Ejecuta TEST_NEGAR.py"""
        try:
            import subprocess
            import os
            
            self.add_system_message("❌ Ejecutando movimiento de negar...")
            
            # Ruta al archivo TEST_NEGAR.py
            script_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "TEST_NEGAR.py")
            
            # Ruta al intérprete de Python del entorno virtual
            python_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "env", "bin", "python")
            
            # Ejecutar el script en background
            subprocess.Popen(
                [python_path, script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            self.add_system_message("✓ Movimiento de negar iniciado")
            
        except Exception as e:
            logger.error(f"Error in negar: {e}")
            self.add_system_message(f"❌ Error al ejecutar negar: {e}")
    
    def volver_al_origen(self):
        """Return servos to center position and stop all movements"""
        try:
            import subprocess
            import time
            
            self.add_system_message("🛑 Deteniendo todos los movimientos...")
            
            # Detener todos los procesos de movimiento de forma robusta
            for pattern in [
                "TEST_ASENTIR.py", "TEST_NEGAR.py", "TEST_ROLL.py",
                "test_boca.py", "test_seguir_rostro.py", "TEST_SEGUIMIENTO_FINAL.PY"
            ]:
                try:
                    subprocess.run(["pkill", "-9", "-f", pattern], check=False)
                except Exception:
                    pass
            
            # Pequeña pausa para asegurar que los procesos se detengan
            time.sleep(0.2)
            
            self.add_system_message("🎯 Volviendo al origen (centro)...")
            
            # Establecer posiciones de centro específicas para cada servo
            # Pin 15 = 180° (hombro derecho)
            # Pin 14 = 60°  (pitch control)
            # Pin 13 = 135° (cuello/yaw)
            # Pin 12 = 180° (hombro izquierdo)
            # Pin 5  = 50°  (boca)
            
            if self.servo_service and self.servo_service.is_initialized:
                try:
                    # Usar ServoKit directamente parac control preciso
                    if hasattr(self.servo_service, 'kit'):
                        self.servo_service.kit.servo[15].angle = 180  # Hombro derecho
                        time.sleep(0.05)
                        self.servo_service.kit.servo[14].angle =  50# Pitch control
                        time.sleep(0.05)
                        self.servo_service.kit.servo[13].angle = 160  # Cuello
                        time.sleep(0.05)
                        self.servo_service.kit.servo[12].angle = 180  # Hombro izquierdo
                        time.sleep(0.05)
                        self.servo_service.kit.servo[5].angle = 50    # Boca
                        
                        self.add_system_message("✓ Servos en posición de origen (15:180°, 14:60°, 13:135°, 12:180°, 5:50°)")
                    else:
                        self.add_system_message("⚠️ Servos detenidos pero no se pudo centrar (ServoKit no disponible)")
                except Exception as e:
                    logger.error(f"Error setting servo positions: {e}")
                    self.add_system_message(f"⚠️ Procesos detenidos pero error al centrar servos: {e}")
            else:
                self.add_system_message("⚠️ Procesos detenidos (servos no inicializados)")
            
            # Reiniciar la cámara si estaba liberada (con verificación)
            if hasattr(self, 'camera_service') and self.camera_service:
                try:
                    self.add_system_message("📷 Reiniciando cámara del servicio principal...")
                    # Asegurar que está detenida antes de iniciar
                    self.camera_service.stop_camera()
                    time.sleep(0.2)
                    started = self.camera_service.start_camera()
                    time.sleep(0.3)
                    if not started:
                        self.add_system_message("⚠️ No se pudo iniciar la cámara (verifica conexión)")
                    else:
                        # Si YOLO estaba habilitado, asegurar modelo cargado
                        if getattr(self.camera_service, 'yolo_enabled', True) and not getattr(self.camera_service, 'model_loaded', False):
                            self.camera_service.load_yolo_model("yolov8n.pt")
                    self.add_system_message("✓ Servicio de cámara restaurado")
                except Exception as e:
                    self.add_system_message(f"⚠️ No se pudo reiniciar la cámara: {e}")
            
        except Exception as e:
            logger.error(f"Error in volver_al_origen: {e}")
            QMessageBox.critical(
                self,
                "Error",
                f"Error al volver al origen:\n{str(e)}"
            )
    
    def emergency_close(self):
        """Emergency close: Force quit application immediately"""
        reply = QMessageBox.question(
            self,
            "⛔ Cerrar Programa",
            "¿Estás seguro de que quieres cerrar el programa?\n\n"
            "Esta opción cierra la aplicación inmediatamente,\n"
            "incluso si está congelada.\n\n"
            "Se intentará limpiar los recursos antes de cerrar.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            try:
                self.add_system_message("⛔ Cerrando aplicación...")
                
                # Try to cleanup resources
                if hasattr(self, 'servo_service') and self.servo_service:
                    try:
                        self.servo_service.cleanup()
                        logger.info("Servos cleaned up")
                    except:
                        pass
                
                if hasattr(self, 'camera_service') and self.camera_service:
                    try:
                        self.camera_service.release()
                        logger.info("Camera released")
                    except:
                        pass
                
                logger.info("Emergency close initiated by user")
                
                # Force quit
                import sys
                sys.exit(0)
                
            except Exception as e:
                logger.error(f"Error during emergency close: {e}")
                # Force quit anyway
                import os
                os._exit(0)
    
    def emergency_close_shortcut(self):
        """Emergency close triggered by Ctrl+Q shortcut"""
        logger.warning("⌨️ Emergency close shortcut activated (Ctrl+Q)")
        
        # Show notification
        try:
            self.add_system_message("⚠️ Atajo de emergencia Ctrl+Q activado - Cerrando programa...")
        except:
            pass
        
        # Try cleanup without confirmation dialog (shortcut implies urgency)
        try:
            if hasattr(self, 'servo_service') and self.servo_service:
                try:
                    self.servo_service.cleanup()
                    logger.info("Servos cleaned up")
                except:
                    pass
            
            if hasattr(self, 'camera_service') and self.camera_service:
                try:
                    self.camera_service.release()
                    logger.info("Camera released")
                except:
                    pass
            
            logger.info("Emergency close via shortcut (Ctrl+Q)")
            
            # Force quit
            import sys
            sys.exit(0)
            
        except Exception as e:
            logger.error(f"Error during emergency close: {e}")
            # Force quit anyway
            import os
            os._exit(0)
    
    def force_quit(self):
        """Force quit without any cleanup (Ctrl+Shift+Q) - most aggressive"""
        logger.critical("⚠️⚠️⚠️ FORCE QUIT ACTIVATED (Ctrl+Shift+Q) - NO CLEANUP ⚠️⚠️⚠️")
        
        try:
            self.add_system_message("🔴 FORCE QUIT - Terminando inmediatamente...")
        except:
            pass
        
        # Immediate termination without cleanup
        import os
        os._exit(0)
    
    def demo_buscar_objeto(self):
        """Demo: Search for specific object using YOLO and head movement"""
        # TODO: Implement object search
        # 1. Ask user what object to search for (or use voice)
        # 2. Move head scanning horizontally
        # 3. Use YOLO to detect objects
        # 4. When found, stop and center on object
        # 5. Announce by voice: "Encontré [objeto]"
        print("Demo IA: Buscar objeto")
        QMessageBox.information(
            self,
            "Demo IA - Buscar Objeto",
            "🔍 Iniciando búsqueda de objeto\n\n"
            "El robot escaneará el entorno moviendo la cabeza\n"
            "y usará YOLO para detectar el objeto solicitado.\n\n"
            "Funcionalidad próximamente..."
        )
    
    def demo_seguir_persona(self):
        """Demo: Track and follow a person using face detection - Ejecuta TEST_SEGUIMIENTO_FINAL.PY"""
        try:
            self.add_system_message("👤 Iniciando seguimiento facial dual...")
            
            # IMPORTANTE: Liberar la cámara si está siendo usada por el camera_service
            if hasattr(self, 'camera_service') and self.camera_service:
                try:
                    self.add_system_message("📷 Deteniendo cámara del servicio principal...")
                    # Intento robusto de liberación
                    self.camera_service.stop_camera()
                    for _ in range(10):
                        # Esperar liberación completa
                        if not self.camera_service.is_available():
                            break
                        time.sleep(0.1)
                    self.add_system_message("✓ Cámara liberada")
                except Exception as e:
                    self.add_system_message(f"⚠️ Error al liberar cámara: {e}")
            
            # Obtener la ruta base del proyecto
            base_dir = os.path.dirname(os.path.dirname(__file__))
            
            # Ruta al archivo TEST_SEGUIMIENTO_FINAL.PY
            script_path = os.path.join(base_dir, "TEST_SEGUIMIENTO_FINAL.PY")
            
            # Ruta al intérprete de Python del entorno virtual
            python_path = os.path.join(base_dir, "env", "bin", "python")
            
            self.add_system_message(f"🔍 Verificando archivos...")
            self.add_system_message(f"📁 Base dir: {base_dir}")
            self.add_system_message(f"📄 Script: {script_path}")
            self.add_system_message(f"🐍 Python: {python_path}")
            
            # Verificar que los archivos existen
            if not os.path.exists(script_path):
                self.add_system_message(f"❌ ERROR: No se encuentra el script")
                self.add_system_message(f"   Ruta buscada: {script_path}")
                return
            
            if not os.path.exists(python_path):
                self.add_system_message(f"❌ ERROR: No se encuentra el intérprete Python")
                self.add_system_message(f"   Ruta buscada: {python_path}")
                return
            
            self.add_system_message(f"✓ Archivos encontrados")
            self.add_system_message(f"� Ejecutando proceso...")
            
            # Asegurar que no haya otro proceso de seguimiento activo
            try:
                subprocess.run(["pkill", "-9", "-f", "TEST_SEGUIMIENTO_FINAL.PY"], check=False)
            except Exception:
                pass

            # Ejecutar el script en background sin capturar salida (para ver errores en terminal)
            process = subprocess.Popen(
                [python_path, script_path],
                cwd=base_dir,
                stdout=None,
                stderr=None
            )
            
            # Dar un momento para que el proceso inicie
            time.sleep(0.5)
            
            # Verificar si el proceso sigue corriendo
            poll_result = process.poll()
            if poll_result is not None:
                self.add_system_message(f"❌ El proceso terminó inmediatamente con código: {poll_result}")
                self.add_system_message(f"⚠️ Revisa la terminal para ver errores")
            else:
                self.add_system_message(f"✓ Proceso iniciado correctamente")
                self.add_system_message(f"🔢 PID: {process.pid}")
                self.add_system_message(f"👁️ El robot seguirá tu cara en X e Y")
                self.add_system_message(f"⏹️ Presiona 'Volver al Origen' para detener")
            
        except Exception as e:
            logger.error(f"Error in demo_seguir_persona: {e}")
            self.add_system_message(f"❌ Error al ejecutar seguimiento: {e}")
            self.add_system_message(f"📋 Detalles: {traceback.format_exc()}")
    
    def demo_completa_ia(self):
        """Demo: Complete AI demonstration combining all capabilities"""
        # TODO: Implement complete AI demo sequence
        # 1. Voice intro: "Hola, soy un robot inteligente"
        # 2. Scan environment (horizontal movement)
        # 3. Detect and count objects with YOLO
        # 4. Describe scene using LLM + YOLO data
        # 5. Track detected person
        # 6. Answer a question about the environment
        print("Demo IA: Demostración completa")
        QMessageBox.information(
            self,
            "Demo IA - Completa",
            "🎯 Iniciando demostración completa de IA\n\n"
            "Secuencia:\n"
            "1. Presentación por voz\n"
            "2. Escaneo del entorno\n"
            "3. Detección y conteo de objetos\n"
            "4. Descripción de la escena\n"
            "5. Seguimiento de persona\n"
            "6. Respuesta a pregunta\n\n"
            "Funcionalidad próximamente..."
        )
    
    def check_services_status(self):
        """Check and display the status of available services"""
        chatgpt_status = "✓" if self.chatgpt_service.is_available() else "✗"
        ollama_status = "✓" if self.ollama_service.is_available() else "✗"
        audio_status = "✓" if self.audio_service and self.audio_service.is_microphone_available() else "✗"
        
        status_msg = f"ChatGPT: {chatgpt_status} | Ollama: {ollama_status} | Audio: {audio_status}"
        
        if not self.chatgpt_service.is_available():
            status_msg += " (Set OPENAI_API_KEY in .env)"
        if not self.ollama_service.is_available():
            status_msg += " (Start Ollama: ollama serve)"
        if not (self.audio_service and self.audio_service.is_microphone_available()):
            status_msg += " (No microphone detected)"
        
        self.add_system_message(status_msg)
        self.update_status_indicators()
    
    def update_status_indicators(self):
        """Update the status indicators in the chat tab"""
        # Safety check: ensure UI elements are created
        if not hasattr(self, 'ai_status_indicator') or not hasattr(self, 'model_selector'):
            return
        
        # Update AI status indicator based on current model
        try:
            current_model = self.model_selector.currentText()
            ai_available = False
            
            if "GPT" in current_model:
                ai_available = self.chatgpt_service and self.chatgpt_service.is_available()
            elif "llama" in current_model or "mistral" in current_model:
                ai_available = self.ollama_service and self.ollama_service.is_available()
            elif "groq" in current_model:
                ai_available = hasattr(self, 'groq_service') and self.groq_service and self.groq_service.is_available()
            
            # Set indicator color
            indicator_color = "#27ae60" if ai_available else "#95a5a6"
            indicator_tooltip = "IA Conectada" if ai_available else "IA No Disponible"
            
            self.ai_status_indicator.setStyleSheet(f"""
                QLabel {{
                    color: {indicator_color};
                    font-size: 14pt;
                    padding: 0 5px;
                }}
            """)
            self.ai_status_indicator.setToolTip(indicator_tooltip)
            
            # Update camera status indicator
            if hasattr(self, 'camera_status_indicator'):
                camera_available = (self.camera_service is not None and 
                                   hasattr(self.camera_service, 'is_available') and 
                                   self.camera_service.is_available())
                cam_color = "#27ae60" if camera_available else "#95a5a6"
                cam_tooltip = "Cámara Activa" if camera_available else "Cámara No Disponible"
                
                self.camera_status_indicator.setStyleSheet(f"""
                    QLabel {{
                        color: {cam_color};
                        font-size: 10pt;
                        padding: 0 3px;
                    }}
                """)
                self.camera_status_indicator.setToolTip(cam_tooltip)
        except Exception as e:
            logger.error(f"Error updating status indicators: {e}")
    
    def add_system_message(self, message: str):
        """Add a system message to the chat display"""
        self.chat_display.append(
            f'<div style="'
            f'background-color: #ecf0f1; '
            f'color: #7f8c8d; '
            f'padding: 8px 12px; '
            f'margin: 8px 0; '
            f'border-left: 3px solid #95a5a6; '
            f'border-radius: 6px; '
            f'font-size: 9pt; '
            f'font-style: italic;">'
            f'ℹ️ {message}'
            f'</div>'
        )
        self.scroll_to_bottom()
    
    def add_user_message(self, message: str):
        """Add a user message to the chat display"""
        self.total_messages += 1
        self.update_message_counter()
        
        # Limit chat display to last 50 messages to prevent memory bloat
        self._cleanup_chat_display_if_needed()
        
        self.chat_display.append(
            f'<div style="'
            f'background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); '
            f'color: white; '
            f'padding: 12px 16px; '
            f'margin: 10px 50px 10px 10px; '
            f'border-radius: 18px 18px 4px 18px; '
            f'box-shadow: 0 2px 8px rgba(0,0,0,0.1); '
            f'font-size: 10pt;">'
            f'<div style="font-weight: 600; margin-bottom: 4px; font-size: 8pt; opacity: 0.9;">TÚ</div>'
            f'{message}'
            f'</div>'
        )
        self.scroll_to_bottom()
    
    def add_assistant_message(self, message: str):
        """Add an assistant message to the chat display"""
        self.total_messages += 1
        self.update_message_counter()
        
        # Limit chat display to last 50 messages to prevent memory bloat
        self._cleanup_chat_display_if_needed()
        
        self.chat_display.append(
            f'<div style="'
            f'background: linear-gradient(135deg, #3498db 0%, #2ecc71 100%); '
            f'color: white; '
            f'padding: 12px 16px; '
            f'margin: 10px 10px 10px 50px; '
            f'border-radius: 18px 18px 18px 4px; '
            f'box-shadow: 0 2px 8px rgba(0,0,0,0.1); '
            f'font-size: 10pt;">'
            f'<div style="font-weight: 600; margin-bottom: 4px; font-size: 8pt; opacity: 0.9;">🤖 FRANKEINSTEIN</div>'
            f'{message}'
            f'</div>'
        )
        self.scroll_to_bottom()
    
    def _cleanup_chat_display_if_needed(self):
        """Remove old messages from chat display to prevent memory bloat"""
        # Keep only last 50 messages (25 exchanges) in UI
        if self.total_messages > 50:
            doc = self.chat_display.document()
            # Count blocks (each message is a block)
            block_count = doc.blockCount()
            if block_count > 50:
                # Remove oldest blocks
                cursor = QTextCursor(doc)
                cursor.movePosition(QTextCursor.Start)
                for _ in range(block_count - 50):
                    cursor.select(QTextCursor.BlockUnderCursor)
                    cursor.removeSelectedText()
                    cursor.deleteChar()  # Remove the newline
    
    def update_message_counter(self):
        """Update the message counter in the UI"""
        if hasattr(self, 'message_counter'):
            self.message_counter.setText(f"{self.total_messages} mensajes")
    
    def scroll_to_bottom(self):
        """Scroll chat display to the bottom"""
        cursor = self.chat_display.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.chat_display.setTextCursor(cursor)
    
    @Slot()
    def on_model_changed(self):
        """Handle model selection change"""
        model = self.model_selector.currentText()
        self.add_system_message(f"Cambiado a modelo {model}")
        self.update_status_indicators()
        
        # Check if selected model is available
        if model == "ChatGPT" and not self.chatgpt_service.is_available():
            QMessageBox.warning(
                self,
                "ChatGPT No Disponible",
                "La clave API de ChatGPT no está configurada.\n\nPor favor, establece OPENAI_API_KEY en el archivo .env."
            )
        elif model == "Ollama" and not self.ollama_service.is_available():
            QMessageBox.warning(
                self,
                "Ollama No Disponible",
                "Ollama no está ejecutándose.\n\nPor favor, inicia Ollama con: ollama serve"
            )
    
    @Slot()
    def send_message(self):
        """Handle sending a message"""
        message = self.message_input.text().strip()
        
        if not message:
            return
        
        # Store message before clearing input
        self.last_user_message = message
        
        # Get selected model
        selected_model = self.model_selector.currentText()
        
        # Select appropriate service
        if selected_model == "ChatGPT":
            service = self.chatgpt_service
            if not service.is_available():
                self.add_system_message("Error: ChatGPT is not available. Check your API key.")
                return
        elif selected_model == "Groq":
            service = self.groq_service
            if not service.is_available():
                self.add_system_message("Error: Groq API key not configured. Get free key at: https://console.groq.com/keys")
                return
        else:  # Ollama
            service = self.ollama_service
            if not service.is_available():
                self.add_system_message("Error: Ollama is not running. Start it with 'ollama serve'")
                return
        
        # Detect if user is asking about vision (if auto-vision is enabled)
        vision_context = None
        if self.auto_vision_enabled:
            vision_keywords = ['qué ves', 'que ves', 'describe', 'imagen', 'cámara', 'camara', 
                              'objetos', 'objeto', 'what do you see', 'camera', 'image', 
                              'detecta', 'detectas', 'hay en la imagen', 'en la foto', 'mira']
            is_vision_query = any(keyword in message.lower() for keyword in vision_keywords)
            
            # Get vision context if this is a vision-related query
            if is_vision_query and self.camera_service:
                # Read latest detections safely and build context
                try:
                    self.detections_lock.acquire()
                    if hasattr(self.camera_service, 'get_detection_summary'):
                        vision_context = self.camera_service.get_detection_summary()
                    else:
                        names = [d.get('class', 'objeto') for d in (self.latest_detections or [])]
                        if names:
                            vision_context = f"Objetos detectados: {', '.join(names[:5])}"
                        else:
                            vision_context = "No se detectan objetos por ahora"
                finally:
                    try:
                        self.detections_lock.release()
                    except:
                        pass
                # Show visual indicator that vision mode is active
                self.add_system_message(f"📷 {vision_context}")
        
        # Add user message to display
        self.add_user_message(message)
        
        # Clear input
        self.message_input.clear()
        
        # Disable send and voice buttons while processing
        self.send_button.setEnabled(False)
        self.voice_button.setEnabled(False)
        self.status_label.setText("Thinking...")
        self.status_label.setStyleSheet("color: orange; font-style: italic;")
        
        # Create worker thread for API call with vision context
        self.current_worker = ChatWorker(service, message, self.conversation_history.copy(), vision_context)
        self.current_worker.response_ready.connect(self.on_response_received)
        self.current_worker.error_occurred.connect(self.on_error_occurred)
        self.current_worker.finished.connect(self.on_worker_finished)
        self.current_worker.start()
    
    @Slot(str)
    def on_response_received(self, response: str):
        """Handle successful response from chat service"""
        # Detect and execute movement commands embedded in response
        processed_response, movements = self.extract_movement_commands(response)
        
        # Display cleaned response
        self.add_assistant_message(processed_response)
        
        # Execute movements if any were detected
        if movements:
            self.execute_movements(movements)
        
        # Speak response if voice is enabled (use cleaned response)
        if self.voice_response_enabled and self.audio_service:
            # Start natural head movements while speaking
            self.simulate_natural_talking_movements(processed_response)
            
            # Start speech (non-blocking)
            self.audio_service.speak(processed_response, blocking=False)
        
        # Update conversation history
        # Note: We need to store the last user message separately since input is cleared
        if hasattr(self, 'last_user_message'):
            self.conversation_history.append({"role": "user", "content": self.last_user_message})
            self.conversation_history.append({"role": "assistant", "content": response})
        
        # Keep only last 10 exchanges
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]
    
    def extract_movement_commands(self, response: str) -> tuple:
        """
        Extract movement commands from LLM response
        
        Commands format: [MOVER:nombre_movimiento]
        Examples: [MOVER:saludar], [MOVER:asentir], [MOVER:negar]
        
        Returns:
            tuple: (cleaned_response, list_of_movements)
        """
        import re
        
        # Find all movement commands
        pattern = r'\[MOVER:(\w+)\]'
        movements = re.findall(pattern, response)
        
        # Remove commands from response
        cleaned_response = re.sub(pattern, '', response).strip()
        
        # Also detect natural language commands
        natural_movements = self.detect_natural_movement_intent(response)
        if natural_movements:
            movements.extend(natural_movements)
        
        return cleaned_response, movements
    
    def detect_natural_movement_intent(self, text: str) -> list:
        """
        Detect movement intentions from natural language
        
        Examples:
        - "voy a saludar" → ['saludar']
        - "voy a asentir" → ['asentir']
        - "déjame saludar" → ['saludar']
        """
        text_lower = text.lower()
        detected = []
        
        # Movement keywords mapping
        movement_keywords = {
            'saludar': ['saludar', 'saludo', 'te saludo', 'voy a saludar'],
            'asentir': ['asentir', 'asiento', 'voy a asentir', 'diré que sí'],
            'negar': ['negar', 'niego', 'voy a negar', 'diré que no', 'shake my head'],
            'mirar_arriba': ['mirar arriba', 'mirar hacia arriba', 'miro arriba'],
            'mirar_abajo': ['mirar abajo', 'mirar hacia abajo', 'miro abajo'],
            'curiosear': ['curiosear', 'mirar alrededor', 'explorar', 'voy a explorar']
        }
        
        for movement, keywords in movement_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    if movement not in detected:
                        detected.append(movement)
                    break
        
        return detected
    
    def simulate_natural_talking_movements(self, text: str):
        """
        Generate natural head movements while speaking
        Movements are synchronized with estimated speech duration
        Uses DIRECT ServoKit control for reliability
        """
        import random
        import threading
        import time
        
        # Estimate speech duration (roughly 130 words per minute = 2.2 words per second)
        word_count = len(text.split())
        estimated_duration = max(word_count / 2.2, 3.0)  # Minimum 3 seconds
        
        logger.info(f"🎭 Starting natural talking movements for {estimated_duration:.1f}s ({word_count} words)")
        
        def movement_thread():
            try:
                # Import ServoKit directly for guaranteed control
                from adafruit_servokit import ServoKit
                
                # Initialize ServoKit
                kit = ServoKit(channels=16, address=0x40)
                
                # Aplicar márgenes de seguridad PWM globales
                PWM_MIN_SAFE = 650
                PWM_MAX_SAFE = 2000
                for channel in range(16):
                    try:
                        kit.servo[channel].set_pulse_width_range(PWM_MIN_SAFE, PWM_MAX_SAFE)
                    except:
                        pass
                logger.info(f"✓ PWM safety margins applied: {PWM_MIN_SAFE}-{PWM_MAX_SAFE}μs")
                
                # Servo configuration (REAL ROBOT PINS)
                SERVO_YAW = 13       # Horizontal (cuello)
                SERVO_PITCH = 14     # Vertical
                SERVO_ROLL_LEFT = 12
                SERVO_ROLL_RIGHT = 15
                
                # Center positions
                CENTER_YAW = 135
                CENTER_PITCH = 120
                CENTER_ROLL = 155
                
                # Define natural movement range (subtle movements)
                yaw_range = 20  # +/- degrees from center
                pitch_range = 15  # +/- degrees from center
                roll_range = 10  # +/- degrees from center
                
                start_time = time.time()
                movement_count = 0
                
                logger.info(f"✓ ServoKit initialized, starting movements from center positions")
                
                while time.time() - start_time < estimated_duration:
                    # Random subtle movements
                    yaw_offset = random.uniform(-yaw_range, yaw_range)
                    pitch_offset = random.uniform(-pitch_range, pitch_range)
                    roll_offset = random.uniform(-roll_range, roll_range)
                    
                    target_yaw = CENTER_YAW + yaw_offset
                    target_pitch = CENTER_PITCH + pitch_offset
                    target_roll = CENTER_ROLL + roll_offset
                    
                    # Clamp to safe ranges
                    target_yaw = max(90, min(180, target_yaw))
                    target_pitch = max(60, min(180, target_pitch))
                    target_roll = max(130, min(180, target_roll))
                    
                    # Move servos
                    kit.servo[SERVO_YAW].angle = target_yaw
                    kit.servo[SERVO_PITCH].angle = target_pitch
                    kit.servo[SERVO_ROLL_LEFT].angle = target_roll
                    kit.servo[SERVO_ROLL_RIGHT].angle = target_roll
                    
                    # Pause between movements (natural rhythm)
                    pause_duration = random.uniform(0.6, 1.2)
                    time.sleep(pause_duration)
                    
                    movement_count += 1
                    
                    # Occasionally return closer to center (more natural)
                    if movement_count % 4 == 0:
                        kit.servo[SERVO_YAW].angle = CENTER_YAW
                        kit.servo[SERVO_PITCH].angle = CENTER_PITCH
                        kit.servo[SERVO_ROLL_LEFT].angle = CENTER_ROLL
                        kit.servo[SERVO_ROLL_RIGHT].angle = CENTER_ROLL
                        time.sleep(0.4)
                
                # Return to center position at the end
                kit.servo[SERVO_YAW].angle = CENTER_YAW
                kit.servo[SERVO_PITCH].angle = CENTER_PITCH
                kit.servo[SERVO_ROLL_LEFT].angle = CENTER_ROLL
                kit.servo[SERVO_ROLL_RIGHT].angle = CENTER_ROLL
                
                logger.info(f"✓ Natural talking movements completed ({movement_count} movements in {estimated_duration:.1f}s)")
                
            except Exception as e:
                logger.error(f"❌ Error in natural talking movements: {e}")
                import traceback
                logger.error(traceback.format_exc())
        
        # Start movement thread
        thread = threading.Thread(target=movement_thread, daemon=True)
        thread.start()
    
    def execute_movements(self, movements: list):
        """Execute a list of movements"""
        for movement in movements:
            method_name = f'movimiento_{movement}'
            if hasattr(self, method_name):
                try:
                    # Log the movement
                    logger.info(f"🤖 Auto-executing movement: {movement}")
                    self.add_system_message(f"🤖 [ACCIÓN AUTOMÁTICA] Ejecutando: {movement}")
                    
                    # Execute the movement
                    getattr(self, method_name)()
                    
                    # Update conversation history with action context
                    self.conversation_history.append({
                        "role": "system",
                        "content": f"[El robot ha ejecutado el movimiento: {movement}]"
                    })
                except Exception as e:
                    logger.error(f"Error executing movement {movement}: {e}")
                    self.add_system_message(f"⚠️ Error ejecutando {movement}: {str(e)}")
    
    @Slot(str)
    def on_error_occurred(self, error: str):
        """Handle error from chat service"""
        self.add_system_message(f"Error: {error}")
    
    @Slot()
    def on_worker_finished(self):
        """Handle worker thread completion"""
        self.send_button.setEnabled(True)
        self.voice_button.setEnabled(True)
        self.status_label.setText("Ready")
        self.status_label.setStyleSheet("color: green; font-style: italic;")
        self.current_worker = None
    
    @Slot()
    def clear_chat(self):
        """Clear chat history"""
        self.chat_display.clear()
        self.conversation_history = []
        self.add_system_message("Chat cleared. Start a new conversation!")
    
    @Slot()
    def start_voice_input(self):
        """Start listening to microphone"""
        if not self.audio_service:
            self.add_system_message("Error: Audio service not available")
            return
        
        if not self.audio_service.is_microphone_available():
            QMessageBox.warning(
                self,
                "Microphone Not Available",
                "No microphone detected.\n\nPlease connect a microphone and try again."
            )
            return
        
        # Disable voice button while listening
        self.voice_button.setEnabled(False)
        self.send_button.setEnabled(False)
        self.voice_button.setText("🎤 Listening...")
        self.status_label.setText("Listening to microphone...")
        self.status_label.setStyleSheet("color: blue; font-style: italic;")
        
        # Start audio worker
        self.audio_worker = AudioWorker(self.audio_service, timeout=self.listen_timeout)
        self.audio_worker.audio_recognized.connect(self.on_audio_recognized)
        self.audio_worker.error_occurred.connect(self.on_audio_error)
        self.audio_worker.status_update.connect(self.on_audio_status)
        self.audio_worker.finished.connect(self.on_audio_finished)
        self.audio_worker.start()
    
    @Slot(str)
    def on_audio_recognized(self, text: str):
        """Handle recognized speech"""
        self.message_input.setText(text)
        self.add_system_message(f"🎤 Recognized: {text}")
        # Automatically send the message
        self.send_message()
    
    @Slot(str)
    def on_audio_error(self, error: str):
        """Handle audio recognition error"""
        self.add_system_message(f"🎤 {error}")
    
    @Slot(str)
    def on_audio_status(self, status: str):
        """Handle audio status update"""
        self.status_label.setText(status)
    
    @Slot()
    def on_audio_finished(self):
        """Handle audio worker completion"""
        self.voice_button.setEnabled(True)
        self.send_button.setEnabled(True)
        self.voice_button.setText("🎤 Voice")
        if not self.current_worker:  # Only reset status if not processing chat
            self.status_label.setText("Ready")
            self.status_label.setStyleSheet("color: green; font-style: italic;")
        self.audio_worker = None
    
    @Slot()
    def toggle_voice_response(self):
        """Toggle voice response on/off"""
        self.voice_response_enabled = self.voice_response_button.isChecked()
        if self.voice_response_enabled:
            self.voice_response_button.setText("🔊 On")
            self.add_system_message("Voice responses enabled")
        else:
            self.voice_response_button.setText("🔇 Off")
            self.add_system_message("Voice responses disabled")
            # Stop any current speech
            if self.audio_service:
                self.audio_service.stop_speaking()
    
    def toggle_camera_preview(self):
        """Toggle camera preview on/off to save resources (YOLO still runs in background)"""
        self.camera_preview_active = not self.camera_preview_active
        
        if self.camera_preview_active:
            # Preview activated
            self.camera_toggle_btn.setText("⏸️ Pausar")
            self.camera_toggle_btn.setStyleSheet("""
                QPushButton {
                    background-color: #e67e22;
                    color: white;
                    border: none;
                    padding: 10px 18px;
                    border-radius: 8px;
                    font-weight: 600;
                    font-size: 11pt;
                }
                QPushButton:hover {
                    background-color: #d35400;
                }
                QPushButton:pressed {
                    background-color: #ba4a00;
                }
            """)
            self.add_system_message("📹 Preview de cámara activado")
            self.update_status_indicators()
        else:
            # Preview paused
            self.camera_toggle_btn.setText("▶️ Activar")
            self.camera_toggle_btn.setStyleSheet("""
                QPushButton {
                    background-color: #27ae60;
                    color: white;
                    border: none;
                    padding: 10px 18px;
                    border-radius: 8px;
                    font-weight: 600;
                    font-size: 11pt;
                }
                QPushButton:hover {
                    background-color: #229954;
                }
                QPushButton:pressed {
                    background-color: #1e8449;
                }
            """)
            self.video_label.setText("⏸️ Preview Pausado\n\n✅ YOLO activo en segundo plano\n🎤 Audio optimizado\n\nClick '▶️ Activar' para ver el video")
            self.video_label.setStyleSheet("""
                QLabel {
                    background-color: #1a1a1a;
                    color: #95a5a6;
                    border: 2px solid #34495e;
                    border-radius: 10px;
                    font-size: 11pt;
                    padding: 20px;
                }
            """)
            self.add_system_message("⏸️ Preview pausado (YOLO sigue activo)")
            self.update_status_indicators()
    
    def on_tts_engine_changed(self, index):
        """Handle TTS engine selection change"""
        if not self.audio_service:
            return
        
        engine_map = {
            0: "pyttsx3",
            1: "gtts"
        }
        
        engine_type = engine_map.get(index, "pyttsx3")
        
        if self.audio_service.set_tts_engine(engine_type):
            engine_names = {
                "pyttsx3": "pyttsx3 (Offline)",
                "gtts": "Google TTS (Online, Alta Calidad)"
            }
            self.add_system_message(f"🎤 Motor de voz cambiado a: {engine_names[engine_type]}")
            
            # Update speed slider visibility (only for pyttsx3)
            if engine_type == "pyttsx3":
                self.voice_speed_slider.setEnabled(True)
                self.voice_speed_row_label.setEnabled(True)
                self.voice_speed_label.setEnabled(True)
            else:
                self.voice_speed_slider.setEnabled(False)
                self.voice_speed_row_label.setEnabled(False)
                self.voice_speed_label.setEnabled(False)
        else:
            self.add_system_message("❌ No se pudo cambiar el motor de voz (gTTS no disponible)")
            self.tts_engine_combo.setCurrentIndex(0)  # Revert to pyttsx3

    
    def start_camera(self):
        """Initialize and start camera with YOLO"""
        try:
            # Start camera
            if self.camera_service.start_camera():
                self.camera_status_label.setText("📷 Camera: Loading YOLO model...")
                self.camera_status_label.setStyleSheet("color: orange; font-style: italic;")
                
                # Load YOLO model
                if self.camera_service.load_yolo_model("yolov8n.pt"):
                    self.camera_status_label.setText("📷 Camera: Active with YOLO")
                    self.camera_status_label.setStyleSheet("color: green; font-style: italic;")
                    
                    # Start video worker thread
                    self.video_worker = VideoWorker(self.camera_service, self)
                    self.video_worker.frame_ready.connect(self.update_frame)
                    self.video_worker.error_occurred.connect(self.on_video_error)
                    self.video_worker.start()
                else:
                    self.camera_status_label.setText("📷 Camera: Active (YOLO failed to load)")
                    self.camera_status_label.setStyleSheet("color: orange; font-style: italic;")
            else:
                self.camera_status_label.setText("📷 Camera: Not available")
                self.camera_status_label.setStyleSheet("color: red; font-style: italic;")
                self.video_label.setText("No camera detected")
        except Exception as e:
            self.camera_status_label.setText(f"📷 Camera: Error - {str(e)}")
            self.camera_status_label.setStyleSheet("color: red; font-style: italic;")
    
    @Slot(QImage, list)
    def update_frame(self, image: QImage, detections: list):
        """Update video frame in the UI"""
        # Scale image to fit label
        pixmap = QPixmap.fromImage(image)
        scaled_pixmap = pixmap.scaled(
            self.video_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.video_label.setPixmap(scaled_pixmap)
        
        # Update detections (thread-safe)
        try:
            self.detections_lock.acquire()
            self.latest_detections = detections
        finally:
            try:
                self.detections_lock.release()
            except:
                pass
        if detections:
            detection_text = f"Detections ({len(detections)}): "
            detection_names = [d['class'] for d in detections[:5]]  # Show first 5
            detection_text += ", ".join(detection_names)
            if len(detections) > 5:
                detection_text += f", +{len(detections) - 5} more"
            self.detection_label.setText(detection_text)
        else:
            self.detection_label.setText("Detections: None")
    
    @Slot(str)
    def on_video_error(self, error: str):
        """Handle video error"""
        self.camera_status_label.setText(f"📷 Camera: Error")
        self.camera_status_label.setStyleSheet("color: red; font-style: italic;")
    
    def closeEvent(self, event):
        """Handle application close"""
        # Stop audio
        if self.audio_service is not None:
            self.audio_service.stop_speaking()
        
        # Stop audio worker
        if self.audio_worker is not None:
            self.audio_worker.wait()
        
        # Stop video worker
        if self.video_worker is not None:
            self.video_worker.stop()
            self.video_worker.wait()
        
        # Stop camera
        if self.camera_service is not None:
            self.camera_service.stop_camera()
        
        # Cleanup servos
        if self.servo_service is not None:
            self.servo_service.cleanup()
        
        event.accept()


def main():
    """Main application entry point"""
    import signal
    
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle("Fusion")
    
    window = ChatWindow()
    window.show()
    
    # Manejar Ctrl+C para detener servos
    def signal_handler(sig, frame):
        print("\n⚠️ Ctrl+C detectado - Deteniendo servos...")
        try:
            # Detener todos los procesos de servos
            import subprocess
            subprocess.run(["pkill", "-f", "TEST_ASENTIR.py"], check=False)
            subprocess.run(["pkill", "-f", "TEST_NEGAR.py"], check=False)
            subprocess.run(["pkill", "-f", "TEST_ROLL.py"], check=False)
            
            # Centrar servos si el servicio está disponible
            if window.servo_service and window.servo_service.is_initialized:
                window.servo_service.move_to_center()
        except Exception as e:
            print(f"Error al detener servos: {e}")
        
        print("✅ Finalizando aplicación...")
        app.quit()
        sys.exit(0)
    
    # Registrar handler de señales
    signal.signal(signal.SIGINT, signal_handler)
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
