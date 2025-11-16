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
from PySide6.QtCore import Qt, QThread, Signal, Slot, QTimer
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
        """Capture frames and run YOLO detection (always), render video only when preview is active"""
        while self.running:
            try:
                # ALWAYS process YOLO for detections (needed for LLM)
                success, frame, detections = self.camera_service.get_frame_with_detection()
                
                if success and frame is not None:
                    # Store detections for LLM (always, even when preview is off)
                    if detections:
                        self.main_window.latest_detections = detections
                    
                    # Only render video to UI if preview is active (saves CPU/RAM)
                    if self.main_window.camera_preview_active:
                        # Convert BGR to RGB
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        h, w, ch = rgb_frame.shape
                        bytes_per_line = ch * w
                        
                        # Convert to QImage
                        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                        self.frame_ready.emit(qt_image, detections)
                    else:
                        # Preview off: process slower to save even more resources
                        self.msleep(200)
                else:
                    self.msleep(30)  # Wait a bit before trying again
                    
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
        self.voice_response_enabled = True  # Enable voice responses by default
        self.camera_preview_active = False  # Camera preview OFF by default to save resources
        
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
        chat_main_layout.setSpacing(10)
        chat_main_layout.setContentsMargins(10, 10, 10, 10)
        
        # Top bar with model selector
        top_bar_layout = QHBoxLayout()
        
        model_label = QLabel("Model:")
        top_bar_layout.addWidget(model_label)
        
        self.model_selector = QComboBox()
        self.model_selector.addItems(["Groq", "Ollama", "ChatGPT"])
        self.model_selector.setMinimumWidth(150)
        self.model_selector.currentTextChanged.connect(self.on_model_changed)
        top_bar_layout.addWidget(self.model_selector)
        
        top_bar_layout.addStretch()
        
        # Camera status
        self.camera_status_label = QLabel("📷 Camera: Initializing...")
        self.camera_status_label.setStyleSheet("color: #666; font-style: italic;")
        top_bar_layout.addWidget(self.camera_status_label)
        
        chat_main_layout.addLayout(top_bar_layout)
        
        # Main content: Two columns (Chat + Camera)
        content_layout = QHBoxLayout()
        content_layout.setSpacing(15)
        
        # LEFT COLUMN: Chat interface
        chat_column = QWidget()
        chat_layout = QVBoxLayout(chat_column)
        chat_layout.setSpacing(10)
        chat_layout.setContentsMargins(0, 0, 0, 0)
        
        # Chat title
        chat_title = QLabel("💬 Chat")
        chat_title_font = QFont()
        chat_title_font.setPointSize(12)
        chat_title_font.setBold(True)
        chat_title.setFont(chat_title_font)
        chat_layout.addWidget(chat_title)
        
        # Status label
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: green; font-style: italic;")
        chat_layout.addWidget(self.status_label)
        
        # Chat display area
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setFont(QFont("Segoe UI", 10))
        self.chat_display.setStyleSheet("""
            QTextEdit {
                background-color: #ffffff;
                border: 2px solid #e1e8ed;
                border-radius: 8px;
                padding: 15px;
                color: #2c3e50;
            }
        """)
        chat_layout.addWidget(self.chat_display)
        
        # Input area
        input_layout = QHBoxLayout()
        
        self.message_input = QLineEdit()
        self.message_input.setPlaceholderText("Escribe tu mensaje aquí...")
        self.message_input.setFont(QFont("Segoe UI", 10))
        self.message_input.setStyleSheet("""
            QLineEdit {
                padding: 12px 15px;
                border: 2px solid #e1e8ed;
                border-radius: 8px;
                background-color: #ffffff;
                color: #2c3e50;
            }
            QLineEdit:focus {
                border: 2px solid #3498db;
            }
        """)
        self.message_input.returnPressed.connect(self.send_message)
        input_layout.addWidget(self.message_input)
        
        self.send_button = QPushButton("Enviar")
        self.send_button.setFont(QFont("Segoe UI", 10, QFont.Bold))
        self.send_button.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                padding: 12px 24px;
                border: none;
                border-radius: 8px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #21618c;
            }
            QPushButton:disabled {
                background-color: #bdc3c7;
            }
        """)
        self.send_button.clicked.connect(self.send_message)
        input_layout.addWidget(self.send_button)
        
        # Voice input button
        self.voice_button = QPushButton("🎤 Voz")
        self.voice_button.setFont(QFont("Segoe UI", 10))
        self.voice_button.setStyleSheet("""
            QPushButton {
                background-color: #9b59b6;
                color: white;
                padding: 12px 20px;
                border: none;
                border-radius: 8px;
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
        self.voice_button.clicked.connect(self.start_voice_input)
        input_layout.addWidget(self.voice_button)
        
        # Voice response toggle button
        self.voice_response_button = QPushButton("🔊 Audio")
        self.voice_response_button.setFont(QFont("Segoe UI", 10))
        self.voice_response_button.setCheckable(True)
        self.voice_response_button.setChecked(True)
        self.voice_response_button.setStyleSheet("""
            QPushButton {
                color: white;
                padding: 12px 20px;
                border: none;
                border-radius: 8px;
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
        self.voice_response_button.clicked.connect(self.toggle_voice_response)
        input_layout.addWidget(self.voice_response_button)
        
        self.clear_button = QPushButton("Limpiar")
        self.clear_button.setFont(QFont("Segoe UI", 10))
        self.clear_button.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                padding: 12px 20px;
                border: none;
                border-radius: 8px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
        """)
        self.clear_button.clicked.connect(self.clear_chat)
        input_layout.addWidget(self.clear_button)
        
        chat_layout.addLayout(input_layout)
        
        # Welcome message
        self.add_system_message(
            "Welcome to AI Chat Assistant! Select a model and start chatting.\n\n"
            "⌨️ Atajos de teclado:\n"
            "  • Ctrl+Q - Cerrar programa (con limpieza)\n"
            "  • Ctrl+Shift+Q - Forzar cierre inmediato"
        )
        
        # Add chat column to content
        content_layout.addWidget(chat_column, stretch=1)
        
        # RIGHT COLUMN: Camera feed with YOLO
        camera_column = QWidget()
        camera_layout = QVBoxLayout(camera_column)
        camera_layout.setSpacing(10)
        camera_layout.setContentsMargins(0, 0, 0, 0)
        
        # Camera title
        camera_title = QLabel("📹 Live Camera + YOLO Detection")
        camera_title_font = QFont()
        camera_title_font.setPointSize(12)
        camera_title_font.setBold(True)
        camera_title.setFont(camera_title_font)
        camera_layout.addWidget(camera_title)
        
        # Video display
        self.video_label = QLabel()
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setMaximumSize(640, 480)
        self.video_label.setStyleSheet("""
            QLabel {
                background-color: #000;
                border: 2px solid #ddd;
                border-radius: 5px;
            }
        """)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setText("Initializing camera...")
        self.video_label.setStyleSheet("""
            QLabel {
                background-color: #000;
                color: white;
                border: 2px solid #ddd;
                border-radius: 5px;
                font-size: 14px;
            }
        """)
        camera_layout.addWidget(self.video_label)
        
        # Camera preview toggle button
        self.camera_toggle_btn = QPushButton("▶️ Activar Preview")
        self.camera_toggle_btn.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                border: none;
                padding: 10px;
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
        self.camera_toggle_btn.clicked.connect(self.toggle_camera_preview)
        camera_layout.addWidget(self.camera_toggle_btn)
        
        # Detection info
        self.detection_label = QLabel("Detections: None")
        self.detection_label.setStyleSheet("color: #666; font-style: italic;")
        self.detection_label.setWordWrap(True)
        camera_layout.addWidget(self.detection_label)
        
        camera_layout.addStretch()
        
        # Add camera column to content
        content_layout.addWidget(camera_column, stretch=1)
        
        # Add content layout to main layout
        chat_main_layout.addLayout(content_layout)
        
        return chat_widget
    
    def create_settings_tab(self):
        """Create the settings/configuration tab"""
        settings_widget = QWidget()
        
        # Create scroll area for settings
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        
        # Container widget for scroll area
        container = QWidget()
        main_layout = QVBoxLayout(container)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # Title
        title_label = QLabel("⚙️ Configuración del Asistente")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title_label.setFont(title_font)
        main_layout.addWidget(title_label)
        
        # ========== LLM CONFIGURATION ==========
        llm_group = QGroupBox("🤖 Configuración de Modelos LLM")
        llm_group.setStyleSheet("QGroupBox { font-weight: bold; font-size: 12pt; }")
        llm_layout = QFormLayout()
        llm_layout.setSpacing(10)
        
        # Ollama model selector
        self.ollama_model_combo = QComboBox()
        self.ollama_model_combo.addItems(["llama3.2:1b", "mistral:7b"])
        self.ollama_model_combo.setCurrentText("llama3.2:1b")
        llm_layout.addRow("Modelo Ollama:", self.ollama_model_combo)
        
        # Groq model selector
        self.groq_model_combo = QComboBox()
        self.groq_model_combo.addItems([
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768",
            "gemma2-9b-it"
        ])
        self.groq_model_combo.setCurrentText("llama-3.3-70b-versatile")
        llm_layout.addRow("Modelo Groq:", self.groq_model_combo)
        
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
        save_keys_btn.clicked.connect(self.save_api_keys)
        llm_layout.addRow("", save_keys_btn)
        
        llm_group.setLayout(llm_layout)
        main_layout.addWidget(llm_group)
        
        # ========== CAMERA CONFIGURATION ==========
        camera_group = QGroupBox("📷 Configuración de Cámara y YOLO")
        camera_group.setStyleSheet("QGroupBox { font-weight: bold; font-size: 12pt; }")
        camera_layout = QFormLayout()
        camera_layout.setSpacing(10)
        
        # YOLO confidence slider
        yolo_conf_layout = QHBoxLayout()
        self.yolo_confidence_slider = QSlider(Qt.Horizontal)
        self.yolo_confidence_slider.setMinimum(10)
        self.yolo_confidence_slider.setMaximum(90)
        self.yolo_confidence_slider.setValue(50)
        self.yolo_confidence_slider.setTickPosition(QSlider.TicksBelow)
        self.yolo_confidence_slider.setTickInterval(10)
        self.yolo_confidence_label = QLabel("0.50")
        self.yolo_confidence_slider.valueChanged.connect(
            lambda v: self.yolo_confidence_label.setText(f"{v/100:.2f}")
        )
        yolo_conf_layout.addWidget(self.yolo_confidence_slider)
        yolo_conf_layout.addWidget(self.yolo_confidence_label)
        camera_layout.addRow("Confianza YOLO:", yolo_conf_layout)
        
        # Resolution selector
        self.resolution_combo = QComboBox()
        self.resolution_combo.addItems(["320x240", "640x480", "1280x720"])
        self.resolution_combo.setCurrentText("640x480")
        camera_layout.addRow("Resolución:", self.resolution_combo)
        
        # Enable YOLO checkbox
        self.enable_yolo_checkbox = QCheckBox("Habilitar detección YOLO")
        self.enable_yolo_checkbox.setChecked(True)
        camera_layout.addRow("", self.enable_yolo_checkbox)
        
        camera_group.setLayout(camera_layout)
        main_layout.addWidget(camera_group)
        
        # ========== AUDIO CONFIGURATION ==========
        audio_group = QGroupBox("🎤 Configuración de Audio")
        audio_group.setStyleSheet("QGroupBox { font-weight: bold; font-size: 12pt; }")
        audio_layout = QFormLayout()
        audio_layout.setSpacing(10)
        
        # TTS Engine selector
        tts_engine_layout = QHBoxLayout()
        self.tts_engine_combo = QComboBox()
        self.tts_engine_combo.addItems(["pyttsx3 (Offline)", "Google TTS (Online)"])
        self.tts_engine_combo.setCurrentIndex(0)  # Default to pyttsx3
        self.tts_engine_combo.currentIndexChanged.connect(self.on_tts_engine_changed)
        tts_engine_layout.addWidget(self.tts_engine_combo)
        
        # Status indicator for gTTS availability
        self.gtts_status_label = QLabel()
        if self.audio_service and self.audio_service.is_gtts_available():
            self.gtts_status_label.setText("✅ gTTS disponible")
            self.gtts_status_label.setStyleSheet("color: green; font-weight: bold;")
        else:
            self.gtts_status_label.setText("❌ gTTS no disponible")
            self.gtts_status_label.setStyleSheet("color: red; font-weight: bold;")
            self.tts_engine_combo.setItemData(1, 0, Qt.UserRole - 1)  # Disable gTTS option
        tts_engine_layout.addWidget(self.gtts_status_label)
        
        audio_layout.addRow("Motor de voz:", tts_engine_layout)
        
        # Voice volume slider
        volume_layout = QHBoxLayout()
        self.voice_volume_slider = QSlider(Qt.Horizontal)
        self.voice_volume_slider.setMinimum(0)
        self.voice_volume_slider.setMaximum(100)
        self.voice_volume_slider.setValue(95)  # Optimized to 95%
        self.voice_volume_slider.setTickPosition(QSlider.TicksBelow)
        self.voice_volume_slider.setTickInterval(20)
        self.voice_volume_label = QLabel("95%")
        self.voice_volume_slider.valueChanged.connect(
            lambda v: self.voice_volume_label.setText(f"{v}%")
        )
        volume_layout.addWidget(self.voice_volume_slider)
        volume_layout.addWidget(self.voice_volume_label)
        audio_layout.addRow("Volumen de voz:", volume_layout)
        
        # Voice speed slider (only for pyttsx3)
        speed_layout = QHBoxLayout()
        self.voice_speed_slider = QSlider(Qt.Horizontal)
        self.voice_speed_slider.setMinimum(50)
        self.voice_speed_slider.setMaximum(200)
        self.voice_speed_slider.setValue(130)  # Optimized to 130 (more natural)
        self.voice_speed_slider.setTickPosition(QSlider.TicksBelow)
        self.voice_speed_slider.setTickInterval(25)
        self.voice_speed_label = QLabel("130 (Optimizado)")
        self.voice_speed_slider.valueChanged.connect(
            lambda v: self.voice_speed_label.setText(f"{v}")
        )
        speed_layout.addWidget(self.voice_speed_slider)
        speed_layout.addWidget(self.voice_speed_label)
        self.voice_speed_row_label = QLabel("Velocidad (pyttsx3):")
        audio_layout.addRow(self.voice_speed_row_label, speed_layout)
        
        # Listen timeout
        self.listen_timeout_spin = QSpinBox()
        self.listen_timeout_spin.setMinimum(3)
        self.listen_timeout_spin.setMaximum(10)
        self.listen_timeout_spin.setValue(5)
        self.listen_timeout_spin.setSuffix(" segundos")
        audio_layout.addRow("Timeout escucha:", self.listen_timeout_spin)
        
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
        """Apply configuration changes"""
        try:
            # Apply Ollama model
            ollama_model = self.ollama_model_combo.currentText()
            if hasattr(self.ollama_service, 'model'):
                self.ollama_service.model = ollama_model
            
            # Apply Groq model
            groq_model = self.groq_model_combo.currentText()
            if hasattr(self.groq_service, 'model'):
                self.groq_service.model = groq_model
            
            # Apply YOLO confidence
            yolo_confidence = self.yolo_confidence_slider.value() / 100.0
            if hasattr(self, 'camera_service') and self.camera_service:
                self.camera_service.confidence = yolo_confidence
            
            # Apply resolution
            resolution = self.resolution_combo.currentText()
            if hasattr(self, 'camera_service') and self.camera_service:
                width, height = map(int, resolution.split('x'))
                # Store for next camera restart
                self.camera_service.target_width = width
                self.camera_service.target_height = height
            
            # Apply YOLO enable/disable
            yolo_enabled = self.enable_yolo_checkbox.isChecked()
            if hasattr(self, 'camera_service') and self.camera_service:
                self.camera_service.yolo_enabled = yolo_enabled
            
            # Apply audio settings
            if hasattr(self, 'audio_service') and self.audio_service:
                volume = self.voice_volume_slider.value() / 100.0
                speed = self.voice_speed_slider.value()
                
                # Update TTS engine properties
                try:
                    if hasattr(self.audio_service, 'tts_engine'):
                        self.audio_service.tts_engine.setProperty('volume', volume)
                        self.audio_service.tts_engine.setProperty('rate', speed)
                except Exception as e:
                    print(f"Error setting audio properties: {e}")
            
            # Store listen timeout for next voice input
            self.listen_timeout = self.listen_timeout_spin.value()
            
            # Store auto-vision setting
            self.auto_vision_enabled = self.auto_vision_checkbox.isChecked()
            
            QMessageBox.information(
                self,
                "Configuración Aplicada",
                "✓ Los cambios se han aplicado correctamente.\n\n"
                f"• Modelo Ollama: {ollama_model}\n"
                f"• Modelo Groq: {groq_model}\n"
                f"• Confianza YOLO: {yolo_confidence:.2f}\n"
                f"• Resolución: {resolution}\n"
                f"• YOLO: {'Habilitado' if yolo_enabled else 'Deshabilitado'}\n"
                f"• Volumen: {int(volume*100)}%\n"
                f"• Velocidad: {speed}%\n"
                f"• Auto-visión: {'Habilitada' if self.auto_vision_enabled else 'Deshabilitada'}"
            )
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
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
            
            # Detener todos los procesos de movimiento
            subprocess.run(["pkill", "-f", "TEST_ASENTIR.py"], check=False)
            subprocess.run(["pkill", "-f", "TEST_NEGAR.py"], check=False)
            subprocess.run(["pkill", "-f", "TEST_ROLL.py"], check=False)
            subprocess.run(["pkill", "-f", "test_boca.py"], check=False)
            subprocess.run(["pkill", "-f", "test_seguir_rostro.py"], check=False)
            subprocess.run(["pkill", "-f", "TEST_SEGUIMIENTO_FINAL.PY"], check=False)
            
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
            
            # Reiniciar la cámara si estaba liberada
            if hasattr(self, 'camera_service') and self.camera_service:
                try:
                    self.add_system_message("📷 Reiniciando cámara del servicio principal...")
                    self.camera_service.start_camera()
                    time.sleep(0.3)
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
                    self.camera_service.stop_camera()
                    time.sleep(0.5)  # Dar tiempo a que se libere completamente
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
            
            # Ejecutar el script en background sin capturar salida (para ver errores en terminal)
            process = subprocess.Popen(
                [python_path, script_path],
                cwd=base_dir,
                stdout=None,  # No redirigir stdout, que vaya a la terminal
                stderr=None   # No redirigir stderr, que vaya a la terminal
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
    
    def add_system_message(self, message: str):
        """Add a system message to the chat display"""
        self.chat_display.append(f'<div style="color: #888; font-style: italic; margin: 5px 0;">{message}</div>')
    
    def add_user_message(self, message: str):
        """Add a user message to the chat display"""
        self.chat_display.append(
            f'<div style="background-color: #e3f2fd; padding: 10px; margin: 5px 0; border-radius: 5px;">'
            f'<strong>You:</strong> {message}'
            f'</div>'
        )
        self.scroll_to_bottom()
    
    def add_assistant_message(self, message: str):
        """Add an assistant message to the chat display"""
        self.chat_display.append(
            f'<div style="background-color: #f1f8e9; padding: 10px; margin: 5px 0; border-radius: 5px;">'
            f'<strong>Assistant:</strong> {message}'
            f'</div>'
        )
        self.scroll_to_bottom()
    
    def scroll_to_bottom(self):
        """Scroll chat display to the bottom"""
        cursor = self.chat_display.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.chat_display.setTextCursor(cursor)
    
    @Slot()
    def on_model_changed(self):
        """Handle model selection change"""
        model = self.model_selector.currentText()
        self.add_system_message(f"Switched to {model}")
        
        # Check if selected model is available
        if model == "ChatGPT" and not self.chatgpt_service.is_available():
            QMessageBox.warning(
                self,
                "ChatGPT Not Available",
                "ChatGPT API key not configured.\n\nPlease set OPENAI_API_KEY in the .env file."
            )
        elif model == "Ollama" and not self.ollama_service.is_available():
            QMessageBox.warning(
                self,
                "Ollama Not Available",
                "Ollama is not running.\n\nPlease start Ollama with: ollama serve"
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
                vision_context = self.camera_service.get_detection_summary()
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
            self.camera_toggle_btn.setText("⏸️ Pausar Preview")
            self.camera_toggle_btn.setStyleSheet("""
                QPushButton {
                    background-color: #ff9800;
                    color: white;
                    border: none;
                    padding: 8px;
                    border-radius: 4px;
                    font-weight: bold;
                    font-size: 13px;
                }
                QPushButton:hover {
                    background-color: #e68900;
                }
                QPushButton:pressed {
                    background-color: #cc7a00;
                }
            """)
            self.add_system_message("📹 Preview de cámara activado (YOLO sigue activo)")
        else:
            # Preview paused
            self.camera_toggle_btn.setText("▶️ Activar Preview")
            self.camera_toggle_btn.setStyleSheet("""
                QPushButton {
                    background-color: #4CAF50;
                    color: white;
                    border: none;
                    padding: 8px;
                    border-radius: 4px;
                    font-weight: bold;
                    font-size: 13px;
                }
                QPushButton:hover {
                    background-color: #45a049;
                }
                QPushButton:pressed {
                    background-color: #3d8b40;
                }
            """)
            self.video_label.setText("Preview pausado\n\n✅ YOLO sigue detectando objetos\n🎤 Audio optimizado\n\n'¿Qué ves?' funcionará normalmente\n\nClick 'Activar Preview' para ver el video")
            self.video_label.setStyleSheet("""
                QLabel {
                    background-color: #000;
                    color: #888;
                    border: 2px solid #ddd;
                    border-radius: 5px;
                    font-size: 13px;
                }
            """)
            self.add_system_message("⏸️ Preview pausado (YOLO activo en background, audio optimizado)")
    
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
        
        # Update detections
        self.latest_detections = detections
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
