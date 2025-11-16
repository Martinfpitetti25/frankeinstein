"""
Audio service for speech recognition and text-to-speech
"""
import speech_recognition as sr
import pyttsx3
import logging
from typing import Optional, Tuple
import threading
import os
import tempfile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import Google TTS (optional dependency)
try:
    from gtts import gTTS
    import pygame
    GTTS_AVAILABLE = True
    logger.info("Google TTS (gTTS) available")
except ImportError:
    GTTS_AVAILABLE = False
    logger.warning("Google TTS not available. Install with: pip install gtts pygame")


class AudioService:
    """Service for handling speech recognition and text-to-speech"""
    
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = None
        self.tts_engine = None
        self.tts_lock = threading.Lock()
        self.is_speaking = False
        
        # TTS engine selection
        self.tts_engine_type = "pyttsx3"  # Default: pyttsx3 or gtts
        
        # Initialize pyttsx3 engine
        try:
            self.tts_engine = pyttsx3.init()
            self._configure_tts()
            logger.info("pyttsx3 TTS engine initialized")
        except Exception as e:
            logger.error(f"Failed to initialize pyttsx3 engine: {str(e)}")
        
        # Initialize pygame mixer for gTTS playback
        if GTTS_AVAILABLE:
            try:
                pygame.mixer.init()
                logger.info("pygame mixer initialized for gTTS")
            except Exception as e:
                logger.warning(f"Failed to initialize pygame mixer: {str(e)}")
    
    def _configure_tts(self):
        """Configure pyttsx3 TTS engine settings (optimized)"""
        if self.tts_engine:
            try:
                # Optimized settings for more natural speech
                self.tts_engine.setProperty('rate', 130)  # Slower = more natural (was 150)
                self.tts_engine.setProperty('volume', 0.95)  # Slightly higher volume
                
                # Try to use a better voice if available
                voices = self.tts_engine.getProperty('voices')
                if voices:
                    # Prefer Spanish voice if available, otherwise use first available
                    spanish_voice = None
                    female_voice = None
                    
                    for voice in voices:
                        voice_name_lower = voice.name.lower()
                        
                        # Look for Spanish voices first
                        if 'spanish' in voice_name_lower or 'español' in voice_name_lower:
                            spanish_voice = voice
                            break
                        
                        # Fallback: prefer female voices (usually sound better)
                        if 'female' in voice_name_lower or 'mujer' in voice_name_lower:
                            female_voice = voice
                    
                    # Priority: Spanish > Female > First available
                    selected_voice = spanish_voice or female_voice or voices[0]
                    self.tts_engine.setProperty('voice', selected_voice.id)
                    logger.info(f"Using pyttsx3 voice: {selected_voice.name}")
                    
            except Exception as e:
                logger.warning(f"Error configuring pyttsx3 TTS: {str(e)}")
    
    def is_microphone_available(self) -> bool:
        """Check if a microphone is available"""
        try:
            mic_list = sr.Microphone.list_microphone_names()
            return len(mic_list) > 0
        except Exception as e:
            logger.error(f"Error checking microphone availability: {str(e)}")
            return False
    
    def get_microphone_list(self) -> list:
        """Get list of available microphones"""
        try:
            return sr.Microphone.list_microphone_names()
        except Exception as e:
            logger.error(f"Error getting microphone list: {str(e)}")
            return []
    
    def listen_once(self, timeout: int = 5, phrase_time_limit: int = 10) -> Tuple[bool, Optional[str]]:
        """
        Listen to microphone once and convert speech to text
        
        Args:
            timeout: Maximum time to wait for speech to start (seconds)
            phrase_time_limit: Maximum time for the phrase (seconds)
        
        Returns:
            Tuple of (success, transcribed_text)
        """
        if not self.is_microphone_available():
            logger.error("No microphone available")
            return False, "No microphone detected"
        
        try:
            with sr.Microphone() as source:
                logger.info("Listening...")
                
                # Adjust for ambient noise
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                
                # Listen for audio
                try:
                    audio = self.recognizer.listen(
                        source,
                        timeout=timeout,
                        phrase_time_limit=phrase_time_limit
                    )
                except sr.WaitTimeoutError:
                    logger.warning("Listening timeout - no speech detected")
                    return False, "No speech detected (timeout)"
                
                logger.info("Processing audio...")
                
                # Try to recognize speech using Google Speech Recognition
                try:
                    # Try Spanish first, then English
                    try:
                        text = self.recognizer.recognize_google(audio, language='es-ES')
                        logger.info(f"Recognized (Spanish): {text}")
                        return True, text
                    except:
                        text = self.recognizer.recognize_google(audio, language='en-US')
                        logger.info(f"Recognized (English): {text}")
                        return True, text
                
                except sr.UnknownValueError:
                    logger.warning("Could not understand audio")
                    return False, "Could not understand audio"
                except sr.RequestError as e:
                    logger.error(f"Speech recognition service error: {str(e)}")
                    return False, f"Speech recognition error: {str(e)}"
        
        except Exception as e:
            logger.error(f"Error during speech recognition: {str(e)}")
            return False, f"Error: {str(e)}"
    
    def speak(self, text: str, blocking: bool = False):
        """
        Convert text to speech using selected engine
        
        Args:
            text: Text to speak
            blocking: If True, wait for speech to complete. If False, speak in background
        """
        if not text or text.strip() == "":
            return
        
        # Use selected TTS engine
        if self.tts_engine_type == "gtts" and GTTS_AVAILABLE:
            self._speak_gtts(text, blocking)
        else:
            # Fallback to pyttsx3
            self._speak_pyttsx3(text, blocking)
    
    def _speak_pyttsx3(self, text: str, blocking: bool = False):
        """Speak using pyttsx3 engine"""
        if not self.tts_engine:
            logger.error("pyttsx3 engine not initialized")
            return
        
        def _speak():
            with self.tts_lock:
                try:
                    self.is_speaking = True
                    logger.info(f"Speaking (pyttsx3): {text[:50]}...")
                    self.tts_engine.say(text)
                    self.tts_engine.runAndWait()
                    self.is_speaking = False
                    logger.info("Finished speaking")
                except Exception as e:
                    logger.error(f"Error during pyttsx3 TTS: {str(e)}")
                    self.is_speaking = False
        
        if blocking:
            _speak()
        else:
            # Speak in a separate thread to avoid blocking
            thread = threading.Thread(target=_speak, daemon=True)
            thread.start()
    
    def _speak_gtts(self, text: str, blocking: bool = False):
        """Speak using Google TTS (gTTS) - requires internet"""
        if not GTTS_AVAILABLE:
            logger.warning("gTTS not available, falling back to pyttsx3")
            self._speak_pyttsx3(text, blocking)
            return
        
        def _speak():
            with self.tts_lock:
                try:
                    self.is_speaking = True
                    logger.info(f"Speaking (gTTS): {text[:50]}...")
                    
                    # Create TTS object with Spanish language
                    tts = gTTS(text=text, lang='es', slow=False)
                    
                    # Save to temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
                        temp_file = fp.name
                        tts.save(temp_file)
                    
                    # Play audio using pygame
                    pygame.mixer.music.load(temp_file)
                    pygame.mixer.music.play()
                    
                    # Wait for playback to finish
                    while pygame.mixer.music.get_busy():
                        pygame.time.Clock().tick(10)
                    
                    # Clean up temp file
                    try:
                        os.unlink(temp_file)
                    except:
                        pass
                    
                    self.is_speaking = False
                    logger.info("Finished speaking (gTTS)")
                    
                except Exception as e:
                    logger.error(f"Error during gTTS TTS: {str(e)}")
                    logger.info("Falling back to pyttsx3")
                    self.is_speaking = False
                    # Fallback to pyttsx3 on error
                    self._speak_pyttsx3(text, blocking=True)
        
        if blocking:
            _speak()
        else:
            # Speak in a separate thread to avoid blocking
            thread = threading.Thread(target=_speak, daemon=True)
            thread.start()
    
    def set_tts_engine(self, engine_type: str):
        """
        Set TTS engine type
        
        Args:
            engine_type: "pyttsx3" or "gtts"
        """
        if engine_type == "gtts" and not GTTS_AVAILABLE:
            logger.warning("gTTS not available, keeping pyttsx3")
            return False
        
        self.tts_engine_type = engine_type
        logger.info(f"TTS engine set to: {engine_type}")
        return True
    
    def get_tts_engine(self) -> str:
        """Get current TTS engine type"""
        return self.tts_engine_type
    
    def is_gtts_available(self) -> bool:
        """Check if Google TTS is available"""
        return GTTS_AVAILABLE
    
    def stop_speaking(self):
        """Stop current speech"""
        if self.tts_engine and self.is_speaking:
            try:
                self.tts_engine.stop()
                self.is_speaking = False
                logger.info("Speech stopped")
            except Exception as e:
                logger.error(f"Error stopping speech: {str(e)}")
    
    def set_speech_rate(self, rate: int):
        """Set speech rate (words per minute)"""
        if self.tts_engine:
            try:
                self.tts_engine.setProperty('rate', rate)
                logger.info(f"Speech rate set to {rate}")
            except Exception as e:
                logger.error(f"Error setting speech rate: {str(e)}")
    
    def set_volume(self, volume: float):
        """Set volume (0.0 to 1.0)"""
        if self.tts_engine:
            try:
                volume = max(0.0, min(1.0, volume))  # Clamp between 0 and 1
                self.tts_engine.setProperty('volume', volume)
                logger.info(f"Volume set to {volume}")
            except Exception as e:
                logger.error(f"Error setting volume: {str(e)}")
    
    def test_microphone(self) -> Tuple[bool, str]:
        """Test microphone by recording a short sample"""
        try:
            logger.info("Testing microphone...")
            with sr.Microphone() as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                logger.info("Microphone test successful")
                return True, "Microphone is working"
        except Exception as e:
            logger.error(f"Microphone test failed: {str(e)}")
            return False, f"Microphone test failed: {str(e)}"
    
    def test_speakers(self) -> Tuple[bool, str]:
        """Test speakers by playing a short sound"""
        try:
            logger.info("Testing speakers...")
            self.speak("Audio test", blocking=True)
            return True, "Speakers are working"
        except Exception as e:
            logger.error(f"Speaker test failed: {str(e)}")
            return False, f"Speaker test failed: {str(e)}"
    
    def __del__(self):
        """Cleanup on deletion"""
        if self.tts_engine:
            try:
                self.tts_engine.stop()
            except:
                pass
