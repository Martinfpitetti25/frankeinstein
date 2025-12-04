"""
Chat service classes for ChatGPT and Ollama integration
"""
import os
from openai import OpenAI
import requests
from typing import Optional
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.logger import get_logger

logger = get_logger(__name__)


class ChatGPTService:
    """Service for interacting with ChatGPT API"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if self.api_key:
            self.client = OpenAI(api_key=self.api_key)
        else:
            self.client = None
        self.system_prompt = "Eres un asistente de IA útil y amigable."
    
    def is_available(self) -> bool:
        """Check if the service is properly configured"""
        return self.client is not None and self.api_key is not None
    
    def send_message(self, message: str, conversation_history: list = None, vision_context: str = None) -> str:
        """
        Send a message to ChatGPT and get response
        
        Args:
            message: The user message
            conversation_history: List of previous messages in format [{"role": "user/assistant", "content": "..."}]
            vision_context: Optional context about what the camera sees
        
        Returns:
            The assistant's response
        """
        if not self.is_available():
            return "Error: ChatGPT API key not configured. Please set OPENAI_API_KEY in .env file"
        
        try:
            messages = conversation_history or []
            
            # Add vision context if provided
            if vision_context:
                enhanced_message = f"""Contexto: Tienes acceso a una cámara con detección de objetos YOLO.
{vision_context}

Basándote en esta información de la cámara, responde a la siguiente pregunta del usuario:
{message}"""
                messages.append({"role": "user", "content": enhanced_message})
            else:
                messages.append({"role": "user", "content": message})
            
            # Add system prompt at the beginning if not already there
            if not messages or messages[0].get("role") != "system":
                messages.insert(0, {"role": "system", "content": self.system_prompt})
            
            logger.debug(f"Sending message to ChatGPT (length: {len(message)} chars)")
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=1000,
                temperature=0.7
            )
            
            response_text = response.choices[0].message.content
            logger.info(f"ChatGPT response received (length: {len(response_text)} chars)")
            return response_text
        
        except Exception as e:
            logger.error(f"Error communicating with ChatGPT: {str(e)}", exc_info=True)
            return f"Error communicating with ChatGPT: {str(e)}"


class OllamaService:
    """Service for interacting with local Ollama installation"""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.model = "llama3.2:1b"  # Default model
        self.system_prompt = "Eres un asistente de IA útil y amigable."
    
    def is_available(self) -> bool:
        """Check if Ollama is running locally"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def get_available_models(self) -> list:
        """Get list of available Ollama models"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return [model["name"] for model in data.get("models", [])]
            return []
        except:
            return []
    
    def set_model(self, model_name: str):
        """Set the active model"""
        self.model = model_name
    
    def send_message(self, message: str, conversation_history: list = None, vision_context: str = None) -> str:
        """
        Send a message to Ollama and get response
        
        Args:
            message: The user message
            conversation_history: List of previous messages (not used for simple implementation)
            vision_context: Optional context about what the camera sees
        
        Returns:
            The assistant's response
        """
        if not self.is_available():
            return "Error: Ollama is not running. Please start Ollama service (ollama serve)"
        
        try:
            # Add vision context if provided
            prompt = message
            if vision_context:
                # Simple and direct approach for better model understanding
                prompt = f"""{vision_context}

Pregunta: {message}"""
            
            # Prepend system prompt
            full_prompt = f"{self.system_prompt}\n\n{prompt}"
            
            payload = {
                "model": self.model,
                "prompt": full_prompt,
                "stream": False
            }
            
            logger.debug(f"Sending message to Ollama (model: {self.model}, length: {len(message)} chars)")
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                response_text = response.json().get("response", "No response received")
                logger.info(f"Ollama response received (length: {len(response_text)} chars)")
                return response_text
            else:
                logger.error(f"Ollama returned status code {response.status_code}")
                return f"Error: Ollama returned status code {response.status_code}"
        
        except requests.exceptions.Timeout:
            logger.warning("Ollama request timed out")
            return "Error: Request timed out. The model might be too slow or not responding."
        except Exception as e:
            logger.error(f"Error communicating with Ollama: {str(e)}", exc_info=True)
            return f"Error communicating with Ollama: {str(e)}"
