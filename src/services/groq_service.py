"""
Groq service for fast LLM inference
"""
import os
from groq import Groq
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GroqService:
    """Service for interacting with Groq API"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if self.api_key:
            self.client = Groq(api_key=self.api_key)
        else:
            self.client = None
        self.model = "llama-3.3-70b-versatile"  # Default model (very powerful and fast)
        self.system_prompt = "Eres un asistente de IA útil y amigable."
    
    def is_available(self) -> bool:
        """Check if the service is properly configured"""
        return self.client is not None and self.api_key is not None
    
    def set_model(self, model_name: str):
        """
        Set the active model
        
        Available models:
        - llama-3.3-70b-versatile (recommended, newest and most intelligent)
        - llama-3.1-8b-instant (faster, still good)
        - mixtral-8x7b-32768 (excellent, large context)
        - gemma2-9b-it (fast and efficient)
        """
        self.model = model_name
        logger.info(f"Groq model set to: {model_name}")
    
    def send_message(self, message: str, conversation_history: list = None, vision_context: str = None) -> str:
        """
        Send a message to Groq and get response
        
        Args:
            message: The user message
            conversation_history: List of previous messages in format [{"role": "user/assistant", "content": "..."}]
            vision_context: Optional context about what the camera sees
        
        Returns:
            The assistant's response
        """
        if not self.is_available():
            return "Error: Groq API key not configured. Get free API key at: https://console.groq.com/keys"
        
        try:
            messages = conversation_history or []
            
            # Add vision context if provided
            if vision_context:
                enhanced_message = f"""{vision_context}

Pregunta: {message}"""
                messages.append({"role": "user", "content": enhanced_message})
            else:
                messages.append({"role": "user", "content": message})
            
            # Add system prompt at the beginning if not already there
            if not messages or messages[0].get("role") != "system":
                messages.insert(0, {"role": "system", "content": self.system_prompt})
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=1000,
                temperature=0.7
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            error_str = str(e)
            
            # Handle specific errors
            if "rate_limit" in error_str.lower() or "429" in error_str:
                return "❌ Error Groq: Límite de velocidad alcanzado.\n\nEspera unos segundos e intenta de nuevo.\nPlan gratuito: 30 requests/minuto"
            elif "invalid_api_key" in error_str.lower() or "401" in error_str:
                return "❌ Error Groq: API Key inválida.\n\nSolución:\n1. Ve a https://console.groq.com/keys\n2. Genera una nueva API key (es gratis)\n3. Actualiza GROQ_API_KEY en el archivo .env"
            elif "quota" in error_str.lower():
                return "❌ Error Groq: Límite diario alcanzado.\n\nPlan gratuito tiene límites diarios.\nPuedes:\n1. Esperar hasta mañana\n2. Usar Ollama (ilimitado)"
            else:
                return f"❌ Error de Groq:\n{error_str}\n\nVisita https://console.groq.com para más detalles"
    
    def get_available_models(self) -> list:
        """Get list of available Groq models"""
        return [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768",
            "gemma2-9b-it"
        ]
