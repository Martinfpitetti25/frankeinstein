"""
Script de prueba para verificar los servicios de ChatGPT y Ollama
"""
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))
from services import ChatGPTService, OllamaService

def test_chatgpt():
    """Prueba el servicio de ChatGPT"""
    print("\n" + "="*50)
    print("🤖 Probando ChatGPT...")
    print("="*50)
    
    service = ChatGPTService()
    
    if not service.is_available():
        print("❌ ChatGPT no está disponible")
        print("   Verifica tu API key en el archivo .env")
        return False
    
    print("✅ ChatGPT está disponible")
    print("📤 Enviando mensaje de prueba...")
    
    try:
        response = service.send_message("Di 'Hola' en una palabra")
        print(f"📥 Respuesta: {response}")
        return True
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

def test_ollama():
    """Prueba el servicio de Ollama"""
    print("\n" + "="*50)
    print("🦙 Probando Ollama...")
    print("="*50)
    
    service = OllamaService()
    
    if not service.is_available():
        print("❌ Ollama no está disponible")
        print("   Asegúrate de que Ollama esté corriendo (ollama serve)")
        return False
    
    print("✅ Ollama está disponible")
    
    # Listar modelos
    models = service.get_available_models()
    print(f"📦 Modelos disponibles: {', '.join(models)}")
    
    if not models:
        print("⚠️  No hay modelos instalados")
        print("   Descarga uno con: ollama pull llama3.2:1b")
        return False
    
    print(f"🎯 Usando modelo: {service.model}")
    print("📤 Enviando mensaje de prueba...")
    
    try:
        response = service.send_message("Say 'Hello' in one word")
        print(f"📥 Respuesta: {response}")
        return True
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

def main():
    """Ejecuta las pruebas"""
    load_dotenv()
    
    print("\n🔍 Verificando servicios de chat...")
    
    chatgpt_ok = test_chatgpt()
    ollama_ok = test_ollama()
    
    print("\n" + "="*50)
    print("📊 RESUMEN")
    print("="*50)
    print(f"ChatGPT: {'✅ Funcionando' if chatgpt_ok else '❌ No disponible'}")
    print(f"Ollama:  {'✅ Funcionando' if ollama_ok else '❌ No disponible'}")
    print()
    
    if chatgpt_ok and ollama_ok:
        print("🎉 ¡Ambos servicios están funcionando correctamente!")
        print("   Puedes iniciar la aplicación con: python src/main.py")
    elif chatgpt_ok or ollama_ok:
        print("⚠️  Solo un servicio está disponible")
        print("   La aplicación funcionará pero solo con el servicio activo")
    else:
        print("❌ Ningún servicio está disponible")
        print("   Configura al menos uno antes de usar la aplicación")

if __name__ == "__main__":
    main()
