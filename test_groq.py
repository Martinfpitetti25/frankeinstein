#!/usr/bin/env python3
"""
Test script for Groq integration
"""
import sys
import os
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from services.groq_service import GroqService

print("="*70)
print("TEST DE GROQ - Verificación de API")
print("="*70)

# Cargar .env
load_dotenv()

# Crear servicio
service = GroqService()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    print("\n❌ NO se encontró GROQ_API_KEY en el archivo .env")
    print("\n📝 Para obtener tu API key GRATIS:")
    print("   1. Ve a: https://console.groq.com/keys")
    print("   2. Crea una cuenta (gratis, sin tarjeta)")
    print("   3. Genera una API key")
    print("   4. Agrégala al archivo .env:")
    print("      GROQ_API_KEY=gsk_tu_key_aqui")
    print("\n📖 Lee GROQ_SETUP.md para instrucciones detalladas")
    sys.exit(1)

print(f"\n✓ API Key encontrada: {api_key[:20]}...")
print(f"✓ Servicio disponible: {'Sí' if service.is_available() else 'No'}")
print(f"✓ Modelo por defecto: {service.model}")

# Listar modelos disponibles
print("\n" + "-"*70)
print("Modelos disponibles en Groq:")
print("-"*70)
for i, model in enumerate(service.get_available_models(), 1):
    print(f"{i}. {model}")

# Probar envío de mensaje
print("\n" + "-"*70)
print("Enviando mensaje de prueba...")
print("-"*70 + "\n")

try:
    response = service.send_message("Di solo: 'Groq funciona perfectamente'")
    
    if response.startswith("❌"):
        print(response)
        print("\n💡 Verifica tu API key en: https://console.groq.com/keys")
    else:
        print(f"✅ ÉXITO! Groq respondió:")
        print(f"   {response}")
        print("\n" + "="*70)
        print("✅ Groq está configurado y funcionando correctamente!")
        print("="*70)
        print("\n💡 Ahora puedes usar Groq en la aplicación:")
        print("   1. python src/main.py")
        print("   2. Selecciona 'Groq' del menú desplegable")
        print("   3. ¡Disfruta respuestas rápidas e inteligentes!")
        
except Exception as e:
    print(f"❌ Error: {str(e)}")
    print("\n💡 Soluciones:")
    print("   - Verifica tu conexión a internet")
    print("   - Verifica que tu API key sea válida")
    print("   - Ve a: https://console.groq.com/keys")
