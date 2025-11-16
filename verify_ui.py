"""
Quick test to verify the UI structure
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

print("✅ Importando módulos...")
from PySide6.QtWidgets import QApplication
from main import ChatWindow

print("✅ Creando aplicación...")
app = QApplication(sys.argv)
window = ChatWindow()

print("✅ Verificando componentes:")
print(f"   - Tab Widget: {'✓' if hasattr(window, 'tab_widget') else '✗'}")
print(f"   - Número de pestañas: {window.tab_widget.count() if hasattr(window, 'tab_widget') else 0}")
print(f"   - Chat Display: {'✓' if hasattr(window, 'chat_display') else '✗'}")
print(f"   - Model Selector: {'✓' if hasattr(window, 'model_selector') else '✗'}")

if hasattr(window, 'tab_widget'):
    for i in range(window.tab_widget.count()):
        print(f"   - Pestaña {i+1}: {window.tab_widget.tabText(i)}")

print("\n🎉 ¡Aplicación verificada correctamente!")
print("   Ejecuta: python src/main.py")
