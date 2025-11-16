#!/bin/bash
# Script de lanzamiento para Robot AI Assistant
# Uso: ./run.sh

echo "🤖 Iniciando Robot AI Assistant..."
echo ""

# Ir al directorio del proyecto
cd /home/isaecluster/robot_ai

# Activar entorno virtual
echo "⏳ Activando entorno virtual..."
source env/bin/activate

# Verificar que se activó correctamente
if [ -z "$VIRTUAL_ENV" ]; then
    echo "❌ Error: No se pudo activar el entorno virtual"
    exit 1
fi

echo "✓ Entorno virtual activado"
echo ""

# Lanzar la aplicación
echo "🚀 Lanzando aplicación..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

python src/main.py

# Capturar código de salida
EXIT_CODE=$?

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Aplicación cerrada correctamente"
else
    echo "⚠️  La aplicación terminó con errores (código: $EXIT_CODE)"
fi

exit $EXIT_CODE
