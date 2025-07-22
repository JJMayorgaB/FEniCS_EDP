import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from datetime import datetime

# Obtener el directorio donde está este script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Crear directorio de salida relativo al script
output_dir = os.path.join(script_dir, "..", "..", "figures", "metrics")
os.makedirs(output_dir, exist_ok=True)

# Buscar archivos de métricas automáticamente
def find_metrics_file():
    # Buscar en el directorio metrics_results que está junto a este script
    metrics_dir = os.path.join(script_dir, "metrics_results")
    pattern = os.path.join(metrics_dir, "wave_performance_*")
    local_metrics_files = glob.glob(pattern)
    
    if not local_metrics_files:
        print("❌ No se encontraron archivos de métricas.")
        print("   Opciones disponibles:")
        print("   1. Ejecutar primero: bash metrics.sh (desde scripts/wave_eq_parallel/)")
        print("   2. Verificar que exista el directorio metrics_results/")
        print(f"   Buscando en: {metrics_dir}")
        print(f"   Script ejecutándose desde: {os.getcwd()}")
        print(f"   Directorio del script: {script_dir}")
        return None
    
    # Si hay múltiples archivos, usar el más reciente
    latest_file = max(local_metrics_files, key=os.path.getmtime)
    print(f"✅ Usando archivo de métricas: {latest_file}")
    return latest_file

# Buscar el archivo de métricas
metrics_file = find_metrics_file()
if metrics_file is None:
    exit(1)

try:
    # Leer los datos desde archivo
    cores, speedup, efficiency = np.loadtxt(metrics_file, unpack=True)
    print(f"📊 Datos cargados: {len(cores)} puntos de medición")
    
except Exception as e:
    print(f"❌ Error al leer el archivo {metrics_file}: {e}")
    print("   Verificar el formato del archivo. Debe contener 3 columnas: cores speedup efficiency")
    exit(1)

# Configurar matplotlib para modo no interactivo (evitar errores de display)
plt.ioff()

# Crear la figura con dos subtramas
plt.figure(figsize=(12, 5))

# Gráfico de Speedup
plt.subplot(1, 2, 1)
plt.plot(cores, speedup, 'bo-', label='Speedup medido')
plt.plot(cores, cores, 'r--', label='Speedup ideal')
plt.xlabel('Número de cores')
plt.ylabel('Speedup')
plt.title('Curva de Speedup para interference.py')
plt.grid(True)
plt.legend()

# Gráfico de Efficiency
plt.subplot(1, 2, 2)
plt.plot(cores, efficiency, 'go-')
plt.xlabel('Número de cores')
plt.ylabel('Efficiency (%)')
plt.title('Curva de Efficiency para interference.py')
plt.grid(True)

# Ajustar layout y guardar figura
plt.tight_layout()
fig_filename = os.path.join(output_dir, f"interference_metrics_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
plt.savefig(fig_filename)
print(f"✅ Figura guardada: {fig_filename}")

# Mostrar resumen de los datos
print(f"\n📊 Resumen de métricas:")
print(f"   Cores probados: {cores.min():.0f} - {cores.max():.0f}")
print(f"   Speedup máximo: {speedup.max():.2f}x")
print(f"   Eficiencia máxima: {efficiency.max():.1f}%")
print(f"   Eficiencia final ({cores.max():.0f} cores): {efficiency[-1]:.1f}%")

print("\n✅ Todas las figuras generadas exitosamente!")
print(f"📁 Directorio de salida: {os.path.abspath(output_dir)}")