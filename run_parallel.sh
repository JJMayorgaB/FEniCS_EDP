#!/bin/bash

# Script para ejecutar interpolation_parallel.py con OpenMPI
# 
# IMPORTANTE: Este script está diseñado para ejecutarse dentro de los contenedores
# proporcionados (Docker o Apptainer). Para usarlo:
#
# Con Docker:
#   docker run -it --rm -v $(pwd):/workspace fenics-edp
#   ./run_parallel.sh
#
# Con Apptainer:
#   apptainer shell fenics.sif
#   ./run_parallel.sh
#
# Usage: ./run_parallel.sh [number_of_processes]

echo "================================================"
echo "FEniCS EDP - Ejecución Paralela con MPI"
echo "================================================"
echo "Verificando entorno..."

# Verificar que estamos en el entorno correcto
if ! command -v mpirun &> /dev/null; then
    echo "❌ ERROR: mpirun no encontrado"
    echo "   Asegúrese de ejecutar este script dentro del contenedor Docker o Apptainer"
    echo "   Consulte el README.md para instrucciones detalladas"
    exit 1
fi

if ! python3 -c "import dolfinx" 2>/dev/null; then
    echo "❌ ERROR: DOLFINx no encontrado"
    echo "   Asegúrese de ejecutar este script dentro del contenedor Docker o Apptainer"
    exit 1
fi

echo "✅ Entorno MPI y FEniCS verificado correctamente"
echo ""

echo "================================================"
echo "Ejecutando interpolation_parallel.py con 8 procesos MPI"
echo "================================================"
mpirun -np 8 python3 scripts/interpolation_parallel.py

echo ""
echo "================================================"
echo "Ejecutando interpolation_parallel.py con 4 procesos MPI"
echo "================================================"
mpirun -np 4 python3 scripts/interpolation_parallel.py

echo ""
echo "================================================"
echo "Ejecutando interpolation_parallel.py con 1 proceso MPI"
echo "================================================"
mpirun -np 1 python3 scripts/interpolation_parallel.py

echo ""
echo "================================================"
echo "Ejecución paralela completada exitosamente"
echo "================================================"
echo ""
echo "Archivos generados en:"
echo "  📁 figures/interpolation/"
echo "  📁 post_data/interpolation/"
echo ""
echo "Para visualizar resultados:"
echo "  🔍 Ver archivos PNG en figures/interpolation/"
echo "  📊 Abrir archivos .bp en ParaView desde post_data/interpolation/"
