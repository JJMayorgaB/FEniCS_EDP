#!/bin/bash

# Script para ejecutar demo_interpolation_parallel.py con OpenMPI
# Usage: ./run_parallel.sh [number_of_processes]

# Número de procesos (por defecto 4)
NPROCS=${1:-4}

echo "Ejecutando demo_interpolation_parallel.py con $NPROCS procesos MPI"
echo "================================================"

# Ejecutar con mpirun
mpirun -np $NPROCS python3 demo_interpolation_parallel.py

echo "================================================"
echo "Ejecución paralela completada"
