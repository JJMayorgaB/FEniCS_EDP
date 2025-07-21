#!/bin/bash

# Script para ejecutar interpolation_parallel.py con OpenMPI
# Usage: ./run_parallel.sh [number_of_processes]

echo "================================================"
echo "Ejecutando interpolation_parallel.py con 8 procesos MPI"
echo "================================================"
mpirun -np 8 python3 scripts/interpolation_parallel.py
echo "================================================"
echo "Ejecutando interpolation_parallel.py con 4 procesos MPI"
echo "================================================"
mpirun -np 4 python3 scripts/interpolation_parallel.py
echo "================================================"
echo "Ejecutando interpolation_parallel.py con 1 proceso MPI"
echo "================================================"
mpirun -np 1 python3 scripts/interpolation_parallel.py
echo "================================================"
echo "Ejecución paralela completada"
