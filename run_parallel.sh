#!/bin/bash

# Script para ejecutar demo_interpolation_parallel.py con OpenMPI
# Usage: ./run_parallel.sh [number_of_processes]

echo "================================================"
echo "Ejecutando demo_interpolation_parallel.py con 4 procesos MPI"
echo "================================================"
mpirun -np 4 python3 demo_interpolation_parallel.py
echo "================================================"
echo "Ejecutando demo_interpolation_parallel.py con 3 procesos MPI"
echo "================================================"
mpirun -np 3 python3 demo_interpolation_parallel.py
echo "================================================"
echo "Ejecutando demo_interpolation_parallel.py con 2 procesos MPI"
echo "================================================"
mpirun -np 2 python3 demo_interpolation_parallel.py
echo "================================================"
echo "Ejecutando demo_interpolation_parallel.py con 1 procesos MPI"
echo "================================================"
mpirun -np 1 python3 demo_interpolation_parallel.py
echo "================================================"
echo "Ejecución paralela completada"
