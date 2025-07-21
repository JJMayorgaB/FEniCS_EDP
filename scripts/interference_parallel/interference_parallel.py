# Wave Interference Solver - Parallel Performance Analysis Version
# Optimized for measuring parallel performance metrics with two Gaussian sources

from mpi4py import MPI
import numpy as np
import time
import ufl
from petsc4py import PETSc

from dolfinx import fem, mesh, default_scalar_type
from dolfinx.fem.petsc import assemble_vector, assemble_matrix, create_vector, apply_lifting, set_bc

def solve_wave_interference_2d_performance():
    """
    Resuelve la ecuación de onda 2D con interferencia de dos fuentes gaussianas:
    ∂²u/∂t² - v²(∂²u/∂x² + ∂²u/∂y²) = f₁(x,y,t) + f₂(x,y,t)
    
    Optimizada para medición de performance paralelo.
    Returns execution time for performance analysis.
    """
    
    # Get MPI communicator information
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Parámetros físicos generales
    v = 1.5      # Velocidad de onda
    
    # Parámetros de la primera fuente
    A1 = 15.0         # Amplitud de la fuente 1
    omega1 = 6.0      # Frecuencia angular de la fuente 1
    x01, y01 = 0.5, 0.5  # Centro de la fuente gaussiana 1
    sigma1 = 0.08     # Ancho de la fuente gaussiana 1
    
    # Parámetros de la segunda fuente
    A2 = 15.0         # Amplitud de la fuente 2
    omega2 = 8.0      # Frecuencia angular de la fuente 2
    x02, y02 = 1.5, 0.5  # Centro de la fuente gaussiana 2
    sigma2 = 0.08     # Ancho de la fuente gaussiana 2
    
    # Parámetros temporales - optimizados para carga computacional
    t = 0.0      # Tiempo inicial
    T = 3.0      # Tiempo final reducido para benchmark
    num_steps = 1500  # Número de pasos para carga significativa
    dt = T / num_steps
    
    # Crear dominio 2D rectangular distribuido - malla más densa para mejor carga
    nx, ny = 200, 100  # Malla más densa para mayor carga computacional
    domain = mesh.create_rectangle(
        comm, 
        [np.array([0.0, 0.0]), np.array([2.0, 1.0])],
        [nx, ny], 
        cell_type=mesh.CellType.triangle
    )
    V = fem.functionspace(domain, ("Lagrange", 1))
    
    # Print mesh information only from rank 0
    if rank == 0:
        total_dofs = V.dofmap.index_map.size_global
        print(f"Total DOFs: {total_dofs}, Processes: {size}")
        
        # Calcular tamaño de malla característico para verificar estabilidad
        hx = 2.0 / nx
        hy = 1.0 / ny
        h = min(hx, hy)
        courant_number = v * dt / h
        print(f"Courant number: {courant_number:.4f}")
        if courant_number > 1.0:
            print(f"WARNING: Courant number > 1, numerical instability possible")
    
    # Condiciones de frontera (bordes fijos)
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.full(x.shape[1], True, dtype=bool))
    bc = fem.dirichletbc(PETSc.ScalarType(0), 
                        fem.locate_dofs_topological(V, fdim, boundary_facets), V)
    
    # Funciones para almacenar soluciones en tiempo
    u_n2 = fem.Function(V, dtype=default_scalar_type)  # u^{n-1}
    u_n1 = fem.Function(V, dtype=default_scalar_type)  # u^n
    u_n = fem.Function(V, dtype=default_scalar_type)   # u^{n+1}
    
    # Condiciones iniciales con pulsos cerca de cada fuente
    def initial_displacement(x):
        # Pulso inicial cerca de la primera fuente
        r1_sq = (x[0] - 0.4)**2 + (x[1] - 0.4)**2
        pulse1 = 0.3 * np.exp(-30 * r1_sq) * np.sin(8 * np.pi * np.sqrt(r1_sq + 1e-10))
        
        # Pulso inicial cerca de la segunda fuente
        r2_sq = (x[0] - 1.6)**2 + (x[1] - 0.6)**2
        pulse2 = 0.3 * np.exp(-30 * r2_sq) * np.sin(8 * np.pi * np.sqrt(r2_sq + 1e-10))
        
        return pulse1 + pulse2
    
    # Aplicar condiciones iniciales
    u_n2.interpolate(initial_displacement)
    u_n1.interpolate(initial_displacement)
    
    # Definir funciones de forma y coordenadas
    u = ufl.TrialFunction(V)
    v_test = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)
    
    # Funciones fuente gaussianas moduladas temporalmente
    def source_terms(t_val):
        # Fuente 1
        r1_sq = (x[0] - x01)**2 + (x[1] - y01)**2
        gaussian1 = ufl.exp(-r1_sq / (2 * sigma1**2))
        temporal1 = ufl.cos(omega1 * t_val)
        f1 = A1 * gaussian1 * temporal1
        
        # Fuente 2
        r2_sq = (x[0] - x02)**2 + (x[1] - y02)**2
        gaussian2 = ufl.exp(-r2_sq / (2 * sigma2**2))
        temporal2 = ufl.cos(omega2 * t_val)
        f2 = A2 * gaussian2 * temporal2
        
        # Suma de las dos fuentes
        return f1 + f2
    
    # Forma bilineal para el esquema implícito
    a = (u * v_test * ufl.dx + 
         (dt**2 * v**2) * ufl.dot(ufl.grad(u), ufl.grad(v_test)) * ufl.dx)
    
    # Ensamblaje de la matriz del sistema
    bilinear_form = fem.form(a)
    A_matrix = assemble_matrix(bilinear_form, bcs=[bc])
    A_matrix.assemble()
    
    # Configurar solver lineal
    solver = PETSc.KSP().create(comm)
    solver.setOperators(A_matrix)
    solver.setType(PETSc.KSP.Type.PREONLY)
    solver.getPC().setType(PETSc.PC.Type.LU)
    
    # Vector para el lado derecho
    b = create_vector(fem.form(u * v_test * ufl.dx))
    
    # Sincronizar antes de comenzar medición y añadir warmup
    comm.Barrier()
    
    # Warmup: algunas iteraciones para preparar el sistema
    if rank == 0:
        print("Warming up...")
    
    for warm in range(3):
        t_warm = warm * dt
        f_warm = source_terms(t_warm)
        L_warm = ((2.0 * u_n1 - u_n2) * v_test * ufl.dx + 
                  dt**2 * f_warm * v_test * ufl.dx)
        linear_form_warm = fem.form(L_warm)
    
    comm.Barrier()
    if rank == 0:
        print("Starting interference computation...")
    
    start_time = time.time()
    
    # Loop temporal principal - SIN visualización
    for n in range(1, num_steps + 1):
        t = n * dt
        
        # Actualizar término fuente con ambas fuentes
        f_current = source_terms(t)
        
        # Lado derecho del sistema
        L = ((2.0 * u_n1 - u_n2) * v_test * ufl.dx + 
             dt**2 * f_current * v_test * ufl.dx)
        
        linear_form = fem.form(L)
        
        # Ensamblar lado derecho
        with b.localForm() as loc_b:
            loc_b.set(0)
        assemble_vector(b, linear_form)
        
        # Aplicar condiciones de frontera
        apply_lifting(b, [bilinear_form], [[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b, [bc])
        
        # Resolver sistema lineal
        solver.solve(b, u_n.x.petsc_vec)
        u_n.x.scatter_forward()
        
        # Actualizar soluciones anteriores
        u_n2.x.array[:] = u_n1.x.array[:]
        u_n1.x.array[:] = u_n.x.array[:]
        
        # Mostrar progreso ocasionalmente (solo desde rank 0)
        if rank == 0 and n % (num_steps // 10) == 0:
            progress = (n / num_steps) * 100
            max_displacement = np.max(np.abs(u_n.x.array))
            print(f"  Progress: {progress:.0f}%, t={t:.3f}s, |u|_max={max_displacement:.4f}")
    
    # Sincronizar y medir tiempo final
    comm.Barrier()
    end_time = time.time()
    
    execution_time = end_time - start_time
    
    return execution_time, u_n

def run_interference_benchmark():
    """
    Ejecuta benchmark de interferencia y calcula métricas paralelas
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank == 0:
        print(f"=== Wave Interference 2D Parallel Performance Benchmark ===")
        print(f"Running on {size} MPI processes")
        print("Two Gaussian sources with interference patterns")
        print("Starting computation...")
    
    # Ejecutar simulación y medir tiempo
    exec_time, solution = solve_wave_interference_2d_performance()
    
    if rank == 0:
        print(f"Execution completed in {exec_time:.4f} seconds")
        
        # Calcular algunas métricas adicionales de la solución
        max_displacement = np.max(np.abs(solution.x.array))
        print(f"Final maximum displacement: {max_displacement:.6f}")
        
        # Guardar resultado para análisis posterior
        result_data = {
            'cores': size,
            'time': exec_time,
            'dofs': solution.function_space.dofmap.index_map.size_global,
            'max_displacement': max_displacement
        }
        
        return result_data
    
    return None

def analyze_interference_theory():
    """
    Análisis teórico de la interferencia de ondas (solo información)
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        print("\n=== Theoretical Interference Analysis ===")
        print("Equation: ∂²u/∂t² - v²∇²u = f₁(x,y,t) + f₂(x,y,t)")
        print("where: f_i(x,y,t) = A_i·exp(-r_i²/2σ_i²)·cos(ω_i·t)")
        print()
        print("Expected phenomena:")
        print("  • Constructive interference: waves in phase")
        print("  • Destructive interference: waves in anti-phase")
        print("  • Diffraction patterns: due to Gaussian source nature")
        print("  • Beat patterns: if ω₁ ≠ ω₂, beat frequency = |ω₂-ω₁|/2π")
        print("  • Standing wave patterns: in certain regions with reflections")
        print()
        print("Parallel computational considerations:")
        print("  • 2D domain decomposition across MPI processes")
        print("  • Matrix assembly and linear solve are main bottlenecks")
        print("  • Communication overhead in ghost point updates")
        print("  • Load balancing depends on mesh partitioning")

def calculate_interference_metrics(times_data):
    """
    Calcula speedup y eficiencia para el problema de interferencia
    
    Args:
        times_data: dict con {cores: time} para cada número de cores
    """
    print("\n=== Wave Interference Parallel Performance Metrics ===")
    print("Cores\tTime(s)\t\tSpeedup\t\tEfficiency(%)")
    print("-" * 55)
    
    # Tiempo de referencia (serial o menor número de cores)
    base_cores = min(times_data.keys())
    base_time = times_data[base_cores]
    
    for cores in sorted(times_data.keys()):
        exec_time = times_data[cores]
        speedup = base_time / exec_time if exec_time > 0 else 0
        efficiency = (speedup / cores) * 100 if cores > 0 else 0
        
        print(f"{cores}\t{exec_time:.4f}\t\t{speedup:.2f}\t\t{efficiency:.1f}")

def main():
    """
    Función principal optimizada para benchmarking de interferencia
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    try:
        # Análisis teórico inicial (solo rank 0)
        if rank == 0:
            analyze_interference_theory()
        
        # Ejecutar benchmark
        result = run_interference_benchmark()
        
        if rank == 0 and result:
            # Mostrar información del run actual
            print(f"\nCurrent run results:")
            print(f"Cores: {result['cores']}")
            print(f"Execution time: {result['time']:.4f} seconds")
            print(f"Total DOFs: {result['dofs']}")
            print(f"DOFs per core: {result['dofs'] // result['cores']}")
            print(f"Max displacement: {result['max_displacement']:.6f}")
            
            # Información para colectar datos completos
            print(f"\nTo collect full performance data, run this script with different core counts:")
            print(f"mpirun -np 1 python interference_parallel.py")
            print(f"mpirun -np 2 python interference_parallel.py")
            print(f"mpirun -np 4 python interference_parallel.py")
            print(f"mpirun -np 8 python interference_parallel.py")
            print(f"mpirun -np 16 python interference_parallel.py")
            print(f"\nOr use the automated metrics script:")
            print(f"bash interference_metrics.sh")
        
    except Exception as e:
        if rank == 0:
            print(f"Error during execution: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()