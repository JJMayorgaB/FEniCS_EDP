# Wave Equation Solver - Performance Analysis Version
# Optimized for measuring parallel performance metrics

from mpi4py import MPI
import numpy as np
import time
import ufl
from petsc4py import PETSc

from dolfinx import fem, mesh, default_scalar_type
from dolfinx.fem.petsc import assemble_vector, assemble_matrix, create_vector, apply_lifting, set_bc

def solve_wave_equation_performance():
    """
    Resuelve la ecuación de onda optimizada para medición de performance:
    ∂²u/∂x² - (1/v²)∂²u/∂t² = A*sin(kx - ωt)
    
    Returns execution time for performance analysis.
    """
    
    # Get MPI communicator information
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Parámetros físicos
    v = 2.0      # Velocidad de onda
    A = 1.0      # Amplitud de la fuente
    k = np.pi    # Número de onda
    omega = k * v  # Frecuencia angular
    
    # Parámetros temporales - aumentados para mejor medición
    t = 0.0      # Tiempo inicial
    T = 2.0      # Tiempo final
    num_steps = 2000  # Más pasos para carga computacional significativa
    dt = T / num_steps
    
    # Crear dominio 1D distribuido - malla más densa
    nx = 1000  # Malla mucho más densa para mejor carga computacional
    domain = mesh.create_interval(comm, nx, [0.0, 2.0])
    V = fem.functionspace(domain, ("Lagrange", 1))
    
    # Print mesh information only from rank 0
    if rank == 0:
        total_dofs = V.dofmap.index_map.size_global
        print(f"Total DOFs: {total_dofs}, Processes: {size}")
    
    # Condiciones de frontera (extremos fijos)
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.full(x.shape[1], True, dtype=bool))
    bc = fem.dirichletbc(PETSc.ScalarType(0), 
                        fem.locate_dofs_topological(V, fdim, boundary_facets), V)
    
    # Funciones para almacenar soluciones
    u_n2 = fem.Function(V, dtype=default_scalar_type)  # u^{n-1}
    u_n1 = fem.Function(V, dtype=default_scalar_type)  # u^n
    u_n = fem.Function(V, dtype=default_scalar_type)   # u^{n+1}
    
    # Condiciones iniciales
    def initial_displacement(x):
        return np.sin(np.pi * x[0]) * np.exp(-10 * (x[0] - 1)**2)
    
    # Interpolación de condiciones iniciales
    u_n2.interpolate(initial_displacement)
    u_n1.interpolate(initial_displacement)
    
    # Formas bilineales y lineales
    u = ufl.TrialFunction(V)
    v_test = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)
    
    # Término fuente A*sin(kx - ωt)
    def source_term(t_val):
        return A * ufl.sin(k * x[0] - omega * t_val)
    
    # Forma bilinear para el esquema implícito
    a = (u * v_test * ufl.dx + 
         (dt**2 * v**2) * ufl.dot(ufl.grad(u), ufl.grad(v_test)) * ufl.dx)
    
    # Ensamblaje de matrices
    bilinear_form = fem.form(a)
    A_matrix = assemble_matrix(bilinear_form, bcs=[bc])
    A_matrix.assemble()
    
    # Configurar solver
    solver = PETSc.KSP().create(comm)
    solver.setOperators(A_matrix)
    solver.setType(PETSc.KSP.Type.PREONLY)
    solver.getPC().setType(PETSc.PC.Type.LU)
    
    # Vector para el lado derecho
    b = create_vector(fem.form(u * v_test * ufl.dx))
    
    # Sincronizar antes de comenzar medición y añadir warmup
    comm.Barrier()
    
    # Warmup: una iteración para preparar el sistema
    if rank == 0:
        print("Warming up...")
    
    # Pequeño warmup
    for warm in range(2):
        t_warm = warm * dt
        f_warm = source_term(t_warm)
        L_warm = ((2.0 * u_n1 - u_n2) * v_test * ufl.dx + 
                  dt**2 * f_warm * v_test * ufl.dx)
        linear_form_warm = fem.form(L_warm)
    
    comm.Barrier()
    if rank == 0:
        print("Starting actual computation...")
    
    start_time = time.time()
    
    # Loop temporal principal - SIN visualización
    for n in range(1, num_steps + 1):
        t = n * dt
        
        # Término del lado derecho
        f_current = source_term(t)
        L = ((2.0 * u_n1 - u_n2) * v_test * ufl.dx + 
             dt**2 * f_current * v_test * ufl.dx)
        
        linear_form = fem.form(L)
        
        # Ensamblaje del lado derecho
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
        u_n2.x.array[:] = u_n1.x.array
        u_n1.x.array[:] = u_n.x.array
    
    # Sincronizar y medir tiempo final
    comm.Barrier()
    end_time = time.time()
    
    execution_time = end_time - start_time
    
    return execution_time, u_n

def run_performance_benchmark():
    """
    Ejecuta benchmark de performance y calcula métricas paralelas
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank == 0:
        print(f"=== Wave Equation Parallel Performance Benchmark ===")
        print(f"Running on {size} MPI processes")
        print("Starting computation...")
    
    # Ejecutar simulación y medir tiempo
    exec_time, solution = solve_wave_equation_performance()
    
    if rank == 0:
        print(f"Execution completed in {exec_time:.4f} seconds")
        
        # Guardar resultado para análisis posterior
        result_data = {
            'cores': size,
            'time': exec_time,
            'dofs': solution.function_space.dofmap.index_map.size_global
        }
        
        return result_data
    
    return None

def calculate_metrics(times_data):
    """
    Calcula speedup y eficiencia dados los tiempos de ejecución
    
    Args:
        times_data: dict con {cores: time} para cada número de cores
    """
    print("\n=== Parallel Performance Metrics ===")
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
    Función principal optimizada para benchmarking
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    try:
        # Ejecutar benchmark
        result = run_performance_benchmark()
        
        if rank == 0 and result:
            # Mostrar información del run actual
            print(f"\nCurrent run results:")
            print(f"Cores: {result['cores']}")
            print(f"Execution time: {result['time']:.4f} seconds")
            print(f"Total DOFs: {result['dofs']}")
            print(f"DOFs per core: {result['dofs'] // result['cores']}")
            
            # Ejemplo de cómo usar los resultados
            print(f"\nTo collect full performance data, run this script with different core counts:")
            print(f"mpirun -np 1 python wave_eq_parallel_metrics.py")
            print(f"mpirun -np 2 python wave_eq_parallel_metrics.py")
            print(f"mpirun -np 4 python wave_eq_parallel_metrics.py")
            print(f"mpirun -np 8 python wave_eq_parallel_metrics.py")
            print(f"mpirun -np 16 python wave_eq_parallel_metrics.py")
        
    except Exception as e:
        if rank == 0:
            print(f"Error during execution: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()