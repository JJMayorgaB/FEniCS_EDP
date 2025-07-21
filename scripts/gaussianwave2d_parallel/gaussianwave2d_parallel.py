# Gaussian Wave 2D - Parallel Version with OpenMPI
# Optimized for parallel execution with MPI following best practices

from mpi4py import MPI
import numpy as np
import os
import ufl
from petsc4py import PETSc
import matplotlib as mpl
import matplotlib.pyplot as plt
import imageio

from dolfinx import fem, mesh, io, plot, default_scalar_type
from dolfinx.fem.petsc import assemble_vector, assemble_matrix, create_vector, apply_lifting, set_bc

def solve_wave_equation_2d_parallel():
    """
    Resuelve la ecuación de onda 2D en paralelo usando MPI:
    ∂²u/∂t² - v²(∂²u/∂x² + ∂²u/∂y²) = A·exp(-[(x-x₀)² + (y-y₀)²]/(2σ²))·cos(ωt)
    
    Usando el método de elementos finitos en espacio y diferencias finitas en tiempo.
    """
    
    # Get MPI communicator information
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank == 0:
        print(f"=== Gaussian Wave 2D Solver - Parallel Version ===")
        print(f"Running on {size} MPI processes")
        # Create directories if they don't exist
        os.makedirs("figures/wave_parallel", exist_ok=True)
        os.makedirs("post_data/wave_parallel", exist_ok=True)
    
    # Parámetros físicos
    v = 1.5      # Velocidad de onda
    A = 10.0     # Amplitud de la fuente
    omega = 8.0  # Frecuencia angular de la fuente
    x0, y0 = 0.5, 0.5  # Centro de la fuente gaussiana
    sigma = 0.1  # Ancho de la fuente gaussiana
    
    # Parámetros temporales
    t = 0.0      # Tiempo inicial
    T = 3.0      # Tiempo final
    num_steps = 500
    dt = T / num_steps
    
    # Crear dominio 2D rectangular distribuido - automáticamente particionado entre procesos
    nx, ny = 80, 80
    domain = mesh.create_rectangle(
        comm, 
        [np.array([0.0, 0.0]), np.array([1.0, 1.0])],
        [nx, ny], 
        cell_type=mesh.CellType.triangle
    )
    V = fem.functionspace(domain, ("Lagrange", 1))
    
    # Print mesh information
    if rank == 0:
        total_dofs = V.dofmap.index_map.size_global
        total_cells = domain.topology.index_map(domain.topology.dim).size_global
        print(f"Total DOFs: {total_dofs}")
        print(f"Total cells: {total_cells}")
        print(f"Domain partitioned among {size} processes")
        
    # Print local info for each process (synchronized)
    for i in range(size):
        if rank == i:
            local_dofs = V.dofmap.index_map.size_local
            local_cells = domain.topology.index_map(domain.topology.dim).size_local
            print(f"Process {rank} has {local_dofs} DOFs and {local_cells} cells")
        comm.Barrier()
    
    # Calcular tamaño de malla característico
    hx = 1.0 / nx  # Tamaño en dirección x
    hy = 1.0 / ny  # Tamaño en dirección y
    h = min(hx, hy)  # Tamaño característico (el menor)
    
    # Calcular número de Courant
    courant_number = v * dt / h
    
    if rank == 0:
        print(f"\nConfiguración de la simulación:")
        print(f"  Velocidad de onda: {v}")
        print(f"  Amplitud fuente: {A}")
        print(f"  Frecuencia angular: {omega}")
        print(f"  Centro fuente: ({x0}, {y0})")
        print(f"  Ancho gaussiano: {sigma}")
        print(f"  Pasos temporales: {num_steps}")
        print(f"  dt = {dt:.6f}")
        print(f"  Discretización espacial: h ≈ {h:.6f}")
        print(f"  Número de Courant: C = {courant_number:.4f}")
        if courant_number > 1.0:
            print(f"  ⚠️  ADVERTENCIA: C > 1, puede haber inestabilidad numérica")
        else:
            print(f"  ✓ C ≤ 1, esquema numéricamente estable")
    
    # Condiciones de frontera (bordes fijos) - aplicadas en paralelo
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.full(x.shape[1], True, dtype=bool))
    bc = fem.dirichletbc(PETSc.ScalarType(0), 
                        fem.locate_dofs_topological(V, fdim, boundary_facets), V)
    
    # Funciones para almacenar soluciones en tiempo (distribuidas)
    u_n2 = fem.Function(V, dtype=default_scalar_type)  # u^{n-1}
    u_n1 = fem.Function(V, dtype=default_scalar_type)  # u^n
    u_n = fem.Function(V, dtype=default_scalar_type)   # u^{n+1} (solución actual)
    
    # Condiciones iniciales paralelas
    def initial_displacement(x):
        # Pulso inicial cerca del centro
        r_sq = (x[0] - 0.3)**2 + (x[1] - 0.3)**2
        return 0.5 * np.exp(-50 * r_sq) * np.sin(10 * np.pi * np.sqrt(r_sq))
    
    def initial_velocity(x):
        return np.zeros_like(x[0])
    
    # Aplicar condiciones iniciales en paralelo
    u_n2.interpolate(initial_displacement)
    u_n1.interpolate(initial_displacement)
    
    comm.Barrier()
    if rank == 0:
        print("Initial conditions set in parallel")
    
    # Setup XDMF para salida paralela
    filename_base = "post_data/wave_parallel/wave_2d_solution_parallel"
    xdmf = io.XDMFFile(comm, f"{filename_base}.xdmf", "w")
    xdmf.write_mesh(domain)  # Escribir malla PRIMERO
    
    # Configuración de PyVista para visualización paralela
    try:
        import pyvista
        
        if rank == 0:
            print("Configurando PyVista para visualización paralela...")
            pyvista.OFF_SCREEN = True
        
        # Setup visualización 3D con PyVista - cada proceso tiene su parte
        cells, types, x_coords = plot.vtk_mesh(V)
        local_grid = pyvista.UnstructuredGrid(cells, types, x_coords)
        
        # Solo el proceso 0 maneja el plotter y GIF
        if rank == 0:
            gif_frames = []
            
            # Configuración de colores y escala
            colormap = mpl.colormaps.get_cmap("RdBu_r").resampled(50)
            
            # Escala fija para visualización consistente
            u_range = 0.8
        
    except ImportError:
        if rank == 0:
            print("PyVista not available, using matplotlib only")
        local_grid = None
    
    # Definir funciones de forma y coordenadas
    u = ufl.TrialFunction(V)
    v_test = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)
    
    # Función fuente gaussiana modulada temporalmente
    def source_term(t_val):
        r_sq = (x[0] - x0)**2 + (x[1] - y0)**2
        gaussian = ufl.exp(-r_sq / (2 * sigma**2))
        temporal = ufl.cos(omega * t_val)
        return A * gaussian * temporal
    
    # Forma bilineal para el esquema implícito
    a = (u * v_test * ufl.dx + 
         (dt**2 * v**2) * ufl.dot(ufl.grad(u), ufl.grad(v_test)) * ufl.dx)
    
    # Escribir condición inicial (DESPUÉS de escribir la malla)
    u_n.x.array[:] = u_n1.x.array
    u_n.name = "displacement"
    xdmf.write_function(u_n, t)
    
    # Ensamblaje paralelo de la matriz del sistema
    if rank == 0:
        print("Assembling system matrices in parallel...")
    
    bilinear_form = fem.form(a)
    A_matrix = assemble_matrix(bilinear_form, bcs=[bc])
    A_matrix.assemble()
    
    # Configurar solver paralelo
    solver = PETSc.KSP().create(comm)
    solver.setOperators(A_matrix)
    solver.setType(PETSc.KSP.Type.PREONLY)
    solver.getPC().setType(PETSc.PC.Type.LU)
    
    # Vector para el lado derecho
    b = create_vector(fem.form(u * v_test * ufl.dx))
    
    # Variables para monitoreo de energía
    energies = []
    times = []
    
    comm.Barrier()
    if rank == 0:
        print("\nComenzando simulación de onda 2D en paralelo...")
        print("Progreso: ", end="", flush=True)
    
    # Crear frame inicial si tenemos PyVista
    if local_grid is not None and rank == 0:
        try:
            # Gather initial data from all processes
            local_grid.point_data["u"] = u_n1.x.array.real
            local_grid_data = {
                'x': x_coords,
                'values': u_n1.x.array.real,
                'cells': cells,
                'types': types
            }
            
            all_grid_data = comm.gather(local_grid_data, root=0)
            
            if all_grid_data is not None:
                # Combinar grids de todos los procesos
                combined_grids = []
                for i, grid_data in enumerate(all_grid_data):
                    proc_grid = pyvista.UnstructuredGrid(
                        grid_data['cells'], 
                        grid_data['types'], 
                        grid_data['x']
                    )
                    proc_grid.point_data["u"] = grid_data['values']
                    combined_grids.append(proc_grid)
                
                if len(combined_grids) > 1:
                    combined_grid = combined_grids[0]
                    for grid in combined_grids[1:]:
                        combined_grid = combined_grid.merge(grid)
                    combined_grid = combined_grid.clean(tolerance=1e-10)
                else:
                    combined_grid = combined_grids[0]
                
                # Crear superficie deformada para visualización 3D inicial
                warped = combined_grid.warp_by_scalar("u", factor=0.3)
                
                plotter = pyvista.Plotter(off_screen=True)
                renderer = plotter.add_mesh(warped, show_edges=False, lighting=True,
                                           cmap=colormap, 
                                           clim=[-u_range, u_range], opacity=0.9)
                
                # Agregar malla de referencia en el plano
                plotter.add_mesh(combined_grid, style="wireframe", color="gray", opacity=0.2)
                
                plotter.add_title(f"Propagación de Onda 2D - Parallel ({size} procs)", font_size=18, color="black")
                plotter.show_axes()
                
                # Configurar vista isométrica
                plotter.camera_position = [(1.5, 1.5, 1.0), (0.5, 0.5, 0.0), (0, 0, 1)]
                
                frame_filename = f"figures/wave_parallel/frame_000.png"
                plotter.screenshot(frame_filename)
                plotter.close()
                gif_frames.append(frame_filename)
                
        except Exception as e:
            print(f"Initial PyVista frame creation failed: {e}")
    else:
        # Para procesos que no son el 0, solo participar en gather
        if local_grid is not None:
            local_grid.point_data["u"] = u_n1.x.array.real
            local_grid_data = {
                'x': x_coords,
                'values': u_n1.x.array.real,
                'cells': cells,
                'types': types
            }
            comm.gather(local_grid_data, root=0)
    
    # Loop principal en tiempo - paralelo
    for n in range(1, num_steps + 1):
        t = n * dt
        
        # Actualizar término fuente
        f_current = source_term(t)
        
        # Lado derecho del sistema: 2u^n - u^{n-1} + dt²f^{n+1}
        L = ((2.0 * u_n1 - u_n2) * v_test * ufl.dx + 
             dt**2 * f_current * v_test * ufl.dx)
        
        linear_form = fem.form(L)
        
        # Ensamblar lado derecho en paralelo
        with b.localForm() as loc_b:
            loc_b.set(0)
        assemble_vector(b, linear_form)
        
        # Aplicar condiciones de frontera en paralelo
        apply_lifting(b, [bilinear_form], [[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b, [bc])
        
        # Resolver sistema lineal en paralelo
        solver.solve(b, u_n.x.petsc_vec)
        u_n.x.scatter_forward()
        
        # Calcular energía total del sistema con menor frecuencia para mejor rendimiento
        if n % 25 == 0:  # Reducido de cada 10 a cada 25 pasos
            # Energía cinética aproximada: (1/2) * (∂u/∂t)²
            u_dot_approx = (u_n.x.array - u_n1.x.array) / dt
            local_kinetic_energy = 0.5 * np.sum(u_dot_approx**2) * (1.0 / len(u_dot_approx))
            
            # Energía potencial: (1/2) * v² * |∇u|²
            grad_u_form = fem.form(v**2 * ufl.dot(ufl.grad(u_n), ufl.grad(u_n)) * ufl.dx)
            local_potential_energy = 0.5 * fem.assemble_scalar(grad_u_form)
            
            # Sumar energías de todos los procesos de forma más eficiente
            local_energies = np.array([local_kinetic_energy, local_potential_energy])
            global_energies = np.zeros(2)
            comm.Allreduce(local_energies, global_energies, op=MPI.SUM)
            
            global_kinetic = global_energies[0] / size
            global_potential = global_energies[1]
            total_energy = global_kinetic + global_potential
            
            if rank == 0:
                energies.append(total_energy)
                times.append(t)
        
        # Actualizar soluciones anteriores
        u_n2.x.array[:] = u_n1.x.array[:]
        u_n1.x.array[:] = u_n.x.array[:]
        
        # Actualizar visualización y guardar datos - Optimizado
        if n % 20 == 0:  # Reducido de cada 8 a cada 20 pasos para mejor rendimiento
            xdmf.write_function(u_n, t)
            
            # Crear frame para PyVista de forma más eficiente
            if local_grid is not None and n % 40 == 0:  # Frames menos frecuentes para GIF
                # Solo proceso 0 coordina la creación del frame
                if rank == 0:
                    try:
                        # Gather data from all processes de forma más eficiente
                        local_grid.point_data["u"] = u_n.x.array.real
                        local_grid_data = {
                            'values': u_n.x.array.real,  # Solo valores, reutilizar geometría
                        }
                        
                        all_values = comm.gather(u_n.x.array.real, root=0)
                        
                        if all_values is not None:
                            # Combinar valores más eficientemente
                            combined_values = np.concatenate(all_values)
                            
                            # Usar grid global pre-computado (definir una vez)
                            if hasattr(solve_wave_equation_2d_parallel, 'global_grid'):
                                global_grid = solve_wave_equation_2d_parallel.global_grid
                            else:
                                # Primera vez: crear grid global y guardarlo
                                all_grid_data = comm.gather({
                                    'x': x_coords, 'cells': cells, 'types': types
                                }, root=0)
                                
                                combined_grids = []
                                for grid_data in all_grid_data:
                                    proc_grid = pyvista.UnstructuredGrid(
                                        grid_data['cells'], grid_data['types'], grid_data['x']
                                    )
                                    combined_grids.append(proc_grid)
                                
                                global_grid = combined_grids[0]
                                for grid in combined_grids[1:]:
                                    global_grid = global_grid.merge(grid)
                                global_grid = global_grid.clean(tolerance=1e-10)
                                
                                # Guardar para reutilizar
                                solve_wave_equation_2d_parallel.global_grid = global_grid
                            
                            # Actualizar solo los valores
                            global_grid.point_data["u"] = combined_values
                            
                            # Crear frame con menor calidad para velocidad
                            warped_new = global_grid.warp_by_scalar("u", factor=0.3)
                            
                            plotter = pyvista.Plotter(off_screen=True)
                            plotter.add_mesh(warped_new, show_edges=False, lighting=True,
                                           cmap=colormap, clim=[-u_range, u_range], opacity=0.9)
                            plotter.add_mesh(global_grid, style="wireframe", color="gray", opacity=0.1)
                            plotter.add_title(f"Onda 2D - Parallel ({size}) - t={t:.3f}s", 
                                            font_size=16, color="black")
                            plotter.camera_position = [(1.5, 1.5, 1.0), (0.5, 0.5, 0.0), (0, 0, 1)]
                            
                            frame_filename = f"figures/wave_parallel/frame_{len(gif_frames):03d}.png"
                            plotter.screenshot(frame_filename, window_size=[800, 600])  # Menor resolución
                            plotter.close()
                            gif_frames.append(frame_filename)
                            
                    except Exception as e:
                        print(f"Frame creation failed at step {n}: {e}")
                else:
                    # Otros procesos solo envían valores, no geometría
                    comm.gather(u_n.x.array.real, root=0)
        
        # Mostrar progreso sincronizado
        if n % (num_steps // 20) == 0:
            # Calcular máximo global
            local_max_displacement = np.max(np.abs(u_n.x.array))
            global_max_displacement = comm.allreduce(local_max_displacement, op=MPI.MAX)
            
            if rank == 0:
                print(" ", end="", flush=True)
                if n % (num_steps // 10) == 0:
                    print(f"\n  t={t:.3f}, |u|_max={global_max_displacement:.4f}")
    
    # Sincronizar antes de crear salidas finales
    comm.Barrier()
    
    if rank == 0:
        print("\n")
    
    # Crear GIF final de forma más eficiente (solo proceso 0)
    if rank == 0 and local_grid is not None:
        try:
            print(f"Creating animation GIF from {len(gif_frames)} frames...")
            gif_path = "figures/wave_parallel/wave_2d_propagation_parallel.gif"
            
            if gif_frames:
                # Usar configuración optimizada para GIF más pequeño y rápido
                with imageio.get_writer(gif_path, mode='I', duration=0.2, subrectangles=True, loop=0) as writer:
                    for frame_file in gif_frames:
                        if os.path.exists(frame_file):
                            image = imageio.imread(frame_file)
                            writer.append_data(image)
                
                print(f"Animation saved: {gif_path}")
                
                # Limpiar frames temporales con threading si hay muchos
                if len(gif_frames) > 30:
                    import threading
                    def cleanup_frames(frame_list):
                        for frame_file in frame_list:
                            if os.path.exists(frame_file):
                                os.remove(frame_file)
                    
                    cleanup_thread = threading.Thread(target=cleanup_frames, args=(gif_frames,))
                    cleanup_thread.start()
                    cleanup_thread.join()
                else:
                    for frame_file in gif_frames:
                        if os.path.exists(frame_file):
                            os.remove(frame_file)
            
        except Exception as e:
            print(f"Error creating GIF: {e}")
    
    # Crear visualización final usando matplotlib para análisis de energía
    if rank == 0:
        try:
            print("Creating final analysis plots...")
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # Gráfico de energía vs tiempo
            if energies and times:
                ax1.plot(times, energies, 'b-', linewidth=2, label='Total Energy')
                ax1.set_xlabel('Time (s)', fontsize=12)
                ax1.set_ylabel('Energy', fontsize=12)
                ax1.set_title(f'Energy Conservation - 2D Wave (Parallel {size} processes)', fontsize=14)
                ax1.grid(True, alpha=0.3)
                ax1.legend()
                
                # Calcular variación de energía
                energy_variation = (max(energies) - min(energies)) / max(energies) * 100
                ax1.text(0.02, 0.98, f'Energy variation: {energy_variation:.2f}%', 
                        transform=ax1.transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            else:
                ax1.text(0.5, 0.5, 'No energy data available', 
                        transform=ax1.transAxes, ha='center', va='center')
            
            # Información de parámetros
            ax2.text(0.1, 0.9, f'Physical Parameters:', fontsize=14, weight='bold', transform=ax2.transAxes)
            ax2.text(0.1, 0.8, f'Wave velocity (v): {v} m/s', fontsize=12, transform=ax2.transAxes)
            ax2.text(0.1, 0.7, f'Source amplitude (A): {A}', fontsize=12, transform=ax2.transAxes)
            ax2.text(0.1, 0.6, f'Angular frequency (ω): {omega} rad/s', fontsize=12, transform=ax2.transAxes)
            ax2.text(0.1, 0.5, f'Source center: ({x0}, {y0})', fontsize=12, transform=ax2.transAxes)
            ax2.text(0.1, 0.4, f'Gaussian width (σ): {sigma}', fontsize=12, transform=ax2.transAxes)
            
            ax2.text(0.6, 0.9, f'Numerical Parameters:', fontsize=14, weight='bold', transform=ax2.transAxes)
            ax2.text(0.6, 0.8, f'Time step (dt): {dt:.6f} s', fontsize=12, transform=ax2.transAxes)
            ax2.text(0.6, 0.7, f'Total time: {T} s', fontsize=12, transform=ax2.transAxes)
            ax2.text(0.6, 0.6, f'Grid size: {nx}×{ny}', fontsize=12, transform=ax2.transAxes)
            ax2.text(0.6, 0.5, f'MPI processes: {size}', fontsize=12, transform=ax2.transAxes)
            ax2.text(0.6, 0.4, f'Courant number: {courant_number:.4f}', fontsize=12, transform=ax2.transAxes)
            ax2.text(0.6, 0.3, f'Mesh size (h): {h:.6f}', fontsize=12, transform=ax2.transAxes)
            
            ax2.axis('off')
            plt.tight_layout()
            
            analysis_plot_path = f"figures/wave_parallel/wave_2d_analysis_parallel_{size}procs.png"
            plt.savefig(analysis_plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"Analysis plot saved: {analysis_plot_path}")
            
        except Exception as e:
            print(f"Error creating analysis plots: {e}")
    
    # Limpiar y cerrar
    xdmf.close()
    
    # Calcular estadísticas finales
    local_max_final = np.max(np.abs(u_n.x.array))
    global_max_final = comm.allreduce(local_max_final, op=MPI.MAX)
    
    # Sincronización final
    comm.Barrier()
    
    if rank == 0:
        print("Simulación completada!")
        print(f"Archivos generados:")
        print(f"  - Datos: {filename_base}.xdmf")
        print(f"  - Datos auxiliares: {filename_base}.h5")
        print(f"  - Análisis: figures/wave_parallel/wave_2d_analysis_parallel_{size}procs.png")
        if local_grid is not None:
            print(f"  - Animación: figures/wave_parallel/wave_2d_propagation_parallel.gif")
        
        print(f"\nEstadísticas finales:")
        print(f"  • Desplazamiento máximo final: {global_max_final:.6f}")
        if energies:
            energy_variation = (max(energies) - min(energies)) / max(energies) * 100
            print(f"  • Variación de energía: {energy_variation:.2f}%")
    
    # Cleanup PyVista
    try:
        import pyvista
        if rank == 0:
            pyvista.close_all()
    except:
        pass
    
    comm.Barrier()
    
    return u_n, domain, energies, times

def analyze_wave_properties_parallel():
    """
    Análisis teórico de la ecuación de onda 2D - versión paralela
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank == 0:
        print("\n=== Análisis de la Ecuación de Onda 2D - Versión Paralela ===")
        print("Ecuación: ∂²u/∂t² - v²(∂²u/∂x² + ∂²u/∂y²) = A·exp(-r²/2σ²)·cos(ωt)")
        print()
        print("Propiedades:")
        print("  • Tipo: Ecuación hiperbólica de segundo orden")
        print("  • Dimensiones: 2D espacial + 1D temporal")
        print("  • Velocidad de propagación: v = 1.5 m/s")
        print("  • Fuente: Gaussiana pulsante con frecuencia ω = 8.0 rad/s")
        print("  • Longitud de onda típica: λ ≈ 2πv/ω ≈ 1.18 m")
        print("  • Condiciones de frontera: Dirichlet homogéneas (bordes fijos)")
        print()
        print("Características numéricas:")
        print(f"  • Número de Courant: C = v·dt/h")
        print(f"  • Para estabilidad se requiere: C ≤ 1")
        print(f"  • Discretización: Elementos finitos P1 + Diferencias finitas")
        print(f"  • h = min(Lx/nx, Ly/ny) donde Lx,Ly son dimensiones del dominio")
        print()
        print("Paralelización:")
        print(f"  • MPI processes: {size}")
        print(f"  • Automatic domain partitioning with FEniCSx")
        print(f"  • Parallel linear solvers (PETSc)")
        print(f"  • Parallel I/O (XDMF)")

def main():
    """
    Función principal para ejecutar la simulación paralela
    """
    try:
        # Análisis inicial
        analyze_wave_properties_parallel()
        
        # Ejecutar simulación
        u_final, domain, energies, times = solve_wave_equation_2d_parallel()
        
        # Información final
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        
        if rank == 0:
            print(f"\n=== Parallel Execution Summary ({size} processes) ===")
            print("2D Gaussian wave equation solved successfully in parallel!")
            print("Check the generated figures and data files.")
        
    except Exception as e:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        
        if rank == 0:
            print(f"Error during parallel execution: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()