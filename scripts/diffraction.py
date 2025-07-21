import pyvista
import ufl
import numpy as np
import os
from petsc4py import PETSc
from mpi4py import MPI
import matplotlib as mpl

from dolfinx import fem, mesh, io, plot
from dolfinx.fem.petsc import assemble_vector, assemble_matrix, create_vector, apply_lifting, set_bc

def solve_wave_diffraction_2d():
    """
    Resuelve la ecuación de onda 2D con difracción por un obstáculo cuadrado:
    ∂²u/∂t² - v²(∂²u/∂x² + ∂²u/∂y²) = A·exp(-[(x-x₀)² + (y-y₀)²]/(2σ²))·cos(ωt)
    
    Con un obstáculo cuadrado en (0.4, 0.7) × (0.35, 0.65)
    """
    
    # Parámetros físicos
    v = 1.0      # Velocidad de onda
    A = 15.0     # Amplitud de la fuente
    omega = 12.0  # Frecuencia angular de la fuente
    x0, y0 = 0.05, 0.5  # Centro de la fuente (lado izquierdo)
    sigma = 0.05  # Ancho de la fuente gaussiana
    
    # Parámetros temporales
    t = 0.0      # Tiempo inicial
    T = 2.5      # Tiempo final
    num_steps = 800
    dt = T / num_steps
    
    # Crear dominio 2D rectangular con mayor resolución
    nx, ny = 120, 80
    domain = mesh.create_rectangle(
        MPI.COMM_WORLD, 
        [np.array([0.0, 0.0]), np.array([1.5, 1.0])],
        [nx, ny], 
        cell_type=mesh.CellType.triangle
    )
    V = fem.functionspace(domain, ("Lagrange", 1))
    
    # Calcular parámetros de estabilidad
    hx = 1.5 / nx
    hy = 1.0 / ny
    h = min(hx, hy)
    courant_number = v * dt / h
    
    print(f"=== Simulación de Difracción de Ondas 2D ===")
    print(f"Configuración:")
    print(f"  Dominio: [0, 1.5] × [0, 1.0]")
    print(f"  Obstáculo: [0.4, 0.7] × [0.35, 0.65]")
    print(f"  Velocidad de onda: {v}")
    print(f"  Amplitud fuente: {A}")
    print(f"  Frecuencia angular: {omega}")
    print(f"  Centro fuente: ({x0}, {y0})")
    print(f"  Discretización: h ≈ {h:.6f}")
    print(f"  Pasos temporales: {num_steps}")
    print(f"  dt = {dt:.6f}")
    print(f"  Número de Courant: C = {courant_number:.4f}")
    
    if courant_number > 0.5:  # Más restrictivo para estabilidad
        print(f"  ⚠️  ADVERTENCIA: C > 0.5, ajustar dt para mejor estabilidad")
    else:
        print(f"  ✓ C ≤ 0.5, esquema estable")
    
    # Crear función para marcar el obstáculo en las celdas
    def obstacle_marker(x):
        """Marcar celdas que están dentro del obstáculo"""
        return ((x[0] >= 0.4) & (x[0] <= 0.7) & 
                (x[1] >= 0.35) & (x[1] <= 0.65))
    
    # Crear una función para identificar nodos del obstáculo
    obstacle_dofs = []
    coords = V.tabulate_dof_coordinates()
    for i, coord in enumerate(coords):
        if (0.4 <= coord[0] <= 0.7) and (0.35 <= coord[1] <= 0.65):
            obstacle_dofs.append(i)
    
    print(f"  Nodos en obstáculo: {len(obstacle_dofs)}")

    # Marcar celdas del obstáculo
    cell_markers = fem.Function(fem.functionspace(domain, ("DG", 0)))
    cell_markers.interpolate(obstacle_marker)
    
    # Crear subdominio sin el obstáculo
    # En lugar de condiciones de frontera complejas, usaremos un enfoque de penalización
    
    # Definir funciones para identificar fronteras del dominio
    def boundary_walls(x):
        """Fronteras del dominio (bordes exteriores)"""
        tol = 1e-14
        return (np.isclose(x[0], 0.0, atol=tol) | np.isclose(x[0], 1.5, atol=tol) | 
                np.isclose(x[1], 0.0, atol=tol) | np.isclose(x[1], 1.0, atol=tol))
    
    # Aplicar condiciones de frontera solo en los bordes del dominio
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary_walls)
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc_walls = fem.dirichletbc(PETSc.ScalarType(0), boundary_dofs, V)
    bcs = [bc_walls]
    
    print(f"  Frontera del dominio: {len(boundary_facets)} facetas")
    
    # Funciones para almacenar soluciones
    u_n2 = fem.Function(V)  # u^{n-1}
    u_n1 = fem.Function(V)  # u^n
    u_n = fem.Function(V)   # u^{n+1}
    
    # Condiciones iniciales - sin perturbación inicial para ver mejor la difracción
    u_n2.x.array[:] = 0.0
    u_n1.x.array[:] = 0.0
    
    # Crear directorios
    if MPI.COMM_WORLD.rank == 0:
        os.makedirs("figures/wave", exist_ok=True)
        os.makedirs("post_data/wave", exist_ok=True)
    MPI.COMM_WORLD.barrier()
    
    # Configuración de PyVista
    pyvista.OFF_SCREEN = True
    if hasattr(pyvista, 'start_xvfb'):
        pyvista.start_xvfb(wait=0.5)
    
    # Definir formas variacionales
    u = ufl.TrialFunction(V)
    v_test = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)
    
    # Fuente continua gaussiana - simplificada
    def source_term(t_val):
        # Fuente en el lado izquierdo que genera ondas
        r_sq = (x[0] - x0)**2 + (x[1] - y0)**2
        gaussian = ufl.exp(-r_sq / (2 * sigma**2))
        
        # Usar función temporal simple sin condicionales complejas
        if t_val < 1.5:
            temporal_factor = 1.0
        else:
            temporal_factor = np.exp(-2*(t_val-1.5))
        
        temporal = ufl.cos(omega * t_val)
        return A * gaussian * temporal * temporal_factor
    
    # Forma bilineal sin penalización - simplificada
    a = (u * v_test * ufl.dx + 
         (dt**2 * v**2) * ufl.dot(ufl.grad(u), ufl.grad(v_test)) * ufl.dx)

    # Setup archivos de salida
    xdmf = io.XDMFFile(domain.comm, "post_data/wave/wave_diffraction_2d.xdmf", "w")
    xdmf.write_mesh(domain)
    
    # Setup visualización
    cells, types, x_coords = plot.vtk_mesh(V)
    grid = pyvista.UnstructuredGrid(cells, types, x_coords)
    
    # Agregar campo para marcar el obstáculo
    obstacle_mask = np.zeros(len(grid.points))
    for i, point in enumerate(grid.points):
        if (0.4 <= point[0] <= 0.7) and (0.35 <= point[1] <= 0.65):
            obstacle_mask[i] = 1.0
    grid.point_data["obstacle"] = obstacle_mask
    
    plotter = pyvista.Plotter(off_screen=True)
    plotter.open_gif("figures/wave/wave_diffraction_2d.gif", fps=15)
    
    # Configuración de visualización
    colormap = mpl.colormaps.get_cmap("seismic").resampled(50)
    sargs = dict(title_font_size=16, label_font_size=12, fmt="%.2f", color="black",
                 position_x=0.02, position_y=0.1, width=0.15, height=0.8)
    
    # Escala de colores más apropiada
    u_range = 0.8
    grid.point_data["u"] = u_n1.x.array
    
    # Crear superficie para visualización
    warped = grid.warp_by_scalar("u", factor=0.3)
    
    # Añadir malla de la onda
    wave_actor = plotter.add_mesh(warped, show_edges=False, lighting=True,
                                 cmap=colormap, scalar_bar_args=sargs,
                                 clim=[-u_range, u_range], opacity=0.9,
                                 scalars="u")
    
    # Añadir obstáculo como superficie sólida
    if np.any(obstacle_mask > 0.5):
        obstacle_grid = grid.extract_points(obstacle_mask > 0.5)
        if len(obstacle_grid.points) > 0:
            # Mantener el obstáculo al mismo nivel que la malla (z=0)
            obstacle_surface = obstacle_grid.copy()
            obstacle_surface.points[:, 2] = 0.0  # Mantener en z=0
            
            # Añadir el obstáculo como superficie plana con color distintivo
            plotter.add_mesh(obstacle_surface, color="darkgray", opacity=0.95,
                            show_edges=True, edge_color="black", line_width=3,
                            render_points_as_spheres=False)
            
            # Opcional: añadir un borde más visible
            obstacle_edges = obstacle_surface.extract_feature_edges()
            if len(obstacle_edges.points) > 0:
                plotter.add_mesh(obstacle_edges, color="red", line_width=4, 
                                render_lines_as_tubes=True, opacity=1.0)
    
    plotter.add_title("Difracción de Ondas 2D - Obstáculo Cuadrado", 
                     font_size=14, color="black")
    plotter.show_axes()
    
    # Vista isométrica mejorada
    plotter.camera_position = [(2.2, 1.8, 1.8), (0.75, 0.5, 0.2), (0, 0, 1)]
    
    # Condición inicial
    u_n.x.array[:] = u_n1.x.array
    u_n.name = "displacement"
    xdmf.write_function(u_n, t)
    plotter.write_frame()
    
    # Ensamblaje del sistema
    bilinear_form = fem.form(a)
    A_matrix = assemble_matrix(bilinear_form, bcs=bcs)
    A_matrix.assemble()
    
    # Solver mejorado
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A_matrix)
    solver.setType(PETSc.KSP.Type.CG)  # Usar gradiente conjugado
    solver.getPC().setType(PETSc.PC.Type.HYPRE)  # Precondicionador más eficiente
    solver.setTolerances(rtol=1e-8)
    
    b = create_vector(fem.form(u * v_test * ufl.dx))
    
    # Variables de monitoreo
    max_displacements = []
    times_recorded = []
    
    print("\nIniciando simulación...")
    print("Progreso: ", end="", flush=True)
    
    # Loop principal
    for n in range(1, num_steps + 1):
        t = n * dt
        
        # Término fuente
        f_current = source_term(t)
        
        # Lado derecho simplificado
        L = ((2.0 * u_n1 - u_n2) * v_test * ufl.dx + 
             dt**2 * f_current * v_test * ufl.dx)
        
        linear_form = fem.form(L)
        
        # Ensamblar
        with b.localForm() as loc_b:
            loc_b.set(0)
        assemble_vector(b, linear_form)
        
        # Aplicar condiciones de frontera
        apply_lifting(b, [bilinear_form], [bcs])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b, bcs)
        
        # Resolver
        solver.solve(b, u_n.x.petsc_vec)
        u_n.x.scatter_forward()
        
        # Forzar u=0 en el obstáculo después de resolver (enfoque directo)
        for dof_idx in obstacle_dofs:
            u_n.x.array[dof_idx] = 0.0
        
        # Monitoreo
        if n % 20 == 0:
            max_disp = np.max(np.abs(u_n.x.array))
            max_displacements.append(max_disp)
            times_recorded.append(t)
        
        # Actualizar soluciones
        u_n2.x.array[:] = u_n1.x.array[:]
        u_n1.x.array[:] = u_n.x.array[:]
        
        # Visualización y salida
        if n % 8 == 0:  # Más frames para mejor animación
            # Actualizar visualización
            grid.point_data["u"] = u_n.x.array
            warped_new = grid.warp_by_scalar("u", factor=0.3)
            warped.points[:, :] = warped_new.points
            warped.point_data["u"][:] = u_n.x.array
            
            xdmf.write_function(u_n, t)
            plotter.write_frame()
        
        # Progreso
        if n % (num_steps // 25) == 0:
            print("█", end="", flush=True)
            if n % (num_steps // 5) == 0:
                max_disp = np.max(np.abs(u_n.x.array))
                if domain.comm.rank == 0:
                    print(f"\n  t={t:.3f}s, |u|_max={max_disp:.4f}")
    
    print("\n")
    plotter.close()
    xdmf.close()
    
    print("¡Simulación completada!")
    print(f"\nArchivos generados:")
    print(f"  • Animación: figures/wave/wave_diffraction_2d.gif")
    print(f"  • Datos XDMF: post_data/wave/wave_diffraction_2d.xdmf")
    print(f"  • Datos HDF5: post_data/wave/wave_diffraction_2d.h5")
    
    return u_n, domain, max_displacements, times_recorded

def analyze_diffraction_physics():
    """
    Análisis físico del fenómeno de difracción
    """
    print("\n=== Análisis de Difracción de Ondas ===")
    print("Fenómeno: Difracción por obstáculo rectangular")
    print()
    print("Características físicas:")
    print("  • Ecuación: ∂²u/∂t² - v²∇²u = fuente")
    print("  • Velocidad de propagación: v = 1.0 m/s")
    print("  • Obstáculo: superficie rígida (u = 0)")
    print("  • Dimensiones obstáculo: 0.3 × 0.3 m")
    print("  • Fuente: Gaussiana pulsante en x = 0.05")
    print()
    print("Efectos esperados:")
    print("  • Reflexión en la superficie del obstáculo")
    print("  • Difracción en los bordes (esquinas)")
    print("  • Formación de sombra acústica detrás del obstáculo")
    print("  • Interferencia entre ondas reflejadas y difractadas")
    print("  • Patrones de interferencia constructiva/destructiva")
    print()
    print("Aplicaciones:")
    print("  • Acústica arquitectónica")
    print("  • Diseño de barreras acústicas")
    print("  • Radar y sonar")
    print("  • Óptica ondulatoria")

if __name__ == "__main__":
    try:
        # Análisis teórico
        analyze_diffraction_physics()
        
        # Ejecutar simulación
        u_final, domain, max_displacements, times_recorded = solve_wave_diffraction_2d()
        
        # Estadísticas finales
        if domain.comm.rank == 0:
            final_max = np.max(np.abs(u_final.x.array))
            print(f"\nEstadísticas finales:")
            print(f"  • Desplazamiento máximo final: {final_max:.6f}")
            print(f"  • Puntos de malla: {len(u_final.x.array)}")
            
            if max_displacements:
                energy_peak = max(max_displacements)
                energy_final = max_displacements[-1] if max_displacements else 0
                print(f"  • Máximo durante simulación: {energy_peak:.6f}")
                if energy_peak > 0:
                    print(f"  • Amortiguación aparente: {((energy_peak-energy_final)/energy_peak*100):.1f}%")
        
    except Exception as e:
        print(f"Error durante la ejecución: {e}")
        import traceback
        traceback.print_exc()