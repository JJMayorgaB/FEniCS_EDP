import pyvista
import ufl
import numpy as np
import os
from petsc4py import PETSc
from mpi4py import MPI
import matplotlib as mpl
import matplotlib.pyplot as plt

from dolfinx import fem, mesh, io, plot
from dolfinx.fem.petsc import assemble_vector, assemble_matrix, create_vector, apply_lifting, set_bc

def solve_wave_interference_2d():
    """
    Resuelve la ecuación de onda 2D con interferencia de dos fuentes gaussianas:
    ∂²u/∂t² - v²(∂²u/∂x² + ∂²u/∂y²) = f₁(x,y,t) + f₂(x,y,t)
    
    donde: f_i(x,y,t) = A_i·exp(-[(x-x_i0)² + (y-y_i0)²]/(2σ_i²))·cos(ω_i·t)
    
    Usando el método de elementos finitos en espacio y diferencias finitas en tiempo.
    """
    
    # Parámetros físicos generales
    v = 1.5      # Velocidad de onda
    
    # Parámetros de la primera fuente (más alejada y más intensa)
    A1 = 15.0         # Amplitud aumentada de la fuente 1
    omega1 = 6.0      # Frecuencia angular de la fuente 1
    x01, y01 = 0.5, 0.5  # Centro de la fuente gaussiana 1 (ajustado para separación de 1m)
    sigma1 = 0.08     # Ancho de la fuente gaussiana 1
    
    # Parámetros de la segunda fuente (más alejada y más intensa)
    A2 = 15.0         # Amplitud aumentada de la fuente 2
    omega2 = 8.0      # Frecuencia angular de la fuente 2
    x02, y02 = 1.5, 0.5  # Centro de la fuente gaussiana 2 (ajustado para separación de 1m)
    sigma2 = 0.08     # Ancho de la fuente gaussiana 2
    
    # Parámetros temporales
    t = 0.0      # Tiempo inicial
    T = 5.0      # Tiempo final aumentado para ver más interferencia
    num_steps = 1000  # Más pasos para mejor resolución
    dt = T / num_steps
    
    # Crear dominio 2D rectangular (2x1 en lugar de 1x1)
    nx, ny = 150, 75  # Malla rectangular adaptada al nuevo dominio
    domain = mesh.create_rectangle(
        MPI.COMM_WORLD, 
        [np.array([0.0, 0.0]), np.array([2.0, 1.0])],  # Dominio rectangular 2x1
        [nx, ny], 
        cell_type=mesh.CellType.triangle
    )
    V = fem.functionspace(domain, ("Lagrange", 1))
    
    # Calcular tamaño de malla característico
    hx = 2.0 / nx  # Ajustado para el nuevo ancho
    hy = 1.0 / ny
    h = min(hx, hy)
    
    # Calcular número de Courant
    courant_number = v * dt / h
    
    # Calcular longitudes de onda características
    lambda1 = 2 * np.pi * v / omega1
    lambda2 = 2 * np.pi * v / omega2
    
    print(f"=== Configuración de Interferencia de Ondas 2D ===")
    print(f"Velocidad de onda: {v}")
    print(f"Dominio: rectangular 2.0 x 1.0 m")  # Información del nuevo dominio
    print(f"")
    print(f"Fuente 1:")
    print(f"  Amplitud: {A1}")
    print(f"  Frecuencia angular: {omega1} rad/s")
    print(f"  Centro: ({x01}, {y01})")
    print(f"  Ancho gaussiano: {sigma1}")
    print(f"  Longitud de onda: λ₁ ≈ {lambda1:.3f} m")
    print(f"")
    print(f"Fuente 2:")
    print(f"  Amplitud: {A2}")
    print(f"  Frecuencia angular: {omega2} rad/s")
    print(f"  Centro: ({x02}, {y02})")
    print(f"  Ancho gaussiano: {sigma2}")
    print(f"  Longitud de onda: λ₂ ≈ {lambda2:.3f} m")
    print(f"")
    print(f"Separación entre fuentes: {np.sqrt((x02-x01)**2 + (y02-y01)**2):.3f} m")
    print(f"Diferencia de frecuencias: Δω = {abs(omega2-omega1):.2f} rad/s")
    if abs(omega2-omega1) > 0:
        beat_period = 2*np.pi/abs(omega2-omega1)
        print(f"Período de batimiento esperado: T_beat ≈ {beat_period:.3f} s")
    print(f"")
    print(f"Discretización:")
    print(f"  Pasos temporales: {num_steps}")
    print(f"  dt = {dt:.6f} s")
    print(f"  Discretización espacial: h ≈ {h:.6f} m")
    print(f"  Número de Courant: C = {courant_number:.4f}")
    if courant_number > 1.0:
        print(f"  ⚠️  ADVERTENCIA: C > 1, puede haber inestabilidad numérica")
    else:
        print(f"  ✓ C ≤ 1, esquema numéricamente estable")
    
    # Condiciones de frontera (bordes fijos)
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.full(x.shape[1], True, dtype=bool))
    bc = fem.dirichletbc(PETSc.ScalarType(0), 
                        fem.locate_dofs_topological(V, fdim, boundary_facets), V)
    
    # Funciones para almacenar soluciones en tiempo
    u_n2 = fem.Function(V)  # u^{n-1}
    u_n1 = fem.Function(V)  # u^n
    u_n = fem.Function(V)   # u^{n+1} (solución actual)
    
    # Condiciones iniciales con pulsos cerca de cada fuente
    def initial_displacement(x):
        # Pulso inicial cerca de la primera fuente
        r1_sq = (x[0] - 0.4)**2 + (x[1] - 0.4)**2
        pulse1 = 0.3 * np.exp(-30 * r1_sq) * np.sin(8 * np.pi * np.sqrt(r1_sq))
        
        # Pulso inicial cerca de la segunda fuente
        r2_sq = (x[0] - 1.6)**2 + (x[1] - 0.6)**2
        pulse2 = 0.3 * np.exp(-30 * r2_sq) * np.sin(8 * np.pi * np.sqrt(r2_sq))
        
        return pulse1 + pulse2
    
    def initial_velocity(x):
        return np.zeros_like(x[0])
    
    # Aplicar condiciones iniciales
    u_n2.interpolate(initial_displacement)
    u_n1.interpolate(initial_displacement)
    
    # Para velocidad inicial cero: u^{n-1} ≈ u^n
    # Para velocidad inicial no cero, usar:
    # u^{n-1} = u^n - dt * v_0 + (dt²/2) * [v²∇²u + f]
    
    # Crear directorios de salida
    if MPI.COMM_WORLD.rank == 0:
        os.makedirs("figures/wave", exist_ok=True)
        os.makedirs("post_data/wave", exist_ok=True)
    MPI.COMM_WORLD.barrier()
    
    # Configuración de PyVista
    pyvista.OFF_SCREEN = True
    pyvista.start_xvfb(wait=0.5)
    
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
    
    # Setup XDMF para salida
    xdmf = io.XDMFFile(domain.comm, "post_data/wave/wave_interference_2d.xdmf", "w")
    xdmf.write_mesh(domain)
    
    # Setup visualización 3D con PyVista
    cells, types, x_coords = plot.vtk_mesh(V)
    grid = pyvista.UnstructuredGrid(cells, types, x_coords)
    
    plotter = pyvista.Plotter(off_screen=True)
    plotter.open_gif("figures/wave/wave_interference_2d.gif", fps=12)
    
    # Configuración de colores y escala
    colormap = mpl.colormaps.get_cmap("seismic").resampled(100)  # Mejor para interferencia
    sargs = dict(title_font_size=20, label_font_size=15, fmt="%.3f", color="black",
                 position_x=0.02, position_y=0.1, width=0.15, height=0.8)
    
    # Escala dinámica para visualización (aumentada para las nuevas amplitudes)
    u_range = 3.0  # Aumentado para capturar las amplitudes mayores
    grid.point_data["u"] = u_n1.x.array
    
    # Crear superficie deformada para visualización 3D
    warped = grid.warp_by_scalar("u", factor=0.2)
    
    renderer = plotter.add_mesh(warped, show_edges=False, lighting=True,
                               cmap=colormap, scalar_bar_args=sargs,
                               clim=[-u_range, u_range], opacity=0.9)
    
    # Agregar puntos para marcar las fuentes
    source_points = np.array([[x01, y01, 0], [x02, y02, 0]])
    plotter.add_points(source_points, color='yellow', point_size=15, 
                      render_points_as_spheres=True)
    
    # Agregar malla de referencia en el plano
    plotter.add_mesh(grid, style="wireframe", color="gray", opacity=0.1)
    
    plotter.add_title("Interferencia de Ondas 2D - Dos Fuentes Gaussianas", 
                     font_size=16, color="black")
    plotter.show_axes()
    
    # Configurar vista isométrica ajustada para dominio rectangular
    plotter.camera_position = [(3.0, 2.0, 1.5), (1.0, 0.5, 0.0), (0, 0, 1)]  # Vista ajustada
    
    # Escribir condición inicial
    u_n.x.array[:] = u_n1.x.array
    u_n.name = "displacement"
    xdmf.write_function(u_n, t)
    plotter.write_frame()
    
    # Ensamblaje de la matriz del sistema
    bilinear_form = fem.form(a)
    A_matrix = assemble_matrix(bilinear_form, bcs=[bc])
    A_matrix.assemble()
    
    # Configurar solver lineal
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A_matrix)
    solver.setType(PETSc.KSP.Type.PREONLY)
    solver.getPC().setType(PETSc.PC.Type.LU)
    
    # Vector para el lado derecho
    b = create_vector(fem.form(u * v_test * ufl.dx))
    
    # Variables para monitoreo
    energies = []
    max_amplitudes = []
    times_recorded = []
    interference_data = []
    
    # Punto medio para observar interferencia (ajustado al nuevo dominio)
    mid_point = [(x01 + x02)/2, (y01 + y02)/2]  # Punto medio entre las fuentes
    
    print("\nComenzando simulación de interferencia de ondas 2D...")
    print("Progreso: ", end="", flush=True)
    
    # Loop principal en tiempo
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
        
        # Monitoreo cada ciertos pasos
        if n % 10 == 0:
            # Energía cinética aproximada
            u_dot_approx = (u_n.x.array - u_n1.x.array) / dt
            kinetic_energy = 0.5 * np.sum(u_dot_approx**2) * (1.0 / len(u_dot_approx))
            
            # Energía potencial
            grad_u_form = fem.form(v**2 * ufl.dot(ufl.grad(u_n), ufl.grad(u_n)) * ufl.dx)
            potential_energy = 0.5 * fem.assemble_scalar(grad_u_form)
            
            total_energy = kinetic_energy + potential_energy
            max_amplitude = np.max(np.abs(u_n.x.array))
            
            energies.append(total_energy)
            max_amplitudes.append(max_amplitude)
            times_recorded.append(t)
        
        # Actualizar soluciones anteriores
        u_n2.x.array[:] = u_n1.x.array[:]
        u_n1.x.array[:] = u_n.x.array[:]
        
        # Actualizar visualización y guardar datos
        if n % 6 == 0:  # Cada 6 pasos para mejor resolución temporal
            # Actualizar escala dinámica basada en el máximo actual
            current_max = np.max(np.abs(u_n.x.array))
            if current_max > u_range * 0.8:
                u_range = current_max * 1.2
                renderer.scalar_bar.SetLookupTable(
                    renderer.mapper.lookup_table)
            
            grid.point_data["u"] = u_n.x.array
            warped_new = grid.warp_by_scalar("u", factor=0.2)
            warped.points[:, :] = warped_new.points
            warped.point_data["u"][:] = u_n.x.array
            
            # Actualizar rango de colores
            warped.set_active_scalars("u")
            plotter.update_scalar_bar_range([-u_range, u_range])
            
            xdmf.write_function(u_n, t)
            plotter.write_frame()
        
        # Mostrar progreso
        if n % (num_steps // 25) == 0:
            print("█", end="", flush=True)
            if domain.comm.rank == 0 and n % (num_steps // 5) == 0:
                max_displacement = np.max(np.abs(u_n.x.array))
                print(f"\n  t={t:.3f}s, |u|_max={max_displacement:.4f}, E={total_energy:.4f}")
    
    print("\n")
    plotter.close()
    xdmf.close()
    
    print("Simulación de interferencia completada!")
    print(f"Archivos generados:")
    print(f"  - Animación: figures/wave/wave_interference_2d.gif")
    print(f"  - Datos: post_data/wave/wave_interference_2d.xdmf")
    
    return u_n, domain, energies, times_recorded, max_amplitudes

def analyze_interference_theory():
    """
    Análisis teórico de la interferencia de ondas
    """
    print("\n=== Análisis Teórico de Interferencia de Ondas ===")
    print("Ecuación: ∂²u/∂t² - v²∇²u = f₁(x,y,t) + f₂(x,y,t)")
    print("donde: f_i(x,y,t) = A_i·exp(-r_i²/2σ_i²)·cos(ω_i·t)")
    print()
    print("Fenómenos esperados:")
    print("  • Interferencia constructiva: cuando las ondas están en fase")
    print("  • Interferencia destructiva: cuando las ondas están en contrafase")
    print("  • Patrones de difracción: debido a la naturaleza gaussiana de las fuentes")
    print("  • Batimiento: si ω₁ ≠ ω₂, frecuencia de batimiento = |ω₂-ω₁|/2π")
    print("  • Ondas estacionarias: en ciertas regiones con reflexiones")
    print()
    print("Consideraciones numéricas:")
    print("  • Condición CFL: v·dt/h ≤ 1 para estabilidad")
    print("  • Resolución espacial: h << min(λ₁, λ₂) para capturar detalles")
    print("  • Duración: T >> max(2π/ω₁, 2π/ω₂) para observar patrones completos")

if __name__ == "__main__":
    try:
        # Análisis teórico inicial
        analyze_interference_theory()
        
        # Ejecutar simulación de interferencia
        u_final, domain, energies, times, max_amps = solve_wave_interference_2d()

    except Exception as e:
        print(f"Error durante la ejecución: {e}")
        import traceback
        traceback.print_exc()