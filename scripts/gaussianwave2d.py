import pyvista
import ufl
import numpy as np
import os
from petsc4py import PETSc
from mpi4py import MPI
import matplotlib as mpl

from dolfinx import fem, mesh, io, plot
from dolfinx.fem.petsc import assemble_vector, assemble_matrix, create_vector, apply_lifting, set_bc

def solve_wave_equation_2d():
    """
    Resuelve la ecuación de onda 2D:
    ∂²u/∂t² - v²(∂²u/∂x² + ∂²u/∂y²) = A·exp(-[(x-x₀)² + (y-y₀)²]/(2σ²))·cos(ωt)
    
    Usando el método de elementos finitos en espacio y diferencias finitas en tiempo.
    """
    
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
    
    # Crear dominio 2D rectangular
    nx, ny = 80, 80
    domain = mesh.create_rectangle(
        MPI.COMM_WORLD, 
        [np.array([0.0, 0.0]), np.array([1.0, 1.0])],
        [nx, ny], 
        cell_type=mesh.CellType.triangle
    )
    V = fem.functionspace(domain, ("Lagrange", 1))
    
    # Calcular tamaño de malla característico
    # Para un dominio rectangular [0,1] x [0,1] con nx x ny elementos
    hx = 1.0 / nx  # Tamaño en dirección x
    hy = 1.0 / ny  # Tamaño en dirección y
    h = min(hx, hy)  # Tamaño característico (el menor)
    
    # Calcular número de Courant
    courant_number = v * dt / h
    
    print(f"Configuración de la simulación:")
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
    
    # Condiciones iniciales
    def initial_displacement(x):
        # Pulso inicial cerca del centro
        r_sq = (x[0] - 0.3)**2 + (x[1] - 0.3)**2
        return 0.5 * np.exp(-50 * r_sq) * np.sin(10 * np.pi * np.sqrt(r_sq))
    
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
        os.makedirs("figures", exist_ok=True)
        os.makedirs("post_data", exist_ok=True)
    MPI.COMM_WORLD.barrier()  # Esperar a que se creen los directorios
    
    # Configuración de PyVista
    pyvista.OFF_SCREEN = True
    pyvista.start_xvfb(wait=0.5)
    
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
    # Ecuación discretizada: (u^{n+1} - 2u^n + u^{n-1})/dt² = v²∇²u^{n+1} + f^{n+1}
    # Reorganizando: u^{n+1} - dt²v²∇²u^{n+1} = 2u^n - u^{n-1} + dt²f^{n+1}
    
    a = (u * v_test * ufl.dx + 
         (dt**2 * v**2) * ufl.dot(ufl.grad(u), ufl.grad(v_test)) * ufl.dx)
    
    # Setup XDMF para salida
    xdmf = io.XDMFFile(domain.comm, "post_data/wave_2d_solution.xdmf", "w")
    xdmf.write_mesh(domain)  # Escribir malla PRIMERO
    
    # Setup visualización 3D con PyVista
    cells, types, x_coords = plot.vtk_mesh(V)
    grid = pyvista.UnstructuredGrid(cells, types, x_coords)
    
    plotter = pyvista.Plotter(off_screen=True)
    plotter.open_gif("figures/wave_2d_propagation.gif", fps=10)
    
    # Configuración de colores y escala
    colormap = mpl.colormaps.get_cmap("RdBu_r").resampled(50)
    sargs = dict(title_font_size=20, label_font_size=15, fmt="%.3f", color="black",
                 position_x=0.02, position_y=0.1, width=0.15, height=0.8)
    
    # Escala fija para visualización consistente
    u_range = 0.8
    grid.point_data["u"] = u_n1.x.array
    
    # Crear superficie deformada para visualización 3D
    warped = grid.warp_by_scalar("u", factor=0.3)
    
    renderer = plotter.add_mesh(warped, show_edges=False, lighting=True,
                               cmap=colormap, scalar_bar_args=sargs,
                               clim=[-u_range, u_range], opacity=0.9)
    
    # Agregar malla de referencia en el plano
    plotter.add_mesh(grid, style="wireframe", color="gray", opacity=0.2)
    
    plotter.add_title("Propagación de Onda 2D", font_size=18, color="black")
    plotter.show_axes()
    
    # Configurar vista isométrica
    plotter.camera_position = [(1.5, 1.5, 1.0), (0.5, 0.5, 0.0), (0, 0, 1)]
    
    # Escribir condición inicial (DESPUÉS de escribir la malla)
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
    
    # Variables para monitoreo de energía
    energies = []
    times = []
    
    print("\nComenzando simulación de onda 2D...")
    print("Progreso: ", end="", flush=True)
    
    # Loop principal en tiempo
    for n in range(1, num_steps + 1):
        t = n * dt
        
        # Actualizar término fuente
        f_current = source_term(t)
        
        # Lado derecho del sistema: 2u^n - u^{n-1} + dt²f^{n+1}
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
        
        # Calcular energía total del sistema
        if n % 10 == 0:
            # Energía cinética aproximada: (1/2) * (∂u/∂t)²
            u_dot_approx = (u_n.x.array - u_n1.x.array) / dt
            kinetic_energy = 0.5 * np.sum(u_dot_approx**2) * (1.0 / len(u_dot_approx))
            
            # Energía potencial: (1/2) * v² * |∇u|²
            grad_u_form = fem.form(v**2 * ufl.dot(ufl.grad(u_n), ufl.grad(u_n)) * ufl.dx)
            potential_energy = 0.5 * fem.assemble_scalar(grad_u_form)
            
            total_energy = kinetic_energy + potential_energy
            energies.append(total_energy)
            times.append(t)
        
        # Actualizar soluciones anteriores
        u_n2.x.array[:] = u_n1.x.array[:]
        u_n1.x.array[:] = u_n.x.array[:]
        
        # Actualizar visualización y guardar datos
        if n % 8 == 0:  # Cada 8 pasos para controlar tamaño del GIF
            grid.point_data["u"] = u_n.x.array
            warped_new = grid.warp_by_scalar("u", factor=0.3)
            warped.points[:, :] = warped_new.points
            warped.point_data["u"][:] = u_n.x.array
            
            xdmf.write_function(u_n, t)
            plotter.write_frame()
        
        # Mostrar progreso
        if n % (num_steps // 20) == 0:
            print(" ", end="", flush=True)
            max_displacement = np.max(np.abs(u_n.x.array))
            if domain.comm.rank == 0 and n % (num_steps // 10) == 0:
                print(f"\n  t={t:.3f}, |u|_max={max_displacement:.4f}")
    
    print("\n")
    plotter.close()
    xdmf.close()
    
    print("Simulación completada!")
    print(f"Archivos generados:")
    print(f"  - Animación: figures/wave_2d_propagation.gif")
    print(f"  - Datos: post_data/wave_2d_solution.xdmf")
    print(f"  - Datos auxiliares: post_data/wave_2d_solution.h5")
    
    return u_n, domain, energies, times

def analyze_wave_properties():
    """
    Análisis teórico de la ecuación de onda 2D
    """
    print("\n=== Análisis de la Ecuación de Onda 2D ===")
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

if __name__ == "__main__":
    try:
        # Análisis inicial
        analyze_wave_properties()
        
        # Ejecutar simulación
        u_final, domain, energies, times = solve_wave_equation_2d()
        
        # Estadísticas finales
        if domain.comm.rank == 0:
            max_final = np.max(np.abs(u_final.x.array))
            print(f"\nEstadísticas finales:")
            print(f"  • Desplazamiento máximo final: {max_final:.6f}")
            if energies:
                energy_variation = (max(energies) - min(energies)) / max(energies) * 100
                print(f"  • Variación de energía: {energy_variation:.2f}%")
        
    except Exception as e:
        print(f"Error durante la ejecución: {e}")
        import traceback
        traceback.print_exc()