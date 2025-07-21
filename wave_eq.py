import pyvista
import ufl
import numpy as np
import os
from petsc4py import PETSc
from mpi4py import MPI
import matplotlib as mpl

from dolfinx import fem, mesh, io, plot
from dolfinx.fem.petsc import assemble_vector, assemble_matrix, create_vector, apply_lifting, set_bc

def solve_wave_equation():
    """
    Resuelve la ecuación de onda:
    ∂²u/∂x² - (1/v²)∂²u/∂t² = A*sin(kx - ωt)
    
    Usando el método de diferencias finitas en tiempo con esquema de Newmark.
    """
    
    # Parámetros físicos
    v = 2.0      # Velocidad de onda
    A = 1.0      # Amplitud de la fuente
    k = np.pi    # Número de onda
    omega = k * v  # Frecuencia angular
    
    # Parámetros temporales
    t = 0.0      # Tiempo inicial
    T = 5.0      # Tiempo final
    num_steps = 1000
    dt = T / num_steps
    
    # Crear dominio 1D
    nx = 100
    domain = mesh.create_interval(MPI.COMM_WORLD, nx, [0.0, 2.0])
    V = fem.functionspace(domain, ("Lagrange", 1))
    
    # Condiciones de frontera (extremos fijos)
    def boundary(x, on_boundary):
        return on_boundary
    
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.full(x.shape[1], True, dtype=bool))
    bc = fem.dirichletbc(PETSc.ScalarType(0), 
                        fem.locate_dofs_topological(V, fdim, boundary_facets), V)
    
    # Funciones para almacenar soluciones
    u_n2 = fem.Function(V)  # u^{n-1}
    u_n1 = fem.Function(V)  # u^n
    u_n = fem.Function(V)   # u^{n+1} (solución actual)
    
    # Condiciones iniciales
    def initial_displacement(x):
        return np.sin(np.pi * x[0]) * np.exp(-10 * (x[0] - 1)**2)
    
    def initial_velocity(x):
        return np.zeros_like(x[0])
    
    u_n2.interpolate(initial_displacement)
    u_n1.interpolate(initial_displacement)
    
    # Para la condición inicial de velocidad, usamos:
    # u^{n-1} = u^n - dt * v_0 + (dt²/2) * (v² * ∂²u/∂x²)
    # Simplificación: u^{n-1} ≈ u^n (velocidad inicial cero)
    
    # Crear directorio de salida
    os.makedirs("figures", exist_ok=True)
    os.makedirs("post_data", exist_ok=True)
    
    # Configuración para gráfico cartesiano en lugar de PyVista
    import matplotlib.pyplot as plt
    import imageio
    
    # Obtener coordenadas x para el gráfico
    dofs_x = V.tabulate_dof_coordinates()[:, 0]  # Solo coordenada x
    x_sorted_indices = np.argsort(dofs_x)
    x_coords_sorted = dofs_x[x_sorted_indices]
    
    # Configurar matplotlib para modo no interactivo
    plt.ioff()
    
    # Lista para almacenar frames del GIF
    gif_frames = []
    
    # Configuración del gráfico
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 2.0)
    ax.set_ylim(-2.5, 2.5)
    ax.set_xlabel('Posición x (m)', fontsize=12)
    ax.set_ylabel('Desplazamiento u(x,t)', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Forma bilineal y lineal usando esquema de diferencias finitas en tiempo
    u = ufl.TrialFunction(V)
    v_test = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)
    
    # Término fuente A*sin(kx - ωt)
    def source_term(t_val):
        return A * ufl.sin(k * x[0] - omega * t_val)
    
    # Forma bilineal para el esquema implícito de segundo orden en tiempo
    # (u^{n+1} - 2u^n + u^{n-1})/dt² = v² * ∂²u^{n+1}/∂x² + f^{n+1}
    
    a = (u * v_test * ufl.dx + 
         (dt**2 * v**2) * ufl.dot(ufl.grad(u), ufl.grad(v_test)) * ufl.dx)
    
    # Setup XDMF para salida
    xdmf = io.XDMFFile(domain.comm, "post_data/wave_solution.xdmf", "w")
    xdmf.write_mesh(domain)
    
    # Setup visualización
    cells, types, x_coords = plot.vtk_mesh(V)
    grid = pyvista.UnstructuredGrid(cells, types, x_coords)
    
    plotter = pyvista.Plotter(off_screen=True)
    plotter.open_gif("figures/wave_propagation.gif", fps=20)
    
    colormap = mpl.colormaps.get_cmap("seismic").resampled(25)
    sargs = dict(title_font_size=20, label_font_size=15, fmt="%.2f", color="black",
                 position_x=0.1, position_y=0.8, width=0.8, height=0.1)
    
    # Escala fija para visualización
    u_range = 2.0
    grid.point_data["u"] = u_n1.x.array
    
    renderer = plotter.add_mesh(grid, show_edges=True, lighting=False,
                               cmap=colormap, scalar_bar_args=sargs,
                               clim=[-u_range, u_range])
    
    plotter.add_title("Propagación de Onda 1D", font_size=16, color="black")
    plotter.view_xy()
    
    # Escribir condición inicial
    u_n.x.array[:] = u_n1.x.array
    u_n.name = "displacement"
    xdmf.write_function(u_n, t)
    
    # Crear frame inicial
    u_values_sorted = u_n1.x.array[x_sorted_indices]
    ax.clear()
    ax.plot(x_coords_sorted, u_values_sorted, 'b-', linewidth=2, label=f't = {t:.3f} s')
    ax.set_xlim(0, 2.0)
    ax.set_ylim(-2.5, 2.5)
    ax.set_xlabel('Posición x (m)', fontsize=12)
    ax.set_ylabel('Desplazamiento u(x,t)', fontsize=12)
    ax.set_title('Propagación de Onda 1D', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Guardar frame inicial
    frame_filename = f"figures/frame_000.png"
    plt.savefig(frame_filename, dpi=100, bbox_inches='tight')
    gif_frames.append(frame_filename)
    
    # Ensamblaje de matrices
    bilinear_form = fem.form(a)
    A_matrix = assemble_matrix(bilinear_form, bcs=[bc])
    A_matrix.assemble()
    
    # Configurar solver
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A_matrix)
    solver.setType(PETSc.KSP.Type.PREONLY)
    solver.getPC().setType(PETSc.PC.Type.LU)
    
    # Vector para el lado derecho
    b = create_vector(fem.form(u * v_test * ufl.dx))
    
    print("Comenzando simulación de la ecuación de onda...")
    
    # Loop temporal
    for n in range(1, num_steps + 1):
        t = n * dt
        
        # Término del lado derecho con esquema de diferencias finitas
        # (2u^n - u^{n-1})/dt² + f^{n+1}
        f_current = source_term(t)
        
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
        u_n2.x.array[:] = u_n1.x.array
        u_n1.x.array[:] = u_n.x.array
        
        # Crear frame para GIF cada pocos pasos
        if n % 5 == 0:
            xdmf.write_function(u_n, t)
            
            # Crear gráfico cartesiano
            u_values_sorted = u_n.x.array[x_sorted_indices]
            ax.clear()
            ax.plot(x_coords_sorted, u_values_sorted, 'b-', linewidth=2, label=f't = {t:.3f} s')
            ax.set_xlim(0, 2.0)
            ax.set_ylim(-2.5, 2.5)
            ax.set_xlabel('Posición x (m)', fontsize=12)
            ax.set_ylabel('Desplazamiento u(x,t)', fontsize=12)
            ax.set_title('Propagación de Onda 1D', fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Guardar frame
            frame_filename = f"figures/frame_{len(gif_frames):03d}.png"
            plt.savefig(frame_filename, dpi=100, bbox_inches='tight')
            gif_frames.append(frame_filename)
            
            if n % 20 == 0:
                print(f"Tiempo: {t:.3f}, Máximo desplazamiento: {np.max(np.abs(u_n.x.array)):.3f}")
    
    # Crear GIF a partir de los frames
    print("Creando animación GIF...")
    with imageio.get_writer("figures/wave_propagation.gif", mode='I', duration=0.1) as writer:
        for frame_file in gif_frames:
            image = imageio.imread(frame_file)
            writer.append_data(image)
    
    # Limpiar frames temporales
    for frame_file in gif_frames:
        os.remove(frame_file)
    
    plt.close(fig)
    xdmf.close()
    
    print("Simulación completada!")
    print(f"Archivos generados:")
    print(f"  - Animación: figures/wave_propagation.gif")
    print(f"  - Datos: post_data/wave_solution.xdmf")
    
    return u_n, domain

# Función para análisis de la solución
def analyze_wave_solution():
    """
    Función adicional para análizar propiedades de la onda
    """
    print("\n=== Análisis de la Ecuación de Onda ===")
    print("Ecuación: ∂²u/∂x² - (1/v²)∂²u/∂t² = A*sin(kx - ωt)")
    print(f"Velocidad de onda (v): 2.0 m/s")
    print(f"Número de onda (k): π rad/m")
    print(f"Frecuencia angular (ω): 2π rad/s")
    print(f"Longitud de onda (λ): {2*np.pi/np.pi:.2f} m")
    print(f"Período (T): {2*np.pi/(np.pi*2):.2f} s")

if __name__ == "__main__":
    try:
        analyze_wave_solution()
        u_final, domain = solve_wave_equation()
        
    except Exception as e:
        print(f"Error durante la ejecución: {e}")
        import traceback
        traceback.print_exc()