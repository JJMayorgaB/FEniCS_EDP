from petsc4py import PETSc
from mpi4py import MPI
import ufl
from dolfinx import mesh, fem, io, plot
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, create_vector, set_bc
import numpy
import matplotlib as mpl
import pyvista
import os

t = 0  # Start time
T = 2  # End time
num_steps = 20  # Number of time steps
dt = (T - t) / num_steps  # Time step size
alpha = 3
beta = 1.2

nx, ny = 5, 5
domain = mesh.create_unit_square(MPI.COMM_WORLD, nx, ny, mesh.CellType.triangle)
V = fem.functionspace(domain, ("Lagrange", 1))

class exact_solution():
    def __init__(self, alpha, beta, t):
        self.alpha = alpha
        self.beta = beta
        self.t = t

    def __call__(self, x):
        return 1 + x[0]**2 + self.alpha * x[1]**2 + self.beta * self.t


u_exact = exact_solution(alpha, beta, t)

u_D = fem.Function(V)
u_D.interpolate(u_exact)
tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)
boundary_facets = mesh.exterior_facet_indices(domain.topology)
bc = fem.dirichletbc(u_D, fem.locate_dofs_topological(V, fdim, boundary_facets))

u_n = fem.Function(V)
u_n.interpolate(u_exact)

f = fem.Constant(domain, beta - 2 - 2 * alpha)

u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
F = u * v * ufl.dx + dt * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx - (u_n + dt * f) * v * ufl.dx
a = fem.form(ufl.lhs(F))
L = fem.form(ufl.rhs(F))

A = assemble_matrix(a, bcs=[bc])
A.assemble()
b = create_vector(L)
uh = fem.Function(V)

solver = PETSc.KSP().create(domain.comm)
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.PREONLY)
solver.getPC().setType(PETSc.PC.Type.LU)

# Create directories if they don't exist
os.makedirs("figures/heat", exist_ok=True)
os.makedirs("post_data/heat", exist_ok=True)

# Setup PyVista visualization
pyvista.OFF_SCREEN = True
pyvista.start_xvfb(wait=0.5)

cells, types, x = plot.vtk_mesh(V)
grid = pyvista.UnstructuredGrid(cells, types, x)

plotter = pyvista.Plotter(off_screen=True)
plotter.open_gif("figures/heat/heat_solution.gif", fps=10)

# Setup XDMF output
xdmf = io.XDMFFile(domain.comm, "post_data/heat/heat_solution.xdmf", "w")
xdmf.write_mesh(domain)

# Initial visualization setup
grid.point_data["uh"] = uh.x.array
# Eliminar warping para visualización 2D
# warped = grid.warp_by_scalar("uh", factor=0.1)

colormap = mpl.colormaps.get_cmap("plasma").resampled(25)
sargs = dict(title_font_size=25, label_font_size=20, fmt="%.2f", color="black",
             position_x=0.1, position_y=0.8, width=0.8, height=0.1)

# Get initial range for consistent color scaling
u_D.interpolate(u_exact)
u_min, u_max = numpy.min(u_D.x.array), numpy.max(u_D.x.array)

# Usar grid directamente sin warping
renderer = plotter.add_mesh(grid, show_edges=True, lighting=False,
                            cmap=colormap, scalar_bar_args=sargs,
                            clim=[u_min, u_max])

plotter.add_title("Solución de la Ecuación del Calor", font_size=16, color="black")

# Configurar vista superior (2D)
plotter.view_xy()

# Write initial condition
xdmf.write_function(uh, t)
plotter.write_frame()

for n in range(num_steps):
    # Update Diriclet boundary condition
    u_exact.t += dt
    u_D.interpolate(u_exact)

    # Update the right hand side reusing the initial vector
    with b.localForm() as loc_b:
        loc_b.set(0)
    assemble_vector(b, L)

    # Apply Dirichlet boundary condition to the vector
    apply_lifting(b, [a], [[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b, [bc])

    # Solve linear problem
    solver.solve(b, uh.x.petsc_vec)
    uh.x.scatter_forward()

    # Update solution at previous time step (u_n)
    u_n.x.array[:] = uh.x.array
    
    # Update visualization - solo actualizar los datos escalares
    grid.point_data["uh"] = uh.x.array
    
    # Write to file and create frame
    xdmf.write_function(uh, u_exact.t)
    plotter.write_frame()

plotter.close()
xdmf.close()

V_ex = fem.functionspace(domain, ("Lagrange", 2))
u_ex = fem.Function(V_ex)
u_ex.interpolate(u_exact)
error_L2 = numpy.sqrt(domain.comm.allreduce(fem.assemble_scalar(fem.form((uh - u_ex)**2 * ufl.dx)), op=MPI.SUM))
if domain.comm.rank == 0:
    print(f"L2-error: {error_L2:.2e}")

# Compute values at mesh vertices
error_max = domain.comm.allreduce(numpy.max(numpy.abs(uh.x.array - u_D.x.array)), op=MPI.MAX)
if domain.comm.rank == 0:
    print(f"Error_max: {error_max:.2e}")