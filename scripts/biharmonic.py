import importlib.util
import os

if importlib.util.find_spec("petsc4py") is not None:
    import dolfinx

    if not dolfinx.has_petsc:
        print("This demo requires DOLFINx to be compiled with PETSc enabled.")
        exit(0)
    from petsc4py.PETSc import ScalarType  # type: ignore
else:
    print("This demo requires petsc4py.")
    exit(0)

from mpi4py import MPI

# +
import dolfinx
import ufl
from dolfinx import fem, io, mesh, plot
from dolfinx.fem.petsc import LinearProblem
from dolfinx.mesh import CellType, GhostMode
from ufl import CellDiameter, FacetNormal, avg, div, dS, dx, grad, inner, jump, pi, sin

msh = mesh.create_rectangle(
    comm=MPI.COMM_WORLD,
    points=((0.0, 0.0), (1.0, 1.0)),
    n=(32, 32),
    cell_type=CellType.triangle,
    ghost_mode=GhostMode.shared_facet,
)

V = fem.functionspace(msh, ("Lagrange", 2))

tdim = msh.topology.dim
msh.topology.create_connectivity(tdim - 1, tdim)
facets = mesh.exterior_facet_indices(msh.topology)

dofs = fem.locate_dofs_topological(V=V, entity_dim=1, entities=facets)

bc = fem.dirichletbc(value=ScalarType(0), dofs=dofs, V=V)

alpha = ScalarType(8.0)
h = CellDiameter(msh)
n = FacetNormal(msh)
h_avg = (h("+") + h("-")) / 2.0


u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
x = ufl.SpatialCoordinate(msh)
f = 4.0 * pi**4 * sin(pi * x[0]) * sin(pi * x[1])

a = (
    inner(div(grad(u)), div(grad(v))) * dx
    - inner(avg(div(grad(u))), jump(grad(v), n)) * dS
    - inner(jump(grad(u), n), avg(div(grad(v)))) * dS
    + alpha / h_avg * inner(jump(grad(u), n), jump(grad(v), n)) * dS
)

L = inner(f, v) * dx


problem = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh = problem.solve()

with io.XDMFFile(msh.comm, "post_data/biharmonic/biharmonic.xdmf", "w") as file:
    V1 = fem.functionspace(msh, ("Lagrange", 1))
    u1 = fem.Function(V1)
    u1.interpolate(uh)
    file.write_mesh(msh)
    file.write_function(u1)

try:
    import pyvista

    print("Configurando PyVista para modo headless...")
    
    # Create directories if they don't exist
    os.makedirs("figures/biharmonic", exist_ok=True)
    os.makedirs("post_data/biharmonic", exist_ok=True)
    
    pyvista.OFF_SCREEN = True
    pyvista.start_xvfb(wait=0.5)
    
    cells, types, x = plot.vtk_mesh(V)
    grid = pyvista.UnstructuredGrid(cells, types, x)
    grid.point_data["u"] = uh.x.array.real
    grid.set_active_scalars("u")
    
    # Crear plotter en modo off_screen
    plotter = pyvista.Plotter(off_screen=True)
    plotter.add_mesh(grid, show_edges=True)
    warped = grid.warp_by_scalar()
    plotter.add_mesh(warped)
    
    # Tomar screenshot
    plotter.show(screenshot="figures/biharmonic/biharmonic.png")
    print("Imagen guardada como: figures/biharmonic/biharmonic.png")
    
except ModuleNotFoundError:
    print("'pyvista' is required to visualise the solution.")
    print("To install pyvista with pip: 'python3 -m pip install pyvista'.")
# -