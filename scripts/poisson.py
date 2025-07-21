import importlib.util

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
import numpy as np
import os

import ufl
from dolfinx import fem, io, mesh, plot
from dolfinx.fem.petsc import LinearProblem


msh = mesh.create_rectangle(
    comm=MPI.COMM_WORLD,
    points=((0.0, 0.0), (2.0, 1.0)),
    n=(32, 16),
    cell_type=mesh.CellType.triangle,
)
V = fem.functionspace(msh, ("Lagrange", 1))

facets = mesh.locate_entities_boundary(
    msh,
    dim=(msh.topology.dim - 1),
    marker=lambda x: np.isclose(x[0], 0.0) | np.isclose(x[0], 2.0),
)

dofs = fem.locate_dofs_topological(V=V, entity_dim=1, entities=facets)


bc = fem.dirichletbc(value=ScalarType(0), dofs=dofs, V=V)

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
x = ufl.SpatialCoordinate(msh)
f = 10 * ufl.exp(-((x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2) / 0.02)
g = ufl.sin(5 * x[0])
a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = ufl.inner(f, v) * ufl.dx + ufl.inner(g, v) * ufl.ds

problem = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh = problem.solve()
assert isinstance(uh, fem.Function)

with io.XDMFFile(msh.comm, "post_data/poisson/poisson.xdmf", "w") as file:
    file.write_mesh(msh)
    file.write_function(uh)

try:
    import pyvista

    print("Configurando PyVista para modo headless...")
    
    # Create directories if they don't exist
    os.makedirs("figures/poisson", exist_ok=True)
    os.makedirs("post_data/poisson", exist_ok=True)
    
    pyvista.OFF_SCREEN = True
    pyvista.start_xvfb(wait=0.5)
    
    cells, types, x = plot.vtk_mesh(V)
    grid = pyvista.UnstructuredGrid(cells, types, x)
    grid.point_data["u"] = uh.x.array.real
    grid.set_active_scalars("u")
    
    plotter = pyvista.Plotter(off_screen=True)
    plotter.add_mesh(grid, show_edges=True)
    warped = grid.warp_by_scalar()
    plotter.add_mesh(warped)
    
    plotter.show(screenshot="figures/poisson/uh_poisson.png")
    print("Imagen guardada como: figures/poisson/uh_poisson.png")
    
except ModuleNotFoundError:
    print("'pyvista' is required to visualise the solution.")
    print("To install pyvista with pip: 'python3 -m pip install pyvista'.")
# -