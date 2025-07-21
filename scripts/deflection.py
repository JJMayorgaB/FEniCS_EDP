import gmsh
import numpy as np
import os

# Create directories if they don't exist
os.makedirs("figures", exist_ok=True)
os.makedirs("post_data", exist_ok=True)

gmsh.initialize()


membrane = gmsh.model.occ.addDisk(0, 0, 0, 1, 1)
gmsh.model.occ.synchronize()

gdim = 2
gmsh.model.addPhysicalGroup(gdim, [membrane], 1)

gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.05)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.05)
gmsh.model.mesh.generate(gdim)

from dolfinx.io import gmshio
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI

gmsh_model_rank = 0
mesh_comm = MPI.COMM_WORLD
domain, cell_markers, facet_markers = gmshio.model_to_mesh(gmsh.model, mesh_comm, gmsh_model_rank, gdim=gdim)


from dolfinx import fem

V = fem.functionspace(domain, ("Lagrange", 1))


import ufl
from dolfinx import default_scalar_type

x = ufl.SpatialCoordinate(domain)
beta = fem.Constant(domain, default_scalar_type(12))
R0 = fem.Constant(domain, default_scalar_type(0.3))
p = 4 * ufl.exp(-(beta**2) * (x[0] ** 2 + (x[1] - R0) ** 2))


import numpy as np


def on_boundary(x):
    return np.isclose(np.sqrt(x[0] ** 2 + x[1] ** 2), 1)


boundary_dofs = fem.locate_dofs_geometrical(V, on_boundary)

bc = fem.dirichletbc(default_scalar_type(0), boundary_dofs, V)

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = p * v * ufl.dx
problem = LinearProblem(
    a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
)
uh = problem.solve()


Q = fem.functionspace(domain, ("Lagrange", 5))
expr = fem.Expression(p, Q.element.interpolation_points())
pressure = fem.Function(Q)
pressure.interpolate(expr)

from dolfinx.plot import vtk_mesh

# +
try:
    import pyvista
    
    print("Configurando PyVista para modo headless...")
    pyvista.OFF_SCREEN = True

    topology, cell_types, x = vtk_mesh(V)
    grid = pyvista.UnstructuredGrid(topology, cell_types, x)

    grid.point_data["u"] = uh.x.array
    warped = grid.warp_by_scalar("u", factor=25)

    plotter = pyvista.Plotter(off_screen=True)
    plotter.add_mesh(warped, show_edges=True, show_scalar_bar=True, scalars="u")
    plotter.screenshot("figures/deflection.png")
    plotter.close()
    print("Imagen de deflexión guardada como: figures/deflection.png")

    load_plotter = pyvista.Plotter(off_screen=True)
    p_grid = pyvista.UnstructuredGrid(*vtk_mesh(Q))
    p_grid.point_data["p"] = pressure.x.array.real
    warped_p = p_grid.warp_by_scalar("p", factor=0.5)
    warped_p.set_active_scalars("p")
    load_plotter.add_mesh(warped_p, show_scalar_bar=True)
    load_plotter.view_xy()
    load_plotter.screenshot("figures/load.png")
    load_plotter.close()
    print("Imagen de carga guardada como: figures/load.png")

except ModuleNotFoundError:
    print("'pyvista' is required to visualise the solution.")
    print("To install pyvista with pip: 'python3 -m pip install pyvista'.")
except Exception as e:
    print(f"Error en visualización con PyVista: {e}")
# -


tol = 0.001 
y = np.linspace(-1 + tol, 1 - tol, 101)
points = np.zeros((3, 101))
points[1] = y
u_values = []
p_values = []

from dolfinx import geometry

bb_tree = geometry.bb_tree(domain, domain.topology.dim)

cells = []
points_on_proc = []
cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)
for i, point in enumerate(points.T):
    if len(colliding_cells.links(i)) > 0:
        points_on_proc.append(point)
        cells.append(colliding_cells.links(i)[0])


points_on_proc = np.array(points_on_proc, dtype=np.float64)
u_values = uh.eval(points_on_proc, cells)
p_values = pressure.eval(points_on_proc, cells)


import matplotlib.pyplot as plt

fig = plt.figure()
plt.plot(
    points_on_proc[:, 1],
    50 * u_values,
    "k",
    linewidth=2,
    label="Deflection ($\\times 50$)",
)
plt.plot(points_on_proc[:, 1], p_values, "b--", linewidth=2, label="Load")
plt.grid(True)
plt.xlabel("y")
plt.legend()

plt.savefig(f"figures/membrane_rank{MPI.COMM_WORLD.rank:d}.png")


import dolfinx.io
from pathlib import Path

pressure.name = "Load"
uh.name = "Deflection"
results_folder = Path("post_data/results_membrane")
results_folder.mkdir(exist_ok=True, parents=True)
with dolfinx.io.VTXWriter(
    MPI.COMM_WORLD, results_folder / "membrane_pressure.bp", [pressure], engine="BP4"
) as vtx:
    vtx.write(0.0)
with dolfinx.io.VTXWriter(
    MPI.COMM_WORLD, results_folder / "membrane_deflection.bp", [uh], engine="BP4"
) as vtx:
    vtx.write(0.0)