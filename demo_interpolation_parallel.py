# Interpolation and IO - Parallel Version with OpenMPI
# Optimized for parallel execution with MPI

from mpi4py import MPI
import numpy as np

from dolfinx import default_scalar_type, plot
from dolfinx.fem import Function, functionspace
from dolfinx.mesh import CellType, create_rectangle, locate_entities

def main():
    # Get MPI communicator information
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    print(size)

    if rank == 0:
        print(f"Running on {size} MPI processes")
    
    # Create a distributed mesh - automatically partitioned across processes
    # Increase mesh resolution for better parallel scaling
    nx, ny = 32, 32  # Increased from 16x16 for better parallelization
    msh = create_rectangle(comm, ((0.0, 0.0), (1.0, 1.0)), (nx, ny), CellType.triangle)
    
    # Print global info only once
    if rank == 0:
        print(f"Mesh created with {msh.topology.index_map(msh.topology.dim).size_global} cells")
    
    # Print local info for each process (synchronized)
    for i in range(size):
        if rank == i:
            print(f"Process {rank} has {msh.topology.index_map(msh.topology.dim).size_local} local cells")
        comm.Barrier()  # Wait for each process to print in order
    
    # Create a Nédélec function space (automatically distributed)
    V = functionspace(msh, ("Nedelec 1st kind H(curl)", 1))
    u = Function(V, dtype=default_scalar_type)
    
    # Each process finds its local cells
    tdim = msh.topology.dim
    cells0 = locate_entities(msh, tdim, lambda x: x[0] <= 0.5)
    cells1 = locate_entities(msh, tdim, lambda x: x[0] >= 0.5)
    
    if rank == 0:
        print("Interpolating functions in parallel...")
    
    # Parallel interpolation - each process works on its local cells
    u.interpolate(lambda x: np.vstack((x[0], x[1])), cells0)
    u.interpolate(lambda x: np.vstack((x[0] + 1, x[1])), cells1)
    
    # Synchronize after interpolation
    comm.Barrier()
    
    # Create discontinuous Lagrange space for visualization
    gdim = msh.geometry.dim
    V0 = functionspace(msh, ("Discontinuous Lagrange", 1, (gdim,)))
    u0 = Function(V0, dtype=default_scalar_type)
    
    # Parallel interpolation from H(curl) to DG space
    u0.interpolate(u)
    
    # Parallel I/O - each process writes its portion
    try:
        from dolfinx.io import VTXWriter
        
        # Create filename with MPI-aware naming
        filename = f"output_nedelec_parallel"
        
        if rank == 0:
            print("Writing parallel output...")
            
        with VTXWriter(comm, f"{filename}.bp", u0, "bp4") as f:
            f.write(0.0)
            
        if rank == 0:
            print("Parallel output written successfully")
            
    except ImportError:
        if rank == 0:
            print("ADIOS2 required for parallel VTX output")

    # Parallel visualization (only on rank 0 for display)
    if rank == 0:
        try:
            import pyvista
            
            print("Configurando PyVista para modo headless...")
            pyvista.OFF_SCREEN = True
            
            # Inicializar Xvfb solo si es necesario
            try:
                pyvista.start_xvfb(wait=0.5)
                print("Xvfb iniciado correctamente")
            except Exception as e:
                print(f"Warning: No se pudo iniciar Xvfb: {e}")

            print("Creating visualization...")
            
            # Gather data from all processes for visualization
            cells, types, x = plot.vtk_mesh(V0)
            grid = pyvista.UnstructuredGrid(cells, types, x)
            values = np.zeros((x.shape[0], 3), dtype=np.float64)
            values[:, : msh.topology.dim] = u0.x.array.reshape(x.shape[0], msh.topology.dim).real
            grid.point_data["u"] = values
            
            # Agregar magnitud como dato escalar para mejor visualización
            magnitude = np.linalg.norm(values[:, :msh.topology.dim], axis=1)
            grid.point_data["magnitude"] = magnitude

            # Crear plotter principal con 4 subplots
            pl = pyvista.Plotter(shape=(2, 2), off_screen=True)

            pl.subplot(0, 0)
            pl.add_text("magnitude (parallel)", font_size=12, color="black", position="upper_edge")
            pl.add_mesh(grid.copy(), scalars="magnitude", show_edges=True, cmap="viridis")

            pl.subplot(0, 1)
            glyphs = grid.glyph(orient="u", factor=0.08)
            pl.add_text("vector glyphs (parallel)", font_size=12, color="black", position="upper_edge")
            pl.add_mesh(glyphs, show_scalar_bar=False)
            pl.add_mesh(grid.copy(), style="wireframe", line_width=2, color="black")

            pl.subplot(1, 0)
            pl.add_text("x-component (parallel)", font_size=12, color="black", position="upper_edge")
            pl.add_mesh(grid.copy(), scalars="u", component=0, show_edges=True, cmap="coolwarm")

            pl.subplot(1, 1)
            pl.add_text("y-component (parallel)", font_size=12, color="black", position="upper_edge")
            pl.add_mesh(grid.copy(), scalars="u", component=1, show_edges=True, cmap="coolwarm")

            pl.view_xy()
            pl.link_views()
            
            # Guardar imagen principal
            pl.screenshot("uh_interpolation_parallel_4plots.png")
            print("Screenshot principal guardado: uh_interpolation_parallel_4plots.png")

            # Crear una visualización adicional simple
            plotter_simple = pyvista.Plotter(off_screen=True)
            plotter_simple.add_text("Magnitude - Parallel Nedelec", font_size=14, color="black")
            plotter_simple.add_mesh(grid, scalars="magnitude", show_edges=True, cmap="plasma")
            plotter_simple.view_xy()
            plotter_simple.screenshot("uh_magnitude_parallel.png")
            print("Screenshot magnitud guardado: uh_magnitude_parallel.png")
                
        except ModuleNotFoundError:
            print("pyvista is required to visualise the solution")
    
    # Final synchronization
    comm.Barrier()
    if rank == 0:
        print("Parallel execution completed successfully")

if __name__ == "__main__":
    main()
