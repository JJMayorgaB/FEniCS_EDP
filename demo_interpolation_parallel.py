# Interpolation and IO - Parallel Version with OpenMPI
# Optimized for parallel execution with MPI - VISUALIZATION FIXED

from mpi4py import MPI
import numpy as np
import os

from dolfinx import default_scalar_type, plot
from dolfinx.fem import Function, functionspace
from dolfinx.mesh import CellType, create_rectangle, locate_entities

def main():

    # Get MPI communicator information
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank == 0:
        print(f"Running on {size} MPI processes")
        # Create directories if they don't exist
        os.makedirs("figures", exist_ok=True)
        os.makedirs("post_data", exist_ok=True)
    
    # Create a distributed mesh - automatically partitioned across processes
    # Increase mesh resolution for better parallel scaling
    nx, ny = 16, 16 
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
        filename = f"post_data/output_nedelec_parallel"
        
        if rank == 0:
            print("Writing parallel output...")
            
        with VTXWriter(comm, f"{filename}.bp", u0, "bp4") as f:
            f.write(0.0)
            
        if rank == 0:
            print("Parallel output written successfully")
            
    except ImportError:
        if rank == 0:
            print("ADIOS2 required for parallel VTX output")

    # FIXED: Parallel visualization with proper data gathering
    try:
        import pyvista
        
        if rank == 0:
            print("Configurando PyVista para modo headless...")
            pyvista.OFF_SCREEN = True
            
            # Inicializar Xvfb solo si es necesario
            try:
                pyvista.start_xvfb(wait=0.5)
                print("Xvfb iniciado correctamente")
            except Exception as e:
                print(f"Warning: No se pudo iniciar Xvfb: {e}")

            print("Creating visualization...")
        
        # Create VTK mesh data for each process
        cells, types, x = plot.vtk_mesh(V0)
        
        # Prepare local function values
        local_values = np.zeros((x.shape[0], 3), dtype=np.float64)
        local_values[:, :msh.topology.dim] = u0.x.array.reshape(x.shape[0], msh.topology.dim).real
        
        # Method 1: Individual process visualization (as fallback)
        if rank == 0:
            print("Creating individual process visualization...")
            
            # Create local visualization for process 0
            grid = pyvista.UnstructuredGrid(cells, types, x)
            grid.point_data["u"] = local_values
            magnitude = np.linalg.norm(local_values[:, :msh.topology.dim], axis=1)
            grid.point_data["magnitude"] = magnitude
            
            # Save individual process visualization
            plotter = pyvista.Plotter(off_screen=True)
            plotter.add_text(f"Process 0 - Magnitude (of {size} procs)", font_size=14, color="black")
            plotter.add_mesh(grid, scalars="magnitude", show_edges=True, cmap="plasma")
            plotter.view_xy()
            plotter.screenshot(f"figures/process0_magnitude_{size}procs.png")
            print(f"Process 0 visualization saved: figures/process0_magnitude_{size}procs.png")
        
        # Method 2: Optimized approach using MPI gather operations
        try:
            if rank == 0:
                print("Attempting global visualization using MPI gather...")
            
            # Create local grid data structure for gathering
            local_grid_data = {
                'x': x,
                'values': local_values,
                'cells': cells,
                'types': types
            }
            
            # Gather all grid data to process 0
            all_grid_data = comm.gather(local_grid_data, root=0)
            
            if rank == 0:
                print(f"Gathered data from {len(all_grid_data)} processes")
                
                # Create grids from gathered data
                local_grids = []
                for i, grid_data in enumerate(all_grid_data):
                    # Create grid for this process
                    proc_grid = pyvista.UnstructuredGrid(
                        grid_data['cells'], 
                        grid_data['types'], 
                        grid_data['x']
                    )
                    proc_grid.point_data["u"] = grid_data['values']
                    proc_magnitude = np.linalg.norm(
                        grid_data['values'][:, :msh.topology.dim], axis=1
                    )
                    proc_grid.point_data["magnitude"] = proc_magnitude
                    
                    local_grids.append(proc_grid)
                    print(f"Process {i} grid: {proc_grid.n_points} points, {proc_grid.n_cells} cells")
                
                # Combine all grids using PyVista's merge functionality
                if len(local_grids) > 1:
                    print("Merging grids from all processes...")
                    combined_grid = local_grids[0]
                    for grid in local_grids[1:]:
                        combined_grid = combined_grid.merge(grid)
                    
                    # Clean up duplicate points
                    combined_grid = combined_grid.clean(tolerance=1e-10)
                    
                    print(f"Combined grid created with {combined_grid.n_points} points and {combined_grid.n_cells} cells")
                    
                    # Create combined visualization
                    pl = pyvista.Plotter(shape=(2, 2), off_screen=True)
                    
                    pl.subplot(0, 0)
                    pl.add_text(f"Magnitude (parallel {size} procs)", font_size=12, color="black", position="upper_edge")
                    pl.add_mesh(combined_grid.copy(), scalars="magnitude", show_edges=True, cmap="viridis")
                    
                    pl.subplot(0, 1)
                    glyphs = combined_grid.glyph(orient="u", factor=0.08)
                    pl.add_text(f"Vector glyphs (parallel {size} procs)", font_size=12, color="black", position="upper_edge")
                    pl.add_mesh(glyphs, show_scalar_bar=False)
                    pl.add_mesh(combined_grid.copy(), style="wireframe", line_width=2, color="black")
                    
                    pl.subplot(1, 0)
                    pl.add_text(f"X-component (parallel {size} procs)", font_size=12, color="black", position="upper_edge")
                    pl.add_mesh(combined_grid.copy(), scalars="u", component=0, show_edges=True, cmap="coolwarm")
                    
                    pl.subplot(1, 1)
                    pl.add_text(f"Y-component (parallel {size} procs)", font_size=12, color="black", position="upper_edge")
                    pl.add_mesh(combined_grid.copy(), scalars="u", component=1, show_edges=True, cmap="coolwarm")
                    
                    pl.view_xy()
                    pl.link_views()
                    
                    # Save main image
                    pl.screenshot(f"figures/uh_interpolation_parallel_{size}procs_4plots.png")
                    print(f"Global visualization saved: figures/uh_interpolation_parallel_{size}procs_4plots.png")
                    
                    # Also create a simple magnitude plot
                    plotter_simple = pyvista.Plotter(off_screen=True)
                    plotter_simple.add_text(f"Magnitude - Parallel Nedelec ({size} procs)", font_size=14, color="black")
                    plotter_simple.add_mesh(combined_grid, scalars="magnitude", show_edges=True, cmap="plasma")
                    plotter_simple.view_xy()
                    plotter_simple.screenshot(f"figures/uh_magnitude_parallel_{size}procs.png")
                    print(f"Simple magnitude visualization saved: figures/uh_magnitude_parallel_{size}procs.png")
                    
                else:
                    # Only one process - use local grid
                    grid = local_grids[0]
                    pl = pyvista.Plotter(shape=(2, 2), off_screen=True)
                    
                    pl.subplot(0, 0)
                    pl.add_text(f"Magnitude (single process)", font_size=12, color="black", position="upper_edge")
                    pl.add_mesh(grid.copy(), scalars="magnitude", show_edges=True, cmap="viridis")
                    
                    pl.subplot(0, 1)
                    glyphs = grid.glyph(orient="u", factor=0.08)
                    pl.add_text(f"Vector glyphs (single process)", font_size=12, color="black", position="upper_edge")
                    pl.add_mesh(glyphs, show_scalar_bar=False)
                    pl.add_mesh(grid.copy(), style="wireframe", line_width=2, color="black")
                    
                    pl.subplot(1, 0)
                    pl.add_text(f"X-component (single process)", font_size=12, color="black", position="upper_edge")
                    pl.add_mesh(grid.copy(), scalars="u", component=0, show_edges=True, cmap="coolwarm")
                    
                    pl.subplot(1, 1)
                    pl.add_text(f"Y-component (single process)", font_size=12, color="black", position="upper_edge")
                    pl.add_mesh(grid.copy(), scalars="u", component=1, show_edges=True, cmap="coolwarm")
                    
                    pl.view_xy()
                    pl.link_views()
                    
                    pl.screenshot(f"figures/uh_interpolation_parallel_{size}procs_4plots.png")
                    print(f"Single process visualization saved: figures/uh_interpolation_parallel_{size}procs_4plots.png")
                
        except Exception as e:
            if rank == 0:
                print(f"Global visualization failed: {e}")
                print("Using individual process visualization instead")
                
    except ModuleNotFoundError:
        if rank == 0:
            print("pyvista is required to visualise the solution")
    
    # Final synchronization
    comm.Barrier()
    if rank == 0:
        print("Parallel execution completed successfully")

if __name__ == "__main__":
    main()