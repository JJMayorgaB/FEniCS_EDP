FROM dolfinx/dolfinx:stable

# Dependency versions
ARG GMSH_VERSION=4_12_2

WORKDIR /tmp

# Install system dependencies
RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get -qq update && \
    apt-get -y install \
    libglu1 \
    libxcursor-dev \
    libxft2 \
    libxinerama1 \
    libfltk1.3-dev \
    libfreetype6-dev  \
    libgl1-mesa-dev \
    libocct-foundation-dev \
    libocct-data-exchange-dev \
    git \
    wget \
    curl \
    ninja-build \
    htop && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install and configure GMSH
#RUN git clone -b gmsh_${GMSH_VERSION} --single-branch --depth 1 https://gitlab.onelab.info/gmsh/gmsh.git && \
 #   cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DENABLE_BUILD_DYNAMIC=1 -B build-dir -S gmsh && \
  #  cmake --build build-dir && \
   # cmake --install build-dir && \
    # Move gmsh python package to correct location
    #export SP_DIR=$(python3 -c 'import sys, site; sys.stdout.write(site.getsitepackages()[0])') && \
   #mv /usr/local/lib/gmsh.py ${SP_DIR}/ && \
    #mv /usr/local/lib/gmsh*.dist-info ${SP_DIR}/ && \
    # Verify GMSH installation
    #gmsh --version && \
    #python3 -c "import gmsh; print('GMSH Python module loaded successfully')" && \
    # Clean up
    #rm -rf /tmp/*

# Install system packages for headless operation
RUN apt-get update && apt-get install -y \
    xvfb \
    x11-utils \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN python3 -m pip install --no-cache-dir \
    cython \
    pkgconfig \
    mpi4py \
    pyvista \
    matplotlib \
    seaborn \
    jupyter \
    jupyterlab \
    ipywidgets \
    trame \
    trame-vuetify \
    trame-vtk

# Install HDF5 with MPI support
#RUN export HDF5_MPI=ON && \
 #   export HDF5_PKGCONFIG_NAME="hdf5" && \
  #  python3 -m pip install --no-cache-dir --no-binary=h5py --no-build-isolation h5py -vv && \
   # python3 -m pip install --no-cache-dir meshio
    
RUN cd /tmp && curl -sS https://starship.rs/install.sh > install_starship.sh  &&  sh install_starship.sh --yes
RUN echo 'eval "$(starship init bash)"' >> ~/.bashrc

# Set working directory
WORKDIR /workspace

# Set environment variables for runtime
#ENV HDF5_MPI=ON
#ENV HDF5_PKGCONFIG_NAME=hdf5
ENV XDG_RUNTIME_DIR=/tmp
ENV MESA_GL_VERSION_OVERRIDE=3.3

# Expose common ports for Jupyter
EXPOSE 8888

# Default command
CMD ["/bin/bash"]