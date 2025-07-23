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
    bc \
    htop && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

#Install Paraview
RUN apt-get update && \
    apt-get install -y paraview

# Install system packages for headless operation
RUN apt-get update && apt-get install -y \
    xvfb \
    x11-utils \
    libxcb-xinerama0 \
    libxkbcommon-x11-0 \
    libxcb-render0 \
    libxcb-shm0 \
    libgl1-mesa-dri \
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
    imageio \
    trame \
    trame-vuetify \
    trame-vtk
    
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