FROM dolfinx/dolfinx:stable

# Add your GMSH and other dependencies
WORKDIR /tmp
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
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install gmsh (same as your second dockerfile)
ARG GMSH_VERSION=4_12_2
RUN git clone -b gmsh_${GMSH_VERSION} --single-branch --depth 1 https://gitlab.onelab.info/gmsh/gmsh.git && \
    cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DENABLE_BUILD_DYNAMIC=1 -B build-dir -S gmsh && \
    cmake --build build-dir && \
    cmake --install build-dir && \
    rm -rf /tmp/*

# Move gmsh python package
RUN export SP_DIR=$(python3 -c 'import sys, site; sys.stdout.write(site.getsitepackages()[0])') \
    && mv /usr/local/lib/gmsh.py ${SP_DIR}/ \
    && mv /usr/local/lib/gmsh*.dist-info ${SP_DIR}/

# Install Python packages for visualization and analysis
RUN python3 -m pip install --no-cache-dir \
    pyvista \
    meshio \
    matplotlib \
    seaborn \
    jupyter \
    jupyterlab \
    ipywidgets \
    trame \
    trame-vuetify \
    trame-vtk


WORKDIR /workspace

# Default command
CMD ["/bin/bash"]