# FEniCS EDP - Simulaciones de Ecuaciones Diferenciales Parciales

Este repositorio contiene una colección completa de simulaciones numéricas de Ecuaciones Diferenciales Parciales (EDP) implementadas con FEniCS/DOLFINx. El proyecto incluye simulaciones de ondas, difusión, ecuaciones del calor, interferencia, difracción y más, con capacidades de visualización avanzada y ejecución en paralelo.

## 📁 Estructura del Proyecto

```
FEniCS_EDP/
├── scripts/                          # Scripts principales de simulación
│   ├── wave_eq.py                   # Ecuación de onda 1D
│   ├── gaussianwave2d.py            # Ecuación de onda 2D con fuente gaussiana
│   ├── interference.py              # Interferencia de ondas 2D
│   ├── diffraction.py               # Difracción de ondas por obstáculo
│   ├── heat.py                      # Ecuación del calor
│   ├── diffusion.py                 # Proceso de difusión
│   ├── poisson.py                   # Ecuación de Poisson
│   ├── biharmonic.py                # Ecuación biharmonica
│   ├── deflection.py                # Deflexión de membrana
│   ├── interpolation_parallel.py    # Interpolación paralela con MPI
│   ├── wave_eq_parallel/           # Versiones paralelas de ecuación de onda
│   │   └── metrics.sh              # Script de métricas de rendimiento
│   └── interference_parallel/      # Versiones paralelas de interferencia
│       └── interference_metrics.sh # Script de métricas de interferencia
├── figures/                         # Imágenes y animaciones generadas
│   ├── wave/                       # Visualizaciones de ondas
│   ├── heat/                       # Visualizaciones de ecuación del calor
│   ├── diffusion/                  # Visualizaciones de difusión
│   ├── poisson/                    # Visualizaciones de Poisson
│   ├── biharmonic/                 # Visualizaciones biharmonicas
│   ├── deflection/                 # Visualizaciones de deflexión
│   └── interpolation/              # Visualizaciones de interpolación
├── post_data/                      # Datos de post-procesamiento (XDMF/HDF5)
│   ├── wave/                       # Datos de simulaciones de ondas
│   ├── heat/                       # Datos de ecuación del calor
│   ├── diffusion/                  # Datos de difusión
│   ├── poisson/                    # Datos de Poisson
│   ├── biharmonic/                 # Datos biharmonicos
│   ├── deflection/                 # Datos de deflexión
│   └── interpolation/              # Datos de interpolación
├── run_parallel.sh                 # Script de ejecución paralela
├── Dockerfile                      # Configuración Docker (RECOMENDADO)
├── fenics.def                      # Definición Apptainer/Singularity (RECOMENDADO)
└── README.md                       # Este archivo
```

## 🚀 Instalación y Configuración

> **⚠️ IMPORTANTE**: Para garantizar compatibilidad completa y evitar problemas de dependencias, se **recomienda encarecidamente** utilizar las opciones de contenedores (Docker o Apptainer) incluidas en este repositorio.

### 🐳 Opción 1: Docker (RECOMENDADO)

El repositorio incluye un `Dockerfile` preconfigurado con todas las dependencias necesarias.

#### Prerrequisitos
- Docker instalado en su sistema
- Al menos 4GB de RAM disponibles
- 5GB de espacio libre en disco

#### Instrucciones de uso:

```bash
# 1. Clonar el repositorio
git clone https://github.com/JJMayorgaB/FEniCS_EDP
cd FEniCS_EDP

# 2. Construir la imagen Docker (solo necesario la primera vez)
docker build -t fenics-edp .

# 3. Ejecutar el contenedor con montaje del directorio actual
docker run -it --rm \
    -v $(pwd):/workspace \
    -e DISPLAY=:99 \
    --name fenics-container \
    fenics-edp

# 4. Una vez dentro del contenedor, configurar el display virtual
Xvfb :99 -screen 0 1024x768x24 -ac +extension GLX +render -noreset &
export DISPLAY=:99

# 5. Ejecutar cualquier script
python3 scripts/wave_eq.py
./run_parallel.sh
```

### 📦 Opción 2: Apptainer/Singularity (RECOMENDADO PARA HPC)

El repositorio incluye un archivo `fenics.def` para crear contenedores Apptainer, ideal para clusters de HPC.

#### Prerrequisitos
- Apptainer (antes Singularity) instalado
- Privilegios de construcción (sudo) o acceso a un builder remoto

#### Instrucciones de uso:

```bash
# 1. Clonar el repositorio
git clone https://github.com/JJMayorgaB/FEniCS_EDP
cd FEniCS_EDP

# 2. Construir el contenedor SIF (solo necesario la primera vez)
sudo apptainer build fenics.sif fenics.def

# Alternativamente, si no tiene privilegios sudo:
apptainer build --remote fenics.sif fenics.def

# 3. Ejecutar el contenedor
apptainer shell fenics.sif

# 4. Una vez dentro del contenedor, el display ya está configurado
# Ejecutar directamente los scripts
python3 scripts/wave_eq.py
./run_parallel.sh
```

## 🔧 Configuración del Entorno

### Variables de Entorno (ya configuradas en contenedores)

Los contenedores incluyen todas las variables de entorno preconfiguradas. No es necesario configurar nada manualmente.

### Verificación de la Instalación

Ejecute estos comandos dentro del contenedor para verificar que todo funciona correctamente:

```bash
# Verificar FEniCS
python3 -c "import dolfinx; print('DOLFINx version:', dolfinx.__version__)"
# Verificar MPI
mpirun --version
# Verificar PyVista
python3 -c "import pyvista; print('PyVista version:', pyvista.__version__)"
# Verificar visualización headless
python3 -c "import pyvista; pyvista.OFF_SCREEN=True; sphere = pyvista.Sphere(); print('PyVista headless: OK')"
```

## 📋 Scripts de Simulación

> **📝 Nota sobre Autoría**: 
> - Los scripts de ondas (`wave_eq.py`, `gaussianwave2d.py`, `interference.py`, `diffraction.py`) son **implementaciones propias** desarrolladas específicamente para este proyecto.
> - Los demás scripts están basados y adaptados del excelente [FEniCSx Tutorial](https://jsdokken.com/dolfinx-tutorial/index.html) de Jørgen S. Dokken.

### 1. Ecuación de Onda 1D (`wave_eq.py`) 🔬 **IMPLEMENTACIÓN PROPIA**

Simula la propagación de ondas unidimensionales con fuente externa.

**Ecuación:** `∂²u/∂t² - v²∂²u/∂x² = A·sin(kx - ωt)`

```bash
# Ejecución básica
python3 scripts/wave_eq.py

# Con animación deshabilitada (más rápido)
# Modificar create_animation=False en el script
```

**Archivos generados:**
- `figures/wave/wave_propagation.gif` - Animación de la propagación
- `post_data/wave/wave_solution.xdmf` - Datos para ParaView

### 2. Onda 2D Gaussiana (`gaussianwave2d.py`) 🔬 **IMPLEMENTACIÓN PROPIA**

Simula ondas bidimensionales con fuente puntual gaussiana.

**Ecuación:** `∂²u/∂t² - v²∇²u = A·exp(-r²/2σ²)·cos(ωt)`

```bash
python3 scripts/gaussianwave2d.py
```

**Parámetros configurables:**
- Velocidad de onda: `v = 1.5 m/s`
- Amplitud: `A = 10.0`
- Frecuencia: `ω = 8.0 rad/s`

**Archivos generados:**
- `figures/wave/wave_2d_propagation.gif`
- `post_data/wave/wave_2d_solution.xdmf`

### 3. Interferencia de Ondas (`interference.py`) 🔬 **IMPLEMENTACIÓN PROPIA**

Simula la interferencia entre dos fuentes gaussianas con diferentes frecuencias.

```bash
python3 scripts/interference.py
```

**Características:**
- Dos fuentes con frecuencias ω₁ = 6.0 y ω₂ = 8.0 rad/s
- Dominio rectangular 2.0 × 1.0 m
- Observación de patrones de batimiento

**Archivos generados:**
- `figures/wave/wave_interference_2d.gif`
- `post_data/wave/wave_interference_2d.xdmf`

### 4. Difracción de Ondas (`diffraction.py`) 🔬 **IMPLEMENTACIÓN PROPIA**

Simula la difracción de ondas por un obstáculo cuadrado.

```bash
python3 scripts/diffraction.py
```

**Características:**
- Obstáculo cuadrado en región [0.4, 0.7] × [0.35, 0.65]
- Fuente en el borde izquierdo
- Visualización de sombra acústica y efectos de difracción

**Archivos generados:**
- `figures/wave/wave_diffraction_2d.gif`
- `post_data/wave/wave_diffraction_2d.xdmf`

### 5. Ecuación del Calor (`heat.py`) 📖 *Basado en FEniCSx Tutorial*

Resuelve la ecuación del calor con solución analítica conocida.

**Ecuación:** `∂u/∂t - ∇²u = f`

```bash
python3 scripts/heat.py
```

**Archivos generados:**
- `figures/heat/heat_solution.gif`
- `post_data/heat/heat_solution.xdmf`

### 6. Proceso de Difusión (`diffusion.py`) 📖 *Basado en FEniCSx Tutorial*

Simula un proceso de difusión con condición inicial gaussiana.

```bash
python3 scripts/diffusion.py
```

**Archivos generados:**
- `figures/diffusion/u_time.gif`
- `post_data/diffusion/diffusion.xdmf`

### 7. Ecuación de Poisson (`poisson.py`) 📖 *Basado en FEniCSx Tutorial*

Resuelve la ecuación de Poisson con condiciones de frontera Dirichlet.

**Ecuación:** `-∇²u = f` con `u = 0` en ∂Ω

```bash
python3 scripts/poisson.py
```

**Archivos generados:**
- `figures/poisson/uh_poisson.png`
- `post_data/poisson/poisson.xdmf`

### 8. Ecuación Biharmonica (`biharmonic.py`) 📖 *Basado en FEniCSx Tutorial*

Resuelve la ecuación biharmonica usando formulación discontinua de Galerkin.

**Ecuación:** `∇⁴u = f`

```bash
python3 scripts/biharmonic.py
```

### 9. Deflexión de Membrana (`deflection.py`) 📖 *Basado en FEniCSx Tutorial*

Simula la deflexión de una membrana circular bajo carga distribuida.

```bash
python3 scripts/deflection.py
```

**Requiere:** GMSH para generación de malla

### 10. Interpolación Paralela (`interpolation_parallel.py`) 📖 *Basado en FEniCSx Tutorial*

Demuestra interpolación en espacios de elementos finitos usando MPI.

```bash
# Ejecución con diferentes números de procesos
mpirun -np 1 python3 scripts/interpolation_parallel.py
mpirun -np 4 python3 scripts/interpolation_parallel.py
mpirun -np 8 python3 scripts/interpolation_parallel.py
```

## ⚡ Ejecución en Paralelo

### Script de Ejecución Automática

```bash
# Ejecutar script paralelo con diferentes configuraciones
bash run_parallel.sh
```

### Métricas de Rendimiento

#### Para Ecuación de Onda:
```bash
cd scripts/wave_eq_parallel/
bash metrics.sh
```

#### Para Interferencia:
```bash
cd scripts/interference_parallel/
bash interference_metrics.sh
```

Estos scripts automáticamente:
- Ejecutan las simulaciones con 1, 2, 4, 6, 8, 10, 12, 14, 16, 17 núcleos
- Calculan speedup y eficiencia paralela
- Generan reportes de rendimiento
- Identifican efectos de oversubscription

**Archivos generados:**
- `metrics_results/wave_performance_YYYYMMDD_HHMMSS.txt`
- `interference_results/interference_performance_YYYYMMDD_HHMMSS.txt`

## 📊 Visualización y Post-procesamiento

### Formatos de Salida

1. **XDMF/HDF5** (.xdmf + .h5): Para ParaView/VisIt
2. **GIF** (.gif): Animaciones para documentación
3. **PNG** (.png): Imágenes estáticas
4. **TXT** (.txt): Datos de métricas de rendimiento

### Visualización con ParaView

```bash
# Abrir archivos XDMF en ParaView
paraview post_data/wave/wave_solution.xdmf
paraview post_data/wave/wave_interference_2d.xdmf
```

### Visualización Programática

Los scripts incluyen visualización automática con PyVista:
- Renderizado headless (sin display)
- Generación automática de GIFs
- Escalas de color optimizadas
- Vistas isométricas configuradas

## 🔧 Configuración y Optimización

### Variables de Entorno

```bash
# Para renderizado headless
export PYVISTA_OFF_SCREEN=true
export DISPLAY=:99

# Para MPI
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

# Para Mesa (software rendering)
export MESA_GL_VERSION_OVERRIDE=3.3
export LIBGL_ALWAYS_SOFTWARE=1
```

### Parámetros de Simulación

Cada script incluye parámetros configurables al inicio:

```python
# Ejemplo de wave_eq.py
v = 2.0          # Velocidad de onda
A = 1.0          # Amplitud
num_steps = 1000 # Pasos temporales
nx = 100         # Resolución espacial
```

### Optimización de Rendimiento

1. **Deshabilitar visualización** para mayor velocidad:
   ```python
   solve_function(create_visualization=False)
   ```

2. **Ajustar resolución** según recursos disponibles
3. **Usar más núcleos** para problemas grandes
4. **Monitorear número de Courant** para estabilidad

## 🐛 Solución de Problemas

### Problemas Comunes

1. **Error de construcción Apptainer**:
   ```bash
   # Verificar versión de Apptainer
   apptainer version
   
   # Limpiar caché si hay problemas
   apptainer cache clean
   ```

2. **Error de display en contenedores**:
   ```bash
   # Dentro del contenedor
   export DISPLAY=:99
   Xvfb :99 -screen 0 1024x768x24 -ac +extension GLX +render -noreset &
   sleep 2
   ```

3. **Error de MPI**:
   ```bash
   export OMPI_ALLOW_RUN_AS_ROOT=1
   ```

4. **Error de permisos en scripts**:
   ```bash
   chmod +x *.sh
   chmod +x scripts/*/*.sh
   ```

### Verificación de Contenedores

**Para Docker:**
```bash
# Verificar que la imagen se construyó correctamente
docker images | grep fenics-edp

# Verificar que el contenedor puede ejecutarse
docker run --rm fenics-edp python3 -c "import dolfinx; print('OK')"
```

**Para Apptainer:**
```bash
# Verificar que el archivo SIF existe
ls -la fenics.sif

# Verificar que el contenedor funciona
apptainer exec fenics.sif python3 -c "import dolfinx; print('OK')"
```

## 📚 Referencias y Documentación

### Recursos Principales
- **[FEniCSx Tutorial](https://jsdokken.com/dolfinx-tutorial/index.html)** - Tutorial principal de Jørgen S. Dokken (fuente de varios scripts)
- **Scripts de Ondas**: Implementaciones propias para este proyecto académico

### Tecnologías de Contenedores
- [Docker Documentation](https://docs.docker.com/)
- [Apptainer Documentation](https://apptainer.org/docs/)
- [FEniCS Container Guide](https://github.com/FEniCS/dolfinx)

### Documentación Oficial
- [FEniCS Project](https://fenicsproject.org/)
- [DOLFINx Documentation](https://docs.fenicsproject.org/)
- [PyVista Documentation](https://docs.pyvista.org/)

### Ecuaciones Implementadas

1. **Ecuación de Onda**: Propagación de ondas acústicas/electromagnéticas
2. **Ecuación del Calor**: Transferencia de calor por conducción
3. **Ecuación de Poisson**: Problemas de potencial electrostático
4. **Ecuación Biharmonica**: Problemas de placas y vigas
5. **Difusión**: Procesos de transporte molecular

### Métodos Numéricos

- **Espacial**: Método de Elementos Finitos (FEM)
- **Temporal**: Diferencias Finitas (esquemas implícitos)
- **Solvers**: PETSc con precondicionadores
- **Parallelización**: Domain decomposition con MPI

## 🧮 Metodología Híbrida FEM-FDM en Scripts de Ondas

### Enfoque Innovador: Combinación FEM + FDM

Los scripts de ondas (`wave_eq.py`, `gaussianwave2d.py`, `interference.py`, `diffraction.py`) implementan una **metodología híbrida** que combina las fortalezas de ambos métodos numéricos:

#### 🔸 **Discretización Espacial: Elementos Finitos (FEM)**

```python
# Espacio de elementos finitos P1 (Lagrange lineales)
V = fem.functionspace(domain, ("Lagrange", 1))

# Forma bilineal para el laplaciano espacial
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
a_spatial = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
```

**Ventajas de FEM para la parte espacial:**
- **Flexibilidad geométrica**: Manejo natural de dominios complejos y obstáculos
- **Condiciones de frontera**: Implementación directa de condiciones Dirichlet/Neumann
- **Precisión local**: Refinamiento adaptativo posible
- **Conservación**: Propiedades de conservación inherentes

#### 🔸 **Discretización Temporal: Diferencias Finitas (FDM)**

```python
# Esquema de Newmark implícito para ecuaciones de segundo orden
# (u^{n+1} - 2u^n + u^{n-1})/dt² = v²∇²u^{n+1} + f^{n+1}

# Reorganizando: u^{n+1} - dt²v²∇²u^{n+1} = 2u^n - u^{n-1} + dt²f^{n+1}
a = (u * v_test * ufl.dx + 
     (dt**2 * v**2) * ufl.dot(ufl.grad(u), ufl.grad(v_test)) * ufl.dx)

L = ((2.0 * u_n1 - u_n2) * v_test * ufl.dx + 
     dt**2 * f_current * v_test * ufl.dx)
```

**Ventajas de FDM para la parte temporal:**
- **Estabilidad**: Esquemas implícitos incondicionalmente estables
- **Eficiencia**: Una sola matriz a ensamblar (independiente del tiempo)
- **Simplicidad**: Implementación directa para ecuaciones hiperbólicas
- **Control de estabilidad**: Fácil monitoreo del número de Courant

#### 🔸 **Sistema Híbrido Resultante**

La ecuación de onda se discretiza como:

```
[M + dt²v²K] u^{n+1} = M(2u^n - u^{n-1}) + dt²F^{n+1}
```

Donde:
- **M**: Matriz de masa (FEM)
- **K**: Matriz de rigidez/laplaciano (FEM)  
- **F**: Vector de fuerzas (evaluado en cada paso temporal)
- **dt**: Paso temporal (FDM)

### Estabilidad y Convergencia

#### Condición CFL Generalizada
```python
# Cálculo automático del número de Courant
hx = domain_width / nx
hy = domain_height / ny
h = min(hx, hy)  # Tamaño característico de elemento
courant_number = v * dt / h

if courant_number > 1.0:
    print("⚠️ ADVERTENCIA: C > 1, posible inestabilidad")
```

**Ventajas del esquema híbrido:**
- **CFL menos restrictiva**: El esquema implícito relaja la condición de estabilidad
- **Preservación de energía**: El método conserva aproximadamente la energía total
- **Dispersión mínima**: Errores de dispersión controlados por FEM espacial

## 🚀 Paralelización con MPI

### Estrategia de Domain Decomposition

La paralelización se implementa mediante **descomposición de dominio automática** de DOLFINx:

#### 🔸 **Particionamiento Automático**

```python
# DOLFINx automáticamente distribuye la malla entre procesos MPI
domain = mesh.create_rectangle(
    MPI.COMM_WORLD,  # Comunicador MPI
    [np.array([0.0, 0.0]), np.array([2.0, 1.0])],
    [nx, ny], 
    cell_type=mesh.CellType.triangle
)
```

**Proceso de paralelización:**
1. **Particionamiento**: La malla se divide en subdominios
2. **Distribución**: Cada proceso MPI recibe una porción
3. **Comunicación**: Intercambio de datos en fronteras entre subdominios
4. **Sincronización**: Operaciones colectivas para ensamblaje y solución

#### 🔸 **Operaciones Paralelas Clave**

```python
# Ensamblaje distribuido de matrices
A_matrix = assemble_matrix(bilinear_form, bcs=[bc])
A_matrix.assemble()  # Comunicación MPI automática

# Ensamblaje distribuido de vectores
assemble_vector(b, linear_form)
b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)

# Solución paralela del sistema lineal
solver.solve(b, u_n.x.petsc_vec)
u_n.x.scatter_forward()  # Actualización de valores fantasma
```

### Análisis de Rendimiento Paralelo

#### 🔸 **Métricas Implementadas**

Los scripts `metrics.sh` calculan automáticamente:

```bash
# Speedup: S(p) = T(1) / T(p)
speedup = tiempo_secuencial / tiempo_paralelo

# Eficiencia: E(p) = S(p) / p × 100%
efficiency = (speedup / num_cores) * 100

# Escalabilidad fuerte: Problema fijo, más cores
# Escalabilidad débil: Problema crece con cores
```

#### 🔸 **Factores que Afectan la Escalabilidad**

**✅ Ventajas para paralelización:**
- **Computación dominante**: Ensamblaje y solución de sistemas lineales
- **Localidad espacial**: Operaciones principalmente locales
- **Comunicación estructurada**: Patrones predecibles de intercambio

**⚠️ Limitaciones típicas:**
- **Overhead de comunicación**: Intercambio en fronteras de subdominios
- **Load balancing**: Distribución desigual de trabajo
- **I/O secuencial**: Escritura de archivos de visualización
- **Oversubscription**: Degradación al exceder cores físicos

#### 🔸 **Optimizaciones Implementadas**

```python
# 1. Reducción de comunicación I/O
if n % 20 == 0:  # Escribir solo cada 20 pasos
    xdmf.write_function(u_n, t)

# 2. Visualización condicional
if create_visualization and n % 15 == 0:  # Frames menos frecuentes
    plotter.write_frame()

# 3. Cálculo de energía espaciado
if n % 25 == 0:  # Monitoreo menos frecuente
    total_energy = kinetic_energy + potential_energy
```

### Configuración para HPC

### Técnicas Avanzadas de Paralelización

#### 🔸 **Precondicionadores Paralelos**

```python
# Configuración optimizada para parallel performance
solver = PETSc.KSP().create(domain.comm)
solver.setType(PETSc.KSP.Type.CG)  # Gradiente conjugado
solver.getPC().setType(PETSc.PC.Type.HYPRE)  # Precondicionador AMG paralelo
solver.setTolerances(rtol=1e-8)
```

#### 🔸 **Balanceado de Carga Dinámico**

```python
# DOLFINx maneja automáticamente:
# - Particionamiento balanceado usando METIS/ParMETIS
# - Migración de datos entre procesos si necesario
# - Optimización de comunicación fantasma
```

**Resultado:** Los scripts de ondas demuestran que la combinación FEM-FDM con paralelización MPI permite simular problemas complejos de propagación de ondas con **alta eficiencia computacional** y **excelente escalabilidad** hasta ~8-16 cores para problemas de tamaño medio.

## 📄 Licencia

Proyecto Académico de Libre Uso.

## 👥 Autores

- **Scripts de Ondas**: Implementación propia para simulaciones académicas de EDP
- **Otros Scripts**: Adaptados del [FEniCSx Tutorial](https://jsdokken.com/dolfinx-tutorial/index.html) de Jørgen S. Dokken
- **Configuración de Contenedores**: Desarrollada específicamente para este proyecto
- Basado en FEniCS/DOLFINx framework
- Optimizado para ejecución en paralelo con MPI
- Configurado para contenedores Docker y Apptainer

### Agradecimientos
- **Jørgen S. Dokken** por el excelente [FEniCSx Tutorial](https://jsdokken.com/dolfinx-tutorial/index.html)
- **FEniCS Project** por el framework de elementos finitos
- **DOLFINx Development Team** por la biblioteca moderna de elementos finitos

---

**Entorno recomendado**: Docker o Apptainer con las configuraciones incluidas en este repositorio.
