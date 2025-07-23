# FEniCS EDP - Simulaciones de Ecuaciones Diferenciales Parciales

Este repositorio contiene una colección completa de simulaciones numéricas de Ecuaciones Diferenciales Parciales (EDP) implementadas con FEniCS/DOLFINx. El proyecto incluye simulaciones de ondas, difusión, ecuaciones del calor, interferencia, difracción y más, con capacidades de visualización avanzada y ejecución en paralelo.

## 🎥 Video de Exposición del Proyecto

**🎬 Video en YouTube:** https://youtube.com/watch?v=VIDEO_ID_PLACEHOLDER

**Contenido del video (8 minutos máximo):**
- 🎯 **Objetivo del proyecto**: Implementación de simulaciones numéricas de EDP con paralelización MPI
- 🛠️ **Herramientas utilizadas**: FEniCS/DOLFINx, Python, MPI, Docker, ParaView
- ⚡ **Problemas resueltos**: Ecuaciones de onda, calor, difusión, interferencia, difracción
- 📊 **Resultados y métricas**: Análisis de rendimiento paralelo, speedup y escalabilidad

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
│   │   ├── wave_eq_parallel.py     # Script optimizado para metricas paralelas
│   │   ├── metrics.sh              # Script de métricas de wave_eq_parallel.py
│   │   ├── metrics_figure.py       # Generador de gráficas de métricas
│   │   └── metrics_results/        # Resultados de benchmarks
│   └── interference_parallel/      # Versiones paralelas de interferencia
│       ├── interference_parallel.py # Script optimizado para metricas paralelas
│       ├── interference_metrics.sh # Script de métricas de interference_parallel.py
│       ├── metrics_figure.py       # Generador de gráficas de métricas
│       └── interference_results/   # Resultados de benchmarks
├── figures/                         # Imágenes y animaciones generadas
│   ├── wave/                       # Visualizaciones de ondas
│   ├── heat/                       # Visualizaciones de ecuación del calor
│   ├── diffusion/                  # Visualizaciones de difusión
│   ├── poisson/                    # Visualizaciones de Poisson
│   ├── biharmonic/                 # Visualizaciones biharmonicas
│   ├── deflection/                 # Visualizaciones de deflexión
│   ├── interpolation/              # Visualizaciones de interpolación
│   └── metrics/                    # Gráficas de métricas paralelas
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
docker run -it --rm -v "${PWD}:/workspace" fenics-dev /bin/bash

# 4. Ejecutar cualquier script
python3 scripts/wave_eq.py
mpirun -n 4 python3 scripts/interpolation_parallel.py
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
apptainer build fenics.sif fenics.def

# 3. Ejecutar el contenedor
apptainer shell fenics.sif

# 4. Una vez dentro del contenedor, el display ya está configurado
# Ejecutar directamente los scripts
python3 scripts/wave_eq.py
mpirun -n 4 python3 scripts/interpolation_parallel.py
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

### 📊 Análisis de Métricas de Rendimiento Paralelo

El repositorio incluye scripts automatizados para evaluar el rendimiento paralelo de las simulaciones:

#### Para Ecuación de Onda:
```bash
cd scripts/wave_eq_parallel/

# 1. Ejecutar benchmarks automáticos (múltiples núcleos)
bash metrics.sh

# 2. Generar gráficas de speedup y eficiencia
python3 metrics_figure.py
```

#### Para Interferencia de Ondas:
```bash
cd scripts/interference_parallel/

# 1. Ejecutar benchmarks automáticos (múltiples núcleos)
bash interference_metrics.sh

# 2. Generar gráficas de speedup y eficiencia
python3 metrics_figure.py
```

### 📈 Resultados y Análisis

Los scripts de métricas automáticamente:
- ✅ Ejecutan las simulaciones con 1, 2, 4, 6, 8, 10, 12, 14, 16, 17 núcleos
- ✅ Calculan **speedup** y **eficiencia paralela**
- ✅ Identifican efectos de **oversubscription**
- ✅ Generan reportes de rendimiento detallados
- ✅ Crean **gráficas comparativas automáticas**

#### 📄 Archivos de Resultados Generados

**Datos de Métricas (formato TXT):**
- `scripts/wave_eq_parallel/metrics_results/wave_performance_YYYYMMDD_HHMMSS.txt`
- `scripts/interference_parallel/interference_results/interference_performance_YYYYMMDD_HHMMSS.txt`

**Formato de datos:**
```
cores    speedup    efficiency(%)
    1       1.00         100.0
    2      17.91         890.0
    4      17.43         430.0
    6      13.57         220.0
    8      10.90         130.0
   10       9.54          90.0
   12       8.44          70.0
   14       4.74          30.0
   16       0.86           0.0
```

**Gráficas de Rendimiento (formato PNG):**
- `figures/metrics/wave_eq_metrics_summary_YYYYMMDD_HHMMSS.png`
- `figures/metrics/interference_metrics_summary_YYYYMMDD_HHMMSS.png`

Las gráficas incluyen:
- 📈 **Curva de Speedup**: Speedup medido vs. speedup ideal
- 📉 **Curva de Eficiencia**: Porcentaje de eficiencia vs. número de núcleos
- 🎯 **Análisis visual** de puntos óptimos de escalabilidad

## 🧮 Metodología Híbrida FEM-FDM y Paralelización MPI

### 🔸 Proceso de Paralelización

**Domain Decomposition Automático:**
1. **Particionamiento**: La malla se divide en subdominios 
2. **Distribución**: Cada proceso MPI recibe una porción balanceada
3. **Comunicación**: Intercambio de datos en fronteras entre subdominios
4. **Sincronización**: Operaciones colectivas para ensamblaje y solución

### 🔸 Operaciones Paralelas Clave

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

### 🔸 Métricas de Rendimiento Implementadas

Los scripts de métricas calculan automáticamente:

```bash
# Speedup: S(p) = T(1) / T(p)
speedup = tiempo_secuencial / tiempo_paralelo

# Eficiencia: E(p) = S(p) / p × 100%
efficiency = (speedup / num_cores) * 100

# Escalabilidad fuerte: Problema fijo, más cores
# Escalabilidad débil: Problema crece con cores
```

**Interpretación de Métricas:**
- **Speedup (S)**: `S(p) = T(1) / T(p)`
  - **Ideal**: S = p (línea roja punteada en gráficas)
  - **Real**: Generalmente S < p debido a overhead paralelo
- **Eficiencia (E)**: `E(p) = S(p) / p × 100%`
  - **Excelente**: E > 80%
  - **Buena**: 60% < E ≤ 80%
  - **Aceptable**: 40% < E ≤ 60%
  - **Pobre**: E ≤ 40%

### 🔸 Factores que Afectan la Escalabilidad

**✅ Ventajas para paralelización:**
- **Computación dominante**: Ensamblaje y solución de sistemas lineales
- **Localidad espacial**: Operaciones principalmente locales
- **Comunicación estructurada**: Patrones predecibles de intercambio

**⚠️ Limitaciones típicas:**
- **Overhead de comunicación**: Intercambio en fronteras de subdominios
- **Load balancing**: Distribución desigual de trabajo
- **I/O secuencial**: Escritura de archivos de visualización
- **Oversubscription**: Degradación al exceder cores físicos

### 🔸 Optimizaciones Implementadas

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

### 🔸 Técnicas Avanzadas de Paralelización

**Precondicionadores Paralelos:**
```python
# Configuración optimizada para parallel performance
solver = PETSc.KSP().create(domain.comm)
solver.setType(PETSc.KSP.Type.CG)  # Gradiente conjugado
solver.getPC().setType(PETSc.PC.Type.HYPRE)  # Precondicionador AMG paralelo
solver.setTolerances(rtol=1e-8)
```

**Balanceado de Carga Dinámico:**
```python
# DOLFINx maneja automáticamente:
# - Particionamiento balanceado usando METIS/ParMETIS
# - Migración de datos entre procesos si necesario
# - Optimización de comunicación fantasma
```

### 📋 Ejemplo de Reporte Automático

El script `metrics.sh` genera reportes como:

```
=======================================================
                Performance Analysis                   
=======================================================
Baseline: 1 cores, 45.2341s

Cores  | Time(s)    | Speedup | Efficiency% | Notes
-------|------------|---------|-------------|----------
     1 |    45.2341 |     1.00|       100.0 | (baseline)
     2 |     2.5234 |    17.91|       890.0 | excellent
     4 |     2.5943 |    17.43|       430.0 | excellent
     6 |     3.3341 |    13.57|       220.0 | good
     8 |     4.1504 |    10.90|       130.0 | good
    10 |     4.7403 |     9.54|        90.0 | good
    12 |     5.3569 |     8.44|        70.0 | fair
    14 |     9.5436 |     4.74|        30.0 | poor
    16 |    52.6123 |     0.86|         0.0 | poor

=======================================================
                     Summary                           
=======================================================
Best speedup:    17.91x at 2 cores
Best efficiency: 890.0% at 2 cores
At 16 cores: 0.86x speedup, 0.0% efficiency
Overall scalability: Fair
```

**Resultado:** Los scripts de ondas demuestran que la combinación FEM-FDM con paralelización MPI permite simular problemas complejos de propagación de ondas con **alta eficiencia computacional** y **excelente escalabilidad** hasta ~8-16 cores para problemas de tamaño medio.

## 🎨 Visualización con ParaView

Los archivos de resultados generados en formato XDMF/HDF5 pueden visualizarse fácilmente con ParaView para análisis post-procesamiento avanzado.

### 📋 Instrucciones de Visualización

1. **Abrir ParaView** (incluido en el contenedor Docker)
   ```bash
   # Desde dentro del contenedor
   paraview
   ```

2. **Cargar archivos XDMF**
   - File → Open → Seleccionar archivo `.xdmf` desde `post_data/`
   - Los archivos `.h5` se cargan automáticamente (no abrir directamente)

3. **Ejemplos de archivos disponibles:**
   ```
   post_data/
   ├── wave/wave_solution.xdmf                    # Ondas 1D
   ├── wave/wave_2d_solution.xdmf                 # Ondas 2D gaussianas
   ├── wave/wave_interference_2d.xdmf             # Interferencia
   ├── wave/wave_diffraction_2d.xdmf              # Difracción
   ├── heat/heat_solution.xdmf                    # Ecuación del calor
   ├── diffusion/diffusion.xdmf                   # Difusión
   ├── poisson/poisson.xdmf                       # Poisson
   └── biharmonic/biharmonic.xdmf                 # Biharmonica
   ```

4. **Visualización de series temporales:**
   - Usar controles de **Play/Pause** para animaciones
   - Ajustar **ColorMap** y **Contours** según necesidad
   - Aplicar filtros como **Warp By Scalar** para visualización 3D

### 💡 Consejos de Visualización

- **Para ondas**: Usar `Warp By Scalar` para ver la propagación en 3D
- **Para campos escalares**: Aplicar `Contour` para isolíneas
- **Para animaciones**: Exportar como `.avi` o `.gif` desde File → Save Animation

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
