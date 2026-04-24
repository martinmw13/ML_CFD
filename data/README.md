### In this directory should be the CFD data. These are pre-processed into CSV files.

The snapshot CSV used by the `mlcfd` package follows the same convention as the thesis notebooks. By default, files are expected at:

- `{input_dir}/modVcropRe{re}.csv`

where `re` is the **Reynolds number** (see `DataConfig` in the code). The filename template is configurable via `snapshot_filename_template` in YAML (default `modVcropRe{re}.csv`).

### The CSV file layout

- Each **column** is a **time snapshot** (one column = one time step’s field).
- Each **row** is a **spatial degree of freedom** on the **cropped** grid (velocity samples indexed over the mesh after cropping).
- The workshop notebooks often used on the order of **2000** columns (2000 snapshots) for a given case; your file may differ.

**Data structure (summary):**


## Original CFD Simulation Parameters

### Mesh
- nx = 257       # X-direction nodes
- ny = 160       # Y-direction nodes

### Domain
- Lx = 20.       # Lx (Size of the box in x-direction)
- Ly = 12.       # Ly (Size of the box in y-direction)

### Cylinder Coordinates
- X_0 = 5        # X coordinate of the center
- Y_0 = 6        # Y coordinate of the center
- r = 0.5        # Cylinder radius

## Data Pre-processing

The snapshots were cropped to reduce data size. The first 30 nx and the first and last 20 ny were removed (u_data[20:-20, 30:]). Thus, the CFD data structure used throughout the code is as follows:

### CFD Data (Cropped) Structure

#### Mesh
- nx = 227       # X-direction nodes
- ny = 120       # Y-direction nodes



#### Domain
- Lx = 17.665369        # Lx (Size of the box in the x-direction)
- Ly = 9.               # Ly (Size of the box in the y-direction)

#### Cylinder Coordinates
- X_0 = 2.66537         # X coordinate of the center
- Y_0 = 4.5             # Y coordinate of the center
- r = 0.5               # Cylinder radius

### Data arrangement
- Each row of the matrix X corresponds to velocity components at spatial locations.
- Each column of the matrix X corresponds to time steps.
- The values in the matrix represent flow velocities at each grid point at a given time.

$$X = \begin{bmatrix}
u_{1,1} & u_{1,2} & \dots & u_{1,n} \\
u_{2,1} & u_{2,2} & \dots & u_{2,n} \\
\vdots & \vdots & \ddots & \vdots \\
\vdots & \vdots &  &  u_{n,m} \\
\end{bmatrix}$$

## Notes for the `mlcfd` pipeline

- Place CSVs under the directory you pass as `data.input_dir` in the run YAML (commonly this `data/` folder).
- Do not commit large CSVs to git: they are listed in the repository [`.gitignore`](../.gitignore).

