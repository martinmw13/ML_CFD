### In this directory should be the CFD data. These are pre-processed into csv files. 

### The CSV file stores data where each column represents a snapshot of dimensions nx x ny, collapsed into a column vector. It contains 2000 columns, which correspond to 2000 snapshots (2000 time-steps).


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

