from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import dataprocess as dp


def read_crop(
    input_dir: str, filename: str, file_ext: str, t_1: int, t_n: int, mesh: dp.Mesh
):
    """Read velocity data from file and crop domain

    Args:
        input_dir (str): _description_
        filename (str): _description_
        file_ext (str): _description_
        t_1 (int): _description_
        t_n (int): _description_
        mesh (dp.Mesh): _description_

    Returns:
        np.ndarray: data matrix containing the velocity data
        with cropped domain.
    """
    nx, ny = mesh.nx, mesh.ny
    u_list = []
    for t in range(t_1, t_n):
        filename_t = filename + str(t) + file_ext
        with open(input_dir + filename_t, "rb") as f:
            u_data = np.fromfile(f, dtype=np.float64, count=nx * ny)
            u_data = dp.assign_zero_to_cyl(u_data, mesh)
            u_data = np.reshape(u_data, (ny, nx))
            u_data = u_data[20:-20, 30:]
            u_data = np.reshape(u_data, ((-1, 1)))
            u_list.append(u_data)
    return np.hstack(u_list)


# Mesh
nx = 257  # X-direction nodes
ny = 160  # Y-direction nodes
# Domain:
Lx = 20  # Lx (Size of the box in x-direction)
Ly = 12  # Ly (Size of the box in y-direction)
# Cylinder coordinates:
X_0 = 5  # X coordinate of the center
Y_0 = 6  # Y coordinate of the center
r = 0.5  # Cylinder radius
input_dir = "data/"
output_dir = "modV/"
filename_ux = ["ux-0", "uy-0", "uz-0"]
file_ext = ".bin"
t_1, t_n = 501, 1500

mesh = dp.Mesh(nx, ny, Lx, Ly, X_0, Y_0, r)

for re in range(50, 200, 5):
    input_dir = f"data{re}/"
    output_dir = f"modV/modV_crop_re{re}"
    U_x = read_crop(input_dir, filename_ux[0], file_ext, t_1, t_n, mesh)
    print(U_x.shape)
    U_y = read_crop(input_dir, filename_ux[1], file_ext, t_1, t_n, mesh)
    # # # X - velocity module
    X = dp.u_module(U_x, U_y)
    # #Save to csv the downsized
    # save_X_csv(output_dir, "modV_cut", X)
    n, m = X.shape
    print("Data matrix X is n by m:", n, "x", m, flush=True)
