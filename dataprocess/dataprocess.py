import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import image
import numpy as np
import pandas as pd
import time
import os
import typing


#################  Classes #################
class Mesh:
    """Mesh object to store the grid and domain information
    of the simulation snapshots
    """

    def __init__(
        self, nx: int, ny: int, Lx: float, Ly: float, X_0: float, Y_0: float, r: float
    ):
        self.nx = nx
        self.ny = ny
        self.Lx = Lx
        self.Ly = Ly
        self.X_0 = X_0
        self.Y_0 = Y_0
        self.r = r
        self.mask = self.mask_cyl()

    def mask_cyl(self) -> np.ndarray:
        """Create a boolean mask for points inside the cylinder

        Returns:
            _type_: np.array
        """
        # Create an array of indx representing points in the grid
        indx = np.arange(0, self.nx * self.ny)
        # Calculate the x, y coordinates of each point in the grid
        x = (indx % self.nx) * (
            self.Lx / (self.nx - 1)
        )  # nx and ny have to be odd to map correctly
        y = (indx // self.nx) * (self.Ly / (self.ny - 1))
        # Calculate the Euclidean distance from the center of the cylinder for each point
        distances = np.sqrt((x - self.X_0) ** 2 + (y - self.Y_0) ** 2)
        # Create a boolean mask for points outside the cylinder
        mask = np.array(distances <= self.r)
        return mask


#################  General functions #################


def make_dir(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory created: {directory_path}")


def save_X_csv(data_path: str, data_matrix: np.ndarray):
    df = pd.DataFrame(data_matrix)
    file_path = data_path
    df.to_csv(file_path, index=False)


def read_X_csv(data_path: str) -> np.ndarray:
    print(f"Reading data from: {data_path}")
    df = pd.read_csv(data_path)
    return df.values.copy()


#################  Cylinder erase and reconstruction #################


def erase_cyl(u_data: np.ndarray, mesh: Mesh):
    """Erases the cylinder from the data snapshot

    Args:
        u_data (np.ndarray): velocity data matrix
        mesh (Mesh): mesh object

    Returns:
        np.ndarray: velocity data matrix without cylinder points
    """
    u_data_filtered = u_data[~mesh.mask]
    return u_data_filtered


def reconstruct_u_data(u_filt: np.ndarray, mesh: Mesh):
    """Reconstructs the velocity data matrix from snapshot
        with cylinder points erased

    Args:
        u_filt (np.ndarray): snapshot with cylinder points erased
        mesh (Mesh): mesh object

    Returns:
        np.ndarray: velocity data matrix with cylinder points
        filled with zeros
    """
    indx = mesh.nx * mesh.ny
    u_list = np.zeros(indx)
    for i in range(indx):
        if ~mesh.mask[i]:
            u_list[i] = u_filt[0]
            u_filt = u_filt[1:]
    return np.array(u_list)


def assign_zero_to_cyl(u_data: np.ndarray, mesh: Mesh):
    mask = mesh.mask
    u_data[mask] = 0
    return u_data


#################  pre-process #################
def u_module(U_x: np.ndarray, U_y: np.ndarray):
    # instanteous velocity module
    return np.sqrt(np.square(U_x) + np.square(U_y))


def subs_mean(X: np.ndarray):
    # subtract the temporal mean of the data set
    X_mean = np.mean(X, axis=1, keepdims=True)
    # Subtract X_mean from X using vectorized operation
    X -= X_mean
    return X, X_mean


#################  Post-process #################


def save_modes(X_modes: np.ndarray, outfile_dir: str, modes: int, mesh: Mesh):
    """Save the spatial modes to a csv file and png for visualization

    Args:
        X_modes (np.ndarray): transformed data matrix
        outfile_dir (string): directory to save the modes
        modes (int): number of modes to save
        mesh (Mesh): mesh object
    """

    make_dir(outfile_dir)

    for i in range(modes):
        U_i = X_modes[:, i]
        U_i = reconstruct_u_data(U_i, mesh)
        # Normalize modes for better visualization
        U_i = (U_i - U_i.min()) / (U_i.max() - U_i.min())
        U_i = U_i.reshape((mesh.ny, mesh.nx))
        # Save modes to csv
        save_X_csv(outfile_dir + f"mode_{i+1}.csv", U_i)

        colors = [
            (0, "cyan"),
            (0.40, "blue"),
            (0.5, "black"),
            (0.60, "red"),
            (1, "yellow"),
        ]
        custom_cmap = LinearSegmentedColormap.from_list("custom_colormap", colors)
        image.imsave(outfile_dir + f"mode_{i+1}.png", U_i, cmap=custom_cmap)


def plot_save_reconst(err_rec: np.ndarray, r_max: int, r_step: int, DRmethod: str):
    """Plot and save the reconstruction error as a function
    of the number of modes used

    Args:
        err_rec (np.ndarray): vector containing the reconstruction error for 1 to r_max modes
        r_max (int): max number of modes to keep
        r_step (int): mode number increment for calculating the reconstruction error
    """
    plt.figure()
    plt.grid(True, which="both")
    plt.plot(range(1, r_max + 1, r_step), err_rec, linestyle="--", marker="o")
    plt.xlabel("Modes used for reconstruction")
    plt.ylabel("Relative reconstruction error")
    plt.title(DRmethod + " reconstruction error")
    plt.semilogy()
    plt.tight_layout()
    plt.savefig(f"{DRmethod}_rec_error.png", dpi=200)
    return
