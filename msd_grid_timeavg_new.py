import MDAnalysis as mda
import os
import numpy as np
import argparse
from scipy.stats import skew
from scipy.optimize import curve_fit

# Setting up command-line arguments for specifying the input files and atom selection.
parser = argparse.ArgumentParser(description='Compute displacement from MD simulation data.')
parser.add_argument('-t', '--tpr', required=True, help='Path to the input .tpr file.')
parser.add_argument('-s', '--trr', required=True, help='Path to the input .trr file.')
parser.add_argument('-x', '--select', default="all", help='Atom selection string. Default is "all".')
args = parser.parse_args()

# Extracting the base filename from the TRR file for later use in output filenames.
trr_filename = os.path.basename(args.trr).split('.')[0]

# Loading the MD simulation data using MDAnalysis.
u = mda.Universe(args.tpr, args.trr)
lipids = u.select_atoms(args.select)                    # Selecting atoms based on input or default 'all'.

# Setting up the simulation box and grid parameters.
L = u.dimensions[0] + 9                                 # Adjusting box length for margin.
grid_size = L / 10                                      # Defining size of each grid square.
grid_index = {i: [] for i in range(10*10)}              # Initializing a dictionary to hold grid indices.
n_grids = 10*10

# Mapping each lipid to a corresponding grid based on initial position.
initial_pos = lipids.positions
ri = lipids.positions[:, :2] + np.array([4, 4])         # Adjusting positions.
for i, (x, y) in enumerate(ri): 
    row_index = x // grid_size
    col_index = y // grid_size
    cell_key = int(row_index*10 + col_index)
    grid_index[cell_key].append(i)

# Initializing an array to store the trajectory data.
lipid_trajectory = []
for ts in u.trajectory:
    lipid_trajectory.append(lipids.positions)
lipid_trajectory = np.array(lipid_trajectory)
lipid_trajectory_2d = lipid_trajectory[:, :, :2]

# Reshaping trajectory data for easier displacement calculation.
xcoor = lipid_trajectory_2d[:, :, :1]
ycoor = lipid_trajectory_2d[:, :, 1:2]
xcoor = xcoor.reshape(xcoor.shape[0], -1)
ycoor = ycoor.reshape(ycoor.shape[0], -1)
xcoor = xcoor.T
ycoor = ycoor.T
timespan = len(xcoor[0, :])

# Function for calculating displacement with periodic boundary conditions.
def distance(g, a, L):
    dr = g - a
    corrx = np.round(dr / L)
    dr = dr - corrx * L
    return dr

# Function for calculating the mean squared displacement (MSD).
def msd(dispX,dispY):
    Ntracer=len(dispX[:,0])
    timespan=len(dispY[0,:])
    time=int(timespan/2)
    x=np.zeros((Ntracer,time))
    for dt in range(1,time+1):

        dpx = dispX - np.roll(dispX,dt,axis=1) 
        dpy = dispY - np.roll(dispY,dt,axis=1)
        dpx[:,:dt] = np.nan
        dpy[:,:dt] = np.nan
        x[:,dt-1] = np.nanmean(dpx**2 + dpy**2,axis=1) 

    xav = np.nanmean(x,axis=0)
    return xav
    
# Calculating displacements and accumulating them.
dxcoordiff = distance(xcoor[:, 1:], xcoor[:, :-1], L)
dycoordiff = distance(ycoor[:, 1:], ycoor[:, :-1], L)
dispX = np.cumsum(dxcoordiff,axis=1)
dispY = np.cumsum(dycoordiff,axis=1)

# Initializing a dictionary to store MSD results for each grid.
MSD_results = {i: [] for i in range(n_grids)}            # Store MSD for each grid
for i in range(n_grids):
    xav=msd(dispX[grid_index[i]],dispY[grid_index[i]])
    MSD_results[i] = xav

  
# Ensuring the output directory exists.
output_directory = "MSD_data"
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Saving MSD results to individual text files for each grid.
for grid_id, msd_values in MSD_results.items():
    output_filename = os.path.join(output_directory, f"{trr_filename}_grid_{grid_id}_MSD.txt")
    with open(output_filename, 'w') as file:
        file.write("TimeStep\tMSD\n")
        for timestep, msd in enumerate(msd_values, start=1):
            file.write(f"{timestep}\t{msd}\n")

# Notifying that the MSD calculations are complete.
print("MSD calculation and saving completed.")
