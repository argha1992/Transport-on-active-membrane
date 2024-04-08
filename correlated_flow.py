"""
This script is designed for the analysis of lipid membrane dynamics from molecular dynamics (MD) simulations. 
It performs the following major tasks:

1. Parses input .tpr (topology) and .trr (trajectory) file paths from the command line arguments. 
   These files are essential for analyzing the dynamics of molecules within the simulation, 
   specifically focusing on lipid molecules.

2. Calculates displacement vectors for lipid molecules across specified time intervals, taking 
   into account the periodic boundary conditions of the simulation box. This is crucial for
   understanding how lipids move relative to each other over time.

3. Computes displacement-displacement correlations (DDC) to assess how the movement of one lipid
   molecule is correlated with the movement of its neighbors. This analysis provides insights 
   into the collective behavior of lipids within the membrane, which is important for understanding 
   membrane fluidity, stability, and interactions with other molecules.
   
4. On a fixed time interval $\tau$, the displacement vectors for the $i^{th}$ and the $j^{th}$ lipids, 
   $\hat{{x}}_i({r}_i,\tau)$ and $\hat{{x}}_j({r}_j,\tau)$ were calculated. The correlation function is 
   then given by,
    
   {g(r, \tau)_x = \frac{1}{N} \sum_{i=1}^{N} \sum_{j=1}^{N} \langle \hat{{x}}_i({r}_i,\tau) \cdot \hat{{x}}_j({r}_j,
                   \tau)\delta(r-|r_{ij}(t_0)|) \rangle}

   where N is the number of particles in the system, $r_{ij}(t_0)$ is the initial separation distance 
   between $i^{th}$ and the $j^{th}$ lipids.
 
5. Outputs the results, including radial distances, radial distribution functions, and displacement-displacement
   correlation functions. 
   These results are saved to a text file for further analysis and visualization.

Example command:
`python correlated_flow.py topology.tpr trajectory.trr

Dependencies:
- MDAnalysis: for loading and manipulating MD simulation data.
- NumPy: for numerical operations.
- SciPy: for statistical functions.
- argparse: for parsing command-line options.

"""

import os
import numpy as np
import MDAnalysis as mda
from scipy.spatial.distance import pdist, squareform
import argparse

# Parse command-line arguments for input files.
def parse_arguments():
    parser = argparse.ArgumentParser(description="Process .tpr and .trr files for correlation analysis.")
    parser.add_argument("tpr_file", help="Input .tpr file path.")
    parser.add_argument("trr_file", help="Input .trr file path.")
    args = parser.parse_args()
    return args.tpr_file, args.trr_file

# Parse input file paths.
tpr_file, trr_file = parse_arguments()

# Calculate displacement vectors, accounting for periodic boundary conditions.
def distance(xi, yi, xt, yt, L):
    dx = xt - xi
    dy = yt - yi
    corrx = np.round(dx / L)
    corry = np.round(dy / L)
    dx = dx - corrx * L
    dy = dy - corry * L
    dr = np.zeros((len(dx), 2))
    dr[:, 0] = dx
    dr[:, 1] = dy
    mag = np.linalg.norm(dr, axis=1)
    dr = dr / mag[:, np.newaxis]
    return dr

# Calculate displacement-displacement correlation for a given atom.
def corr_func(i, dist_nd, unit_disp, r, dr, rho):
    cr_disp_i = np.zeros(2)
    cr_i = 0
    for j in prange(dist_nd.shape[0]):              # Parallelized loop for efficiency
        if i == j:
            continue                                # Skip self-comparison
        dist = dist_nd[i, j]
        if r <= dist <= (r + dr):                   # Check if within specified distance range
            cr_disp_i += unit_disp[j]               # Sum displacements
            cr_i += 1                               # Count neighbors
    return cr_disp_i, cr_i

# Main analysis section.
u = mda.Universe(tpr_file, trr_file)                # Load MD simulation data.
lipids = u.select_atoms("name PO4")                 # Select lipid atoms based on name.
L = 300                                             # Box dimension
dt = 100                                            # Time interval
dr = 0.2                                            # Radial distance increment for correlation functions.
d = 0.2
R = np.arange(0.0001, L/2, dr)                      # Range of radial distances
dim = 2                                             # Dimension (2D for membrane analysis)

gr_en = np.zeros(len(R))                            # radial distribution function
gr_disp_en = np.zeros(len(R))                       # displacement displacement correlation function
ensembles = np.arange(0, 1000, 10)                  # Time frames to analyze
N_ensembles = len(ensembles)
N = len(lipids)                                     # Number of lipids
rho = N / (L**2)                                    # Number density of lipids

# Loop over selected time frames and compute correlations.
for ti in ensembles:
    u.trajectory[ti+dt]                             # Go to time step 'ti + dt'
    final_pos = lipids.positions
    rt = final_pos[:,:2]
    xt = final_pos[:,0]
    yt = final_pos[:,1]

    u.trajectory[ti]                                # Return to time step 'ti'
    initial_pos = lipids.positions
    ri = initial_pos[:,:2]
    xi = initial_pos[:,0]
    yi = initial_pos[:,1]

    # Compute pairwise distances
    dist_nd_sq = np.zeros(N*(N-1)//2)               # Squared distances
    for di in range(dim):
        pos_1d = ri[:, di][:, np.newaxis]
        dist_1d = pdist(pos_1d)
        dist_1d[dist_1d > L * 0.5] -= L             # Correct for periodic boundaries
        dist_nd_sq += dist_1d**2
    dist_nd = np.sqrt(dist_nd_sq)
    dist_nd = squareform(dist_nd)                   # Convert to square form
    
    unit_disp = distance(xi, yi, xt, yt, L)         # Compute unit displacements

    # Initialize arrays for radial distribution (gr_en) and displacement correlation (gr_disp_en)
    gr = np.array([])
    gr_disp = np.array([])
    for r in R:
        cr_disp = 0
        cr = 0
        for i in range(N):
            cr_dot, cr_i = corr_func(i, dist_nd, unit_disp, r, dr, rho)
            cr_disp_i = np.dot(cr_dot, unit_disp[i]) / (2 * np.pi * r * dr * rho)
            cr_disp += cr_disp_i
            cr += cr_i / (2 * np.pi * r * dr * rho)
        cr = cr / N
        cr_disp = cr_disp / N
        gr = np.append(gr, cr)              # Normalize and append to gr array
        gr_disp = np.append(gr_disp, cr_disp)
        
    gr_en += gr                             # Accumulate for ensemble averaging
    gr_disp_en += gr_disp                   # Normalize and append to gr_disp array

# Finalize ensemble-averaged correlation functions.
gr_en = gr_en / N_ensembles
gr_disp_en = gr_disp_en / N_ensembles

# Combine results and prepare for output.
result = np.column_stack((R, gr_en, gr_disp_en))


# Name of the output file and creating directory and saving the result
base_filename = os.path.splitext(os.path.basename(trr_file))[0]
dirname = 'correlation_data_1'
if not os.path.exists(dirname):
    os.makedirs(dirname)
result_file_path = os.path.join(dirname, f'{base_filename}_100_cor.txt')
np.savetxt(result_file_path, result, header='R gr_en, gr_disp_en', comments='', delimiter='\t', fmt='%10.5f')
