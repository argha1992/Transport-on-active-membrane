# 
### Authors
Saurav G. Varma, Argha Mitra

### Year
2024

### References

# Description" 

This collection of Python scripts offers a suite of tools for analyzing the dynamics of lipid membrane molecular dynamics simulation and lattice particle kinetic monte-carlo simulations. 

## Scripts Overview

## Codes for the coarse-grained MARTINI v3.0 lipids (GROMACS formatted .tpr and .trr)

#### `correlated_flow.py`
Calculates displacement vectors and displacement-displacement correlations (DDC) for lipid molecules. It helps in understanding the collective behavior of lipids, essential for studying membrane fluidity and molecule interactions.

#### `msd_grid_timeavg.py`
Computes the time-averaged Mean Squared Displacement (MSD) of lipids or selected atoms on a grid, facilitating the analysis of local-diffusive behavior within the membrane.

## Code for LATTICE simulation 
#### `membrane_sim.py`
Simulates the dynamics of a two-dimensional lattice model to study lipid membranes and embedded asters. It incorporates thermal fluctuations and active processes, allowing for the investigation of phase separation, self-assembly, and active patterning.

#### `MSD_analysis 1.py`
Analyzes the mobility of bound and non-bound tracers in MD simulations. It calculates the MSD for each type of tracer, providing insights into their diffusive behavior under various conditions.

## Dependencies and Software Requirements
- Python 3.10 (or above)
- Python packages: NumPy, SciPy, MDAnalysis, Numba (for `membrane_sim.py`)

## Installation
No installation is required. Directly run the Python programs if the dependencies are installed before. If not then install the necessary libraries (excluding standard libraries), run:
```bash
pip install MDAnalysis numpy scipy numba
```
or 
```bash
conda install -c conda-forge mdanalysis numpy scipy numba
```
## Detailed Usage

Each script within the toolkit is designed to be run from the command line and has specific input requirements. Below you'll find more detailed usage instructions for each script, which include the necessary command-line arguments and any optional parameters you can include to customize the analysis.

### `correlated_flow.py`
To analyze displacement displacement correlations (DDC) for lipid molecules, use:
```bash
python3 correlated_flow.py -t topology.tpr -s trajectory.trr
```
Options:
- `tpr_file`: Path to the input .tpr (topology) file.
- `trr_file`: Path to the input .trr (trajectory) file.

### `msd_grid_timeavg.py`
For computing the Mean Squared Displacement (MSD) of lipids or selected atoms on a grid, execute:
```bash
python3 msd_grid_timeavg.py -t topology.tpr -s trajectory.trr -x "PO4"
```
Options:
- `-t`, `--tpr`: Path to the input .tpr file.
- `-s`, `--trr`: Path to the input .trr file.
- `-x`, `--select`: Atom selection string, defaults to "all".

### `membrane_sim.py`
This script is executed with predefined parameters within the code.
```bash
python3 membrane_sim.py
```
Options:

### `MSD_analysis 1.py`
The script expects an input configuration file specifying simulation parameters such as tracer radius, area fraction, binding probability, and more.
```bash
python3 MSD_analysis 1.py input_parameters.txt
```
## Examples
To help you get started, here are some example commands and expected outcomes:

Correlated Flow Analysis MARTINI lipids: Run correlated_flow.py to analyze how pure DPPC molecules move relative to each other over time within a radius. This script outputs a .txt file containing the radial distribution functions and displacement-displacement correlation as a function of radius.

Mean Squared Displacement Analysis for MARTINI lipids: Use msd_grid_timeavg.py to calculate the time-averaged MSD of pure DPPC lipids in a grid box. This script outputs a .txt file containing the timestep and MSD for each grid.

Equilibrium and non-equilibrium Lattice simulation:


Mean Squared Displacement Analysis for Lattice:

## Output Data
The scripts generate output files containing the analysis results. For example, MSD values, and correlation functions. These files are saved in specified directories (please specify your path for the saving outputs). 

## Contributing
We welcome contributions from the community! If you have suggestions for improvements, bug fixes, or new features, please feel free to submit an issue or pull request on GitHub.

## Citation


## License

This toolkit is available under the MIT License. Feel free to use, modify, and distribute it as per the license conditions.
For more details, see the [LICENSE](LICENSE.md) file in this repository.











