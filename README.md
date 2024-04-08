# Molecular Dynamics Simulation Analysis Toolkit

This collection of Python scripts offers a suite of tools for analyzing the dynamics of lipid membranes and tracer particles in molecular dynamics (MD) simulations. The toolkit is designed for researchers in biophysics, computational chemistry, and related fields, providing insights into membrane fluidity, tracer mobility, and the effects of active processes.

## Scripts Overview

### `correlated_flow.py`
Calculates displacement vectors and displacement-displacement correlations for lipid molecules. It helps in understanding the collective behavior of lipids, essential for studying membrane fluidity and molecule interactions.

### `msd_grid_timeavg.py`
Computes the Mean Squared Displacement (MSD) of lipids or selected atoms on a grid, facilitating the analysis of diffusive behavior within the membrane.

### `membrane_sim.py`
Simulates the dynamics of a two-dimensional lattice model to study lipid membranes and embedded asters. It incorporates thermal fluctuations and active processes, allowing for the investigation of phase separation, self-assembly, and active patterning.

### `MSD_analysis 1.py`
Analyzes the mobility of bound and non-bound tracers in MD simulations. It calculates the MSD for each type of tracer, providing insights into their diffusive behavior under various conditions.

## Installation

Requirements include Python 3 and the following libraries:
- MDAnalysis
- NumPy
- SciPy
- Numba (for `membrane_sim.py`)

To install the necessary libraries (excluding standard libraries), run:

```bash
pip install MDAnalysis numpy scipy numba
```
## Detailed Usage

Each script within the toolkit is designed to be run from the command line and has specific input requirements. Below you'll find more detailed usage instructions for each script, which include the necessary command-line arguments and any optional parameters you can include to customize the analysis.

### `correlated_flow.py`
To analyze displacement vectors and correlations for lipid molecules, use:
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

Correlated Flow Analysis: Run correlated_flow.py to analyze how lipid molecules move relative to each other over time. This script outputs a file containing the radial distribution functions and displacement-displacement correlation functions.

Mean Squared Displacement Analysis: Use msd_grid_timeavg.py to calculate the MSD of selected atoms or lipids. This helps in understanding their diffusive behavior in the membrane environment.

## Output Data
The scripts generate output files containing the analysis results. For example, displacement vectors, MSD values, and correlation functions. These files are saved in specified directories or alongside the input files, depending on the script's configuration.

## Contributing
We welcome contributions from the community! If you have suggestions for improvements, bug fixes, or new features, please feel free to submit an issue or pull request on GitHub.

## License

This toolkit is available under the MIT License. Feel free to use, modify, and distribute it as per the license conditions.
For more details, see the [LICENSE](LICENSE.md) file in this repository.




"""









