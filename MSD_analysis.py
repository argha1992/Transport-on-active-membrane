"""
This Python script is designed to analyze the transport property of lipids in the lattice simulation. It calculates the Mean Squared Displacement (MSD) for each type of lipid.

Key functionalities:
- Reading input parameters from a configuration file specified as a command-line argument, allowing for flexible simulation analysis without modifying the code.
- Loading x and y coordinates of tracer particles from specified data files, supporting analysis of different tracer categories (e.g., 'passive' and 'inert').
- Calculating displacements with consideration of periodic boundary conditions to ensure accurate distance measurements across simulation box boundaries.
- Saving MSD calculations to text files, facilitating further analysis.

Usage:
The script expects the name of an input configuration file as its command-line argument. 

Example command:
python MSD_analysis.py input_parameters.txt

The input configuration file should follow a simple key-value format, with each parameter on a new line. Lines starting with '#' are considered comments and ignored.

Input file example:
# Simulation parameters
R 4
Afrac 0.2
pbind 1.0
runid 0
T 281
aster_scale 10


Dependencies:
- NumPy: For numerical operations and handling large arrays of simulation data.
- sys: For accessing command-line arguments.
"""

# Importing important libraries
import numpy as np
import sys

# Distance function to calculate displacement with periodic boundary condition
def distance(g,a,L):
    dx=g-a
    corrx=np.round(dx/L)
    dx=dx-corrx*L
    return dx

# Main function to calculate the mean squared displacement (MSD).
def msd(dispX,dispY):
    Ntracer=len(dispX[:,0]) # No. of tracer particles as rows
    timespan=len(dispY[0,:]) # No. of timesteps as columns
    time=int(timespan/2) # Statistically appropriate choice for range of lagtime
    x=np.zeros((Ntracer,time)) # MSD matrix for each particle
    for dt in range(1,time+1):

        dpx = dispX - np.roll(dispX,dt,axis=1) 
        dpy = dispY - np.roll(dispY,dt,axis=1)
        dpx[:,:dt] = np.nan
        dpy[:,:dt] = np.nan
        x[:,dt-1] = np.nanmean(dpx**2 + dpy**2,axis=1) # Time averaging

    xav = np.nanmean(x,axis=0) # Ensemble average
    return xav
    


if __name__ == '__main__':
    

    inputfile_name = sys.argv[1] # Asking for a parameter file
    Afrac= 0.2 # Area fraction of asters
    R=4 # Radius of asters assumed to be circle
    pbind=1.0 # Binding probability of passive lipids with asters
    runid=0 # Run id of the replica
    T=281 # Temperature of the system 
    L=100 # Size of our lattice
    with open(inputfile_name, 'r') as inputfile:
        for words in inputfile: # Reading the parameters from the input file
            if(words[0]== '#' or words[0]== '\n'):
                continue
            key = words.split()[0]
            if (key=='start'):
                continue
            if (key== 'end'):
                break
            value = words.split()[1]

            if (key=='R'):
                R = eval(value)
            elif (key=='Afrac'):
                Afrac= eval(value)
            elif (key== 'pbind'):
                pbind = eval(value)
            elif (key == 'runid'):
                runid= eval(value)
            elif (key == 'T'):
                T= eval(value)
            elif (key == 'aster_scale'):
                aster_scale = eval(value)
            else:
                raise ValueError('Unrecognised parameters')



    # Location of the trajectories depending on the input parameters
    filepathsource='/Data/Afrac_%2.1f_R_%d/aster_scale_%d/Data_pbind_%2.1f/Run%d_Active/T_%d/'%(Afrac, R, aster_scale,pbind, runid, T)

    cat='nobind' # Inert lipid category
    
    xcoordiff_tracer=np.loadtxt(filepathsource+'x'+cat+'coordiff_tracer_size='+str(L)+'temperature='+str(T)+'K.npy',dtype=float,delimiter=',') # Trajectory file containing x-coordinate of lipids at different time along column
    ycoordiff_tracer=np.loadtxt(filepathsource+'y'+cat+'coordiff_tracer_size='+str(L)+'temperature='+str(T)+'K.npy',dtype=float,delimiter=',') # Trajectory file containing y-coordinate of lipids at different time along column
    xcoordiff_tracer=xcoordiff_tracer.T
    ycoordiff_tracer=ycoordiff_tracer.T


    dxcoordiff_tracer = distance(xcoordiff_tracer[:, 1:], xcoordiff_tracer[:, :-1],L) # Calculating displacement between each consecutive times with periodic boundary condition
    dycoordiff_tracer = distance(ycoordiff_tracer[:, 1:], ycoordiff_tracer[:, :-1],L)

    dispX = np.cumsum(dxcoordiff_tracer,axis=1) # Calculating total displacement at each time which is sum of all small displacements
    dispY = np.cumsum(dycoordiff_tracer,axis=1)

    xav=msd(dispX,dispY) # Calculating MSD averaged over all particles with time average.
    np.savetxt(filepathsource+'Diffuse'+cat+str(T)+'.txt',xav) # Saving the MSD array
    del dxcoordiff_tracer, dycoordiff_tracer,xav


    cat='bind' # Passive lipid category
    # Above steps are repeated here to obtain MSD for passive lipids
    xcoordiff_tracer=np.loadtxt(filepathsource+'x'+cat+'coordiff_tracer_size='+str(L)+'temperature='+str(T)+'K.npy',dtype=float,delimiter=',')
    ycoordiff_tracer=np.loadtxt(filepathsource+'y'+cat+'coordiff_tracer_size='+str(L)+'temperature='+str(T)+'K.npy',dtype=float,delimiter=',')
    xcoordiff_tracer=xcoordiff_tracer.T
    ycoordiff_tracer=ycoordiff_tracer.T


    dxcoordiff_tracer = distance(xcoordiff_tracer[:, 1:], xcoordiff_tracer[:, :-1],L)
    dycoordiff_tracer = distance(ycoordiff_tracer[:, 1:], ycoordiff_tracer[:, :-1],L)
    dispX = np.cumsum(dxcoordiff_tracer,axis=1)
    dispY = np.cumsum(dycoordiff_tracer,axis=1)

    xav=msd(dispX,dispY)
    np.savetxt(filepathsource+'Diffuse'+cat+str(T)+'.txt',xav)
    del dxcoordiff_tracer, dycoordiff_tracer,xav


    
