"""
This script simulates the dynamics of a two-dimensional lattice model, which is particularly designed to study the behavior of lipid membranes in the presence of active processes of actin aster modeling. The model incorporates both thermal fluctuations and active processes, allowing for the investigation of complex transport of lipids.

Key features of the simulation include:
- A square lattice where each site represents a lipid molecule that can interact with its neighbors. The lattice employs periodic boundary conditions.
- An implementation of the Monte Carlo method to evolve the system in time. This includes standard Metropolis moves to ensure equilibrium behavior and custom moves to introduce active driving forces due to asters.
- Asters, which are modeled as circular regions that can affect the local dynamics of lipids depending on the kind of lipid and their binding state, representing active regions within the membrane.
- A mechanism to track the movement of individual lipids over time, allowing for detailed analysis of the system's dynamical properties.

The simulation proceeds in several phases:
1. Initialization: Set up the lattice with a random configuration of lipids and asters, and define the interaction parameters.
2. Equilibration: Run a series of Monte Carlo steps to bring the system to equilibrium under the given temperature.
3. Active dynamics: Introduce non-equilibrium moves to model the effects of active processes, such as the directed motion of lipids on the asters.
4. Data collection: After the system has reached steady state (N_steady), we can save configurations, energy measurements, and other relevant data for post-simulation analysis.

This script is highly configurable, with parameters for lattice size, temperature, interaction strengths. 

Dependencies: NumPy for numerical operations, SciPy for statistical functions, and Numba for performance optimization through just-in-time compilation.

Usage: The script is executed from the command line: python lattice_simulation.py 
"""


import numpy as np
from math import ceil, pi
from scipy.stats import expon
from numba import njit

# Define a function to get the neighboring spins and their distances
@njit
def getneighbors(k, l, spin):
    nei = []  # List to store neighboring spins
    dist = []  # List to store distances to neighbors
    # Loop through surrounding cells within a distance of 2 units
    for i in range(-2, 3):
        for j in range(-2, 3):
            # Skip the current cell
            if i == 0 and j == 0:
                continue
            # Calculate distance and include if within range
            r = (i**2 + j**2)**0.5
            if r <= 2:
                nei.append(spin[(k + i) % L, (l + j) % L])
                dist.append(r**6)
    return nei, dist

# Calculate pairwise interaction strengths based on temperature
@njit
def pairint(species):
    Tc = 293  # Critical temperature Das et al (PRL 2016)
    Jbeta = Tc / (T * 1.43)  # Scaling factor for interaction strengths wrt Tc
    J = np.zeros((species, species))  # Interaction matrix
    for i in range(species):
        J[i, i] = Jbeta
    return J

# Calculate the energy contribution of a spin and its neighbors
@njit
def nE(i, j, spin):
    neighbors, dist = getneighbors(i, j, spin)  # Get neighbors and distances
    centre = int(spin[i, j])  # Chosen spin
    ener = 0  
    # Calculate energy based on interactions and distances
    for t, m in enumerate(neighbors):
        m = int(m)
        ener += (-1) * J[centre, m] / dist[t]
    return ener

# Calculate the total energy of the system
@njit
def totenergy(spin):
    energy = 0  
    for i in range(L):
        for j in range(i, L):
            energy += nE(i, j, spin)
    return energy / N

# Perform a global Kawasaki moves for equilibration
@njit
def eqmcmove(spin, Tot):
    for i in range(L*L):
        # Select two random positions
        a, b, c, d = np.random.randint(L, size=4)
        # Ensure positions are different
        if (a != c or b != d):
            ei = nE(a, b, spin) + nE(c, d, spin)  # Initial energy
            # Swap spins
            temp1, temp2 = spin[a, b], spin[c, d]
            spin[a, b], spin[c, d] = temp2, temp1
            ef = nE(c, d, spin) + nE(a, b, spin)  # Final energy
            dE = ef - ei  # Change in energy
            # Accept or reject the move based on the Metropolis criterion
            if dE <= 0:
                Tot += dE / N
            elif np.random.random() < np.exp(-dE):
                Tot += dE / N
            else:
                # Revert swap if move is rejected
                spin[a, b], spin[c, d] = temp1, temp2
    return spin, Tot

# Perform Barrier Hop Dynamics(BHD) moves 
def diffmcmove(spin,a,b,Tot,lipids,lipidtemp):
    ind=np.random.randint(4)
    moves=[(a,(b+1)%L),(a,(b-1)%L),((a-1)%L,b),((a+1)%L,b)]
    c,d=moves[ind]
    ei=nE(a,b,spin)+nE(c,d,spin)
    barrier=-ei
    spin[a,b],spin[c,d]=spin[c,d],spin[a,b]
    ef=nE(a,b,spin)+nE(c,d,spin)
    dE=ef-ei
    if barrier<=0:
        Tot+=dE/N 
        s1=lipidtemp[a,b]
        s2=lipidtemp[c,d]
        lipids[s1]=[c,d]
        lipids[s2]=[a,b]
        lipidtemp[a,b]=s2
        lipidtemp[c,d]=s1
    elif (np.random.random()<np.exp(-barrier)):
        Tot+=dE/N
        s1=lipidtemp[a,b]
        s2=lipidtemp[c,d]
        lipids[s1]=[c,d]
        lipids[s2]=[a,b]
        lipidtemp[a,b]=s2
        lipidtemp[c,d]=s1
    else:
        spin[a,b],spin[c,d]=spin[c,d],spin[a,b]
    return spin,Tot,lipids,lipidtemp


# Perform non-equilibrium active moves 
def noneqmcmove(spin,Tot,a,b,dis, lipids,lipidtemp):
    p=0.8 # Biased probablility to move towards aster core
    temp1=spin[a,b] 
    case1=abs(dis[1])==abs(dis[2]) # Performs radial moves on a square lattice depending on 3 cases
    case2=abs(dis[1])>abs(dis[2])
    case3=abs(dis[2])>abs(dis[1])
    if case1:
        c=np.random.choice([a,(a+1*sign(dis[1]))%L])
        if c==a:
            d=(b+1*sign(dis[2]))%L
            # dy=dy-1*sign(dy)
        else:
            d=b
            # dx=dx-1*sign(dx)
    elif case2:
        c=(a+1*sign(dis[1]))%L
        d=b
    elif case3:
        c=a
        d=(b+1*sign(dis[2]))%L
                   
    if p>np.random.random(): # Performing aster moves
        ei=nE(a,b,spin)+nE(c,d,spin)
        spin[a,b]=spin[c,d]
        spin[c,d]=temp1
        s1=lipidtemp[a,b]
        s2=lipidtemp[c,d]
        lipids[s1]=[c,d]
        lipids[s2]=[a,b]
        lipidtemp[a,b]=s2
        lipidtemp[c,d]=s1
        ef=nE(a,b,spin)+nE(c,d,spin)
        dE=ef-ei
        Tot+=dE/N 
              
    else: # Other 3 directions are chosen randomly
        if case1:
            if c==a:
                c=np.random.randint((a-1),(a+2))
                if c>(L-1) or c<0:
                    c=c%L
                if c==a:
                    d=(b-1*sign(dis[2]))%L
                else:
                    d=b
            else:
                d=np.random.randint((b-1),(b+2))
                if d>(L-1) or d<0:
                    d=d%L
                if d==b:
                    c=(a-1*sign(dis[1]))%L
                else:
                    c=a
        elif case2:
            d=np.random.randint((b-1),(b+2))
            if d>(L-1) or d<0:
                d=d%L
            if d==b:
                c=(a-1*sign(dis[1]))%L
            else:
                c=a
        else:
            c=np.random.randint((a-1),(a+2))
            if c>(L-1) or c<0:
                c=c%L
            if c==a:
                d=(b-1*sign(dis[2]))%L
            else:
                d=b
        ei=nE(a,b,spin)+nE(c,d,spin)
        spin[a,b]=spin[c,d]
        spin[c,d]=temp1
        s1=lipidtemp[a,b]
        s2=lipidtemp[c,d]
        lipids[s1]=[c,d]
        lipids[s2]=[a,b]
        lipidtemp[a,b]=s2
        lipidtemp[c,d]=s1
        ef=nE(a,b,spin)+nE(c,d,spin)
        dE=ef-ei
        Tot+=dE/N 
       
    return spin,Tot,lipids,lipidtemp        

# Main dynamics function to orchestrate the combination of thermal and active moves
def dynamics(spin,Tot, lipids,lipidtemp):
    
    for i in range(L*L):
        a=np.random.randint(L)
        b=np.random.randint(L)
       
        if spin[a,b]==1: # Passive lipids
            move=True # Flag for the moves to be executed only once
            aff=np.random.random()
            if aff<Kaff[1]: # Bind state of passive lipids
                for j in range(noaster):
                    if a==coords[j,0] and b==coords[j,1]: # Avoid moves for passive lipids on aster core
                        move=False
                        break

                    dis=distance(coords[j,0], coords[j,1], a, b, L)
                    if dis[0]<=R : # Check for passive lipids in the vicinity of asters
                        spin,Tot,lipids,lipidtemp=noneqmcmove(spin,Tot,a,b,dis, lipids,lipidtemp ) 
                        move=False
                        break
                    elif (R<dis[0]<=(R+1)): # Exchange moves for lipids on the aster edge
                        z=np.random.random()
                        c,d=dirastercore(a,b,dis)
                        q=(1-Kaff[spin[c,d]])
                        if q>z:
                            ei=nE(a,b,spin)+nE(c,d,spin)
                            temp1=spin[a,b]
                            spin[a,b]=spin[c,d]
                            spin[c,d]=temp1
                            s1=lipidtemp[a,b]
                            s2=lipidtemp[c,d]
                            lipids[s1]=[c,d]
                            lipids[s2]=[a,b]
                            lipidtemp[a,b]=s2
                            lipidtemp[c,d]=s1
                            ef=nE(a,b,spin)+nE(c,d,spin)
                            dE=ef-ei
                            Tot+=dE/N
                        move=False
                        break
            if move:    
    
                spin,Tot,lipids,lipidtemp = diffmcmove(spin,a,b,Tot,lipids,lipidtemp) # Perform BHD moves
        else:
            spin,Tot,lipids,lipidtemp =diffmcmove(spin, a, b,Tot, lipids,lipidtemp ) # Perform BHD moves

    return spin,Tot,lipids,lipidtemp 


# Aster coordinates initialisation with no overlap
def is_circle_overlapping(x, y, R, circles):
  
    for cx, cy in circles:
        distance_squared = (distance(cx,cy,x,y,L)[0])**2 
        if distance_squared < (2*R)**2:
            return True
    return False

# Random coordinates designated to aster core on the lattice
def generate_nonoverlapping_circles(noaster, R, L):
   
    circles = []
    while len(circles) < noaster:
        x = np.random.randint(L)
        y = np.random.randint(L)
        if not is_circle_overlapping(x, y, R, circles):
            circles.append([x, y])
    return circles

# Exponential life death process of aster remodeling
def timesample(aster_scale):
    u=ceil(expon.rvs(scale=aster_scale))
    return u

# Check if any aster is overlapping with the existing ones during the remodeling
def coordsdynamics(coords,i,noaster):

    clash=True
    while (clash)==True:
        coords[i]=np.random.randint(L,size=(1,2))
        for j in range(noaster):
            if i==j:
                continue
            dist=distance(coords[i,0],coords[i,1],coords[j,0],coords[j,1],L)[0]
            if dist<(2*R+2):
                clash=True
                break
            else:
                clash=False
    return coords
    
# Main function for aster remodeling dynamics
def asterdynamics(coords,noaster,tp,t_birth,t):
   
    for i in range(noaster):
        if (t-t_birth[i]==tp[i]):
            coords=coordsdynamics(coords, i, noaster)
            t_birth[i]=t
            tp[i]=timesample(aster_scale)
    return coords,t_birth,tp

# Sign function to define the direction towards aster core
@njit
def sign(x):
    if x>0:
        value=1
    elif x<0:
        value=-1
    else:
        value=0
    return value

# Distance with periodic boundary conditions
@njit
def distance(g,h,a,b,L):
    dx=g-a
    dy=h-b
    corrx=np.round(dx/L)
    corry=np.round(dy/L)
    dx=dx-corrx*L
    dy=dy-corry*L
    dr=np.sqrt(dx**2+dy**2)
    return dr,dx,dy

# Function defining direction towards aster core
def dirastercore(a,b,dis):

    case1=abs(dis[1])==abs(dis[2])
    case2=abs(dis[1])>abs(dis[2])
    case3=abs(dis[2])>abs(dis[1])
    if case1:
        c=np.random.choice([a,(a+1*sign(dis[1]))%L])
        if c==a:
          d=(b+1*sign(dis[2]))%L
          # dy=dy-1*sign(dy)
        else:
            d=b
            # dx=dx-1*sign(dx)
    if case2:
      c=(a+1*sign(dis[1]))%L
      d=b
    if case3:
        c=a
        d=(b+1*sign(dis[2]))%L
    return c,d



if __name__ == '__main__':
    
    ###################
    # Define simulation parameters.
    # These include the size of the simulation grid (L), interaction parameters, temperature (T), and other physical constants.
 



    R=8 # Radius of asters
    L=100 # Lattice size
    spinparam=0.5 # Lipids ratio 50:50
    Neq=500 # Number of equilibration Monte Carlo steps
    Nneq=1500 # Nneq - Neq = Number of non-equilibrium Monte Carlo steps to reach steady state
    N_steady=11500 # N_steady - Nneq = Number of non-equilibrium Monte Carlo steps used for production
    Afrac=0.2 # Fraction of area covered by asters
    aster_scale=10 # Mean time of aster remodeling
    species=2 # Number of species in the simulation (Passive and inert)
    pbind=1.0 # Binding probability of passive lipids with asters
    T=311 # Simulation temperature
    runid=100 # Identifier for the simulation run used while averaging
    seed=np.random.randint(1000000) # Seed for random number generation for reproducibility

   
  
    
    # Generate file paths for saving simulation data.
    # Constructs a directory structure based on simulation parameters.
    filepath = '/path/to/save/results'    

    Tphys=310 # Physiological temperature
    np.random.seed(seed=seed)
    # Initialize interaction parameters and spin configuration.
    Kaff=np.zeros(species)
    Kaff[0]=0 # Binding affinity for inert lipids
    Kaff[1]=1 # Binding affinity for passive lipids
    N=L*L # Total number of lipid sites
    
    spin=np.loadtxt('initial_config_L_'+str(L)+'.npy',dtype=int,delimiter=',') # Load initial random lipid configuration for size L*L
    J=pairint(species) # Interaction matrix based on temperature
    Tot=totenergy(spin) # Calculate initial total energy

    # Equilibration phase: Perform global kawasaki moves to equilibrate the system.
    for t in range(Neq):
        spin,Tot=eqmcmove(spin,Tot)


    # Initialize asters: Positions and dynamics parameters.
    noaster=round((Afrac*L*L)/(pi*R**2)) # Number of asters
    coords=np.array(generate_nonoverlapping_circles(noaster, R, L),dtype=int) # Initial aster positions
    tp=np.zeros(noaster,dtype=int) # Lifespan of asters
    for i in range(noaster): 
        tp[i]=timesample(aster_scale) # Exponential probability distribution function for remodeling
    t_birth=np.ones(noaster,dtype=int)*Neq # Birth time of asters

    # Additional initialization for lipid tracking and dynamics.
    lipids=[] # List to store lipid positions
    lipidtemp=np.zeros((L,L),dtype=int) # Temporary array for lipid positions

    bind_index=[] # Passive lipid id
    nobind_index=[] # Inert lipid id
    for i in range(L): # Segragating loop for passive and inert lipids
        for j in range(L):
            lipids.append([i,j])
            lipidtemp[i,j]=L*i+j
            if spin[i,j]==1:
                bind_index.append(lipidtemp[i,j])
            else:
                nobind_index.append(lipidtemp[i,j])
    
    bind_index=np.array(bind_index)
    Nbindtracer=np.int32(0.05*len(bind_index)) # 5% are tracer particles
    bind_tracer_ind=np.random.choice(bind_index,size=Nbindtracer,replace=False)
    nobind_index=np.array(nobind_index)
    Nnobindtracer=np.int32(0.05*len(nobind_index)) # 5% are tracer particles
    nobind_tracer_ind=np.random.choice(nobind_index,size=Nnobindtracer,replace=False)

    # Simulation is performed untill non equilibrium steady state is reached
    for t in range(Neq,Nneq):
        spin,Tot,lipids,lipidtemp=dynamics(spin,Tot,lipids,lipidtemp) # Lipid dynamics
        coords,t_birth,tp=asterdynamics(coords,noaster,tp,t_birth,t) # Aster dynamics
    np.savetxt(filepath+'steady_config_size='+str(L)+'temperature='+str(T)+'K.npy', spin, delimiter=',',fmt='%d') # Saving steady state configuration 

    
    
    # Creating different files to save trajectory of lipids 
    trajbind_x=filepath+'xbindcoordiff_size='+str(L)+'temperature='+str(T)+'K.npy'
    trajbind_y=filepath+'ybindcoordiff_size='+str(L)+'temperature='+str(T)+'K.npy'
    trajnobind_x=filepath+'xnobindcoordiff_size='+str(L)+'temperature='+str(T)+'K.npy'
    trajnobind_y=filepath+'ynobindcoordiff_size='+str(L)+'temperature='+str(T)+'K.npy'
    trajbind_tracer_x=filepath+'xbindcoordiff_tracer_size='+str(L)+'temperature='+str(T)+'K.npy'
    trajbind_tracer_y=filepath+'ybindcoordiff_tracer_size='+str(L)+'temperature='+str(T)+'K.npy'
    trajnobind_tracer_x=filepath+'xnobindcoordiff_tracer_size='+str(L)+'temperature='+str(T)+'K.npy'
    trajnobind_tracer_y=filepath+'ynobindcoordiff_tracer_size='+str(L)+'temperature='+str(T)+'K.npy'
    aster_coords_x = filepath+'aster_coords_x.npy'
    aster_coords_y = filepath+'aster_coords_y.npy'

    f=open(filepath+'thermo'+str(T)+'K.txt','w') # Thermodynamic file to save average energy at every time
    print('Cycle \t Temp \t Energyav',file=f)

    # Further data handling and saving operations here...
    with open(trajbind_x, 'w') as file0, open(trajbind_y, 'w') as file1, open(trajnobind_x, 'w') as file2, open(trajnobind_y, 'w') as file3, open(trajbind_tracer_x, 'w') as file4, open(trajbind_tracer_y, 'w') as file5, open(trajnobind_tracer_x, 'w') as file6, open(trajnobind_tracer_y, 'w') as file7, open(aster_coords_x,'w') as file8, open(aster_coords_y,'w') as file9:       
        for t in range(Nneq,N_steady):
            traj=np.array(list(lipids))
            trajbind=traj[bind_index]
            trajnobind=traj[nobind_index]
            trajbind_tracer=traj[bind_tracer_ind]
            trajnobind_tracer=traj[nobind_tracer_ind]
            trajbind=np.insert(trajbind,0,[t,t],axis=0)
            trajnobind=np.insert(trajnobind,0,[t,t],axis=0)
            np.savetxt(file0, [trajbind[:,0]], fmt="%d", delimiter=',', newline='\n')
            np.savetxt(file1, [trajbind[:,1]], fmt="%d", delimiter=',', newline='\n')
            np.savetxt(file2, [trajnobind[:,0]], fmt="%d", delimiter=',', newline='\n')
            np.savetxt(file3, [trajnobind[:,1]], fmt="%d", delimiter=',', newline='\n')
            np.savetxt(file4, [trajbind_tracer[:,0]], fmt="%d", delimiter=',', newline='\n')
            np.savetxt(file5, [trajbind_tracer[:,1]], fmt="%d", delimiter=',', newline='\n')
            np.savetxt(file6, [trajnobind_tracer[:,0]], fmt="%d", delimiter=',', newline='\n')
            np.savetxt(file7, [trajnobind_tracer[:,1]], fmt="%d", delimiter=',', newline='\n')
            np.savetxt(file8, [coords[:,0]], fmt="%d", delimiter=',', newline='\n')
            np.savetxt(file9, [coords[:,1]], fmt="%d", delimiter=',', newline='\n')
            # Main simulation loop: Perform dynamics for (N_steady - Nneq) steps.
            spin,Tot,lipids,lipidtemp=dynamics(spin,Tot,lipids,lipidtemp)
            coords,t_birth,tp=asterdynamics(coords,noaster,tp,t_birth,t)
            print('%5d \t  %5.1f \t %4.5f'%(t,T,Tot),file=f)
        f.close()
