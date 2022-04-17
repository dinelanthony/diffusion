# OBJECTIVE
# To simulate the diffusion of dye in a square dish of water
# using a PDE (Partial Differential Equation)

# OPERATION
# In the first cell, the first 5 lines will dictate
# the size of the box, as well as the rate of diffusion,
# and the total simulation time
# Line 15 controls the total initial concentration of the dye

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from scipy.optimize import curve_fit

L = 1  # size of "box"
D = 1e-3  # Constant for the PDE
N = 101  # Number of grid points in one-d
dt = 0.01  # time step
tmax = 100  # Total duration
dx = L / N  # change in x
dy = L / N  # change in y
steps = int(tmax / dt) + 1  # Amount of steps to take
nframes = int(tmax / 0.2)  # Total number of frames
iterations = int(steps / nframes)  # Steps between frames

# create initial conditions:
C = np.zeros((N, N))
# Set particles in a blob in the center:
C[N//2, N//2] = 10
k = dt/dx/dx*D  # PDE Constant
Cp = np.zeros((N, N))
ims = [(plt.pcolormesh(C), )]  # Initial frame for animation


def partial_diff():
    """
    Calculate the new phases of the diffusing
    system according to the PDE. Reflecting
    boundary conditions are stored here as well.
    """
    global C, Cp
    Cp[0, 0] = Cp[1, 1]
    Cp[0, -1] = Cp[1, -2]
    Cp[-1, 0] = Cp[-2, 1]
    Cp[-1, -1] = Cp[-2, -2]
    Cp[0, 1:-1] = Cp[1, 1:-1]
    Cp[-1, 1:-1] = Cp[-2, 1:-1]
    Cp[1:-1, 0] = Cp[1:-1, 1]
    Cp[1:-1, -1] = Cp[1:-1, -2]

    ### PDE ###
    Cp[1:-1, 1:-1] = C[1:-1, 1:-1] + k * (C[2:, 1:-1]
                                          + C[0:-2, 1:-1]
                                          + C[1:-1, 2:]
                                          + C[1:-1, 0:-2]
                                          - 4 * C[1:-1, 1:-1])
    C, Cp = Cp, C  # Save C and Cp


def Gaussian(x, sigma, A):
    """
    Returns the value of the Gaussian at a given x
    """
    expo = np.exp(-1 / 2 * ((x - (N / 200)) / sigma) ** 2)
    return A * (1 / (sigma * np.sqrt(2 * np.pi))) * expo


def animThis():
    """
    Completes a given number of iterations
    of the diffusion experiment. Print is
    used to track how many times the function
    has been called
    """
    print(".", end="")
    for i in range(iterations):
        partial_diff()


### STORE FRAMES ###
fig = plt.figure(figsize=(8, 8))
for j in range(nframes):
    animThis()
    ims.append((plt.pcolormesh(C.copy()), ))

### ANIMATION ###
plt.xlim(0, N)
plt.ylim(0, N)
anim = animation.ArtistAnimation(fig, ims, interval=50, repeat=False)
anim.save('diffuse.webm', extra_args=['-vcodec', 'libvpx'])
plt.close()

del anim
del ims

HTML('<video controls autoplay>'
     + '<source src="diffuse.webm" type="video/webm"></video>')