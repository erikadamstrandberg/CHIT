import os
from pathlib import Path

mypath = Path(Path(__file__).resolve().parents[0])

os.chdir(str(mypath))

#%%
import matplotlib.pyplot as plt
import numpy as np

def fft2c(x):
    '''
    2D Fourier transform with shift!
    '''
    return np.fft.fftshift(np.fft.fft2(np.fft.fftshift(x)))

def ifft2c(x):
    '''
    2D inverse Fourier transform with shift!
    '''
    return np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(x)))

def PAS(E1, L, N, a, lam0, n):
    '''
    Propagation of angular spectrum square sampling grid
    '''
    delta_k = 2*np.pi/(N*a)
    kx  = np.arange(-(N/2)*delta_k, (N/2)*delta_k, delta_k)
    ky  = kx
    KX, KY = np.meshgrid(kx,ky)
    
    k = 2*np.pi*n/lam0
    KZ = np.sqrt(k**2 - KX**2 - KY**2, dtype=complex)
    phase_prop = np.exp(1j*KZ*L)

    A = (a**2/(4*np.pi**2))*fft2c(E1)
    B = A*phase_prop
    E2 = (N*delta_k)**2*ifft2c(B)
    
    return E2
#%%

PI = np.pi
RAD = PI/180
CM = 1e-2
MM = 1e-3
UM = 1e-6
NM = 1e-9

#def deflecting_lens_gradient_map(X, Y, n, lam0, anglex, angley, f):
#    return np.exp(-1j*np.mod(2*PI - 2*PI*n/lam0*(X*np.sin(anglex) + Y*np.sin(angley)), 2*PI))*np.exp(-1j*np.mod(2*PI - 2*PI*n/lam0*(np.sqrt(X**2 + Y**2+f**2)-f), 2*PI))

# ### Get deflecting cylindrical lens gradient phase map
# def deflecting_cylindrical_lens_gradient_map(X, Y, n, lam0, anglex, angley, f):
#     return 2*PI - (2*PI*n/lam0)*np.sqrt(X**2 + f**2 - f) + 2*PI*n/lam0*(X*np.sin(anglex) + Y*np.sin(angley))
# ### Get deflecting cylindrical lens gradient phase map

#def deflecting_cylindrical_lens_gradient_map(X, Y, n, lam0, anglex, angley, f):
 #   return 2*PI - (2*PI*n/lam0)*(np.sqrt(X**2 + f**2) - f) + 2*PI*n/lam0*(X*np.sin(anglex) + Y*np.sin(angley))

def deflecting_spherical_lens_gradient_map(X, Y, n, lam0, anglex, angley, f):
    return 2*PI - (2*PI*n/lam0)*(np.sqrt(X**2 + Y**2 + f**2) - f) + 2*PI*n/lam0*(X*np.sin(anglex) + Y*np.sin(angley))

def plot_focused_beam(E_incident, n, NA_objective, ax, ay, Nx, Ny, X, Y, anglex, angley, f):
    
    k = 2*PI*n/lam0
    kx = np.linspace(-2*PI/(ax*2), 2*PI/(ax*2), Nx)/k
    ky = np.linspace(-2*PI/(ay*2), 2*PI/(ay*2), Ny)/k
    
    I_incident = np.abs(E_incident)**2

    theta = np.linspace(0, 2*PI, 1000)
    kx_objective = NA_objective*np.cos(theta)
    ky_objective = NA_objective*np.sin(theta)

    NA_beam = n*np.sin(anglex)
    theta   = np.linspace(0, 2*PI, 1000)
    kx_beam = NA_beam*np.cos(theta)
    ky_beam = NA_beam*np.sin(theta)
    
    T_deflecting_lens = np.exp(1j*deflecting_spherical_lens_gradient_map(X, Y, n, lam0, anglex, angley, f))
    
    E_k = fft2c(E_incident*T_deflecting_lens)
    I_k = np.abs(E_k)**2
    I_k = I_k/np.max(I_k)
        
    fig     = plt.figure(1, figsize=(10,8))
    ax1     = fig.add_subplot(221)
    ax2     = fig.add_subplot(222)
    ax3     = fig.add_subplot(223)
    
    ax_list = [ax1, ax2, ax3]

    extent_r = np.array([x_um.min(), x_um.max(), y_um.min(), y_um.max()])
    extent_k = np.array([kx.min(), kx.max(), ky.min(), ky.max()])

    ax1.imshow(I_incident, extent=extent_r, cmap='plasma')
    
    ax2.imshow(np.angle(T_deflecting_lens), extent=extent_r)

    ax3.imshow(I_k, extent=extent_k, cmap='plasma')
    ax3.plot(kx_objective, ky_objective, color='black', linewidth=0.8)
    ax3.plot(kx_beam, ky_beam, '--', color='yellow', linewidth=0.5)
    
    fontsize_title  = 16
    fontsize_axis   = 15
    
    plt.figure(2)
    plt.imshow(I_k, extent=extent_k, cmap='plasma')
    plt.plot(kx_objective, ky_objective, color='black', linewidth=0.8)
    plt.plot(kx_beam, ky_beam, '--', color='yellow', linewidth=0.5)
    plt.xlabel(r'$k_x/k$', fontsize=fontsize_axis)
    plt.ylabel(r'$k_y/k$', fontsize=fontsize_axis)
    plt.title(r'PAS, spherical lens, 61 $^{\circ}$', fontsize=fontsize_title)
    plt.tight_layout()
    #plt.savefig(mypath/'PAS_cylindrical_61.png', dpi=600, format='png')
    

    
    
# Create beam
# Setup simulation window
Lx = 1*MM
Ly = 1*MM

ax = 420*NM
ay = 420*NM

Nx = int(Lx/ax)
Ny = int(Ly/ay)

x = np.arange(-(Nx/2)*ax, (Nx/2)*ax, ax)
y = np.arange(-(Ny/2)*ay, (Ny/2)*ay, ay)
x_um = x/UM
y_um = y/UM

X, Y = np.meshgrid(x, y)
R    = np.sqrt(X**2 + Y**2)

# Setup loosly focused beam in measurement
lam0        = 976*NM
n_air       = 1
omega0      = 100*UM
offset_x    = 0
offset_y    = 0

NA_objective = 0.95

anglex = 55*RAD
angley = 0*RAD

f = 800*UM

E_incident = np.exp(-((X - offset_x)**2 + (Y - offset_y)**2)/omega0**2)

plot_focused_beam(E_incident, n_air, NA_objective, ax, ay, Nx, Ny, X, Y, anglex, angley, f)



#%%

Lx = 1*MM
Ly = 1*MM

ax = 260*NM
ay = 260*NM

Nx = int(Lx/ax)
Ny = int(Ly/ay)

x = np.arange(-(Nx/2)*ax, (Nx/2)*ax, ax)
y = np.arange(-(Ny/2)*ay, (Ny/2)*ay, ay)
x_um = x/UM
y_um = y/UM

X, Y = np.meshgrid(x, y)
R    = np.sqrt(X**2 + Y**2)

# Setup beam from VCSEL
lam0        = 984*NM
n_air        = 1
n_substrate = 3.48
k_substrate = (2*PI*n_substrate)/lam0 
omega0      = 1.3*UM
offset_x    = 0
offset_y    = 0

z_substrate = 630*UM

rayleigh_length = PI*omega0**2*n_substrate/lam0
omega           = omega0*np.sqrt(1 + (z_substrate/rayleigh_length)**2)
R_curve         = z_substrate*(1 + (rayleigh_length/z_substrate)**2)
gouy_phase      = np.arctan(z_substrate/rayleigh_length)

E_interface = np.exp(-((X - offset_x)**2 + (Y - offset_y)**2)/omega**2)*np.exp(1j*(k_substrate*z_substrate + k_substrate*R**2/(2*R_curve)) - gouy_phase)
print('Waist size at interface: ' + str(round(2*omega/UM, 2)) + ' um')

NA_objective = 0.95

anglex = 40*RAD
angley = 0*RAD

f = 1*MM

E_incident = np.exp(-((X - offset_x)**2 + (Y - offset_y)**2)/omega0**2)

plot_focused_beam(E_incident, n_air, NA_objective, ax, ay, Nx, Ny, X, Y, anglex, angley, f)

#%% Discretized beam!

Lx = 200*UM
Ly = 200*UM

ax = 260*NM
ay = 260*NM

Nx = int(Lx/ax)
Ny = int(Ly/ay)

x = np.arange(-(Nx/2)*ax, (Nx/2)*ax, ax)
y = np.arange(-(Ny/2)*ay, (Ny/2)*ay, ay)
x_um = x/UM
y_um = y/UM

X, Y = np.meshgrid(x, y)
R    = np.sqrt(X**2 + Y**2)

lam0        = 976*NM
n_air       = 1
omega0      = 30*UM
offset_x    = 0
offset_y    = 0

anglex = 55*RAD
angley = 0*RAD

f = 800*UM

T_deflector = deflecting_cylindrical_lens_gradient_map(X, Y, n_air, lam0, anglex, angley, f)
T_cs = np.angle(T_deflector[Ny//2,:])

plt.figure(1)
plt.plot(x_um, T_cs)


