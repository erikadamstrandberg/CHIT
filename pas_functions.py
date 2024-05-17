#%%
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