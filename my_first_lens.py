import numpy as np
import matplotlib.pyplot as plt
import gdsfactory as gf
import scipy.io as io

import os
from pathlib import Path

## CREATE REALATIVE SAVE PATH
save_folder = 'GDS_file'
save_filename = 'Dynamic1.gds'

# Path() is really nice! Universal solution for different operating systems
# Path for save folder
folder_path = Path(Path(__file__).resolve().parents[0])

# Path to .gds file inside save folder
save_folder_path = Path(folder_path, save_folder)

# Create the save folder if it does not exist
if not os.path.exists(str(save_folder_path)):
    os.makedirs(str(save_folder_path))
    
save_path = Path(save_folder_path, save_filename) 

## Another relative savepath, samma folder som ovan
save_filename2 = 'Dynamic1_refpattern.gds'

# Path() is really nice! Universal solution for different operating systems
# Path for save folder
folder_path2 = Path(Path(__file__).resolve().parents[0])

# Path to .gds file inside save folder
save_folder_path2 = Path(folder_path2, save_folder)

# Create the save folder if it does not exist
if not os.path.exists(str(save_folder_path2)):
    os.makedirs(str(save_folder_path2))
    
save_path2 = Path(save_folder_path2, save_filename2) 


#%%

## UNITS (microns is ref)
CM = 10000
MM = 1000
NM = 1e-3

r = 99.84 #metasurface radie
P = 260*NM #period

x = np.arange(-r, r+P, P)
xx, yy = np.meshgrid(x, x)


n = 3.52
lam0 = 973*NM
f = 600

phi = (2*np.pi - (((2*np.pi*n)/lam0)*(np.sqrt(xx**2 + yy**2 + f**2)-f))) % (2*np.pi)

font_size_axis = 15
font_size_title = 17

plt.figure(1)
plt.contourf(x, x, phi, cmap = 'Blues')
cbar = plt.colorbar(ticks=[0,3.14,6.28])
plt.show()
plt.xlabel(r'x, $\mu$m', fontsize = font_size_axis)
plt.ylabel(r'y, $\mu$m', fontsize = font_size_axis)
plt.yticks(fontsize = font_size_axis)
plt.yt
plt.xticks(fontsize = font_size_axis)
plt.title(r'Phase map, collimating lens', fontsize = font_size_title)
cbar.set_label(r'Phase $\Phi$, rad', fontsize = font_size_axis)
cbar.set_ticklabels(['0','$\pi$','$2\pi$'], fontsize = font_size_axis)
#plt.grid(color = 'black', linewidth = 2)
plt.savefig(folder_path/'phasemap.png', dpi=600, format='png')
#cbar.set_yticks([0,3.14,6.28], fontsize = font_size_axis)


# fig, ax = plt.subplots()
# ax.contourf(x, x, phi)
# cbar = fig.colorbar(ticks = [0,3.14,6.28])
# ax.show()
# ax.set_xlabel(r'x $\mu$m', fontsize = font_size_axis)
# ax.set_ylabel(r'y $\mu$m', fontsize = font_size_axis)
# ax.set_yticks(fontsize = font_size_axis)
# ax.set_xticks(fontsize = font_size_axis)
# ax.set_title(r'Phase map, collimating lens', fontsize = font_size_title)
# cbar.set_label(r'Phase $\Phi$', fontsize = font_size_axis)
# cbar.set_ticks([0,180,360],fontsize = font_size_axis)
# cbar.ax.set_yticklabels(['0','$\pi$','$2\pi$'])


# ax.plot(radius[20:], -phi[20:]*180/np.pi+360, '.') #lägg till - innan phi och +360 på slutet
# ax.grid()
# ax.set_xlabel('r [nm]')
# ax.set_ylabel('Phase [deg]')
# ax.legend()
# ax.set_yticks([0,180,360])
# ax.set_yticklabels(['0','$\pi$','$2\pi$'])


#%%

mypath = Path(Path(__file__).resolve().parents[0])

os.chdir(str(mypath))
os.chdir('Parameter_sweep_COMSOL') #Parameter_sweep_COMSOL'
data = np.loadtxt('radius-dyn1_ver2_1-130.txt', skiprows=5)

radius = []
phase = []

for i in range(len(data)):
    radius = np.append(radius, data[i,0])    
    phase = np.append(phase, data[i,8])
    
phase = 2*np.pi -((phase + 2*np.pi) % (2*np.pi)) #(2*np.pi - 8/(180/np.pi)) - 


phase_delay_levels = [phase[34], phase[39], phase[43], phase[47], phase[50], phase[54], phase[22], phase[24], phase[27], phase[30], phase[33]] # phase[57], phase[60]
#phase_delay_levels = [0, np.pi/2, np.pi, (3*np.pi)/2, 2*np.pi] #from 53 är det fel
radius_levels = [radius[34], radius[39], radius[43], radius[47], radius[50], radius[54], radius[22], radius[24], radius[27], radius[30], radius[33]]

#%% creating lens pattern


## TOP CELL
gf.clear_cache()
top = gf.Component('TOP')

##CHIP
#Component, rectangle
chip_size = (1*CM, 1*CM) #x,y
chip_c = gf.components.rectangle(chip_size, layer = (1,0))
chip_c.name = 'chip'

#Reference for chip
chip_r = top.add_ref(chip_c) #adding it to top
chip_r.translate(-chip_size[0]/2, -chip_size[1]/2) #moving the chip so that the middle is in origo

##PATTERN
#Component 0
pattern0_c = gf.components.circle(radius[33]*NM, layer=(2,0))
pattern0_c.name = 'phase0'

#Component 1
pattern1_c = gf.components.circle(radius[39]*NM, layer=(2,0))
pattern1_c.name = 'phase1'

#Component 2
pattern2_c = gf.components.circle(radius[43]*NM, layer=(2,0))
pattern2_c.name = 'phase2'

#Component 3
pattern3_c = gf.components.circle(radius[47]*NM, layer=(2,0))
pattern3_c.name = 'phase3'

#Component 4
pattern4_c = gf.components.circle(radius[50]*NM, layer=(2,0))
pattern4_c.name = 'phase4'

#Component 5
pattern5_c = gf.components.circle(radius[54]*NM, layer=(2,0))
pattern5_c.name = 'phase5'

#Component 6
pattern6_c = gf.components.circle(radius[22]*NM, layer=(2,0))
pattern6_c.name = 'phase6'

#Component 7
pattern7_c = gf.components.circle(radius[24]*NM, layer=(2,0))
pattern7_c.name = 'phase7'

#Component 8
pattern8_c = gf.components.circle(radius[27]*NM, layer=(2,0))
pattern8_c.name = 'phase8'

#Component 9
pattern9_c = gf.components.circle(radius[30]*NM, layer=(2,0))
pattern9_c.name = 'phase9'


phi_discrete = np.zeros([len(phi[0,:]),len(phi[0,:])])


for i in range(len(phi[0,:])):
    for j in range(len(phi[:,0])):
        index = np.argmin(np.abs(phase_delay_levels - phi[i,j]))
        current_levels = phase_delay_levels[index]
        phi_discrete [i,j] = current_levels        
        if (x[i]**2 + x[j]**2) <= x[0]**2:
            if index == 0:            
                pattern_r = top.add_ref(pattern0_c).translate(x[i], x[j])
            if index == 1:            
                pattern_r = top.add_ref(pattern1_c).translate(x[i], x[j])    
            if index == 2:            
                pattern_r = top.add_ref(pattern2_c).translate(x[i], x[j])            
            if index == 3:            
                pattern_r = top.add_ref(pattern3_c).translate(x[i], x[j])
            if index == 4:            
                pattern_r = top.add_ref(pattern4_c).translate(x[i], x[j])
            if index == 5:            
                pattern_r = top.add_ref(pattern5_c).translate(x[i], x[j])
            if index == 6:            
                pattern_r = top.add_ref(pattern6_c).translate(x[i], x[j])
            if index == 7:            
                pattern_r = top.add_ref(pattern7_c).translate(x[i], x[j])
            if index == 8:            
                pattern_r = top.add_ref(pattern8_c).translate(x[i], x[j])
            if index == 9:            
                pattern_r = top.add_ref(pattern9_c).translate(x[i], x[j])
            if index == 10:
                pattern_r = top.add_ref(pattern0_c).translate(x[i], x[j])
                
#%% creating ref pattern

## TOP CELL
gf.clear_cache()
top2 = gf.Component('TOP')

# ##CHIP
# #Component, rectangle
chip_size2 = (1*CM, 1*CM) #x,y
# chip_c = gf.components.rectangle(chip_size, layer = (1,0))
# chip_c.name = 'chip'

# #Reference for chip
# chip_r = top.add_ref(chip_c) #adding it to top
# chip_r.translate(-chip_size[0]/2, -chip_size[1]/2) #moving the chip so that the middle is in origo

N=15
array_space = 5

for i in range(N):
    for j in range(N):
        pattern_r = top2.add_ref(pattern0_c).translate(-2*array_space + P*(i), P*(j))
        
for i in range(N):
    for j in range(N):
        pattern_r = top2.add_ref(pattern1_c).translate(-array_space + P*(i), P*(j))
        
for i in range(N):
    for j in range(N):
        pattern_r = top2.add_ref(pattern2_c).translate(P*(i), P*(j))
        
for i in range(N):
    for j in range(N):
        pattern_r = top2.add_ref(pattern3_c).translate(array_space + P*(i), P*(j))
        
for i in range(N):
    for j in range(N):
        pattern_r = top2.add_ref(pattern4_c).translate(2*array_space + P*(i), P*(j))
        
for i in range(N):
    for j in range(N):
        pattern_r = top2.add_ref(pattern5_c).translate(-2*array_space + P*(i), -array_space + P*(j))
        
for i in range(N):
    for j in range(N):
        pattern_r = top2.add_ref(pattern6_c).translate(-array_space + P*(i), -array_space + P*(j))
        
for i in range(N):
    for j in range(N):
        pattern_r = top2.add_ref(pattern7_c).translate(P*(i), -array_space + P*(j))
        
for i in range(N):
    for j in range(N):
        pattern_r = top2.add_ref(pattern8_c).translate(array_space + P*(i), -array_space + P*(j))
        
for i in range(N):
    for j in range(N):
        pattern_r = top2.add_ref(pattern9_c).translate(2*array_space + P*(i), -array_space + P*(j))



#%% lens pattern --> gds
##WRITE TO GDS
top.write_gds(str(save_path))

#%% ref pattern --> gds
##WRITE TO SECOND GDS FILE, refpattern
top2.write_gds(save_path2)

#%%
plt.figure(2)
plt.contourf(x, x, phi_discrete)
plt.colorbar()
plt.show()
        
#%%

def fft2c(x):

    return np.fft.fftshift(np.fft.fft2(np.fft.fftshift(x)))

def ifft2c(x):

    return np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(x)))


def PAS(E1, L, N, a, lam0, n):
    '''
    Funktion för att propagera E1 sträckan L genom PAS
    
    Funktionen är ej klar! Byt ut alla '.x.' för att få PAS funktionen att fungera
    '''
    
    # Varje sampelpunkt i k-planet motsvarar en plan våg med en viss riktning [kx,ky,kz]
    delta_k = (2*np.pi)/(N*a)                                               # Samplingsavstånd i k-planet
    
    kx      = np.arange(-(N/2)*delta_k, (N/2)*delta_k, delta_k) # Vektor med samplingspunkter i kx-led
    ky      = kx                                                # och ky-led
    
    KX, KY  = np.meshgrid(kx,ky)                                # k-vektorns x- resp y-komponent i varje 
                                                                # sampelpunkt i k-planet
    
    k = (2*np.pi*n)/lam0                                            # k-vektorns längd (skalär) för en plan våg i ett material med brytningsindex n
    
    KZ = np.sqrt(k**2 - KX**2 - KY**2, dtype=complex)                   # k-vektorns z-komponent i varje sampelpunkt.
                                                       # dtype=complex tillåter np.sqrt att evaluera till ett komplext tal
    
    fasfaktor_propagation = np.exp(1j*KZ*L) # Faktor för varje sampelpunkt i k-planet
                                           # multas med för att propagera sträckan L i z-led 

    A  = (a**2)/((2*np.pi)**2) *fft2c(E1)                # Planvågsspektrum i Plan 1
    B  = A*fasfaktor_propagation        # Planvågsspektrum i Plan 2 (Planvågsspektrum i Plan 1 multat med fasfaktorn för propagation)
    E2 = delta_k**2 * N**2 *ifft2c(B)
    
    return E2

#%%
## UNITS (microns is ref)
CM = 10000
MM = 1000
NM = 1e-3

r = 1000   # Metasurface radie
P = 200*NM # Period

x = np.arange(-r, r, P)
xx, yy = np.meshgrid(x, x)

N = 2*r/P

n = 1
lam0 = 973*NM
f = 1000

L = f

phi = (2*np.pi - (((2*np.pi*n)/lam0)*(np.sqrt(xx**2 + yy**2 + f**2) - f ))) % 2*np.pi

R = np.sqrt(xx**2 + yy**2)
omega1 = 44 #radius where the intenstiy falls to 1/e of orirignal value

gauss = np.exp(-R**2/omega1**2)

E1 = gauss*phi #phi_discrete_complex_large
I1 = np.abs(E1)**2
phase_profile = np.angle(E1)


E2 = PAS(E1, L, N, P, lam0, n)         # Propagera med vår PAS funktion
I2 = np.abs(E2)**2/np.max(np.abs(E2)**2)    # Intesitet i plan 2

E2_cs = E2[int(N/2), :]
I2_cs = np.abs(E2_cs)**2/np.max( np.abs(E2_cs)**2)

index = np.argmin(abs(I2_cs-1/np.exp(2)))
print(x[index])


#%%

plt.figure(6)
plt.plot(x, I2_cs)
plt.show()

#%%

plt.figure(7)
plt.imshow(I2)
