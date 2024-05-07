#%%
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import curve_fit
from scipy import interpolate

def linear_fit(x, k, m):
    return k*x + m

def quadratic_fit(x, a, b, c):
    return a*x**2 + b*x + c

def exp_fit(x, a, b, c):
    return a*np.exp(b*x) + c

# r_design = np.array([240, 220, 200, 180])
r_actual = np.array([201, 182, 134, 75])/2
h_28_A_cycles        = np.array([739, 691, 643, 600])

r_fit = np.linspace(25, 120, 240)
popt, pcov = curve_fit(quadratic_fit, r_actual, h_28_A_cycles)

quadratic_fit_curve = quadratic_fit(r_fit, popt[0], popt[1], popt[2])

plt.figure(1)
plt.plot(r_actual, h_28_A_cycles, 'rx', label=r'Data points')
plt.plot(r_fit, quadratic_fit_curve, 'black', label=r'Quadratic fit?')
plt.grid(linewidth=1, alpha=0.3)

plt.title(r'Etching for ?? cycles. r=240 nm ')
plt.xlabel(r'r [nm]')
plt.xlabel(r'heigth [nm]')

#%%
project_path = Path(__file__).parent
data_path    = Path(project_path, 'data')

def create_data_dict(file_path):
    with open(str(file_path), "rb") as f:
        num_lines = sum(1 for _ in f)
        
    header_rows  = 4
    column_split  = []
    column_names = []
    with open(file_path) as f:
        for i, line in enumerate(f):
            if i < header_rows:
                pass
            
            elif i == header_rows:
                colum = line.split()
                for name in colum:
                    column_split.append(name)
                
                for name in column_split:  
                    if sum(map(str.isalpha, name)) > 3 or name == 'r':
                        column_names.append(name)
                        
                # print(column_names)
  
                data_matrix = np.zeros(shape=(num_lines - header_rows - 1, len(column_names)))

            else:
                data_matrix[i - (header_rows + 1),:] = line.split()

                
        data = {}
        for i, name in enumerate(column_names):
            data[name] = data_matrix[:, i]
        
    return data

## C_mesh
## Start fitting from
fit_index_start = 10

## Plot parameters
fontsize_title  = 14
fontsize_axis   = 13
fontsize_legend = 10
outer_pad       = 3
width_pad       = 2
height_pad      = 3

## Data
paths = ['rw_sweep_h_500_600.txt', 
         'rw_sweep_h_600_700.txt', 
         'rw_sweep_h_700_800.txt', 
         'rw_sweep_h_800_900.txt', 
         'rw_sweep_h_900_1000.txt', 
         'rw_sweep_h_1000_1100.txt',
         'rw_sweep_h_1100_1200.txt',
         'rw_sweep_h_1200_1300.txt',
         'rw_sweep_h_1300_1400.txt']

for i, path in enumerate(paths):
    rw_sweep_path = Path(data_path, path)
    
    if i == 0:
        ## Parameters for brute force sweep
        rw_sweep_dict = create_data_dict(rw_sweep_path)
        radius = rw_sweep_dict['r']
        h_meta = rw_sweep_dict['h_meta']
        
        ## Unique parameters values
        unique_radius = np.unique(radius)
        unique_h_meta = np.unique(h_meta)
        
        final_h_meta = unique_h_meta
        
        ## Data matrices
        transmission = np.zeros(shape=(len(unique_h_meta), len(unique_radius)))
        phi_t        = np.zeros(shape=(len(unique_h_meta), len(unique_radius)))


        ## Arange data
        for i in range(len(unique_radius)):
            for j in range(len(unique_h_meta)):
                transmission[j, i] = rw_sweep_dict['transmission'][i*len(unique_h_meta) + j]
                phi_t[j, i]        = rw_sweep_dict['transmission_phi'][i*len(unique_h_meta) + j]
        
    else: 
        start_from = 1    
        current_rw_sweep_dict = create_data_dict(rw_sweep_path)
        
        current_radius = current_rw_sweep_dict['r'][start_from:]
        current_h_meta = current_rw_sweep_dict['h_meta'][start_from:]
        
        current_unique_radius = np.unique(current_radius)
        current_unique_h_meta = np.unique(current_h_meta)
        
        final_h_meta = np.concatenate((final_h_meta, current_unique_h_meta))

        ## Data matrices
        current_transmission = np.zeros(shape=(len(current_unique_h_meta), len(current_unique_radius)))
        current_phi_t        = np.zeros(shape=(len(current_unique_h_meta), len(current_unique_radius)))

        ## Arange data
        for i in range(len(current_unique_radius)):
            for j in range(start_from, len(current_unique_h_meta), 1):
                current_transmission[j, i] = current_rw_sweep_dict['transmission'][i*len(current_unique_h_meta) + j]
                current_phi_t[j, i]        = current_rw_sweep_dict['transmission_phi'][i*len(current_unique_h_meta) + j]
                
        phi_t        = np.concatenate((phi_t, current_phi_t[start_from:,:]))
        transmission = np.concatenate((transmission, current_transmission[start_from:,:]))
       

h_meta = final_h_meta
extent = np.array([radius.min(), radius.max(), h_meta.min(), h_meta.max()])

dr = 0.01
r_inter = np.arange(unique_radius.min(), unique_radius.max(), dr)
phi_t_inter = np.zeros(shape=(len(h_meta), len(r_inter)))
transmission_inter = np.zeros(shape=(len(h_meta), len(r_inter)))

for i in range(phi_t.shape[0]):
    
    f_phi = interpolate.interp1d(unique_radius, phi_t[i, :], kind='linear')
    f_t   = interpolate.interp1d(unique_radius, transmission[i, :], kind='linear')
    phi_t_inter[i, :] = f_phi(r_inter)
    transmission_inter[i, :] = f_t(r_inter)

phi_t = phi_t_inter
transmission = transmission_inter

## -------------------------- Select data for plotting -------------------------- ##

experimental_radius = r_fit
experimental_heigth = quadratic_fit_curve 


phi_cut = np.zeros(len(experimental_radius))
t_cut   = np.zeros(len(experimental_radius))

for i in range(len(experimental_radius)):
    r_current = experimental_radius[i]
    h_current = experimental_heigth[i]
    
    r_index = np.argmin(np.abs(r_current - r_inter))
    h_index = np.argmin(np.abs(h_current - h_meta))
    
    phi_cut[i] = phi_t[h_index, r_index]
    t_cut[i] = transmission[h_index, r_index]



fig     = plt.figure(figsize=(10,8))
ax1     = fig.add_subplot(221)
ax2     = fig.add_subplot(222)
ax3     = fig.add_subplot(223)
ax4     = fig.add_subplot(224)
ax_list = [ax1, ax2, ax3, ax4]

## Figure 00 Phase map
ax1.imshow(phi_t, origin='lower', extent=extent, aspect='auto', cmap='Blues', interpolation='None')
ax1.plot(experimental_radius, experimental_heigth, 'black')
ax1.set_xlabel(r'r [nm]', fontsize=fontsize_axis)
ax1.set_ylabel(r'h [nm]', fontsize=fontsize_axis)
ax1.set_title(r'Phase Map', fontsize=fontsize_title)

## Figure 01 Transmission map 
ax2.imshow(transmission, origin='lower', extent=extent, aspect='auto', cmap='plasma', interpolation='None')
ax2.plot(experimental_radius, experimental_heigth, 'black')
ax2.set_xlabel(r'r [nm]', fontsize=fontsize_axis)
ax2.set_ylabel(r'h [nm]', fontsize=fontsize_axis)
ax2.set_title(r'Transmission Map', fontsize=fontsize_title)

## Figure 10 Choosen phi cut
ax3.plot(experimental_radius, phi_cut, 'red')
ax3.grid(linewidth=1, alpha=0.3)
ax3.set_xlabel(r'r [nm]', fontsize=fontsize_axis)
ax3.set_ylabel(r'$\phi$ [rad]', fontsize=fontsize_axis)
# ax3.set_title(r'Phase r: ' + str(round(unique_radius[min_index_r])) + ' - ' + str(round(unique_radius[max_index_r])) + ' nm h: ' + str(round(unique_h_meta[index_height])) + ' nm', fontsize=fontsize_title)

## Figure 11 Choosen transmission cut
ax4.plot(experimental_radius, t_cut, 'black')
ax4.grid(linewidth=1, alpha=0.3)
ax4.set_xlabel(r'r [nm]', fontsize=fontsize_axis)
ax4.set_ylabel(r'Transmission [-]', fontsize=fontsize_axis)
# ax4.set_title(r'Transmission r: ' + str(round(unique_radius[min_index_r])) + ' - ' + str(round(unique_radius[max_index_r])) + ' nm h: ' + str(round(unique_h_meta[index_height])) + ' nm', fontsize=fontsize_title)

plt.tight_layout(pad=outer_pad, w_pad=width_pad,  h_pad=height_pad)