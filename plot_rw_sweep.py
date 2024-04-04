#%%

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import curve_fit

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
# paths = ['rw_sweep_h_500_600.txt', 'rw_sweep_h_600_700.txt']
paths = ['rw_sweep_h_500_600.txt']
for i, path in enumerate(paths):
    rw_sweep_path = Path(data_path, path)
    
    if i == 0:
        ## Parameters for brute force sweep
        rw_sweep_dict = create_data_dict(rw_sweep_path)
        radius = rw_sweep_dict['r']
        h_meta = rw_sweep_dict['h_meta']
        
        
    else: 
        rw_sweep_dict_temp = create_data_dict(rw_sweep_path)
        radius = np.concatenate((radius, rw_sweep_dict_temp['r']))
        h_meta = np.concatenate((h_meta, rw_sweep_dict_temp['h_meta']))
        
        rw_sweep_dict['reflection']       = np.concatenate((rw_sweep_dict['reflection'], rw_sweep_dict_temp['reflection']))
        rw_sweep_dict['transmission']     = np.concatenate((rw_sweep_dict['transmission'], rw_sweep_dict_temp['transmission']))
        rw_sweep_dict['reflection_phi']   = np.concatenate((rw_sweep_dict['reflection_phi'], rw_sweep_dict_temp['reflection_phi']))
        rw_sweep_dict['transmission_phi'] = np.concatenate((rw_sweep_dict['transmission_phi'], rw_sweep_dict_temp['transmission_phi']))
    
        
    ## Unique parameters values
    unique_radius = np.unique(radius)
    unique_h_meta = np.unique(h_meta)
    
    ## Data matrices
    transmission = np.zeros(shape=(len(unique_h_meta), len(unique_radius)))
    phi_t        = np.zeros(shape=(len(unique_h_meta), len(unique_radius)))


## Arange data
for i in range(len(unique_radius)):
    for j in range(len(unique_h_meta)):
        transmission[j, i] = rw_sweep_dict['transmission'][i*len(unique_h_meta) + j]
        phi_t[j, i]        = rw_sweep_dict['transmission_phi'][i*len(unique_h_meta) + j]


extent = np.array([radius.min(), radius.max(), h_meta.min(), h_meta.max()])
    
## -------------------------- Select data for plotting -------------------------- ##
pick_radius = np.array([70, 120])    
pick_height = 650

if pick_radius[0] < unique_radius.min() or pick_radius[1] > unique_radius.max():
    print('Warning: Picked radiie is out of data range!')
    
## Plot it
min_index_r  = np.argmin(np.abs(unique_radius - pick_radius[0]))
max_index_r  = np.argmin(np.abs(unique_radius - pick_radius[1]))
radius_array = unique_radius[min_index_r:max_index_r + 1]
index_height = np.argmin(np.abs(unique_h_meta - pick_height))

t_cut   = transmission[index_height, min_index_r:max_index_r+1]
phi_cut = phi_t[index_height, min_index_r:max_index_r+1]

fig     = plt.figure(figsize=(10,8))
ax1     = fig.add_subplot(221)
ax2     = fig.add_subplot(222)
ax3     = fig.add_subplot(223)
ax4     = fig.add_subplot(224)
ax_list = [ax1, ax2, ax3, ax4]

## Figure 00 Phase map
ax1.imshow(phi_t, origin='lower', extent=extent, aspect='auto', cmap='Blues')
ax1.plot(radius_array, np.ones(len(radius_array))*pick_height, 'black')
ax1.set_xlabel(r'r [nm]', fontsize=fontsize_axis)
ax1.set_ylabel(r'h [nm]', fontsize=fontsize_axis)
ax1.set_title(r'Phase Map', fontsize=fontsize_title)

## Figure 01 Transmission map 
ax2.imshow(transmission, origin='lower', extent=extent, aspect='auto', cmap='plasma')
ax2.plot(radius_array, np.ones(len(radius_array))*pick_height, 'black')
ax2.set_xlabel(r'r [nm]', fontsize=fontsize_axis)
ax2.set_ylabel(r'h [nm]', fontsize=fontsize_axis)
ax2.set_title(r'Transmission Map', fontsize=fontsize_title)

## Figure 10 Choosen phi cut
ax3.plot(radius_array, phi_cut, 'red')
ax3.grid(linewidth=1, alpha=0.3)
ax3.set_xlabel(r'r [nm]', fontsize=fontsize_axis)
ax3.set_ylabel(r'$\phi$ [rad]', fontsize=fontsize_axis)
ax3.set_title(r'Phase r: ' + str(round(unique_radius[min_index_r])) + ' - ' + str(round(unique_radius[max_index_r])) + ' nm h: ' + str(round(unique_h_meta[index_height])) + ' nm', fontsize=fontsize_title)

## Figure 11 Choosen transmission cut
ax4.plot(radius_array, t_cut, 'black')
ax4.grid(linewidth=1, alpha=0.3)
ax4.set_xlabel(r'r [nm]', fontsize=fontsize_axis)
ax4.set_ylabel(r'Transmission [-]', fontsize=fontsize_axis)
ax4.set_title(r'Transmission r: ' + str(round(unique_radius[min_index_r])) + ' - ' + str(round(unique_radius[max_index_r])) + ' nm h: ' + str(round(unique_h_meta[index_height])) + ' nm', fontsize=fontsize_title)

plt.tight_layout(pad=outer_pad, w_pad=width_pad,  h_pad=height_pad)
