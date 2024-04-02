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
                    if sum(map(str.isalpha, name)) > 3:
                        column_names.append(name)
                
                data_matrix = np.zeros(shape=(num_lines - header_rows - 1, len(column_names)))
           
            else:
                data_matrix[i - (header_rows + 1),:] = line.split()
                
    data = {}
    for i, name in enumerate(column_names):
        data[name] = data_matrix[:, i]
        
    return data
    
def f0_fit_neg(x, f0, f1, alpha):
    return f0 + f1*x**-alpha

def f0_fit_array_neg(x, popt):
    return f0_fit_neg(x, popt[0], popt[1], popt[2])

def f0_fit_pos(x, f0, f1, alpha):
    return f0 + f1*x**alpha

def f0_fit_array_pos(x, popt):
    return f0_fit_pos(x, popt[0], popt[1], popt[2])

def f0_fit_lin(x, f0, f1):
    return f0 + f1*x

def f0_fit_array_lin(x, popt):
    return f0_fit_lin(x, popt[0], popt[1])

def error_function(popt, y_data):
    return np.abs(y_data - popt[0])/popt[0]

    

#%% C_mesh
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
c_mesh_path = Path(data_path, 'c_mesh.txt')
c_mesh_data = create_data_dict(c_mesh_path)

## Start fitting 
## Create fitted curves for zero cell size
c_mesh_fit = c_mesh_data['c_mesh'][fit_index_start:]
popt_T, pcov = curve_fit(f0_fit_neg, c_mesh_fit, c_mesh_data['transmission'][fit_index_start:])
popt_R, pcov = curve_fit(f0_fit_neg, c_mesh_fit, c_mesh_data['reflection'][fit_index_start:])
popt_T_phi, pcov = curve_fit(f0_fit_neg, c_mesh_fit, c_mesh_data['transmission_phi'][fit_index_start:])
popt_R_phi, pcov = curve_fit(f0_fit_neg, c_mesh_fit, c_mesh_data['reflection_phi'][fit_index_start:])

## Figure objects
fig = plt.figure(figsize=(10,8))
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)
ax_list = [ax1, ax2, ax3, ax4]

## Plot 00 T and R
marker_style = 'x'
ax1.plot(c_mesh_data['c_mesh'], c_mesh_data['transmission'], marker_style, color='blue', label=r'Transmission')
ax1.plot(c_mesh_data['c_mesh'][fit_index_start:], f0_fit_array_neg(c_mesh_data['c_mesh'][fit_index_start:], popt_T), color='black', label=r'Transmission fit')

ax1.plot(c_mesh_data['c_mesh'], c_mesh_data['reflection'],   marker_style, color='red', label=r'Reflectance')
ax1.plot(c_mesh_data['c_mesh'][fit_index_start:], f0_fit_array_neg(c_mesh_data['c_mesh'][fit_index_start:], popt_R), color='black', label=r'Transmission fit')

ax1.set_title(r'Convergence T and R', fontsize=fontsize_title)
ax1.set_ylabel(r'T, R', fontsize=fontsize_axis)

## Plot 10 T and R error
ax3.plot(c_mesh_fit, error_function(popt_T, c_mesh_data['transmission'][fit_index_start:]), color='blue', label=r'$e_T$')
ax3.plot(c_mesh_fit, error_function(popt_R, c_mesh_data['reflection'][fit_index_start:]), color='red', label=r'$e_R$')

ax3.set_title(r'Relative error of T and R', fontsize=fontsize_title)
ax3.set_ylabel(r'$e_T$, $e_R$', fontsize=fontsize_axis)

## Plot 01 T_phi and R_phi
ax2.plot(c_mesh_data['c_mesh'], c_mesh_data['transmission_phi'], marker_style, color='blue', label=r'$T_{\phi}$')
ax2.plot(c_mesh_data['c_mesh'][fit_index_start:], f0_fit_array_neg(c_mesh_data['c_mesh'][fit_index_start:], popt_T_phi), color='black', label=r'Transmission fit')

ax2.plot(c_mesh_data['c_mesh'], c_mesh_data['reflection_phi'],  marker_style, color='Red', label=r'$R_{\phi}$')
ax2.plot(c_mesh_data['c_mesh'][fit_index_start:], f0_fit_array_neg(c_mesh_data['c_mesh'][fit_index_start:], popt_R_phi), color='black', label=r'Transmission fit')

ax2.set_title(r'Convergence $T_{\phi}$ and $R_{\phi}$', fontsize=fontsize_title)
ax2.set_ylabel(r'$T_{\phi}$, $R_{\phi}$', fontsize=fontsize_axis)

## Plot 11 T_phi and R_phi error
ax4.plot(c_mesh_fit, error_function(popt_T_phi, c_mesh_data['transmission_phi'][fit_index_start:]), color='blue', label=r'$e_T$')
ax4.plot(c_mesh_fit, error_function(popt_R_phi, c_mesh_data['reflection_phi'][fit_index_start:]), color='red', label=r'$e_R$')

ax4.set_title(r'Relative error of $T_{\phi}$ and $R_{\phi}$', fontsize=fontsize_title)
ax4.set_ylabel(r'$e_R$, $e_R$', fontsize=fontsize_axis)

plt.tight_layout(pad=outer_pad, w_pad=width_pad,  h_pad=height_pad)

## Add elements to all axis
for ax in ax_list:
    ax.legend(fontsize=fontsize_legend)
    ax.grid(linewidth=1, alpha=0.3)
    ax.set_xlabel(r'$c_{mesh}$', fontsize=fontsize_axis)
    
#%% h_air
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
h_air_path = Path(data_path, 'h_air.txt')
h_air_data = create_data_dict(h_air_path)

h_sub_path = Path(data_path, 'h_sub.txt')
h_sub_data = create_data_dict(h_sub_path)

## Start fitting 
## Create fitted curves for zero cell size
h_air_fit = h_air_data['h_air'][fit_index_start:]
h_sub_fit = h_sub_data['h_sub'][fit_index_start:]
popt_T, pcov = curve_fit(f0_fit_pos, h_air_fit, h_air_data['transmission'][fit_index_start:])
popt_R, pcov = curve_fit(f0_fit_pos, h_air_fit, h_air_data['reflection'][fit_index_start:])
popt_Ts, pcov = curve_fit(f0_fit_lin, h_sub_fit, h_sub_data['transmission'][fit_index_start:])
popt_Rs, pcov = curve_fit(f0_fit_lin, h_sub_fit, h_sub_data['reflection'][fit_index_start:])

## Figure objects
fig = plt.figure(figsize=(10,8))
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)
ax_list = [ax1, ax2, ax3, ax4]

## Plot 00 T and R air
marker_style = 'x'
ax1.plot(h_air_data['h_air'], h_air_data['transmission'], marker_style, color='blue', label=r'Transmission')
ax1.plot(h_air_data['h_air'][fit_index_start:], f0_fit_array_pos(h_air_data['h_air'][fit_index_start:], popt_T), color='black', label=r'Transmission fit')

ax1.plot(h_air_data['h_air'], h_air_data['reflection'],   marker_style, color='red', label=r'Reflectance')
ax1.plot(h_air_data['h_air'][fit_index_start:], f0_fit_array_pos(h_air_data['h_air'][fit_index_start:], popt_R), color='black', label=r'Transmission fit')

ax1.set_title(r'Convergence T and R', fontsize=fontsize_title)
ax1.set_ylabel(r'T, R', fontsize=fontsize_axis)

## Plot 10 T and R error
ax3.plot(h_air_fit, error_function(popt_T, h_air_data['transmission'][fit_index_start:]), color='blue', label=r'$e_T$')
ax3.plot(h_air_fit, error_function(popt_R, h_air_data['reflection'][fit_index_start:]), color='red', label=r'$e_R$')

ax3.set_title(r'Relative error of T and R', fontsize=fontsize_title)
ax3.set_ylabel(r'$e_T$, $e_R$', fontsize=fontsize_axis)

## Plot 01 T and R sub
marker_style = 'x'
ax2.plot(h_sub_data['h_sub'], h_sub_data['transmission'], marker_style, color='blue', label=r'Transmission')
ax2.plot(h_sub_data['h_sub'][fit_index_start:], f0_fit_array_lin(h_sub_data['h_sub'][fit_index_start:], popt_Ts), color='black', label=r'Transmission fit')

ax2.plot(h_sub_data['h_sub'], h_sub_data['reflection'],   marker_style, color='red', label=r'Reflectance')
ax2.plot(h_sub_data['h_sub'][fit_index_start:], f0_fit_array_lin(h_sub_data['h_sub'][fit_index_start:], popt_Rs), color='black', label=r'Transmission fit')

ax2.set_title(r'Convergence T and R', fontsize=fontsize_title)
ax2.set_ylabel(r'T, R', fontsize=fontsize_axis)

## Plot 11 T and R error
ax4.plot(h_sub_fit, error_function(popt_Ts, h_sub_data['transmission'][fit_index_start:]), color='blue', label=r'$e_T$')
ax4.plot(h_sub_fit, error_function(popt_Rs, h_sub_data['reflection'][fit_index_start:]), color='red', label=r'$e_R$')

ax4.set_title(r'Relative error of T and R', fontsize=fontsize_title)
ax4.set_ylabel(r'$e_T$, $e_R$', fontsize=fontsize_axis)


plt.tight_layout(pad=outer_pad, w_pad=width_pad, h_pad=height_pad)

## Add elements to all axis
for ax in ax_list:
    ax.legend(fontsize=fontsize_legend)
    ax.grid(linewidth=1, alpha=0.3)
    ax.set_xlabel(r'$h_{air}$', fontsize=fontsize_axis)
