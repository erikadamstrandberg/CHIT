#%%
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


import gdsfactory as gf
gf.clear_cache()
from gdsfactory.generic_tech import get_generic_pdk

PI = np.pi
DEG_TO_RAD = PI/180
RAD_TO_DEG = 180/PI
NM = 1e-3
UM = 1


### Get key with a certain item value in dict
def get_key_with_item_value(dictionary, item_key, item_value):
    for key, value in dictionary.items():
        if item_key in value and value[item_key] == item_value:
            return key
    return None

### Load comsol models from models folder in CHIT
def create_save_folder():
    project_path = Path(__file__).parents[1]
    save_folder_path = Path(project_path, '_output')
    
    if not save_folder_path.exists():
        os.mkdir(save_folder_path)

    return save_folder_path

def load_xlsx(xlsx_filename):
    project_path = Path(__file__).parents[1]
    data_path = Path(project_path, 'data', xlsx_filename)
    
    return pd.read_excel(data_path)
    

### Linear phase gradient to generate angle to dphase map
def linear_phase_gradient_map(X, Y, n, lam0, anglex, angley):
    return 2*PI - 2*PI*n/lam0*(X*np.sin(anglex) + Y*np.sin(angley))

### Generate needed angles from phase gradient
def gradient_to_angle(X, Y, n, lam0, Nnd):
    anglex_array = np.arange(30, 86, 1)
    
    lookup_dict = {}
    dphase_array = np.zeros(len(anglex_array))

    for i in range(len(anglex_array)):
        phase_map = linear_phase_gradient_map(X, Y, n, lam0, anglex_array[i]*DEG_TO_RAD, 0)
        cs_phase_map = phase_map[Nnd//2, :]
        dphase = cs_phase_map[1] - cs_phase_map[0]
        
        lookup_dict[i] = {'dphase' : dphase, 'anglex' : anglex_array[i], 'Pd' : Pd_grating_equation(lam0, anglex_array[i])}
        dphase_array[i] = dphase
    
    return lookup_dict, dphase_array

### Get deflecting cylindrical lens gradient phase map
def deflecting_spherical_lens_gradient_map(X, Y, n, lam0, anglex, angley, f):
    return 2*PI - (2*PI*n/lam0)*(np.sqrt(X**2 + Y**2 + f**2) - f) + 2*PI*n/lam0*(X*np.sin(anglex) + Y*np.sin(angley))

### Grating equation for Pd 
def Pd_grating_equation(lam0, theta):
    return np.round(lam0/(np.sin(theta*DEG_TO_RAD)), 3)

comsol_dataframe = load_xlsx('lens_discretization.xlsx')
comsol_angle = np.array(comsol_dataframe['Deflection angle (deg)'])
comsol_pd    = np.array(comsol_dataframe['P_d, comsol'])
comsol_r     = np.array(comsol_dataframe['r'])
comsol_g1    = np.array(comsol_dataframe['g1'])
comsol_g2    = np.array(comsol_dataframe['g2'])
comsol_g3    = np.array(comsol_dataframe['g3'])
comsol_g4    = np.array(comsol_dataframe['g4'])

### Set size of meta surface 
# L   = 1600*UM
# L_real = 100*UM

L   = 2000*UM
L_real = 250*UM #radie


### Pd does not really matter since it will be reshaped!
Pd  = 240*NM
Pnd = 240*NM
Nd  = int(L/Pd)
Nnd = int(L/Pnd)

x = np.arange(-(Nd/2)*Pd, (Nd/2)*Pd, Pd)
x_for_gen = np.arange(-(Nd/2)*Pd, (Nd/2 + 1)*Pd, Pd)
y = np.arange(-(Nnd/2)*Pnd, (Nnd/2)*Pnd, Pnd)

x_um = x/UM
y_um = y/UM

X, Y = np.meshgrid(x, y)
X_for_gen, Y_for_gen = np.meshgrid(x_for_gen, y)
R    = np.sqrt(X**2 + Y**2)

### Parameters for generated lens
n        = 1
lam0     = 984*NM
anglex   = -60*DEG_TO_RAD
angley   = 0*DEG_TO_RAD
f        = 600*UM
r_offset = -f*np.tan(np.abs(anglex))

(gradient_to_angle_look_up, dphase_array) = gradient_to_angle(X, Y, n, lam0, Nnd)

phase_map = deflecting_spherical_lens_gradient_map(X, Y, n, lam0, anglex, angley, f)
phase_map_for_gen = deflecting_spherical_lens_gradient_map(X_for_gen, Y_for_gen, n, lam0, anglex, angley, f)
cs_phase_map = phase_map_for_gen[Nnd//2, :]
dphase = np.diff(cs_phase_map)

### Generate the needed dphase for needed angles
x_angle_design = np.zeros(len(x))
for i in range(len(dphase)):
    current_dphase = dphase[i]
    current_index = np.argmin(np.abs(current_dphase - dphase_array))
    x_angle_design[i] = gradient_to_angle_look_up[current_index]['anglex']
    
x_angle_design_unique = np.unique(x_angle_design)

plot_phase_map = False
if plot_phase_map:
    fig     = plt.figure(figsize=(10,8))
    ax1     = fig.add_subplot(121)
    extent = [x_um.min(), x_um.max(), y_um.min(), y_um.max()]
    ax1.imshow(np.mod(phase_map_for_gen, 2*np.pi), extent=extent)
    ax2     = fig.add_subplot(122)
    plt.plot(cs_phase_map)
    print('Needed angles: ' + str(x_angle_design_unique))
    

### Find the Fresnel regions 
fresnel_regions = {}
for i, x_angle in enumerate(x_angle_design):
    
    comsol_index = np.argmin(np.abs(comsol_angle - x_angle))
    if not comsol_angle[comsol_index] == x_angle:
        print('Error: Selecting the wrong index in the .xslx-file')
        
    Pd     = comsol_pd[comsol_index]*NM
    g1     = comsol_g1[comsol_index]*NM
    g2     = comsol_g2[comsol_index]*NM
    g3     = comsol_g3[comsol_index]*NM
    g4     = comsol_g4[comsol_index]*NM
    r      = comsol_r[comsol_index]*NM
    region = x[np.where(x_angle_design == x_angle)]
    
    
    number_supercells = np.abs(region.max() - region.min())
    fresnel_regions[x_angle] =  {'from' : np.round(region.min(), 3) - r_offset,
                                 'to' : np.round(region.max(), 3) - r_offset,
                                 'Pd' : np.round(Pd, 3), 
                                 'number' : number_supercells,
                                 'anglex' : x_angle,
                                 'g1' : g1,
                                 'g2' : g2,
                                 'g3' : g3,
                                 'g4' : g4,
                                 'r'  : r}

    
def create_cut_comp(angle, r, g1, g2, g3, g4, radius, width, pnd, radius_ms, layer_dict):
    bool_c = gf.Component('bool')
        
    supercell_ring_c = gf.components.ring(radius=radius, width=width, layer=(layer_dict[key]['layer'], layer_dict[key]['datatype']))
    supercell_ring_r = bool_c << supercell_ring_c
    
    supercell_bool_bot_c = gf.components.rectangle((2*radius_ms, 2*radius_ms),  layer=(layer_dict[key]['layer'], layer_dict[key]['datatype']), centered=True)
    supercell_bool_bot_r = bool_c << supercell_bool_bot_c
    supercell_bool_bot_r.translate(0, -radius_ms).rotate(-180 + angle, (0,0))
    
    supercell_cut_c = gf.geometry.boolean(supercell_ring_r, supercell_bool_bot_r, operation='and', precision=1e-6, layer=(layer_dict[key]['layer'], layer_dict[key]['datatype']))
    
    supercell_bool_bot_rot_r = bool_c << supercell_bool_bot_c
    supercell_bool_bot_rot_r.translate(0, -radius_ms).rotate(-angle, (0,0))
    
    supercell_cut_c = gf.geometry.boolean(supercell_cut_c, supercell_bool_bot_rot_r, operation='and', precision=1e-6, layer=(layer_dict[key]['layer'], layer_dict[key]['datatype']))
    
    circle_cut_c = gf.components.circle(radius=r, layer=(layer_dict[key]['layer'], layer_dict[key]['datatype']))
    left_corner = -radius - width/2 + r
    
    if np.isnan(g4):
        circle_cut_r_g1 = bool_c << circle_cut_c
        circle_cut_r_g1.translate(left_corner + g1, 0)
        supercell_cut_c = gf.geometry.boolean(supercell_cut_c, circle_cut_r_g1, 'not', layer=(layer_dict[key]['layer'], layer_dict[key]['datatype']))
        
        circle_cut_r_g2 = bool_c << circle_cut_c
        circle_cut_r_g2.translate(left_corner + g2, 0)
        supercell_cut_c = gf.geometry.boolean(supercell_cut_c, circle_cut_r_g2, 'not', layer=(layer_dict[key]['layer'], layer_dict[key]['datatype']))
        
        circle_cut_r_g3 = bool_c << circle_cut_c
        circle_cut_r_g3.translate(left_corner + g3, 0)
        supercell_cut_c = gf.geometry.boolean(supercell_cut_c, circle_cut_r_g3, 'not', layer=(layer_dict[key]['layer'], layer_dict[key]['datatype']))
        
    else:
        circle_cut_r_g1 = bool_c << circle_cut_c
        circle_cut_r_g1.translate(left_corner + g1, 0)
        supercell_cut_c = gf.geometry.boolean(supercell_cut_c, circle_cut_r_g1, 'not', layer=(layer_dict[key]['layer'], layer_dict[key]['datatype']))
        
        circle_cut_r_g2 = bool_c << circle_cut_c
        circle_cut_r_g2.translate(left_corner + g2, 0)
        supercell_cut_c = gf.geometry.boolean(supercell_cut_c, circle_cut_r_g2, 'not', layer=(layer_dict[key]['layer'], layer_dict[key]['datatype']))
        
        circle_cut_r_g3 = bool_c << circle_cut_c
        circle_cut_r_g3.translate(left_corner + g3, 0)
        supercell_cut_c = gf.geometry.boolean(supercell_cut_c, circle_cut_r_g3, 'not', layer=(layer_dict[key]['layer'], layer_dict[key]['datatype']))
        
        circle_cut_r_g4 = bool_c << circle_cut_c
        circle_cut_r_g4.translate(left_corner + g4, 0)
        supercell_cut_c = gf.geometry.boolean(supercell_cut_c, circle_cut_r_g4, 'not', layer=(layer_dict[key]['layer'], layer_dict[key]['datatype']))
        
    
    return supercell_cut_c

### ------------------------ Start creating mask ------------------------ ###
save_folder_path = create_save_folder()
ms_name = 'spherical_lens_60'

layer_dict = {'ms'     : {'layer': 1, 'datatype' : 0},
              'labels' : {'layer': 2, 'datatype' : 1}}

component_dict = {}
key = 'ms'
top = gf.Component('TOP') 

end_radius_diff = 0
angles_used_to_create_ms = []
for k, anglex in enumerate(fresnel_regions.keys()):
    bool_c = gf.Component('bool')
    
    Pd     = fresnel_regions[anglex]['Pd']
    g1     = fresnel_regions[anglex]['g1']
    g2     = fresnel_regions[anglex]['g2']
    g3     = fresnel_regions[anglex]['g3']
    g4     = fresnel_regions[anglex]['g4']
    r      = fresnel_regions[anglex]['r']
    region_to = fresnel_regions[anglex]['to']
    region_from = fresnel_regions[anglex]['from']
    
    region_width   = np.abs(region_to - region_from)
    region_radius  = np.abs((region_to + region_from))/2
    region_radius_end = (region_to + region_from)/2
    region_radius_circ = np.pi*region_radius
    
    if k == 0:
        start_from = region_from
        
    if k > 0:
        start_from = region_from + end_radius_diff - 0.24

    angle_RAD = 2*np.arcsin(Pnd/(2*np.abs(region_to)))
    angle_DEG = angle_RAD*RAD_TO_DEG
    
    number_cell_r = region_width/Pd
    number_cell_theta = 360/angle_DEG
    number_cells_added = int(number_cell_theta)
    
    angles_missed = angle_RAD/(number_cell_theta - number_cells_added)
    angles_to_add = angles_missed/(2*number_cells_added)
    
    angle_RAD = angle_RAD + angles_to_add
    angle_DEG = angle_RAD*RAD_TO_DEG
    
    for i in range(int(number_cell_r)):
        current_radius = start_from + i*Pd + Pd/2
        current_width  = Pd
    
        super_cell = create_cut_comp(angle_DEG/2, r, g1, g2, g3, g4, np.abs(current_radius), current_width, Pnd, L, layer_dict)
    
        for j in range(number_cells_added):            
            current_x = r_offset - current_radius*np.cos(angle_RAD*j)
            current_y = current_radius*np.sin(angle_RAD*j)
            current_r = np.sqrt(current_x**2 + current_y**2)

            if current_r < L_real and current_x < L_real:
                print('Added' + ' with angle: ' + str(anglex) + ' with ' + str(number_cells_added) + ' cells')
                angles_used_to_create_ms.append(anglex)
                super_cell_r = top << super_cell
                super_cell_r.rotate(np.pi*RAD_TO_DEG + angle_DEG*j, (0, 0)).translate(-r_offset, 0)
    
    end_radius_diff = current_radius - region_to + Pd/2
    
angles_used_to_create_ms = np.unique(np.array(angles_used_to_create_ms))

print('Angles used to create phase profile: ' + str(x_angle_design_unique))
print('Angles used In the cutout MS: ' + str(x_angle_design_unique))


save_path = Path(save_folder_path, ms_name)
top.write_gds(str(save_path) + '.gds')
 
#%%



