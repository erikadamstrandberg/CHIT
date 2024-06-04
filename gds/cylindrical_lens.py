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
def gradient_to_angle(X, Y, n, lam0, Nnd, phase_discritazation):
    if phase_discritazation:
        anglex_array = np.arange(30, 86, 0.2)
    else:
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
def deflecting_cylindrical_lens_gradient_map(X, Y, n, lam0, anglex, angley, f):
    return 2*PI - (2*PI*n/lam0)*(np.sqrt(X**2 + f**2) - f) + 2*PI*n/lam0*(X*np.sin(anglex) + Y*np.sin(angley))

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
ms_diameter = 300*UM


### Pd does not really matter since it will be reshaped!
Pd  = 50*NM
Pnd = 240*NM
Nd  = int(ms_diameter/Pd)
Nnd = int(ms_diameter/Pnd)

x         = np.arange(-(Nd/2)*Pd, (Nd/2)*Pd, Pd)
x_for_gen = np.arange(-(Nd/2)*Pd, (Nd/2 + 1)*Pd, Pd)
y         = np.arange(-(Nnd/2)*Pnd, (Nnd/2)*Pnd, Pnd)

x_um = x/UM
y_um = y/UM

X, Y                 = np.meshgrid(x, y)
X_for_gen, Y_for_gen = np.meshgrid(x_for_gen, y)
R                    = np.sqrt(X**2 + Y**2)

### Important parameters
focal_plane = 800*UM
anglex      = -61*DEG_TO_RAD

### Fab correction
r_scale_factor = 115/110


### Parameters for generated lens
n            = 1
lam0         = 984*NM
angley       = 0
f            = focal_plane/np.cos(np.abs(anglex))
r_offset     = -f*np.sin(np.abs(anglex))

phase_discritazation = True

(gradient_to_angle_look_up, dphase_array) = gradient_to_angle(X, Y, n, lam0, Nnd, phase_discritazation)
phase_map         = deflecting_cylindrical_lens_gradient_map(X, Y, n, lam0, anglex, angley, f)
phase_map_for_gen = deflecting_cylindrical_lens_gradient_map(X_for_gen, Y_for_gen, n, lam0, anglex, angley, f)
cs_phase_map      = phase_map_for_gen[Nnd//2, :]
dphase            = np.diff(cs_phase_map)

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
    ax1.imshow(np.mod(phase_map_for_gen, 2*np.pi))
    ax2     = fig.add_subplot(122)
    plt.plot(cs_phase_map)
    print('Needed angles: ' + str(x_angle_design_unique))
    
  
### Find the Fresnel regions 
fresnel_regions = {}
x_reshaped = np.zeros(1)
for i, x_angle_current in enumerate(x_angle_design):
    
    if phase_discritazation:
        if np.isclose(x_angle_current, np.round(x_angle_current), 0.0001):
            x_angle_current = round(x_angle_current)
            comsol_index = np.argmin(np.abs(comsol_angle - x_angle_current))
                
            Pd     = comsol_pd[comsol_index]*NM
            g1     = comsol_g1[comsol_index]*NM
            g2     = comsol_g2[comsol_index]*NM
            g3     = comsol_g3[comsol_index]*NM
            g4     = comsol_g4[comsol_index]*NM
            r      = comsol_r[comsol_index]*NM*r_scale_factor
            
            
            # print(np.isclose(x_angle_design, x_angle_current))
            # region = x[np.where(x_angle_design == x_angle_current)]
            # print(region)
            region = x[np.isclose(x_angle_design, x_angle_current, 0.0001)]
            
            # print(np.where(np.isclose(x_angle_design, x_angle_current, 0.0001)))

            number_supercells = int(np.abs(region.max() - region.min())/Pd)
            
            if not comsol_angle[comsol_index] == x_angle_current:
                print('Error: Selecting the wrong index in the .xslx-file')
                print('Could not find: ' + str(x_angle_current))
            else:
                fresnel_regions[x_angle_current] =  {'from'   : np.round(region.min() - r_offset, 3),
                                                      'to'     : np.round(region.max() - r_offset, 3),
                                                      'Pd'     : np.round(Pd, 3), 
                                                      'number' : number_supercells,
                                                      'anglex' : x_angle_current,
                                                      'g1'     : g1,
                                                      'g2'     : g2,
                                                      'g3'     : g3,
                                                      'g4'     : g4,
                                                      'r'      : r}
        else:
            
            
            x_angle_current_above = np.ceil(x_angle_current)
            x_angle_current_below = np.floor(x_angle_current)
            comsol_index_above = np.argmin(np.abs(comsol_angle - x_angle_current_above))
            comsol_index_below = np.argmin(np.abs(comsol_angle - x_angle_current_below))
            
            Pd_above     = comsol_pd[comsol_index_above]*NM
            g1_above     = comsol_g1[comsol_index_above]*NM
            g2_above     = comsol_g2[comsol_index_above]*NM
            g3_above     = comsol_g3[comsol_index_above]*NM
            g4_above     = comsol_g4[comsol_index_above]*NM
            
            Pd_below     = comsol_pd[comsol_index_below]*NM
            g1_below     = comsol_g1[comsol_index_below]*NM
            g2_below     = comsol_g2[comsol_index_below]*NM
            g3_below     = comsol_g3[comsol_index_below]*NM
            g4_below     = comsol_g4[comsol_index_below]*NM
            r            = comsol_r[comsol_index_below]*NM*r_scale_factor
            
            scaling = x_angle_current - np.floor(x_angle_current)
            
            Pd = Pd_above*scaling + Pd_below*(1 - scaling)
            g1 = g1_above*scaling + g1_below*(1 - scaling)
            g2 = g2_above*scaling + g2_below*(1 - scaling)
            g3 = g3_above*scaling + g3_below*(1 - scaling)
            g4 = g4_above*scaling + g4_below*(1 - scaling)
            
            region = x[np.isclose(x_angle_design, x_angle_current, 0.001)]
            number_supercells = int(np.abs(region.max() - region.min())/Pd)
            
            fresnel_regions[x_angle_current] =  {'from'   : np.round(region.min() - r_offset, 3),
                                                  'to'     : np.round(region.max() - r_offset, 3),
                                                  'Pd'     : np.round(Pd, 3), 
                                                  'number' : number_supercells,
                                                  'anglex' : x_angle_current,
                                                  'g1'     : g1,
                                                  'g2'     : g2,
                                                  'g3'     : g3,
                                                  'g4'     : g4,
                                                  'r'      : r}
            
            
            
            
            
    else:
    
        comsol_index = np.argmin(np.abs(comsol_angle - x_angle_current))
            
        Pd     = comsol_pd[comsol_index]*NM
        g1     = comsol_g1[comsol_index]*NM
        g2     = comsol_g2[comsol_index]*NM
        g3     = comsol_g3[comsol_index]*NM
        g4     = comsol_g4[comsol_index]*NM
        r      = comsol_r[comsol_index]*NM*r_scale_factor
        
        region = x[np.where(x_angle_design == x_angle_current)]
        number_supercells = int(np.abs(region.max() - region.min())/Pd)
        
        if not comsol_angle[comsol_index] == x_angle_current:
            print('Error: Selecting the wrong index in the .xslx-file')
            print('Could not find: ' + str(x_angle_current))
        else:
            fresnel_regions[x_angle_current] =  {'from'   : np.round(region.min() - r_offset, 3),
                                                 'to'     : np.round(region.max() - r_offset, 3),
                                                 'Pd'     : np.round(Pd, 3), 
                                                 'number' : number_supercells,
                                                 'anglex' : x_angle_current,
                                                 'g1'     : g1,
                                                 'g2'     : g2,
                                                 'g3'     : g3,
                                                 'g4'     : g4,
                                                 'r'      : r}
    
plot_fresnel_regions = False
if plot_fresnel_regions:
    for key in fresnel_regions.keys():
        print('------------------')
        print('For angle: ' + str(fresnel_regions[key]['anglex']))
        print('Phase-map radius: ' + str((fresnel_regions[key]['to'] + fresnel_regions[key]['from'])/2))
        r_doublechecking = f*np.sin(fresnel_regions[key]['anglex']*DEG_TO_RAD)
        print('Trigonometry radius: ' + str(r_doublechecking))  

### Reshape the x-sampling
for i, key in enumerate(fresnel_regions.keys()):
    region_from = fresnel_regions[key]['from']
    region_to   = fresnel_regions[key]['to']
    if int(np.floor(fresnel_regions[key]['number'])) == 0:
        region_num = 1
    else:
        region_num  = int(np.floor(fresnel_regions[key]['number'])) + 1
        
    region_pd   = fresnel_regions[key]['Pd']
    angle       = fresnel_regions[key]['anglex']

    if i == 0:
        x_reshaped_initial = np.zeros(region_num)
        
        for i in range(region_num):
            x_reshaped_initial[i] = region_from + i*region_pd
        
        angle_reshape_initial = np.ones(len(x_reshaped_initial))*angle
        
        x_reshaped    = x_reshaped_initial
        angle_reshape = angle_reshape_initial

        next_start = x_reshaped_initial[-1]
        last_region_pd = region_pd
        
    else:
        next_start = next_start + last_region_pd 
        x_reshaped_initial = np.zeros(region_num)
        
        for i in range(region_num):
            x_reshaped_initial[i] = next_start + i*region_pd
        
        angle_reshape_initial = np.ones(len(x_reshaped_initial))*angle
        
        x_reshaped = np.concatenate((x_reshaped, x_reshaped_initial))
        angle_reshape = np.concatenate((angle_reshape, angle_reshape_initial))
    
        next_start = x_reshaped_initial[-1]
        last_region_pd = region_pd


X_reshaped, Y_reshaped = np.meshgrid(x_reshaped, y)
R_reshaped             = np.sqrt(X_reshaped**2 + Y_reshaped**2)

### ------------------------ Start creating mask ------------------------ ###
save_folder_path = create_save_folder()

ms_name = 'cylindrical_lens_f' + str(round(focal_plane)) + '_d' + str(round(np.abs(anglex)*RAD_TO_DEG)) + '_r' + str(r_scale_factor)

layer_dict = {'ms'     : {'layer': 1, 'datatype' : 0},
              'labels' : {'layer': 2, 'datatype' : 1}}

component_dict = {}
key = 'ms'

for anglex in fresnel_regions.keys():
    bool_c = gf.Component('bool')
    
    Pd     = fresnel_regions[anglex]['Pd']
    g1     = fresnel_regions[anglex]['g1']
    g2     = fresnel_regions[anglex]['g2']
    g3     = fresnel_regions[anglex]['g3']
    g4     = fresnel_regions[anglex]['g4']
    r      = fresnel_regions[anglex]['r']

    supercell_c  = gf.components.rectangle((Pd, Pnd), layer=(layer_dict[key]['layer'], layer_dict[key]['datatype']))
    supercell_r  = bool_c << supercell_c
    
    if np.isnan(g4):
        circle_cut_c = gf.components.circle(radius=r, layer=(layer_dict[key]['layer'], layer_dict[key]['datatype']))
        
        circle_cut_r_g1 = bool_c << circle_cut_c
        circle_cut_r_g1.translate(Pd - r - g1, Pnd/2)
        supercell_bool = gf.geometry.boolean(supercell_r, circle_cut_r_g1, 'not', layer=(layer_dict[key]['layer'], layer_dict[key]['datatype']))
        
        circle_cut_r_g2 = bool_c << circle_cut_c
        circle_cut_r_g2.translate(Pd - r - g2, Pnd/2)
        supercell_bool = gf.geometry.boolean(supercell_bool, circle_cut_r_g2, 'not', layer=(layer_dict[key]['layer'], layer_dict[key]['datatype']))
        
        circle_cut_r_g3 = bool_c << circle_cut_c
        circle_cut_r_g3.translate(Pd - r - g3, Pnd/2)
        supercell_bool = gf.geometry.boolean(supercell_bool, circle_cut_r_g3, 'not', layer=(layer_dict[key]['layer'], layer_dict[key]['datatype']))
        
        fresnel_regions[anglex].update({'comp' : supercell_bool})
        
    else:
        circle_cut_c = gf.components.circle(radius=r, layer=(layer_dict[key]['layer'], layer_dict[key]['datatype']))
        
        circle_cut_r_g1 = bool_c << circle_cut_c
        circle_cut_r_g1.translate(Pd - r - g1, Pnd/2)
        supercell_bool = gf.geometry.boolean(supercell_r, circle_cut_r_g1, 'not', layer=(layer_dict[key]['layer'], layer_dict[key]['datatype']))
        
        circle_cut_r_g2 = bool_c << circle_cut_c
        circle_cut_r_g2.translate(Pd - r - g2, Pnd/2)
        supercell_bool = gf.geometry.boolean(supercell_bool, circle_cut_r_g2, 'not', layer=(layer_dict[key]['layer'], layer_dict[key]['datatype']))
        
        circle_cut_r_g3 = bool_c << circle_cut_c
        circle_cut_r_g3.translate(Pd - r - g3, Pnd/2)
        supercell_bool = gf.geometry.boolean(supercell_bool, circle_cut_r_g3, 'not', layer=(layer_dict[key]['layer'], layer_dict[key]['datatype']))
        
        circle_cut_r_g4 = bool_c << circle_cut_c
        circle_cut_r_g4.translate(Pd - r - g4, Pnd/2)
        supercell_bool = gf.geometry.boolean(supercell_bool, circle_cut_r_g4, 'not', layer=(layer_dict[key]['layer'], layer_dict[key]['datatype']))
        
        
        fresnel_regions[anglex].update({'comp' : supercell_bool})
        
        
print('-----------------------------------')
print('Generating cylindrical lens with:')
for key in fresnel_regions.keys():
    print('Angle: ' + str(key) + ' with: ' + str(np.ceil(fresnel_regions[key]['number'])) + ' superunitcells')
            
    
top = gf.Component('TOP')    
for i in range(len(x_reshaped)):
    
    angle_current_x = angle_reshape[i]
    current_comp = fresnel_regions[angle_current_x]['comp']
    if i%10 == 0:
        print('Placing row: ' + str(i) + ' of ' + str(len(x_reshaped)) + ' with angle: '  + str(round(angle_current_x)))
        
    for j in range(len(y)):
        current_x = x_reshaped[i] + r_offset
        current_y = y[j]
        current_r = np.sqrt(current_x**2 + current_y**2)
        
        if current_r < ms_diameter/2:
            c_outer_r = top << current_comp
            c_outer_r.translate(x_reshaped[i] + r_offset, y[j]).rotate(180, (0, 0))
      
save_path = Path(save_folder_path, ms_name)
top.write_gds(str(save_path) + '.gds')
 
