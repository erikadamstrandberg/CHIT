#%%
import os
import numpy as np
from pathlib import Path
import mph

PI = np.pi

### Load comsol models from models folder in CHIT
def load_3d_comsol_models(comsol_model):
    project_path = Path(__file__).parents[1]
    comsol_model_path    = Path(project_path, '3d_models', comsol_model)

    return comsol_model_path

comsol_model = 'Supercell_holes_3holes_35deg_com60.mph'
model_to_load = load_3d_comsol_models(comsol_model)

client = mph.start()
model = client.load(model_to_load)

#%% Will clear all solutions from the model in the mph environment

# model.clear()
# model.reset()

#%% Load port variables and save arrays. Maybe make a nicer structure?

T_0_x  = 'ewfd.Torder_0_0'
T_0_y  = 'ewfd.Torder_0_0_orth'
T__1_x = 'ewfd.Torder_n1_0_ip'
T__1_y = 'ewfd.Torder_n1_0_op'
T_1_x  = 'ewfd.Torder_p1_0_ip'  
T_1_y  = 'ewfd.Torder_p1_0_op'
R_0_x  = 'ewfd.Rorder_0_0'
R_0_y  = 'ewfd.Rorder_0_0_orth'
R__1_x = 'ewfd.Rorder_n1_0_ip'
R__1_y = 'ewfd.Rorder_n1_0_op'
R_1_x  = 'ewfd.Rorder_p1_0_ip'  
R_1_y  = 'ewfd.Rorder_p1_0_op'

ports_eval = [T_0_x, T_0_y, T__1_x, T__1_y, T_1_x, T_1_y, R_0_x, R_0_y, R__1_x, R__1_y, R_1_x, R_1_y]

E_x_array = np.zeros(len(ports_eval))
E_y_array = np.zeros(len(ports_eval))

#%% Run Study 1 with x pol light

model.parameter('E_x', '1')
model.parameter('E_y', '0')
model.parameters()

model.mesh()
model.solve('Study 1')

for i, eval_name in enumerate(ports_eval):
    E_x_array[i] = model.evaluate(eval_name)

#%% Run Study 1 with y pol light

model.parameter('E_x', '0')
model.parameter('E_y', '1')
model.parameters()

model.mesh()
model.solve('Study 1')

#%%
for i, eval_name in enumerate(ports_eval):
    E_y_array[i] = model.evaluate(eval_name)



#%% Compare output

for i in range(len(E_y_array)):
    print(E_x_array[i] - E_y_array[i])
    
