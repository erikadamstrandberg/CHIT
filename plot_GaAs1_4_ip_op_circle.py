#%%
import numpy as np
from image_functions import MEL_9000_k_images, list_image_path

PI = np.pi

## Data path
measurment_name = 'GaAs1_4_ref_ip_op'
what_measurement = 0

image_path = list_image_path(measurment_name, what_measurement)
GaAs1_4_image = MEL_9000_k_images(image_path)

middle = (-0.11, -0.25)
# GaAs1_4_image.plot_image()
size = 1

x_pos = 100e-4
y_pos_mid = -150e-4
y_pos_top = 8300e-4
y_pos_bottom = -8900e-4

integrate_over = 60
n = 1
angle = PI/3
GaAs1_4_image.rotate_image(-0.4)
GaAs1_4_image.remove_background(2.8)
GaAs1_4_image.set_image_bounds(middle, size)
#GaAs1_4_image.plot_image()


GaAs1_4_image.plot_3_areas(x_pos, y_pos_mid, y_pos_top, y_pos_bottom, integrate_over)
#GaAs1_4_image.plot_image_and_cs(angle, n,  x_pos, integrate_over)

