#%%
import numpy as np
from image_functions import MEL_9000_k_images, list_image_path

PI = np.pi

## Data path
measurment_name = 'GaAs3_1'
what_measurement = 0

image_path = list_image_path(measurment_name, what_measurement)
image = MEL_9000_k_images(image_path)

middle = (-0.05, -0.25)
# GaAs1_4_image.plot_image()
size = 1

y_pos = 100e-4
y_pos_mid = -150e-4
y_pos_top = 8300e-4
y_pos_bottom = -8900e-4

integrate_over = 60
n = 1
angle = 61
angle_rad = angle*PI/180
image.rotate_image(-2.5)
image.remove_background(0)
image.set_image_bounds(middle, size)


contrast = 20
brightness = 10
# image.plot_image(contrast, brightness)
image.plot_image_and_x_cs(angle_rad, n, y_pos, integrate_over)
# 
# GaAs1_4_image.plot_3_areas(x_pos, y_pos_mid, y_pos_top, y_pos_bottom, integrate_over)

