#%%
#import numpy as np
#from image_functions import MEL_9000_k_images, list_image_path

import os
from pathlib import Path

mypath = Path(Path(__file__).resolve().parents[0])

os.chdir(str(mypath))

#%%
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage
from pathlib import Path

PI = np.pi

def list_image_path(measurement, what_measurement):
    project_path = Path(__file__).parent
    measurement_path    = Path(project_path, 'measurement', measurement)

    images  = os.listdir(measurement_path)
    image_paths = Path(measurement_path, images[what_measurement])
    
    return image_paths
    
## Class for analysin images
class MEL_9000_k_images():
    
    ## Initialize image
    def __init__(self, image_path):
        ## Pixel size in k-space
        self.pixel_size_k = 1.4e-3
        
        self.image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        self.pixels = self.image.shape
    
        self.x_axis = np.arange(-self.pixels[0]*self.pixel_size_k/2, 
                                 self.pixels[0]*self.pixel_size_k/2, 
                                 self.pixel_size_k)
        self.y_axis = np.arange(-self.pixels[1]*self.pixel_size_k/2, 
                                 self.pixels[1]*self.pixel_size_k/2, 
                                 self.pixel_size_k)
        
    ## Rescale axis for better scales
    def rescale_axis(self, constant):
        self.x_axis = self.x_axis/constant
        self.y_axis = self.y_axis/constant
        
    ## Centre image on middle of beam
    def set_image_bounds(self, middle, size):
        size_indx = int(size/self.pixel_size_k)
        mid_x_indx = np.argmin(np.abs(self.x_axis - middle[0]))
        mid_y_indx = np.argmin(np.abs(self.y_axis - middle[1]))
        
        self.x_axis = np.arange(-size, size, 
                                self.pixel_size_k)
        self.y_axis = np.arange(-size, size, 
                                self.pixel_size_k)
        
        self.image = self.image[mid_y_indx - size_indx:mid_y_indx + size_indx,
                                mid_x_indx - size_indx:mid_x_indx + size_indx]
        
    def remove_background(self, background):
        self.image = self.image - background
        
    ## Rotate image
    def rotate_image(self, angle):
        self.image = ndimage.rotate(self.image, angle, reshape=False)
        
    ## Plot raw image
    def plot_image(self, contrast=None, brightness=None):
        fig     = plt.figure(figsize=(10,8))
        ax1     = fig.add_subplot(111)
        extent = np.array([self.x_axis.min(), self.x_axis.max(),
                            self.y_axis.min(), self.y_axis.max()])
        # NA = 0.95
        # theta = np.linspace(0, 2*PI, 1000)
        # outer_bound_k_x = NA*np.cos(theta)
        # outer_bound_k_y = NA*np.sin(theta)
        if contrast and brightness:
            self.image = cv2.convertScaleAbs(self.image, alpha=contrast, beta=brightness)
        
        ax1.imshow(self.image, extent=extent, origin='lower', cmap='plasma')
        # ax1.plot(outer_bound_k_x, outer_bound_k_y, 'black')
        
    ## Plot raw image and select a cross-section to integrate over
    def plot_image_and_y_cs(self, angle, n,  x_pos, integrate_over):
        
        mid_x_indx = np.argmin(np.abs(self.x_axis - x_pos))
        y_cs = self.image[:, mid_x_indx - integrate_over:mid_x_indx + integrate_over]
        y_cs_integrated = np.sum(y_cs, axis=1)
        
        cs_x_axis = self.x_axis[mid_x_indx - integrate_over:mid_x_indx + integrate_over]
        
        NA_outer = 0.95
        theta = np.linspace(0, 2*PI, 1000)
        outer_bound_k_x = NA_outer*np.cos(theta)
        outer_bound_k_y = NA_outer*np.sin(theta)
        
        NA_diffracted = n*np.sin(angle)
        theta = np.linspace(0, 2*PI, 1000)
        diffracted_k_x = NA_diffracted*np.cos(theta)
        diffracted_k_y = NA_diffracted*np.sin(theta)
        
        fontsize_title  = 14
        fontsize_axis   = 13
        fontsize_legend = 10
        outer_pad       = 3
        width_pad       = 2
        height_pad      = 3
        
        fig     = plt.figure(figsize=(10,8))
        ax1     = fig.add_subplot(221)
        ax2     = fig.add_subplot(222)
        ax3     = fig.add_subplot(223)
        ax_list = [ax1, ax2, ax3]
        
        extent    = np.array([self.x_axis.min(), self.x_axis.max(),
                              self.y_axis.min(), self.y_axis.max()])
        extent_cs = np.array([cs_x_axis.min(), cs_x_axis.max(),
                              self.y_axis.min(), self.y_axis.max()])
        
        ## Figure 00 - Raw image with outer bound k and wanted k
        ax1.imshow(self.image, extent=extent, origin='lower', cmap='plasma')
        ax1.plot(outer_bound_k_x, outer_bound_k_y, 'black')
        ax1.plot(diffracted_k_x, diffracted_k_y, '--', color='red')
        ax1.set_xlabel(r'$k_x/k$', fontsize=fontsize_axis)
        ax1.set_ylabel(r'$k_y/k$', fontsize=fontsize_axis)
        ax1.set_title(r'Raw image', fontsize=fontsize_title)
        
        ## Figure 01 - Selected cross-section to integrate over
        ax2.imshow(y_cs, extent=extent_cs, origin='lower', cmap='plasma')
        ax2.set_xlabel(r'$k_x/k$', fontsize=fontsize_axis)
        ax2.set_ylabel(r'$k_y/k$', fontsize=fontsize_axis)
        ax2.set_title(r'Cut section to intergrate over', fontsize=fontsize_title)
        
        ## Figure 10 - Integrated cross-section
        ax3.plot(self.x_axis[0:-1], y_cs_integrated, 'black')
        ax3.grid(linewidth=1, alpha=0.3)
        ax3.set_xlabel(r'$k_x/k$', fontsize=fontsize_axis)
        ax3.set_ylabel(r'Counts [-]', fontsize=fontsize_axis)
        ax3.set_title(r'Integrated cross-section', fontsize=fontsize_title)
        
        plt.tight_layout(pad=outer_pad, w_pad=width_pad, h_pad=height_pad)
        
    def plot_image_and_x_cs(self, angle, n, y_pos, integrate_over):
        
        mid_y_indx = np.argmin(np.abs(self.y_axis - y_pos))
        x_cs = self.image[mid_y_indx - integrate_over:mid_y_indx + integrate_over, :]
        x_cs_integrated = np.sum(x_cs, axis=0)
        
        cs_y_axis = self.y_axis[mid_y_indx - integrate_over:mid_y_indx + integrate_over]
        
        NA_outer = 0.95
        theta = np.linspace(0, 2*PI, 1000)
        outer_bound_k_x = NA_outer*np.cos(theta)
        outer_bound_k_y = NA_outer*np.sin(theta)
        
        NA_diffracted = n*np.sin(angle)
        theta = np.linspace(0, 2*PI, 1000)
        diffracted_k_x = NA_diffracted*np.cos(theta)
        diffracted_k_y = NA_diffracted*np.sin(theta)
        
        fontsize_title  = 14
        fontsize_axis   = 13
        fontsize_legend = 10
        outer_pad       = 3
        width_pad       = 2
        height_pad      = 3
        
        fig     = plt.figure(1, figsize=(10,8))
        ax1     = fig.add_subplot(221)
        ax2     = fig.add_subplot(222)
        ax3     = fig.add_subplot(223)
        ax_list = [ax1, ax2, ax3]
        
        extent    = np.array([self.x_axis.min(), self.x_axis.max(),
                              self.y_axis.min(), self.y_axis.max()])
        extent_cs = np.array([self.x_axis.min(), self.x_axis.max(),
                              cs_y_axis.min(), cs_y_axis.max(),])
        
        ## Figure 00 - Raw image with outer bound k and wanted k
        ax1.imshow(self.image, extent=extent, origin='lower', cmap='plasma')
        ax1.plot(outer_bound_k_x, outer_bound_k_y, 'black')
        ax1.plot(diffracted_k_x, diffracted_k_y, '--', color='red', linewidth=0.5)
        ax1.set_xlabel(r'$k_x/k$', fontsize=fontsize_axis)
        ax1.set_ylabel(r'$k_y/k$', fontsize=fontsize_axis)
        ax1.set_title(r'Raw image', fontsize=fontsize_title)
        
        ## Figure 01 - Selected cross-section to integrate over
        ax2.imshow(x_cs, extent=extent_cs, origin='lower', cmap='plasma')
        ax2.set_xlabel(r'$k_x/k$', fontsize=fontsize_axis)
        ax2.set_ylabel(r'$k_y/k$', fontsize=fontsize_axis)
        ax2.set_title(r'Cut section to intergrate over', fontsize=fontsize_title)
        
        ## Figure 10 - Integrated cross-section
        ax3.plot(self.x_axis[0:-1], x_cs_integrated, 'black')
        ax3.grid(linewidth=1, alpha=0.3)
        ax3.set_xlabel(r'$k_x/k$', fontsize=fontsize_axis)
        ax3.set_ylabel(r'Counts [-]', fontsize=fontsize_axis)
        ax3.set_title(r'Integrated cross-section', fontsize=fontsize_title)
        
        # plt.figure(2)
        # plt.imshow(self.image, extent=extent, origin='lower', cmap='plasma')
        # plt.plot(outer_bound_k_x, outer_bound_k_y, 'black')
        # plt.plot(diffracted_k_x, diffracted_k_y, '--', color='red', linewidth=0.5)
        # plt.xlabel(r'$k_x/k$', fontsize=fontsize_axis)
        # plt.ylabel(r'$k_y/k$', fontsize=fontsize_axis)
        # plt.title(r'Measurement, 61$^{\circ}$', fontsize=fontsize_title)
        # plt.savefig(mypath/'Measurement_61.png', dpi=600, format='png')
        
        
        plt.tight_layout(pad=outer_pad, w_pad=width_pad, h_pad=height_pad)
        
        return x_cs_integrated
        

    def plot_3_areas(self, x_pos, y_pos_mid, y_pos_top, y_pos_bottom, integrate_over, angle):
        
        mid_x_indx = np.argmin(np.abs(self.x_axis - x_pos))
        mid_y_indx = np.argmin(np.abs(self.y_axis - y_pos_mid))
        top_y_indx = np.argmin(np.abs(self.y_axis - y_pos_top))
        bottom_y_indx = np.argmin(np.abs(self.y_axis - y_pos_bottom))
        corner_x_indx = np.argmin(np.abs(self.x_axis))
        
        #vertical
        # xy_cs_background = self.image[corner_x_indx - integrate_over:corner_x_indx + integrate_over, top_y_indx - integrate_over:top_y_indx + integrate_over]
        # xy_cs_middle = self.image[mid_y_indx - integrate_over:mid_y_indx + integrate_over, mid_x_indx - integrate_over:mid_x_indx + integrate_over] - xy_cs_background[1,1]
        # xy_cs_top = self.image[top_y_indx - integrate_over:top_y_indx + integrate_over, mid_x_indx - integrate_over:mid_x_indx + integrate_over] - xy_cs_background[1,1]
        # xy_cs_bottom = self.image[bottom_y_indx - integrate_over:bottom_y_indx + integrate_over, mid_x_indx - integrate_over:mid_x_indx + integrate_over] - xy_cs_background[1,1]
        
        #horizontal
        xy_cs_background = self.image[top_y_indx - integrate_over:top_y_indx + integrate_over, corner_x_indx - integrate_over:corner_x_indx + integrate_over]
        xy_cs_middle = self.image[mid_x_indx - integrate_over:mid_x_indx + integrate_over, mid_y_indx - integrate_over:mid_y_indx + integrate_over] - xy_cs_background[1,1]
        xy_cs_top = self.image[mid_x_indx - integrate_over:mid_x_indx + integrate_over, top_y_indx - integrate_over:top_y_indx + integrate_over] - xy_cs_background[1,1]
        xy_cs_bottom = self.image[mid_x_indx - integrate_over:mid_x_indx + integrate_over, bottom_y_indx - integrate_over:bottom_y_indx + integrate_over] - xy_cs_background[1,1]
        
        # left_x_indx = np.argmin(np.abs(self.y_axis - x_pos_left))
        # right_x_indx = np.argmin(np.abs(self.y_axis - x_pos_right))
        
        
        h, w = xy_cs_middle.shape
        
        radius = 20
        mask = np.zeros_like(xy_cs_middle)
        mask = cv2.circle(mask, (h-30,w-30), radius, (255,255,255), -1)/255
        dst_middle = mask*xy_cs_middle #cv2.bitwise_and(xy_cs_middle, mask)
        dst_top = xy_cs_top*mask
        dst_bottom = xy_cs_bottom*mask
        
        y_cs_integrated = np.sum(xy_cs_top, axis=1)
        x_cs_integrated = np.sum(y_cs_integrated, axis = 0)
        
        NA_diffracted = n*np.sin(angle)
        theta = np.linspace(0, 2*PI, 1000)
        diffracted_k_x = NA_diffracted*np.cos(theta)
        diffracted_k_y = NA_diffracted*np.sin(theta)
        
        
        fontsize_title  = 14
        fontsize_axis   = 13
        fontsize_legend = 10
        outer_pad       = 3
        width_pad       = 2
        height_pad      = 1
    
        fig     = plt.figure(figsize=(15,10))
        ax1     = fig.add_subplot(331)
        ax2     = fig.add_subplot(332)
        ax3     = fig.add_subplot(333)
        ax4     = fig.add_subplot(334)
        ax5     = fig.add_subplot(335)
        ax6     = fig.add_subplot(336)
        ax7     = fig.add_subplot(337)
        ax8     = fig.add_subplot(338)
        ax9     = fig.add_subplot(339)
        #ax_list = [ax1, ax2, ax3, ax4]
        
        extent    = np.array([self.x_axis.min(), self.x_axis.max(),
                              self.y_axis.min(), self.y_axis.max()])
        #extent_cs = np.array([cs_x_axis.min(), cs_x_axis.max(),
                              # self.y_axis.min(), self.y_axis.max()])
        
        ## Figure 00 - Raw image with outer bound k and wanted k
        ax1.imshow(self.image, extent=extent, origin='lower', cmap='plasma')
        # ax1.plot(outer_bound_k_x, outer_bound_k_y, 'black')
        ax1.plot(diffracted_k_x, diffracted_k_y, 'red')
        ax1.set_xlabel(r'$k_x/k$', fontsize=fontsize_axis)
        ax1.set_ylabel(r'$k_y/k$', fontsize=fontsize_axis)
        ax1.set_title(r'Raw image', fontsize=fontsize_title)
        
        ax2.imshow(mask, origin='lower', cmap='plasma')
        ax2.set_xlabel(r'$k_x/k$', fontsize=fontsize_axis)
        ax2.set_ylabel(r'$k_y/k$', fontsize=fontsize_axis)
        ax2.set_title(r'Aperture', fontsize=fontsize_title)
       
        ax3.imshow(xy_cs_background, origin='lower', cmap='plasma')
        ax3.set_xlabel(r'$k_x/k$', fontsize=fontsize_axis)
        ax3.set_ylabel(r'$k_y/k$', fontsize=fontsize_axis)
        ax3.set_title(r'Background', fontsize=fontsize_title)
        
        ax4.imshow(xy_cs_middle, origin='lower', cmap='plasma')
        ax4.set_xlabel(r'$k_x/k$', fontsize=fontsize_axis)
        ax4.set_ylabel(r'$k_y/k$', fontsize=fontsize_axis)
        ax4.set_title(r'Cut section to intergrate over, middle', fontsize=fontsize_title)
        
        ax5.imshow(xy_cs_top, origin='lower', cmap='plasma')
        ax5.set_xlabel(r'$k_x/k$', fontsize=fontsize_axis)
        ax5.set_ylabel(r'$k_y/k$', fontsize=fontsize_axis)
        ax5.set_title(r'Cut section to intergrate over, top', fontsize=fontsize_title)
        
        ax6.imshow(xy_cs_bottom, origin='lower', cmap='plasma')
        ax6.set_xlabel(r'$k_x/k$', fontsize=fontsize_axis)
        ax6.set_ylabel(r'$k_y/k$', fontsize=fontsize_axis)
        ax6.set_title(r'Cut section to intergrate over, bottom', fontsize=fontsize_title)
        
        ax7.imshow(dst_middle, origin='lower', cmap='plasma')
        ax7.set_xlabel(r'$k_x/k$', fontsize=fontsize_axis)
        ax7.set_ylabel(r'$k_y/k$', fontsize=fontsize_axis)
        ax7.set_title(r'Cut section with aperture, middle', fontsize=fontsize_title)
        
        ax8.imshow(dst_top, origin='lower', cmap='plasma')
        ax8.set_xlabel(r'$k_x/k$', fontsize=fontsize_axis)
        ax8.set_ylabel(r'$k_y/k$', fontsize=fontsize_axis)
        ax8.set_title(r'Cut section with aperture, top', fontsize=fontsize_title)
        
        ax9.imshow(dst_bottom, origin='lower', cmap='plasma')
        ax9.set_xlabel(r'$k_x/k$', fontsize=fontsize_axis)
        ax9.set_ylabel(r'$k_y/k$', fontsize=fontsize_axis)
        ax9.set_title(r'Cut section with aperture, bottom', fontsize=fontsize_title)
        

        
        plt.tight_layout(pad=outer_pad, w_pad=width_pad, h_pad=height_pad)
        
        return x_cs_integrated
#%%
#PI = np.pi

## Data path
measurment_name = 'GaAs3_3_right'
what_measurement = 0

image_path = list_image_path(measurment_name, what_measurement)
image = MEL_9000_k_images(image_path)



middle = (0.17, 0.05)
# GaAs1_4_image.plot_image()
size = 1

y_pos = 100e-4
x_pos = 150e-4
y_pos_mid = -50e-4 #2300e-4
y_pos_top = 5800e-4
y_pos_bottom = -5800e-4


integrate_over = 100
n = 1
angle = 35
angle_rad = angle*PI/180
image.rotate_image(-270.3)
image.remove_background(0)
image.set_image_bounds(middle, size)


contrast = 20
brightness = 10
#image.plot_image(contrast, brightness)

tfdeg = image.plot_3_areas(x_pos, y_pos_mid, y_pos_top, y_pos_bottom, integrate_over, angle) #35


#%%
#PI = np.pi

## Data path
measurment_name = 'GaAs3_3_right'
what_measurement = 1

image_path = list_image_path(measurment_name, what_measurement)
image = MEL_9000_k_images(image_path)



middle = (-0.25, -0.06)
# GaAs1_4_image.plot_image()
size = 1

y_pos = 200e-4
x_pos = -20e-4
y_pos_mid = -100e-4
y_pos_top = 6400e-4
y_pos_bottom = -6600e-4


integrate_over = 30
n = 1
angle = 40
angle_rad = angle*PI/180
image.rotate_image(-90.3)
image.remove_background(0)
image.set_image_bounds(middle, size)


contrast = 20
brightness = 10
#image.plot_image(contrast, brightness)

fozdeg = image.plot_3_areas(x_pos, y_pos_mid, y_pos_top, y_pos_bottom, integrate_over, angle) #40

#%%
#PI = np.pi

## Data path
measurment_name = 'GaAs3_3_right'
what_measurement = 2

image_path = list_image_path(measurment_name, what_measurement)
image = MEL_9000_k_images(image_path)



middle = (-0.23, -0.07)
# GaAs1_4_image.plot_image()
size = 1

y_pos = 100e-4
x_pos = 10e-4
y_pos_mid = -200e-4
y_pos_top = 7000e-4
y_pos_bottom = -7400e-4


integrate_over = 50
n = 1
angle = 45
angle_rad = angle*PI/180
image.rotate_image(-90.3)
image.remove_background(0)
image.set_image_bounds(middle, size)


contrast = 20
brightness = 10
#image.plot_image(contrast, brightness)

fofdeg = image.plot_3_areas(x_pos, y_pos_mid, y_pos_top, y_pos_bottom, integrate_over, angle) #45

#%%
#PI = np.pi

## Data path
measurment_name = 'GaAs3_3_right'
what_measurement = 4

image_path = list_image_path(measurment_name, what_measurement)
image = MEL_9000_k_images(image_path)



middle = (-0.26, -0.06)
# GaAs1_4_image.plot_image()
size = 1

y_pos = 100e-4
x_pos = 10e-4
y_pos_mid = -50e-4
y_pos_top = 7700e-4
y_pos_bottom = -6850e-4


integrate_over = 30
n = 1
angle = 50
angle_rad = angle*PI/180
image.rotate_image(-90.3)
image.remove_background(0)
image.set_image_bounds(middle, size)


contrast = 20
brightness = 10
#image.plot_image(contrast, brightness)

fzdeg = image.plot_3_areas(x_pos, y_pos_mid, y_pos_top, y_pos_bottom, integrate_over, angle) #50

#%%
#PI = np.pi

## Data path
measurment_name = 'GaAs3_3_right'
what_measurement = 5

image_path = list_image_path(measurment_name, what_measurement)
image = MEL_9000_k_images(image_path)



middle = (-0.34, -0.06)
# GaAs1_4_image.plot_image()
size = 1

y_pos = 100e-4
x_pos = 10e-4
y_pos_mid = -50e-4
y_pos_top = 8200e-4
y_pos_bottom = -8400e-4


integrate_over = 30
n = 1
angle = 55
angle_rad = angle*PI/180
image.rotate_image(-90.3)
image.remove_background(0)
image.set_image_bounds(middle, size)


contrast = 20
brightness = 10
#image.plot_image(contrast, brightness)

ffdeg =  image.plot_3_areas(x_pos, y_pos_mid, y_pos_top, y_pos_bottom, integrate_over, angle) #55

#%%
#PI = np.pi

## Data path
measurment_name = 'GaAs3_3_right'
what_measurement = 6

image_path = list_image_path(measurment_name, what_measurement)
image = MEL_9000_k_images(image_path)



middle = (-0.25, -0.06)
# GaAs1_4_image.plot_image()
size = 1

y_pos = 100e-4
x_pos = 10e-4
y_pos_mid = -50e-4
y_pos_top = 8750e-4
y_pos_bottom = -8950e-4


integrate_over = 30
n = 1
angle = 61
angle_rad = angle*PI/180
image.rotate_image(-90.3)
image.remove_background(0)
image.set_image_bounds(middle, size)


contrast = 20
brightness = 10
#image.plot_image(contrast, brightness)

sodeg = image.plot_3_areas(x_pos, y_pos_mid, y_pos_top, y_pos_bottom, integrate_over, angle) #61
#%%
#PI = np.pi

## Data path
measurment_name = 'GaAs3_3_right'
what_measurement = 7

image_path = list_image_path(measurment_name, what_measurement)
image = MEL_9000_k_images(image_path)



middle = (-0.19, -0.06)
# GaAs1_4_image.plot_image()
size = 1

y_pos = 100e-4
x_pos = 10e-4
y_pos_mid = -1e-4
y_pos_top = 9050e-4
y_pos_bottom = -9200e-4


integrate_over = 30
n = 1
angle = 65
angle_rad = angle*PI/180
image.rotate_image(-90.3)
image.remove_background(0)
image.set_image_bounds(middle, size)


contrast = 20
brightness = 10
#image.plot_image(contrast, brightness)

sfdeg = image.plot_3_areas(x_pos, y_pos_mid, y_pos_top, y_pos_bottom, integrate_over, angle) #65

#%%
#PI = np.pi

## Data path
measurment_name = 'GaAs3_3_right'
what_measurement = 8

image_path = list_image_path(measurment_name, what_measurement)
image = MEL_9000_k_images(image_path)



middle = (-0.35, -0.06)
# GaAs1_4_image.plot_image()
size = 1

y_pos = 100e-4
x_pos = 10e-4
y_pos_mid = 10e-4
y_pos_top = 9300e-4
y_pos_bottom = -9250e-4


integrate_over =30
n = 1
angle = 70
angle_rad = angle*PI/180
image.rotate_image(-90.3)
image.remove_background(0)
image.set_image_bounds(middle, size)


contrast = 20
brightness = 10
#image.plot_image(contrast, brightness)

sezdeg = image.plot_3_areas(x_pos, y_pos_mid, y_pos_top, y_pos_bottom, integrate_over, angle)  #70

#%%

plt.figure()
plt.plot(tfdeg, label = r'$35^{\circ}$')
plt.plot(fozdeg, label = r'$40^{\circ}$')
plt.plot(fofdeg, label = r'$45^{\circ}$')
plt.plot(fzdeg, label = r'$50^{\circ}$')
plt.plot(ffdeg, label = r'$55^{\circ}$')
plt.plot(sodeg, label = r'$61^{\circ}$')
plt.plot(sfdeg, label = r'$65^{\circ}$')
plt.plot(sezdeg, label = r'$70^{\circ}$')
plt.grid(linewidth=1, alpha=0.3)
plt.xlabel(r'$k_x/k$', fontsize=12)
plt.ylabel(r'Counts [-]', fontsize=12)
plt.legend() 
plt.title(r'Integrated cross-section', fontsize=12)
plt.savefig(mypath/'GaAs3_3_crossection_35-70deg.png', dpi=600, format='png')
