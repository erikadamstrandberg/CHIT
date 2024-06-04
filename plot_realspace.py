#%%
#import numpy as np
#from image_functions import MEL_9000_k_images, list_image_path
import os
from pathlib import Path

mypath = Path(Path(__file__).resolve().parents[0])

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage

PI = np.pi

def list_image_path(measurement, what_measurement):
    project_path = Path(__file__).parent
    measurement_path    = Path(project_path, 'measurement', measurement)

    images  = os.listdir(measurement_path)
    image_paths = Path(measurement_path, images[what_measurement])
    
    return image_paths

def list_all_image_path(measurement):
    project_path = Path(__file__).parent
    measurement_path    = Path(project_path, 'measurement', measurement)

    return os.listdir(measurement_path)



    
## Class for analysin images
class MEL_9000_x_images():
    
    ## Initialize image
    def __init__(self, image_path):
        ## Pixel size in k-space
        self.pixel_size_k = 0.172e-6
        
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
    # def set_image_bounds(self, middle):

    #     mid_x_indx = np.argmin(np.abs(self.x_axis - middle[0]))
    #     mid_y_indx = np.argmin(np.abs(self.y_axis - middle[1]))
        
    #     self.image = self.image[mid_y_indx - size_indx:mid_y_indx + size_indx,
    #                             mid_x_indx - size_indx:mid_x_indx + size_indx]
        
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
        
        fontsize_title  = 16
        fontsize_axis   = 15
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
        ax3.plot(self.x_axis[0:], y_cs_integrated, 'black')
        ax3.grid(linewidth=1, alpha=0.3)
        ax3.set_xlabel(r'$k_y/k$', fontsize=fontsize_axis)
        ax3.set_ylabel(r'Counts [-]', fontsize=fontsize_axis)
        ax3.set_title(r'Integrated cross-section', fontsize=fontsize_title)
        
        plt.tight_layout(pad=outer_pad, w_pad=width_pad, h_pad=height_pad)
        
        return y_cs_integrated/max(y_cs_integrated)
        
    def plot_image_and_x_cs(self, angle, n, y_pos, integrate_over, plot_on):
        
        mid_y_indx = np.argmin(np.abs(self.y_axis - y_pos))
        x_cs = self.image[mid_y_indx - integrate_over:mid_y_indx + integrate_over, :]
        x_cs_integrated = np.sum(x_cs, axis=0)
        
        cs_y_axis = self.y_axis[mid_y_indx - integrate_over:mid_y_indx + integrate_over]

        if plot_on:
            fontsize_title  = 16
            fontsize_axis   = 15
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
            # ax1.plot(outer_bound_k_x, outer_bound_k_y, 'black')
            # ax1.plot(diffracted_k_x, diffracted_k_y, '--', color='red', linewidth=0.5)
            ax1.set_xlabel(r'$k_x/k$', fontsize=fontsize_axis)
            ax1.set_ylabel(r'$k_y/k$', fontsize=fontsize_axis)
            ax1.set_title(r'Raw image', fontsize=fontsize_title)
            
            ## Figure 01 - Selected cross-section to integrate over
            ax2.imshow(x_cs, extent=extent_cs, origin='lower', cmap='plasma')
            ax2.set_xlabel(r'$k_x/k$', fontsize=fontsize_axis)
            ax2.set_ylabel(r'$k_y/k$', fontsize=fontsize_axis)
            ax2.set_title(r'Cut section to intergrate over', fontsize=fontsize_title)
            
            ## Figure 10 - Integrated cross-section
            ax3.plot(self.x_axis[0:], x_cs_integrated, 'black')
            ax3.grid(linewidth=1, alpha=0.3)
            ax3.set_xlabel(r'$k_x/k$', fontsize=fontsize_axis)
            ax3.set_ylabel(r'Counts [-]', fontsize=fontsize_axis)
            ax3.set_title(r'Integrated cross-section', fontsize=fontsize_title)
        
        # plt.figure(2)
        # plt.imshow(self.image, extent=extent, origin='lower', cmap='plasma')
        # # plt.plot(outer_bound_k_x, outer_bound_k_y, 'black')
        # # plt.plot(diffracted_k_x, diffracted_k_y, '--', color='yellow', linewidth=0.5)
        # plt.plot(self.x_axis[0:], 0.7*(x_cs_integrated/max(x_cs_integrated))-0.95, 'red')
        # plt.xlabel(r'$k_x/k$', fontsize=fontsize_axis)
        # plt.ylabel(r'$k_y/k$', fontsize=fontsize_axis)
        # plt.title(r'Measurement, 61$^{\circ}$', fontsize=fontsize_title)     
        # plt.yticks([-0.5, 0, 0.5], fontsize = fontsize_axis)
        # plt.xticks([-0.5, 0, 0.5], fontsize = fontsize_axis)
        # # plt.tight_layout()
        # # plt.savefig(mypath/'Measurement_sph_61.png', dpi=600, format='png')
        
            plt.tight_layout(pad=outer_pad, w_pad=width_pad, h_pad=height_pad)
        
        return x_cs_integrated/max(x_cs_integrated)
        

#PI = np.pi

## Data path
measurment_name = 'sph61_real'
what_measurement = 0

image_path = list_image_path(measurment_name, what_measurement)
image = MEL_9000_x_images(image_path)


middle = (-5, 0.075)
size = 1

y_pos = 0
# x_pos = 200e-4
# y_pos_mid = -50e-4
# y_pos_top = 8600e-4
# y_pos_bottom = -8750e-4

integrate_over = 400
n = 1
angle = 61
angle_rad = angle*PI/180
image.rotate_image(-271)
image.remove_background(0)
# image.set_image_bounds(middle, size)

contrast   = 60
brightness = 60
# image.plot_image(contrast, brightness)

plot_on = True
sodeg_y = image.plot_image_and_x_cs(angle_rad, n, y_pos, integrate_over, plot_on) #61 y


#%%

measurment_name = 'sph61_real'
image_paths = list_all_image_path(measurment_name)

full_beam = np.zeros(shape=(len(image_paths), len(sodeg_y)))


for i, im in enumerate(image_paths):
    print('Loading image: ' + str(i) + ' of ' + str(len(image_paths)))
    project_path = Path(__file__).parent
    measurement_path    = Path(project_path, 'measurement', measurment_name, im)
    image = MEL_9000_x_images(image_path)


    middle = (-5, 0.075)
    size = 1

    y_pos = 0

    integrate_over = 400
    n = 1
    angle = 61
    angle_rad = angle*PI/180
    image.rotate_image(-271)
    image.remove_background(0)
    # image.set_image_bounds(middle, size)

    contrast   = 60
    brightness = 60
    # image.plot_image(contrast, brightness)

    plot_on = False
    sodeg_y = image.plot_image_and_x_cs(angle_rad, n, y_pos, integrate_over, plot_on) #61 y

    