#%%
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
from scipy import ndimage

PI = np.pi

## Data path
project_path = Path(__file__).parent
measurement_path    = Path(project_path, 'measurement', 'data', 'GaAs1_4_ref_ip_op')

## Select image to analyze
what_image_to_analyse = 0
images  = os.listdir(measurement_path)
image_path = Path(measurement_path, images[what_image_to_analyse])

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
    def plot_image(self):
        fig     = plt.figure(figsize=(10,8))
        ax1     = fig.add_subplot(111)
        extent = np.array([self.x_axis.min(), self.x_axis.max(),
                            self.y_axis.min(), self.y_axis.max()])
        # NA = 0.95
        # theta = np.linspace(0, 2*PI, 1000)
        # outer_bound_k_x = NA*np.cos(theta)
        # outer_bound_k_y = NA*np.sin(theta)
        
        ax1.imshow(self.image, extent=extent, origin='lower', cmap='plasma')
        # ax1.plot(outer_bound_k_x, outer_bound_k_y, 'black')
        
    ## Plot raw image and select a cross-section to integrate over
    def plot_image_and_cs(self, angle, n,  x_pos, integrate_over):
        
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
        ax1.plot(diffracted_k_x, diffracted_k_y, 'red')
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
        

    def plot_3_areas(self, x_pos, y_pos_mid, y_pos_top, integrate_over):
        
        mid_x_indx = np.argmin(np.abs(self.x_axis - x_pos))
        mid_y_indx = np.argmin(np.abs(self.y_axis - y_pos_mid))
        top_y_indx = np.argmin(np.abs(self.y_axis - y_pos_top))
        bottom_y_indx = np.argmin(np.abs(self.y_axis - y_pos_bottom))
        xy_cs_middle = self.image[mid_y_indx - integrate_over:mid_y_indx + integrate_over, mid_x_indx - integrate_over:mid_x_indx + integrate_over]
        xy_cs_top = self.image[top_y_indx - integrate_over:top_y_indx + integrate_over, mid_x_indx - integrate_over:mid_x_indx + integrate_over]
        xy_cs_bottom = self.image[bottom_y_indx - integrate_over:bottom_y_indx + integrate_over, mid_x_indx - integrate_over:mid_x_indx + integrate_over]
        
        y_cs_integrated = np.sum(xy_cs_bottom, axis=1)
        x_cs_integrated = np.sum(y_cs_integrated, axis = 0)
    
        fontsize_title  = 14
        fontsize_axis   = 13
        # fontsize_legend = 10
        # outer_pad       = 3
        # width_pad       = 2
        # height_pad      = 3
    
        fig     = plt.figure(figsize=(10,8))
        ax1     = fig.add_subplot(221)
        ax2     = fig.add_subplot(222)
        ax3     = fig.add_subplot(223)
        ax4     = fig.add_subplot(224)
        #ax_list = [ax1, ax2, ax3, ax4]
        
        extent    = np.array([self.x_axis.min(), self.x_axis.max(),
                              self.y_axis.min(), self.y_axis.max()])
        #extent_cs = np.array([cs_x_axis.min(), cs_x_axis.max(),
                             # self.y_axis.min(), self.y_axis.max()])
        
        ## Figure 00 - Raw image with outer bound k and wanted k
        ax1.imshow(self.image, extent=extent, origin='lower', cmap='plasma')
        # ax1.plot(outer_bound_k_x, outer_bound_k_y, 'black')
        # ax1.plot(diffracted_k_x, diffracted_k_y, 'red')
        ax1.set_xlabel(r'$k_x/k$', fontsize=fontsize_axis)
        ax1.set_ylabel(r'$k_y/k$', fontsize=fontsize_axis)
        ax1.set_title(r'Raw image', fontsize=fontsize_title)
        
        ## Figure 01 - Selected cross-section to integrate over
        ax2.imshow(xy_cs_middle, origin='lower', cmap='plasma')
        ax2.set_xlabel(r'$k_x/k$', fontsize=fontsize_axis)
        ax2.set_ylabel(r'$k_y/k$', fontsize=fontsize_axis)
        ax2.set_title(r'Cut section to intergrate over, middle', fontsize=fontsize_title)
        
        ## Figure 10 - Selected cross-section to integrate over
        ax3.imshow(xy_cs_top, origin='lower', cmap='plasma')
        ax3.set_xlabel(r'$k_x/k$', fontsize=fontsize_axis)
        ax3.set_ylabel(r'$k_y/k$', fontsize=fontsize_axis)
        ax3.set_title(r'Cut section to intergrate over, top', fontsize=fontsize_title)
        
        ## Figure 10 - Selected cross-section to integrate over
        ax4.imshow(xy_cs_bottom, origin='lower', cmap='plasma')
        ax4.set_xlabel(r'$k_x/k$', fontsize=fontsize_axis)
        ax4.set_ylabel(r'$k_y/k$', fontsize=fontsize_axis)
        ax4.set_title(r'Cut section to intergrate over, bottom', fontsize=fontsize_title)
        
        return x_cs_integrated
        
        
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
# GaAs1_4_image.plot_image()


GaAs1_4_image.plot_3_areas(x_pos, y_pos_mid, y_pos_top, integrate_over)
#GaAs1_4_image.plot_image_and_cs(angle, n,  x_pos, integrate_over)

