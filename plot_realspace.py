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
        ax3.plot(self.x_axis[0:-1], y_cs_integrated, 'black')
        ax3.grid(linewidth=1, alpha=0.3)
        ax3.set_xlabel(r'$k_y/k$', fontsize=fontsize_axis)
        ax3.set_ylabel(r'Counts [-]', fontsize=fontsize_axis)
        ax3.set_title(r'Integrated cross-section', fontsize=fontsize_title)
        
        plt.tight_layout(pad=outer_pad, w_pad=width_pad, h_pad=height_pad)
        
        return y_cs_integrated/max(y_cs_integrated)
        
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
        
        plt.figure(2)
        plt.imshow(self.image, extent=extent, origin='lower', cmap='plasma')
        plt.plot(outer_bound_k_x, outer_bound_k_y, 'black')
        plt.plot(diffracted_k_x, diffracted_k_y, '--', color='yellow', linewidth=0.5)
        plt.plot(self.x_axis[0:-1], 0.7*(x_cs_integrated/max(x_cs_integrated))-0.95, 'red')
        plt.xlabel(r'$k_x/k$', fontsize=fontsize_axis)
        plt.ylabel(r'$k_y/k$', fontsize=fontsize_axis)
        plt.title(r'Measurement, 61$^{\circ}$', fontsize=fontsize_title)     
        plt.yticks([-0.5, 0, 0.5], fontsize = fontsize_axis)
        plt.xticks([-0.5, 0, 0.5], fontsize = fontsize_axis)
        plt.tight_layout()
        plt.savefig(mypath/'Measurement_sph_61.png', dpi=600, format='png')


        
        
        plt.tight_layout(pad=outer_pad, w_pad=width_pad, h_pad=height_pad)
        
        return x_cs_integrated/max(x_cs_integrated)
        

    def plot_3_areas(self, x_pos, y_pos_mid, y_pos_top, y_pos_bottom, integrate_over):
        
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
        
        y_cs_integrated = np.sum(xy_cs_bottom, axis=1)
        x_cs_integrated = np.sum(y_cs_integrated, axis = 0)
        
        
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
        # ax1.plot(diffracted_k_x, diffracted_k_y, 'red')
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
measurment_name = 'GaAs4_1'
what_measurement = 18

image_path = list_image_path(measurment_name, what_measurement)
image = MEL_9000_k_images(image_path)



middle = (0.23, 0.075)
# GaAs1_4_image.plot_image()
size = 1

y_pos = 100e-4
x_pos = 200e-4
y_pos_mid = -50e-4
y_pos_top = 8600e-4
y_pos_bottom = -8750e-4


integrate_over = 30
n = 1
angle = 61
angle_rad = angle*PI/180
image.rotate_image(-271)
image.remove_background(0)
image.set_image_bounds(middle, size)


contrast = 20
brightness = 10
#image.plot_image(contrast, brightness)

sodeg = image.plot_image_and_x_cs(angle_rad, n, y_pos, integrate_over) #61
sodeg_y = image.plot_image_and_y_cs(angle_rad, n, y_pos, integrate_over) #61 y


#%%
#PI = np.pi

## Data path
measurment_name = 'GaAs4_1'
what_measurement = 27

image_path = list_image_path(measurment_name, what_measurement)
image = MEL_9000_k_images(image_path)

#middle = (0.205, 0.05) #10
middle = (0.09, 0.1) #11
# GaAs1_4_image.plot_image()
size = 1

y_pos = 100e-4
x_pos = 200e-4
y_pos_mid = -50e-4
y_pos_top = 8600e-4
y_pos_bottom = -8750e-4


integrate_over = 30
n = 1
angle = 65
angle_rad = angle*PI/180
image.rotate_image(-270)
image.remove_background(0)
image.set_image_bounds(middle, size)


contrast = 20
brightness = 10
#image.plot_image(contrast, brightness)

sfdeg = image.plot_image_and_x_cs(angle_rad, n, y_pos, integrate_over) #65
sfdeg_y = image.plot_image_and_y_cs(angle_rad, n, y_pos, integrate_over) #65 y

#%%

# so_index = np.argmin(abs(sodeg[1228:1428]-1/np.exp(2)))
# so_index_m = np.argmax(abs(sodeg[1228:1428]))
# so_index_up = so_index_m + (so_index_m-so_index)

# sf_index = np.argmin(abs(sfdeg[1228:1428]-1/np.exp(2)))
# sf_index_m = np.argmax(abs(sfdeg[1228:1428]))
# sf_index_up = sf_index_m + (sf_index_m-sf_index)
# print((sf_index_m-sf_index))

plt.figure(1)
plt.plot(sodeg, label = r'$61^{\circ}$')
plt.plot(sfdeg, label = r'$65^{\circ}$')
plt.grid(linewidth=1, alpha=0.3)
plt.xlabel(r'$k_x/k$', fontsize=15)
plt.ylabel(r'Counts [-]', fontsize=15)
plt.legend(fontsize=15) 
plt.title(r'Integrated cross-section', fontsize=16)
plt.yticks(fontsize = 15)
plt.xticks(fontsize = 15)
plt.tight_layout()
#plt.savefig(mypath/'GaAs4_1_sph_crossection_sph_61-65deg.png', dpi=600, format='png')

# plt.figure(2)
# plt.plot(sodeg[1228:1428], label = r'$61^{\circ}$')
# plt.plot(sfdeg[1228:1428], label = r'$65^{\circ}$')
# plt.axvline(x = so_index, linestyle = '--')
# plt.axvline(x = so_index_up, linestyle = '--')
# plt.axvline(x = sf_index, linestyle = '--', color = 'orange')
# plt.axvline(x = sf_index_up, linestyle = '--', color = 'orange')
# plt.grid(linewidth=1, alpha=0.3)
# plt.xlabel(r'$k_x/k$', fontsize=15)
# plt.ylabel(r'Counts [-]', fontsize=15)
# plt.legend(fontsize=15) 
# plt.title(r'Integrated cross-section (x)', fontsize=16)
# plt.yticks(fontsize = 15)
# plt.xticks(fontsize = 15)
# plt.tight_layout()
# plt.savefig(mypath/'GaAs4_1_sph_crossection_x_61-65deg_zoomed.png', dpi=600, format='png')

#%%

sodeg_y[708:719] = 0 

so_index = np.argmin(abs(sodeg_y[600:800]-1/np.exp(2)))
so_index_m = np.argmax(abs(sodeg_y[600:800]))
so_index_up = so_index_m + (so_index_m-so_index)

sf_index = np.argmin(abs(sfdeg_y[600:800]-1/np.exp(2)))
sf_index_m = np.argmax(abs(sfdeg_y[600:800]))
sf_index_up = sf_index_m + (sf_index_m-sf_index)

print((sf_index_m-sf_index))


plt.figure(2)
plt.plot(sodeg_y, label = r'$61^{\circ}$')
plt.plot(sfdeg_y, label = r'$65^{\circ}$')
plt.grid(linewidth=1, alpha=0.3)
plt.xlabel(r'$k_y/k$', fontsize=15)
plt.ylabel(r'Counts [-]', fontsize=15)
plt.legend(fontsize=15) 
plt.title(r'Integrated cross-section (y)', fontsize=16)
plt.yticks(fontsize = 15)
plt.xticks(fontsize = 15)
plt.tight_layout()
#plt.savefig(mypath/'GaAs4_1_sph_crossection_y_61-65deg.png', dpi=600, format='png')

# plt.figure(4)
# plt.plot(sodeg_y[600:800], label = r'$61^{\circ}$')
# plt.plot(sfdeg_y[600:800], label = r'$65^{\circ}$')
# plt.axvline(x = so_index, linestyle = '--')
# plt.axvline(x = so_index_up, linestyle = '--')
# plt.axvline(x = sf_index, linestyle = '--', color = 'orange')
# plt.axvline(x = sf_index_up, linestyle = '--', color = 'orange')
# plt.grid(linewidth=1, alpha=0.3)
# plt.xlabel(r'$k_y/k$', fontsize=15)
# plt.ylabel(r'Counts [-]', fontsize=15)
# plt.legend(fontsize=15) 
# plt.title(r'Integrated cross-section (y)', fontsize=16)
# plt.yticks(fontsize = 15)
# plt.xticks(fontsize = 15)
# plt.tight_layout()
# #plt.savefig(mypath/'GaAs4_1_sph_crossection_y_61-65deg_zoomed.png', dpi=600, format='png')
