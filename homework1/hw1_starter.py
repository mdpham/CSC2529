import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
from pathlib import Path
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import math
from scipy.ndimage.filters import generic_filter as gf


hw_dir = Path(__file__).parent

# Load images
img1 = io.imread(hw_dir/'image1.png')
img2 = io.imread(hw_dir/'image2.png')

# Part (a)
W = img1.shape[0]       # = 1001 dots
d = np.array([0.4, 2])  # distances (m)
dpi = 300               # dots per inch
dpmm = dpi/25.4         # 25.4mm in an inch
printout = W/dpmm
print('printout length in mm is: ', printout)

# vis_degrees in visual degrees
# view_distance in millimeters
img_extent = lambda vis_degrees, view_distance: 2 * \
    view_distance*np.tan((vis_degrees*np.pi/180)/2)
img_extent_1 = img_extent(1, 400)
img_extent_2 = img_extent(1, 2000)
print('frontal extent size is:')
print(img_extent_1)
print(img_extent_2)

num_pixels = lambda extent: extent*dpmm
num_pixels_1 = num_pixels(img_extent_1)
num_pixels_2 = num_pixels(img_extent_2)
print('number of pixels viewed at 1 degree:')
print(num_pixels_1)
print(num_pixels_2)

#### YOUR CODE HERE ####

# Part (b)

cpd = 5   # Peak contrast sensitivity location (cycles per degree)

#### YOUR CODE HERE ####

# Part (c)
# Hint: fft2, ifft2, fftshift, and ifftshift functions all take an |axes|
# argument to specify the axes for the 2D DFT. e.g. fft2(arr, axes=(1, 2))
# Hint: Check out np.meshgrid.

# Change these to the correct values for the high- and low-pass filters

# https://github.com/j2kun/hybrid-images/blob/main/hybrid-images.py
def gaussian_filter(shape, sigma, high_pass=True):
    num_rows = shape[0]
    num_cols = shape[1]
    center_i = int(num_rows/2) + 1 if num_rows % 2 == 1 else int(num_rows/2)
    center_j = int(num_cols/2) + 1 if num_cols % 2 == 1 else int(num_cols/2)
    def gaussian(i,j):
        coefficient = math.exp(-1.0 * ((i - center_i)**2 + (j - center_j)**2) / (2 * sigma**2))
        return 1 - coefficient if high_pass else coefficient
    return np.array([[gaussian(i,j) for j in range(num_cols)] for i in range(num_rows)])

# https://stackoverflow.com/questions/44865023/how-can-i-create-a-circular-mask-for-a-numpy-array
def create_circular_mask(shape, center=None, radius=None, high_pass=True):
    h = shape[0]
    w = shape[1]
    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    mask = dist_from_center <= radius
    if high_pass:
        mask = np.invert(mask)
    float_mask = mask.astype(float)
    print(high_pass,float_mask)
    return float_mask

    
def make_filter(shape, radius, high_pass=True):
    # sigma = radius
    # gaussian = gaussian_filter(shape, sigma, high_pass)
    # filter = np.stack([gaussian, gaussian, gaussian], axis=-1)
    # 
    circle_mask = create_circular_mask(shape, center=None, radius=radius, high_pass=high_pass)
    filter = np.stack([circle_mask, circle_mask, circle_mask], axis=-1)
    # print(np.shape(filter))
    # print(filter.dtype)
    return filter

sigma = 200

hpf = make_filter(np.shape(img1), sigma, True)
lpf = make_filter(np.shape(img1), sigma, False)

# Apply the filters to create the hybrid image
# hybrid_img = np.zeros_like(img1)  # TODO: Replace
def filter_dft(img, filter):
    shifted_dft = fftshift(fft2(img))
    filtered_dft = np.multiply(shifted_dft, filter)
    return ifft2(ifftshift(filtered_dft)).real
    
def low_pass(img, sigma):
    shape = np.shape(img)
    filter = make_filter(shape, sigma, high_pass=False)
    return filter_dft(img, filter)
def high_pass(img, sigma):
    shape = np.shape(img)
    filter = make_filter(shape, sigma, high_pass=True)
    return filter_dft(img, filter)

def make_hybrid_img(img1, img2, sigma):
    lp_img1 = low_pass(img1, sigma)
    hp_img2 = high_pass(img2, sigma)
    hybrid_img = ifft2(fft2(lp_img1) + fft2(hp_img2)).real
    return hybrid_img

hybrid_img = make_hybrid_img(img1, img2, sigma)

fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(10, 10))
axs[0,0].imshow(img2)
axs[0,0].axis('off')
axs[0,1].imshow(hpf, cmap='gray')
axs[0,1].set_title("High-pass filter")
axs[1,0].imshow(img1)
axs[1,0].axis('off')
axs[1,1].imshow(lpf, cmap='gray')
axs[1,1].set_title("Low-pass filter")
axs[2,0].imshow(high_pass(img1, sigma))
axs[2,1].imshow(low_pass(img2, sigma), cmap='gray')
axs[2,0].axis('off')
axs[2,1].axis('off')
axs[2,0].set_title("High-pass img1")
axs[2,1].set_title("Low-pass img2")
plt.savefig("hpf_lpf.png", bbox_inches='tight') 
io.imsave("hybrid_image.png", np.clip(hybrid_img, a_min=0, a_max=255.))