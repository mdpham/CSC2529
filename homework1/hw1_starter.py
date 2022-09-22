import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
from pathlib import Path
from numpy.fft import fft2, ifft2, fftshift, ifftshift

hw_dir = Path(__file__).parent

# Load images
img1 = io.imread(hw_dir/'image1.png')
img2 = io.imread(hw_dir/'image2.png')

# Part (a)
W = img1.shape[0]       # = 1001 dots
d = np.array([0.4, 2])  # distances (m)
dpi = 300               # dots per inch

#### YOUR CODE HERE ####

# Part (b)
cpd = 5   # Peak contrast sensitivity location (cycles per degree)

#### YOUR CODE HERE ####

# Part (c)
# Hint: fft2, ifft2, fftshift, and ifftshift functions all take an |axes|
# argument to specify the axes for the 2D DFT. e.g. fft2(arr, axes=(1, 2))
# Hint: Check out np.meshgrid.

#### Change these to the correct values for the high- and low-pass filters
hpf = np.zeros_like(img1)  # TODO: Replace
lpf = np.zeros_like(img1)  # TODO: Replace

#### Apply the filters to create the hybrid image
hybrid_img = np.zeros_like(img1)  # TODO: Replace

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
axs[0,0].imshow(img2)
axs[0,0].axis('off')
axs[0,1].imshow(hpf, cmap='gray')
axs[0,1].set_title("High-pass filter")
axs[1,0].imshow(img1)
axs[1,0].axis('off')
axs[1,1].imshow(lpf, cmap='gray')
axs[1,1].set_title("Low-pass filter")
plt.savefig("hpf_lpf.png", bbox_inches='tight')
io.imsave("hybrid_image.png", np.clip(hybrid_img, a_min=0, a_max=255.))
