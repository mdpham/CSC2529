import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import imageio
import cv2

from pdb import set_trace
from pathlib import Path

hdr_dir = Path("hdr_data")

# initialize HDR image with all zeros
hdr = np.zeros((768, 512, 3), dtype=float)

##########################################

# load LDR images from hdr_dir (need to unzip before)

# compute weights

# fuse LDR images using weights, make sure to store your fused HDR using the name hdr
hdr = ...

##########################################

# Normalize
hdr = np.exp(hdr / scale)
hdr *= 0.8371896/np.mean(hdr)  # this makes the mean of the created HDR image match the reference image (totally optional)

# convert to 32 bit floating point format, required for OpenCV
hdr = np.float32(hdr)

# crop boundary - image data here are only captured in some of the exposures, which is why they are indicated in blue in the LDR images
hdr = hdr[29:720, 19:480, :]

# tonemap image and save LDR image using OpenCV's implementation of Drago's tonemapping operator
gamma = 1.0
saturation = 0.7
bias = 0.85
tonemapDrago = cv2.createTonemapDrago(gamma,saturation,bias)
ldrDrago = tonemapDrago.process(hdr)
io.imsave('my_hdr_image_tonemapped.jpg', np.uint8( np.clip(3*ldrDrago, 0, 1) *255))

# write HDR image (can compare to hw4_1_memorial_church.hdr reference image in an external viewer)
hdr = cv2.cvtColor(hdr, cv2.COLOR_BGR2RGB)
cv2.imwrite('my_hdr_image.hdr', hdr)
