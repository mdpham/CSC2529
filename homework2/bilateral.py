from cmath import exp
import numpy as np
from fspecial import fspecial_gaussian_2d

def bilateral2d(img, radius, sigma, sigmaIntensity):

    pad = radius
    # Initialize filtered image to 0
    out = np.zeros_like(img)

    # Pad image to reduce boundary artifacts
    imgPad = np.pad(img, pad)
    

    # Smoothing kernel, gaussian with standard deviation sigma
    # and size (2*radius+1, 2*radius+1)
    # filtSize = (2*radius + 1, 2*radius + 1)
    # spatialKernel = fspecial_gaussian_2d(filtSize, sigma)
    # print(spatialKernel)
    # Define weighting function with img in scope to access
    def w(p1, p2, I): 
        y1, x1 = p1
        y2, x2 = p2
        pixel_norm = np.linalg.norm(np.subtract(p2, p1), ord=2)
        pixel_intensity_norm = np.square(I[y2, x2] - I[y1, x1])
        spatial_distance = np.exp(-pixel_norm/(2*np.square(sigma)))
        intensity_distance = np.exp(-pixel_intensity_norm/(2*np.square(sigmaIntensity)))
        return spatial_distance*intensity_distance
        
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            # Shift original index into padded index space
            y_pad = y+pad
            x_pad = x+pad
            # Go over a window of size (2*radius + 1) around the current pixel,
            # compute weights, sum the weighted intensity.

            # Get neighbourhood of y,x original index
            neighbourhood = imgPad[y_pad-pad:y_pad+pad+1, x_pad-pad:x_pad+pad+1]
            centerVal = imgPad[y_pad, x_pad] # Careful of padding amount!
            # padded index space
            y_nbhd = range(y_pad-pad,y_pad+pad+1)
            x_nbhd = range(x_pad-pad,x_pad+pad+1)
            # print(y_nbhd, x_nbhd)
            # print(np.shape(neighbourhood))
            # Don't forget to normalize by the sum of the weights used.
            W_p = np.zeros_like(neighbourhood)
            # print(W_p)
            for i in range(len(y_nbhd)):
                y_p = y_nbhd[i]
                for j in range(len(x_nbhd)):
                    x_p = x_nbhd[j]
                    W_p[i, j] = w([y_pad, x_pad], [y_p, x_p], imgPad)
            # print(np.sum(W_p))

            normalization_factor = np.sum(W_p)
            local_weighted_sum = np.sum(np.multiply(neighbourhood, W_p))
            new_intensity = np.divide(local_weighted_sum, normalization_factor)
            out[y, x] = new_intensity
    return out
