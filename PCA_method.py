"""
MIT License

Copyright (c) 2024 Nicholas Gordon

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import sys
import os
import spectral
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

"""
Big O notation for principle component analysis is O(p^2 n + p^3) for n sized input with p variables
Covariance matrix calculation is O(p^2 n)
Eigenvalue decomposition is O(p^3)
"""

def COMPUTE_PCA(hyperspectral_image, n_comp):
    """
    PURPOSE:
        This function performs principle component analysis on a hyperspectral image
        The components are calculated compared to each pixel spectra
    EXPECTED INPUT:
        Two inputs are expected, one of which requires specific formatting
        hyperspectral_image should be 3 dimensional and formated as such (bands, x, y) with bands being the spectra, x and y being the coordinates in the image
        n_comp is an integer indicating how many components should be calculated
    OUTPUT:
        Returns a hyperspectral image with the 0th dimension being each principal component, the x and y values are then the calculated principle components 

    EXAMPLE:
    The hyperspectral image of interest is 200 bands with dimensions 500 by 500
    Formated as (band, x, y) with name hyperspectal_in
    The number of components to calculate in this instance are 10 but can be decided using the function below (COMPUTE_COMPONENTS)
    
    pca_image = COMPUTE_PCA(hyperspectral_in, 10)

    The resultant pca_image will have dimensionality (10,500,500)
    Each 0th dimenson represents a principle component.
    """

    # Reshape the image from (n, x, y) to (n, x*y)
    layers, height, width = hyperspectral_image.shape
    reshaped_data = hyperspectral_image.reshape(layers, height * width)
    
    std_data = StandardScaler().fit_transform(reshaped_data)
    
    # Perform PCA on the pixel spectra, in this case because it's an ENVI file that's dimension 0
    pca = PCA(n_components=n_comp)
    p_components = pca.fit_transform(std_data.T)  # Transpose back for PCA... remember that pesky dimenson zero???
    
    # Reshape the principal components back to (components, height, width)
    pca_image = p_components.T.reshape(n_comp, height, width)
    
    return pca_image


def COMPUTE_COMPONENTS(hyperspectral_image, v_percent):
    """
    PURPOSE:
        This function is used to find the number of components needed to describe a set percentage of variance in the original nth dimensional matrix
    EXPECTED INPUT:
        Two inputs, hyperspectral_image and v_percent
        hyperspectral_image should be 3 dimensional and formated as such (bands, x, y) with bands being the spectra, x and y being the coordinates in the image
        v_percent is the variance percentage you want explained by the components, expected input is a float with 2 digit precision
    OUTPUT:
        Returns an integer value that coincides with number of components necessary to describe the requested percentage of variance

    EXAMPLE:
    The hyperspectral image of interest is 200 bands with dimensions 500 by 500
    Formated as (band, x, y) with name hyperspectal_in
    We want to know how many bands are needed to explain 80% of the variance in our original image

    num_components = COMPUTE_COMPONENTS(hyperspectral_in, 0.80)

    The num_components variable will then be an integer value corresponding to number of components
    """

    # Reshape the image from (layer, x, y) to (layer, x*y) we want to compute the principal components of each pixel spectra
    layers, height, width = hyperspectral_image.shape
    reshaped_data = hyperspectral_image.reshape(layers, height * width)

    #standardize the dataset
    standardized_data = StandardScaler().fit_transform(reshaped_data)
    
    pca = PCA(n_components=None)
    pca_fit = pca.fit_transform(standardized_data.T)

    #find the number of components that satisfies v percentage of variance, specified at call
    n_components = 1 + np.argmax(np.cumsum(pca.explained_variance_ratio_) >= v_percent)


    return n_components


"""
First define the file path to the header (HDR) file, it is CRITICAL that the image (IMG) file is in the SAME DIRECTORY WITH THE SAME NAME!!!!

Then read the hyperspectral image to memory
We start with saying we want to move the axis around by transposing our input file and then swapping the last portions of array (np.moveaxis)
This transformation makes working the the data later more intuitive
Format of the original input is (y,x,bands) which is not impossible but weird to work with
After transformation we're left with (bands,x,y)
The spectral.open_image is taking the path to the header file we defined earlier and then loading the accompanying img file
This is why it is CRITICAL to have the .img file in the SAME directory with the SAME name!!!!
"""

img_input_path = sys.argv[1]
if os.path.exists(img_input_path) == False:
    raise Exception("Image file doesn't exist!! Please try again with a real file...")

img_save_path = sys.argv[2]

print("Loading image into memory")
hyperspectral_img_in = np.transpose(spectral.open_image(img_input_path).load(), (2, 0, 1))

print("Calculating PCA")
#call COMPUTE_COMPONENTS for input to PCA computation
num_components = COMPUTE_COMPONENTS(hyperspectral_img_in, 0.90)
#call PCA_RESULTS to create our hyperspectral PCA image
pca_result = COMPUTE_PCA(hyperspectral_img_in, num_components)

for i in tqdm(range(pca_result.shape[0]), desc="Saving components to file"):
    plt.imsave(f'comp{i}_PCA90.tiff', pca_result[i,:,:], cmap='Greys_r')

#transpose the data back to the expected ENVI format for saving
pca_result = np.transpose(pca_result, (1,2,0))

#define the output path for the file we're going to save, note this includes the file name
hdr_metadata = {
    'lines' : pca_result.shape[0], #x
    'samples' : pca_result.shape[1], #y
    'bands' : pca_result.shape[2],
    'interleave' : 'bsq',
    'data_type' : 4, #32bit float
    'byte_order' : 0 #small endian
}

#finally save the file as ENVI img and hdr format, note the funciton only calls for the hdr file name, the img will be auto generated
spectral.envi.save_image(img_save_path, pca_result, metadata=hdr_metadata, interleave='bsq', force=True)
