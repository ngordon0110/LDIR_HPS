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
import numpy as np
import matplotlib.pyplot as plt
import spectral
import pandas as pd
from tqdm import tqdm


def LIB_SEARCH(img_in, library_in, match_thres=0.6, save_thres=.02, legend=False):
    """
    PURPOSE:
        This function will search through input img_in and compare the spectra at each pixel to the USGS specular IR library.

    EXPECTED INPUT:
        ***it's expected that the input img_in and library_in have been normalized using MAX_NORM, derivative and magnitude will be calculated at call***
        there are a total of five inputs expected, two of which expect specific formatting
        ! - img_in should be 3 dimensional and formated as such (bands, x, y) with bands being the spectra, x and y being the coordinates in the image
        ! - library_in should be a pandas dataframe with keys associated to each spectra and the key representing the mineral ID
        match_thresh is a float value used as a cutoff for cosine similarity matching between the img_in pixel spectr and USGS library
        save_thres is a float value used to determine the cutoff for saving similarity map. The percentage is how many pixels of x mineral
            are present in the original image. I.E. if there are 1000 pixels in the image and 20 are a match for calcite then a calcite similarity
            map will be saved.
        finally legend is a boolean value T/F used to determine if the exported image will have a legend including title and color bar.

        ! = required ipnut
        
    OUTPUT:
        In it's current form this function has no particular output but instead will just save the similarity map.
    """

    #vectorization to calculate derivative and magnitude along 0th axis of n-dimensional array
    #the magnitude calculation was originally done with np.linalg.norm but is being cranky as of 11-27-24
    #instead magnitude is calculated manually with the numpy summation and square root function
    img_in = np.gradient(hyperspectral_img_in, axis=0)
    img_in_mag = np.sqrt(np.sum(img_in**2, axis=0))

    #Once again utilize vectorization but this time we want to retain the dataframe keys
    #To do this we spawn sub routines or sub functions with lambda to cast the derivative and magnitude calculations while utilizing the .apply() functionality thereby retaining original dataframe keys
    library = library_in.apply(lambda col: np.gradient(col.values), axis=0)
    library_mag = library.apply(lambda col: np.linalg.norm(col.values), axis=0)

    #with the data preprocessed we can start to search the library for mineral ID matches
    #loop through the library but skip the first column, in this case the USGS libraries first column is just wavenumbers
    for minName in tqdm(library.columns[1:], desc="Searching library"):
        #for each mineral create a cosine similarity map, this does so by using tensordot to calculate the dot product
        #between the current mineral in the library and the input image
        #we use tensordot from numpy because it allows for matrix calculations along specified axes of input arrays
        #because our arrays ***SHOULD*** have the same first axis dimensions we can use vectorization here to speed up the process
        #once we have a dot product we can divide that by the product of the arrays magnitudes using, you guessed it, vectorization
        #to speed up that process
        similarity_map = np.tensordot(library[minName], img_in, axes=([0], [0])) / (img_in_mag * library_mag[minName])
        #once a similarity map exists check to see if any of the pixels correspond to a set match threshold
        if np.any(similarity_map > match_thres):
            #if true then create a mask of the dataset, this creates a 2d array of booleans
            mask = similarity_map > match_thres
            #.sum() acts kind of similar to if statement default logic, so all true booleans are summed effectively
            #creating a number of pixels that match the threshold.
            count = mask.sum()
            #print(f"{minName}:{count}")
            #this checks to see what percent of the image is explained by the matched spectra, threshold set at call
            if count / (img_in.shape[1] * img_in.shape[2]) > save_thres:
                #check if legend is wanted
                if legend:
                    plt.imshow(similarity_map, cmap='turbo', vmin=0, vmax=1)
                    plt.colorbar(label='Similarity to USGS library')
                    plt.title(f"Cosine similarity for {minName}")
                    plt.axis('off')
                    plt.savefig(f'{minName}_similarity.tiff', format='tiff', dpi=400, bbox_inches='tight')
                    plt.close()
                else:
                    plt.imsave(f'{minName}_similarity.tiff', similarity_map, cmap='turbo', vmin=0, vmax=1)



def MAX_NORM(array_in):
    """
    PURPOSE:
        Normalize input array to the maximum value of the input expressed as a percentage

    EXPECTED INPUT:
        The only input is an array, this can be any n>0 dimensional array
        
    OUTPUT:
        returns input array with values along 0th axis normalized to max expressed as a percentage
    """
    
    #utilize vectorization instead of looping through to find max value in first position of array
    max_value = np.max(array_in, axis=0)
    #once again we just cast the arrays to normalize, very neat, thanks python
    out = array_in / max_value
    return out

img_input_path = sys.argv[1]
if os.path.exists(img_input_path) == False:
    raise Exception("Image file doesn't exist!! Please try again with a real file...")

library_input_path = sys.argv[2]
if os.path.exists(library_input_path) == False:
    raise Exception("Library file doesn't exist!! Please try again with a real file...")

#load the datasets to memory while doing some preprocessing
print("Loading image and library into memory")
#transpose the hyperspectral image to match the expected input format
hyperspectral_img_in = MAX_NORM(np.transpose(spectral.open_image(img_input_path).load(), (2,0,1)))
#load library with pandas as a dataframe, keys correspond to csv file headers
USGS_in = MAX_NORM(pd.read_csv(library_input_path))

print("Processing data")
LIB_SEARCH(hyperspectral_img_in, USGS_in, match_thres=float(sys.argv[3]), save_thres=float(sys.argv[4]), legend=bool(sys.argv[5]))
