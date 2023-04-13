# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 01:26:25 2023

@author: rakes
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import skimage.io as io
import skimage.filters as flt
#matplotlib inline
# since we can't use imports
import numpy as np
import scipy.ndimage.filters as flt
import warnings
#from medpy.filter.smoothing import anisotropic_diffusion
import os 
import cv2 as cv
import skimage.color
from skimage import filters
from skimage import feature
from PIL import Image
def anisodiff(img,niter=1,kappa=50,gamma=0.1,step=(1.,1.),sigma=0, option=1,ploton=False):
	"""
	Anisotropic diffusion.

	Usage:
	imgout = anisodiff(im, niter, kappa, gamma, option)

	Arguments:
	        img    - input image
	        niter  - number of iterations
	        kappa  - conduction coefficient 20-100 ?
	        gamma  - max value of .25 for stability
	        step   - tuple, the distance between adjacent pixels in (y,x)
	        option - 1 Perona Malik diffusion equation No 1
	                 2 Perona Malik diffusion equation No 2
	        ploton - if True, the image will be plotted on every iteration

	Returns:
	        imgout   - diffused image.

	kappa controls conduction as a function of gradient.  If kappa is low
	small intensity gradients are able to block conduction and hence diffusion
	across step edges.  A large value reduces the influence of intensity
	gradients on conduction.

	gamma controls speed of diffusion (you usually want it at a maximum of
	0.25)

	step is used to scale the gradients in case the spacing between adjacent
	pixels differs in the x and y axes

	Diffusion equation 1 favours high contrast edges over low contrast ones.
	Diffusion equation 2 favours wide regions over smaller ones.

	Reference: 
	P. Perona and J. Malik. 
	Scale-space and edge detection using ansotropic diffusion.
	IEEE Transactions on Pattern Analysis and Machine Intelligence, 
	12(7):629-639, July 1990.

	Original MATLAB code by Peter Kovesi  
	School of Computer Science & Software Engineering
	The University of Western Australia
	pk @ csse uwa edu au
	<http://www.csse.uwa.edu.au>

	Translated to Python and optimised by Alistair Muldal
	Department of Pharmacology
	University of Oxford
	<alistair.muldal@pharm.ox.ac.uk>

	June 2000  original version.       
	March 2002 corrected diffusion eqn No 2.
	July 2012 translated to Python
	"""

	# ...you could always diffuse each color channel independently if you


	# initialize output array
	img = img.astype('float32')
	imgout = img.copy()

	# initialize some internal variables
	deltaS = np.zeros_like(imgout)
	deltaE = deltaS.copy()
	NS = deltaS.copy()
	EW = deltaS.copy()
	gS = np.ones_like(imgout)
	gE = gS.copy()

	# create the plot figure, if requested
	if ploton:
		import pylab as pl
		from time import sleep

		fig = pl.figure(figsize=(20,5.5),num="Anisotropic diffusion")
		ax1,ax2 = fig.add_subplot(1,2,1),fig.add_subplot(1,2,2)

		ax1.imshow(img,interpolation='nearest')
		ih = ax2.imshow(imgout,interpolation='nearest',animated=True)
		ax1.set_title("Original image")
		ax2.set_title("Iteration 0")

		fig.canvas.draw()

	for ii in np.arange(1,niter):

		# calculate the diffs
		deltaS[:-1,: ] = np.diff(imgout,axis=0)
		deltaE[: ,:-1] = np.diff(imgout,axis=1)

		if 0<sigma:
			deltaSf=flt.gaussian_filter(deltaS,sigma);
			deltaEf=flt.gaussian_filter(deltaE,sigma);
		else: 
			deltaSf=deltaS;
			deltaEf=deltaE;
			
		# conduction gradients (only need to compute one per dim!)
		if option == 1:
			gS = np.exp(-(deltaSf/kappa)**2.)/step[0]
			gE = np.exp(-(deltaEf/kappa)**2.)/step[1]
		elif option == 2:
			gS = 1./(1.+(deltaSf/kappa)**2.)/step[0]
			gE = 1./(1.+(deltaEf/kappa)**2.)/step[1]

		# update matrices
		E = gE*deltaE
		S = gS*deltaS

		# subtract a copy that has been shifted 'North/West' by one
		# pixel. don't as questions. just do it. trust me.
		NS[:] = S
		EW[:] = E
		NS[1:,:] -= S[:-1,:]
		EW[:,1:] -= E[:,:-1]

		# update the image
		imgout += gamma*(NS+EW)

		if ploton:
			iterstring = "Iteration %i" %(ii+1)
			ih.set_data(imgout)
			ax2.set_title(iterstring)
			fig.canvas.draw()
			# sleep(0.01)

	return imgout

def despeckle_black_removal(img):
    binary_mask = img > 105
    # Convert the data type to 'uint8'
    binary_mask = binary_mask.astype('uint8')
    
    #fig, ax = plt.subplots()
    #plt.imshow(binary_mask, cmap="gray")
    #plt.title("Threshold applied")
    
    
    # threshold the image to create a binary image
    ret, threshold = cv.threshold(binary_mask, 105, 255, cv.THRESH_BINARY)
    
    #find all the contours in the binary image
    
    contours, ret = cv.findContours(binary_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    #defining img_out
    img_out=binary_mask
    
    # Iterate through the contours and draw a rectangle around each one
    for contour in contours:
        if cv.contourArea(contour) < 7:
            img_out = cv.drawContours(binary_mask, [contour], -1, (0,0,0), -1)
    if contours:
       return img_out
    else:
        return binary_mask
        
def despeckable_large_only(img):
    # Find all the contours in the image
    contours1, hierarchy = cv.findContours(img, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    
    # Find the contour with the largest area
    largest_contour = max(contours1, key=cv.contourArea)
    
    # Create a mask image filled with zeros
    mask = np.zeros_like(img)
    
    # Draw the largest contour on the mask
    img_only_larg=cv.drawContours(mask, [largest_contour], 0, 255, -1)
    
    # Create an output image using the mask
    output = cv.bitwise_and(img, img, mask=mask)
    
    return output

#path to the folder having original images 
inputfolder = "C:\\Users\\rakes\\Dropbox (ASU)\\registered_images\\G6a\\AMG6a_160000C_Registration"


#path to the folder to save processed images
outputfolder="D:\Processed Registered Images\G6A\AMGa_160000C_Processed"

# get a list of all the image files in the input folder
image_files = [f for f in os.listdir(inputfolder) if f.endswith('.png')]

counter = 0
# iterate through the list of image files
#for image in image_files:
for image_file in image_files:
    img = io.imread(os.path.join(inputfolder, image_file))
    #plt.imshow(img)
    #plt.show()
    
    # Apply filter to image
    #filtered_image = anisodiff(image, niter=8, kappa=10, gamma=0.068182)
    
    # Plot filtered image
    #plt.imshow(filtered_image)
    #plt.show()
    # read the image
    #img = io.imread(os.path.join(inputfolder, image))
    #print(path)
    #for images in os.listdir(path)
    #img=io.imread("C:\\Users\\rakes\\Dropbox (ASU)\\registered_images\\G6a\\AMG6a_050000C_Registration\\AMG6a__rec_Tar00000001.png")
    fimg=anisodiff(img, niter=8, kappa=10, gamma=0.068182)   
    #plt.figure(1)
    #plt.imshow(fimg,cmap=plt.cm.gray)
    #plt.title("Anisotropic filtered image")
    img_no_speckles=despeckle_black_removal(fimg)  
    output=despeckable_large_only(img_no_speckles)
    # Normalize the image
    norm_img_out = cv.normalize(output, None, 0, 255, cv.NORM_MINMAX)
    plt.figure(2)
    plt.imshow(norm_img_out,cmap=plt.cm.gray)
    plt.show()
    
    # Check the value of the white color
    if 255 in np.unique(norm_img_out.ravel()):
        print("White color is 255")
    else:
        print("White color is not 255")
    counter+=1
    print("processed image:",counter)
    #plt.title("1st despeckle")
    # save the processed image in the output folder
    try:
       cv.imwrite(os.path.join(outputfolder, image_file), norm_img_out, [cv.IMWRITE_PNG_COMPRESSION, 9])
    except Exception as e:
       print("Error: ", e)
       