## CSC320 Winter 2017 
## Assignment 2
## (c) Kyros Kutulakos
##
## DISTRIBUTION OF THIS CODE ANY FORM (ELECTRONIC OR OTHERWISE,
## AS-IS, MODIFIED OR IN PART), WITHOUT PRIOR WRITTEN AUTHORIZATION 
## BY THE INSTRUCTOR IS STRICTLY PROHIBITED. VIOLATION OF THIS 
## POLICY WILL BE CONSIDERED AN ACT OF ACADEMIC DISHONESTY

##
## DO NOT MODIFY THIS FILE ANYWHERE EXCEPT WHERE INDICATED
##

import numpy as np
import cv2 as cv

# File psi.py define the psi class. You will need to 
# take a close look at the methods provided in this class
# as they will be needed for your implementation
import psi        

# File copyutils.py contains a set of utility functions
# for copying into an array the image pixels contained in
# a patch. These utilities may make your code a lot simpler
# to write, without having to loop over individual image pixels, etc.
import copyutils

#########################################
## PLACE YOUR CODE BETWEEN THESE LINES ##
#########################################

# If you need to import any additional packages
# place them here. Note that the reference 
# implementation does not use any such packages

#########################################
from scipy.linalg import norm
from numpy import unravel_index
#########################################
#
# Computing the Patch Confidence C(p)
#
# Input arguments: 
#    psiHatP: 
#         A member of the PSI class that defines the
#         patch. See file inpainting/psi.py for details
#         on the various methods this class contains.
#         In particular, the class provides a method for
#         accessing the coordinates of the patch center, etc
#    filledImage:
#         An OpenCV image of type uint8 that contains a value of 255
#         for every pixel in image I whose color is known (ie. either
#         a pixel that was not masked initially or a pixel that has
#         already been inpainted), and 0 for all other pixels
#    confidenceImage:
#         An OpenCV image of type uint8 that contains a confidence 
#         value for every pixel in image I whose color is already known.
#         Instead of storing confidences as floats in the range [0,1], 
#         you should assume confidences are represented as variables of type 
#         uint8, taking values between 0 and 255.
#
# Return value:
#         A scalar containing the confidence computed for the patch center
#

def computeC(psiHatP=None, filledImage=None, confidenceImage=None):
    assert confidenceImage is not None
    assert filledImage is not None
    assert psiHatP is not None
    
    #########################################
    ## PLACE YOUR CODE BETWEEN THESE LINES ##
    #########################################
    #print('>computeC')
    #i think you have to loop all pixels in patch
    width = (psiHatP.radius() * 2) + 1
    
    #NOTE: 0 is black => In fill area => C(p) = 0

    x, y = psiHatP.row() - psiHatP.radius() - 1, psiHatP.col() - psiHatP.radius() - 1

    #print('coords', psiHatP.row(), psiHatP.col())
    #print('x', x, 'y', y)

    confidence = confidenceImage[x : x + width, y : y + width] 
    filled = filledImage[x : x + width, y : y + width]

    #component wise multiplication - any non confident cells will have a value of 0
    c_f = np.multiply(confidence, filled)

    #add up all entries to get the total sum and divide by the area of patch
    C = np.sum(c_f) / (width ** 2)

    #shorter solution: return np.sum(np.multiply(confident, filled)) / (width ** 2)
    #########################################
    
    return C

#########################################
#
# Computing the max Gradient of a patch on the fill front
#
# Input arguments: 
#    psiHatP: 
#         A member of the PSI class that defines the
#         patch. See file inpainting/psi.py for details
#         on the various methods this class contains.
#         In particular, the class provides a method for
#         accessing the coordinates of the patch center, etc
#    filledImage:
#         An OpenCV image of type uint8 that contains a value of 255
#         for every pixel in image I whose color is known (ie. either
#         a pixel that was not masked initially or a pixel that has
#         already been inpainted), and 0 for all other pixels
#    inpaintedImage:
#         A color OpenCV image of type uint8 that contains the 
#         image I, ie. the image being inpainted
#
# Return values:
#         Dy: The component of the gradient that lies along the 
#             y axis (ie. the vertical axis).
#         Dx: The component of the gradient that lies along the 
#             x axis (ie. the horizontal axis).
#
    
def computeGradient(psiHatP=None, inpaintedImage=None, filledImage=None):
    assert inpaintedImage is not None
    assert filledImage is not None
    assert psiHatP is not None
    #########################################
    ## PLACE YOUR CODE BETWEEN THESE LINES ##
    #########################################
    
    # Replace these dummy values with your own code
    Dy = 1
    Dx = 0

    #convert the picture to greyscale
    grey_patch = cv.cvtColor(psiHatP.pixels(), cv.COLOR_BGR2GRAY)

    #get the x and y direction of the image gradient
    sobelx = cv.Sobel(grey_patch, cv.CV_64F, 1, 0, ksize=11)
    sobely = cv.Sobel(grey_patch, cv.CV_64F, 0, 1, ksize=11)

    #square both the x and y components
    f_sobelx = np.multiply(sobelx, sobelx)
    f_sobely = np.multiply(sobely, sobely)

    #add together and get the square root (the magnitude)
    total = np.sqrt(f_sobely + f_sobelx)

    #get the index of the maximum gradient
    index = unravel_index(total.argmax(), total.shape)

    #find the associated x and y gradient components
    Dx = sobelx[index[0]][index[1]]
    Dy = sobely[index[0]][index[1]]

    return Dy, Dx

#########################################
#
# Computing the normal to the fill front at the patch center
#
# Input arguments: 
#    psiHatP: 
#         A member of the PSI class that defines the
#         patch. See file inpainting/psi.py for details
#         on the various methods this class contains.
#         In particular, the class provides a method for
#         accessing the coordinates of the patch center, etc
#    filledImage:
#         An OpenCV image of type uint8 that contains a value of 255
#         for every pixel in image I whose color is known (ie. either
#         a pixel that was not masked initially or a pixel that has
#         already been inpainted), and 0 for all other pixels
#    fillFront:
#         An OpenCV image of type uint8 that whose intensity is 255
#         for all pixels that are currently on the fill front and 0 
#         at all other pixels
#
# Return values:
#         Ny: The component of the normal that lies along the 
#             y axis (ie. the vertical axis).
#         Nx: The component of the normal that lies along the 
#             x axis (ie. the horizontal axis).
#
# Note: if the fill front consists of exactly one pixel (ie. the
#       pixel at the patch center), the fill front is degenerate
#       and has no well-defined normal. In that case, you should
#       set Nx=None and Ny=None
#

def computeNormal(psiHatP=None, filledImage=None, fillFront=None):
    assert filledImage is not None
    assert fillFront is not None
    assert psiHatP is not None

    #########################################
    ## PLACE YOUR CODE BETWEEN THESE LINES ##
    #########################################


    grey_patch = psiHatP.filled() / 255

    sobelx = cv.Sobel(psiHatP.filled(), cv.CV_64F, 1, 0, ksize=5)
    sobely = cv.Sobel(psiHatP.filled(), cv.CV_64F, 0, 1, ksize=5)

    Nx, Ny = 0, 0 

    magnitude = np.sqrt(np.multiply(sobelx, sobelx) + np.multiply(sobely, sobely))

    width = (psiHatP.radius() * 2) + 1
    #if only 1 pixel on the front is filled
    if(np.count_nonzero(psiHatP.filled()) <= 1):
        Nx, Ny = None, None
    else:
    # Replace these dummy values with your own code
        Nx =  - np.sum(sobely) // (width ** 2)
        Ny = np.sum(sobelx) // (width ** 2)
        #allows us to ignore unfilled pixels (I think)
    #########################################

    return Ny, Nx




