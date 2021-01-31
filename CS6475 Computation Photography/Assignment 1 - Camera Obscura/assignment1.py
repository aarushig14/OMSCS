""" Camera Obscura - Post-processing
This file has a number of functions that you need to fill out in order to
complete the assignment. Please write the appropriate code, following the
instructions on which functions you may or may not use.

Notes
-----
You are only allowed to use cv2.imread, c2.imwrite and cv2.copyMakeBorder from 
cv2 library. You should implement convolution on your own.
GENERAL RULES:
    1. DO NOT INCLUDE code that saves, shows, displays, writes the image that
    you are being passed in. Do that on your own if you need to save the images
    but these functions should NOT save the image to disk.
    2. DO NOT import any other libraries aside from those that we provide.
    You should be able to complete the assignment with the given libraries
    (and in many cases without them).
    3. DO NOT change the format of this file. You may NOT change function
    type signatures (not even named parameters with defaults). You may add
    additional code to this file at your discretion, however it is your
    responsibility to ensure that the autograder accepts your submission.
    4. This file has only been tested in the course environment.
    You are responsible for ensuring that your code executes properly in the
    course environment, and that any changes you make outside the
    areas annotated for student code do not impact your performance on the
    autograder system.
"""
import numpy as np
import cv2  
import scipy as sp
import scipy.signal as sg
 

def computeOutput2D(_img,kernel,paddingsize):
    row,col = _img.shape
    output = np.zeros(_img.shape)
    for i in range(paddingsize,row-paddingsize):
        for j in range(paddingsize,col-paddingsize):
            for k in range(-paddingsize,paddingsize+1):
                for l in range(-paddingsize,paddingsize+1):
                    output[i,j] = output[i,j] + _img[i+k,j+l]*kernel[paddingsize+k,paddingsize+l]
    output = np.rint(output)
    output[output<0]=0
    output[output>255]=255
    return output.astype(np.uint8)
 
def applyConvolution(image, filter,borderType=cv2.BORDER_REPLICATE):
    """Apply convolution operation on image with the filter provided. 
    Return a result with the same height and width as the original.
    You may assume that the filter is square of size MxM, where M is odd.
    You may assume the filter is symmetric in x and y.

    You must use create own convolution method, but you are not *required*
    to use nested for loops <yawn>.  *Alternate* options:
    
    1.) You may also implement this with numpy strides to 
    perform the convolution as a broadcast operation.

    2.) You might consider trying to create separable filters.  Although the
    filter you are provided is not guaranteed to be separable, you can 
    approximate a separable filter to a sufficient degree using Singular Value
    Decomposition (Szeliski, section 3.2.1, pg. 102)  The final sentence
    of the final paragraph of that section gives a hint on how to combine 
    ranks for increasingly better solutions.

    Option #2 might be challenging, so if you wish to apply this approach,
    you can ask for more information from your friendly neighborhood TAs 
    on Piazza.  

    Make sure your function can handle 2d or 3d images, the autograder will
    test both.

    Parameters
    ----------
    image : numpy.ndarray
        A numpy array of dimensions (HxWxD) and type np.uint8
    filter: numpy.ndarray
        A numpy array of dimensions (M.N) and type np.float64
    border: cv2 border type. 
    Returns
    -------
    output : numpy.ndarray
        A numpy array of dimensions (HxWxD) and type np.uint8
    """
    # step 1 : add padding
    paddingsize = int((filter.shape[0])/2)
    _img = cv2.copyMakeBorder(image, paddingsize, paddingsize, paddingsize, paddingsize, borderType)
    _img = _img.astype(np.float64)
    
    # step 2 : flip kernel along X and Y
#     filter = filter[::-1,::-1]
    
    # step 3 : calculate o/p matrix
    if (_img.ndim == 3):
        blue, green, red = np.moveaxis(_img, 2, 0)
        blue = computeOutput2D(blue,filter,paddingsize)
        green = computeOutput2D(green,filter,paddingsize)
        red = computeOutput2D(red,filter,paddingsize)
        output = np.dstack([blue,green,red])
    else:
        output = computeOutput2D(_img,filter,paddingsize)
    
    # step 4 : remove padding and return
    return output[paddingsize:-paddingsize:,paddingsize:-paddingsize:]

 
def applyAveragingFilter(image,kernel_size,borderType=cv2.BORDER_REFLECT101):
    """Filter noise from the image by using applyConvolution() and an averaging kernel.
    Parameters
    ----------
    image : numpy.ndarray
        A numpy array of dimensions (HxWxD) and type np.uint8

    kernel_size: uint
        The size of one side of a square-shaped kernel (always odd). You will create the
        filter and apply it to the image using the applyConvolution method.

    borderType: opencv border types
        The approach by which the convolution will handle borders.

    
    Returns
    -------
    output : numpy.ndarray
        A numpy array of dimensions (HxWxD) and type np.uint8
    """
    size = kernel_size*kernel_size
    kernel = (np.ones(size).reshape(kernel_size,kernel_size))/size
    return applyConvolution(image,kernel,borderType)

def getGaussianKernel(kernel_size,sigma = 1.):
    kernel = np.zeros(kernel_size*kernel_size).reshape(kernel_size,kernel_size)
    mid = int((kernel_size-1)/2)
    for i in range(-mid,mid+1):
        for j in range(-mid,mid+1):
            kernel[mid+i][mid+j] = np.power(np.e,-(i*i + j*j)/(2*sigma*sigma))/(2*np.pi*sigma*sigma)
    return kernel/np.sum(kernel)

def applyGaussianFilter(image, kernel_size, sigma = 1., borderType=cv2.BORDER_REFLECT101):
    """Filter noise from the image by using applyConvolution() and a gaussian filter.
    The gaussian filter elements should sum to one.  
    

    Parameters
    ----------
    image : numpy.ndarray
        A numpy array of dimensions (HxWxD) and type np.uint8.  The image may have 1 or 3 channels.
    
    kernel_size: uint
        The size of one side of a square-shaped kernel (always odd).

    sigma: float
        The standard deviation of the gaussian kernel.

    borderType: opencv border types
        The approach by which the convolution will handle borders.
    
    Returns
    -------
    output : numpy.ndarray
        A numpy array of dimensions (HxWxD) and type np.uint8.  
    """
    kernel = getGaussianKernel(kernel_size,sigma)
    return applyConvolution(image,kernel,borderType)

def getSharpKernel(kernel_size,alpha_weight,beta_weight,gaussian_std):
    identical = np.zeros((kernel_size,kernel_size))
    mid = int(kernel_size/2)
    identical[mid,mid] = 1
    kernel = (alpha_weight/(alpha_weight-beta_weight))*identical - (beta_weight/(alpha_weight-beta_weight))*getGaussianKernel(kernel_size,gaussian_std)
    return kernel

def sharpenImage(image,kernel_size,alpha_weight,beta_weight,gaussian_std,borderType=cv2.BORDER_REFLECT101):
    """Sharpen the image. Call applyConvolution with an image sharpening kernel
    and gaurantee an unchanged local mean.

    Here, you will apply a Gaussian blur to the image and subtract the result
    (weighted by beta) from the original image (weighted by alpha),
    with alpha > beta. This final result is divided by the difference of alpha 
    and beta.
    
    Unsharp masking:
    f_sharp = alpha * f - beta * f_blur

    To guarantee an unchanged local mean in the image:
    f_sharp = (alpha * f - beta * f_blur ) / (alpha - beta)

    Thought exercise: How would one go about creating a kernel that  
    can be used with one simple call to applyConvolution?  
    Hint 1:  Consider the identity matrix. 
    Hint 2:  Remember that convolution is associative.
    Hint 3:  Ask TAs on Piazza.
    Note: Not required to implement the solution this way.  

    Parameters
    ----------
    image : numpy.ndarray
        A numpy array of dimensions (HxWxD) and type np.uint8
    
    kernel_size: uint
        The size of one side of the kernel (always odd).
        
    alpha_weight: float 
        A fraction (greater than beta) to weight the original image.

    beta_weight: float
        A fraction (less than alpha) to weight the low-pass filter 
        (Gaussian).
    
    gaussian_std: float
        The standard deviation for the Gaussian filter.

    borderType: opencv border types
        The approach by which the convolution will handle borders.
    
    
    Returns
    -------
    output : numpy.ndarray
        A numpy array of dimensions (HxWxD) and type np.uint8
    """
    kernel = getSharpKernel(kernel_size,alpha_weight,beta_weight,gaussian_std)
    return applyConvolution(image,kernel,borderType)

if __name__ == "__main__":
    original_image = cv2.imread("original_image.jpg", cv2.IMREAD_GRAYSCALE)
    applyAveragingFilter(original_image,3)
    applyGaussianFilter(original_image,3,1.)
    sharpenImage(original_image,5,2,1,1.)
    # WRITE YOUR CODE HERE.
    # Read original_image.jpg and pass it to  applyAveragingFilter(), applyGaussianFilter() and sharpenImage()
    # Submit:
    #   scene.jpg
    #   setup.jpg
    #   original_image.jpg,
    #   gaussian_filtered_image.jpg,
    #   averaging_filtered_image.jpg,
    #   sharp_image.jpg
    # Use parameters you think might improve your image's quality.
    pass
