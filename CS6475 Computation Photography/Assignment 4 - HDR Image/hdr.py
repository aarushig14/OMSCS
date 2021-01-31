""" Building an HDR Image

This file has a number of functions that you need to complete. Please write
the appropriate code, following the instructions on which functions you may
or may not use.

GENERAL RULES:
    1. DO NOT INCLUDE code that saves, shows, displays, writes the image that
    you are being passed in. Do that on your own if you need to save the images
    but the functions should NOT save the image to file.

    2. DO NOT import any other libraries aside from those that we provide.
    You may not import anything else, and you should be able to complete
    the assignment with the given libraries (and in many cases without them).

    3. DO NOT change the format of this file. You may NOT change function
    type signatures (not even named parameters with defaults). You may add
    additional code to this file at your discretion, however it is your
    responsibility to ensure that the autograder accepts your submission.

    4. This file has only been tested in the provided class environment.
    You are responsible for ensuring that your code executes properly in the
    class environment, and that any changes you make outside the
    areas annotated for student code do not impact your performance on the
    autograder system.

Notation
--------
The following symbols are used in "Recovering High Dynamic Range Radiance
Maps from Photographs", by Debevec & Malik (available on Canvas in Files > Papers),
and are used extensively throughout the instructions:

    Z    : pixel intensity value; may be subscripted to indicate position
           in an array, e.g., Zij is the pixel intensity in row i column j
           of a 2D array

    Zmax : maximum pixel intensity (255 for uint8)

    Zmin : minimum pixel intensity (0 for uint8)

    W    : weight of an intensity Z; may be subscripted to indicate
           position in an array, e.g., Wk is the weight of Zk, the
           intensity at position k of a 1D array

    g    : response curve mapping pixel values Z to sensor response

    t    : frame exposure time; may be subscripted to indicate position in
           an array, e.g., ln(tj) is the log of exposure time of frame j

    E    : radiance value of a pixel; may be subscripted to indicate position
           in an array, e.g., ln(Ei) is the log radiance of pixel i
"""
import numpy as np
import scipy as sp
import cv2

import random

Z_MIN = 0
Z_MAX = 255

# from numba import jit
# may be used, not required. (Uncomment and load to use)



def linearWeight(pixel_value):
    """ Linear weighting function based on pixel intensity that reduces the
    weight of pixel values that are near saturation.

    linearWeight(z) is a piecewise linear function that resembles a simple hat,
    given in section 2.1 of Debevek & Malik, as Equation (4).
        z is a pixel intensity value
        Zmax and Zmin are the largest and smallest possible uint8 intensity values.

    Parameters
    ----------
    pixel_value : np.uint8
        A pixel intensity value from 0 to 255

    Returns
    -------
    weight : a single value, of np.float64 type
        The weight corresponding to the input pixel intensity

    """
    # WRITE YOUR CODE HERE.
    limit = 0.5*(Z_MIN + Z_MAX)
    return (pixel_value - Z_MIN) if pixel_value <= limit else (Z_MAX-pixel_value)
    # raise NotImplementedError


def sampleIntensities(images):
    """ Randomly sample pixel intensities from the exposure stack.
    Although D&M used a manual sampling method, we will use the
    following automated method.

    The returned `intensity_values` array has one row for every possible
    pixel value, and one column for each image in the exposure stack. The
    values in the array are filled according to the instructions below.

    Candidate locations are drawn from the middle image of the stack because
    it is expected to be the least likely image to contain over- or
    under-exposed pixels.

    Parameters
    ----------
    images : list<numpy.ndarray>
        A list containing a stack of single-channel (i.e., single color or grayscale)
        layers of an HDR exposure stack

    Returns
    -------
    intensity_values : numpy.array, dtype=np.uint8
        An array containing a uniformly sampled intensity value from each
        exposure layer with shape = (num_intensities, num_images).
        num_intensities is the total number of possible pixel values
        in a uint8 image: one for each value [0...255], inclusive.

    Procedure:
    (1) Initiate the intensity_values array as described above, fill it with zeros
    (2) Using integer division, find the middle image in the exposure stack (images)
        to use as the source for all pixel intensity locations;
    (3) Let mid_img be the middle image in the exposure stack.
    (4) Collect intensity samples from the image stack
        for each possible pixel intensity level Zmin <= Zi <= Zmax:
            (a) Find the locations (r, c) of all candidate pixels in mid_img with value Zi
            (b) If there are no pixels in mid_img with value Zi,
                do nothing, go to the next Zi.
            (c) Else, randomly select one location (r, c) from the candidate pixel locations.
                Set intensity_values[Zi, j] to Zj where:
                    j is the place of the image in images
                    Zj is the intensity of image Ij from the image stack at location (r, c)
    """
    # TODO WRITE YOUR CODE HERE.
    N = Z_MAX - Z_MIN + 1
    P = len(images)
    intensity_values = np.zeros((N, P))
    mid_img = images[int(P/2)]
    for z in range(Z_MIN, Z_MAX+1):
        pixels = np.dstack(np.where(mid_img == z))[0]
        if len(pixels) == 0:
            continue
        ix, iy = random.choices(pixels)[0]
        for j in range(P):
            intensity_values[z, j] = images[j][ix, iy]
    return intensity_values
    # raise NotImplementedError


def computeResponseCurve(intensity_samples, log_exposures, smoothing_lambda, weighting_function):
    """ Find the camera response curve for a single color channel

    The constraints are described in detail in section 2.1 of "Recovering
    High Dynamic Range Radiance Maps from Photographs" by Debevec & Malik
    (available in the course resources material on T-Square). Study the
    constraintMatrixEqns.pdf in the assignment repo and track how the variables
    are used to help in this section.

    The "mat_A_example.png" image file further illustrates the correct structure of
    the constraint matrix. The example was generated for 3 images with 16 colors
    (you need to handle N images with 256 colors). The illustration shows the
    pattern in which pixels should be set by this function; it has a value of one
    in each location that was touched by this function. Your code needs to set
    the appropriate values in the constraint matrix Some entries
    may have a value of 1, but that is not the correct value for all cells.

    You will first fill in mat_A and mat_b with coefficients corresponding to
    an overdetermined system of constraint equations, then solve for the
    response curve by finding the least-squares solution (i.e., solve for x
    in the linear system Ax=b).

        *************************************************************
            NOTE: Use the weighting_function() parameter to get
              the weight, do NOT directly call linearWeight()
        *************************************************************

    Parameters
    ----------
    intensity_samples : numpy.ndarray
        Stack of single channel input values (num_samples x num_images)

    log_exposures : numpy.ndarray
        Log exposure times (size == num_images)

    smoothing_lambda : float
        A constant value used to correct for scale differences between
        data and smoothing terms in the constraint matrix -- source
        paper suggests a value of 100.

    weighting_function : callable
        Function that computes a weight from a pixel intensity

    Returns
    -------
    numpy.ndarray, dtype=np.float64
        Return a vector g(z) where the element at index i is the log exposure
        of a pixel with intensity value z = i (e.g., g[0] is the log exposure
        of z=0, g[1] is the log exposure of z=1, etc.)
    """
    # SETUP CODE PROVIDED
    intensity_range = 255  # difference between min and max possible pixel value for uint8
    num_samples = intensity_samples.shape[0]
    num_images = len(log_exposures)

    # mat_A shape: NxP + [Zmax - (Zmin + 1)] + 1 constraints, N + intensity_range + 1
    mat_A = np.zeros((num_images * num_samples + intensity_range,
                      num_samples + intensity_range + 1), dtype=np.float64)
    mat_b = np.zeros((mat_A.shape[0], 1), dtype=np.float64)
    # END PROVIDED CODE

    # 1. Add data-fitting constraints (the first NxP rows in the array).
    # For each of the k values in the range 0 <= k < intensity_samples.size
    # and the intensities Zij at (i, j) in the intensity_samples array:
    #
    #    Let Wij be the weight of Zij
    #
    #     i. Set mat_A at row k in column Zij to Wij
    #
    #    ii. Set mat_A at row k in column num_samples + i to -Wij
    #
    #   iii. Set mat_b at row k to Wij * log_exposure[j]
    #
    # TODO WRITE YOUR CODE HERE
    k = 0
    for r in range(num_samples):
        for c in range(num_images):
            k += 1
            Zij = int(intensity_samples[r][c])
            Wij = mat_A[k][Zij] = weighting_function(Zij)
            mat_A[k][num_samples + r] = -Wij
            mat_b[k] = Wij * log_exposures[c]

    # -------------------------------------------
    # 2. Add smoothing constraints (the N-2 rows after the data constraints).
    # Beginning in the first row after the last data constraint, loop over each
    # value Zk in the range Zmin+1 <= Zk <= Zmax-1:
    #
    #   Let Wk be the weight of Zk
    #
    #     i. Set mat_A in the current row at column Zk - 1 to
    #        Wk * smoothing_lambda
    #
    #    ii. Set mat_A in the current row at column Zk to
    #        -2 * Wk * smoothing_lambda
    #
    #   iii. Set mat_A in the current row at column Zk + 1 to
    #        Wk * smoothing_lambda
    #
    #   Move to the next row
    #
    #   *** WE STRONGLY RECOMMEND THAT YOU SAVE MAT_A AS AN IMAGE. ***
    #   Compare your mat_A image to the provided sample, the sections
    #   should have a similar, though much larger, pattern.
    #
    # TODO WRITE YOUR CODE HERE
    offset = num_samples * num_images
    for Zk in range(Z_MIN+1, Z_MAX):
        Wk = weighting_function(Zk)
        mat_A[offset + Zk][Zk-1] = Wk * smoothing_lambda
        mat_A[offset + Zk][Zk] = -2 * Wk * smoothing_lambda
        mat_A[offset + Zk][Zk+1] = Wk * smoothing_lambda

    # -------------------------------------------
    # 3. Add color curve centering constraint (the last row of mat_A):
    #       Set the value of mat_A in the last row and
    #       column (Zmax - Zmin) // 2 to the constant 1.
    #
    # TODO WRITE YOUR CODE HERE
    mat_A[-1][(Z_MAX-Z_MIN)//2] = 1

    # print(mat_A.shape, " ", mat_b.shape)
    # out = np.zeros(shape=mat_A.shape, dtype=mat_A.dtype)
    # cv2.normalize(mat_A, out, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    # out[mat_A == 0] = 255
    # out[mat_A != 0] = 0
    # cv2.imwrite("matrix.png", out)


    # -------------------------------------------
    # 4. Solve the system Ax=b. Recall from linear algebra that the solution
    # to a linear system can be obtained:
    #
    #   Ax = b
    #   A^-1 dot A dot x = A^-1 dot b
    #   x = A^-1 dot b
    #
    #   NOTE: "dot" here is the dot product operator. The numpy *
    #         operator performs an element-wise multiplication which is
    #         different. So don't use it -- use np.dot instead.
    #
    #     i. Get the Moore-Penrose psuedo-inverse of mat_A (Numpy has a
    #        function to do this)
    #
    #    ii. Multiply inv_A with mat_b (remember, use dot not *) to get x.
    #        If done correctly, x.shape should be 512 x 1
    #
    # TODO WRITE YOUR CODE HERE
    inv_A = np.linalg.pinv(mat_A)
    x = np.dot(inv_A, mat_b)

    # -------------------------------------------
    # Assuming that you set up your equation so that the first elements of
    # x correspond to g(z); otherwise you can change this to match your
    # constraints

    # CODE PROVIDED (2 LINES, use these or write your own)
    g = x[0:intensity_range + 1]
    # return response curve
    return g[:, 0]
    raise NotImplementedError


def computeRadianceMap(images, log_exposure_times, response_curve, weighting_function):
    """ Calculate a radiance map for each pixel from the response curve.

    Parameters
    ----------
    images : list
        Collection containing a single color layer (i.e., grayscale)
        from each image in the exposure stack. (size == num_images)

    log_exposure_times : numpy.ndarray
        Array containing the log exposure times for each image in the
        exposure stack (size == num_images)

    response_curve : numpy.ndarray
        Least-squares fitted log exposure of each pixel value z

    weighting_function : callable
        Function that computes the weights

    Returns
    -------
    numpy.ndarray(dtype=np.float64)
        The image radiance map (in log space)
    """
    # SETUP CODE PROVIDED
    img_shape = images[0].shape
    img_rad_map = np.zeros(img_shape, dtype=np.float64)
    # END PROVIDED CODE

    # 1. Construct the radiance map -- for each pixel i in the output (note
    #    that "i" is a (row, col) location in this case):
    #
    #     i. Get all Zij values -- the intensities of pixel i from each
    #        image Ik in the input stack
    #
    #    ii. Get all Wij values -- the weight of each Zij (use the weighting
    #        function parameter)
    #
    #   iii. Calculate SumW - the sum of all Wij values for pixel i
    #
    #    iv. If SumW > 0, set pixel i in the output equal to the weighted
    #        average radiance (i.e., sum(Wij * (g(Zij) - ln(tj))) / SumW),
    #        otherwise set it to the log radiance from the middle image in
    #        the exposure stack (i.e., calculate the right hand side of
    #        Eq. 5: ln(Ei) = g(Zij) - ln(tj) from the source paper for
    #        just the middle image, rather than the average of the stack)
    #
    # TODO WRITE YOUR CODE HERE
    Zij = []
    Wij = []
    for i in range(images[0].size):
        r = int(i / img_shape[1])
        c = i % img_shape[1]
        intensities = [image[r][c] for image in images]
        Zij.append(intensities)
        Wij.append([weighting_function(z) for z in intensities])
    SumW = np.sum(Wij, axis=1)
    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            pixel_loc = (i) * img_shape[1] + j
            if SumW[pixel_loc] > 0:
                img_rad_map[i, j] = np.sum(Wij[pixel_loc] *
                                           (np.array([response_curve[int(z)] for z in Zij[pixel_loc]]) -
                                            np.array(log_exposure_times))) / np.sum(SumW[pixel_loc])
            else:
                middle_img = len(images) // 2
                img_rad_map[i, j] = response_curve[int(Zij[pixel_loc][middle_img])] - log_exposure_times[middle_img]
    return img_rad_map
    raise NotImplementedError


def computeHistogram(image):
    """ Calculate a histogram for each image.
    Parameters
    ----------
    image: numpy.ndarray
        the three channel basic_hdr_image image produced by the main function in main.py.

    Returns
    -------
    numpy.ndarray(dtype=np.uint64)
        The image histogram bin counts; which should be an array of shape (256,1)
    """
    # 1. Convert your image array from BGR color-space to HSV color-space.
    #       Then, isolate the Value channel (V) as a numpy.array.
    #       You may convert your image into HSV color-space using cv2.cvtColor.
    # 2. Construct the histogram binning
    #       For each pixel in the V channel of the HSV image passed in
    #       construct an array where each entry in the array is a count
    #       of all pixels with that V value.
    #
    # TODO WRITE YOUR CODE HERE

    hsv_image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2HSV)
    value = hsv_image[:, :, 2]
    freq = np.zeros(Z_MAX-Z_MIN+1)
    for i in range(hsv_image.shape[0]):
        for j in range(hsv_image.shape[1]):
            freq[value[i][j]] += 1
    return freq.astype(dtype=np.uint64)[:,np.newaxis]
    raise NotImplementedError


def computeCumulativeDensity(histogram):
    """ Calculate a cumulative density array for the histogram bins.
    Parameters
    ----------
    histogram: numpy.ndarray(dtype=np.uint64)
        Bins containing each V value in an array

    Returns
    -------
    numpy.ndarray(dtype=np.uint64)
        The cumulative density of the histogram;
        should  be an array of shape (256,1)
    """

    # 1. Construct the cumulative density array
    #       For each histogram bin, compute the cumulative number of pixels.
    #       This is the sum of the pixel count in that bin plus the count
    #       of all previous bins.

    #       This can be thought of as:
    #           cumulative_density[x] = histogram[x] + cumulative_density[x-1]
    #       where x is the current bin value.
    # TODO WRITE YOUR CODE HERE
    cdf = np.zeros(histogram.shape)
    for i in range(histogram.shape[0]):
        cdf[i] = cdf[i-1] + histogram[i]
    return cdf.astype(dtype=np.uint64)
    raise NotImplementedError


def applyHistogramEqualization(image, cumulative_density):
    """ Apply the cumulative density calculated on each pixel in the original image
    Parameters
    ----------
    image: numpy.ndarray
        A numpy array of dimensions (HxWx3) and type np.uint8

    cumulative_density: numpy.ndarray(dtype=np.uint64)
        cumulative density of each possible pixel value in the image

    Returns
    -------
    numpy.ndarray
        A numpy array of dimensions (HxWx3) and type np.uint8
    """
    # 1. Normalize the cumulative density array so that it is scaled
    #       between a minimum value of 0 and a maximum value of 255.
    #       This is sometimes called min-max normalization.
    #       Round the result using np.round (which rounds slightly
    #       differently than what you may have learned long ago).
    #
    # 2. Convert your image into the HSV color space.
    #
    # 3. Loop through each pixel of only the V channel and replace its value
    #       with the value of the normalized cumulative density function
    #       at that value.   i.e. if image[i,j] = k, replace k with ncdf[k]
    #
    # 4. Convert the HSV image with the altered V channel back into
    #       BGR colorspace and return.

    # TODO WRITE YOUR CODE HERE
    ncdf = np.zeros(shape=cumulative_density.shape, dtype=cumulative_density.dtype)
    ncdf = np.round(cv2.normalize(cumulative_density, ncdf.astype(dtype=np.float64), alpha=0, beta=255, norm_type=cv2.NORM_MINMAX))
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    for i in range(hsv_img.shape[0]):
        for j in range(hsv_img.shape[1]):
            hsv_img[i, j, 2] = ncdf[hsv_img[i, j, 2]]
    return cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
    raise NotImplementedError


def bestHDR(image):
    """ In this function, implement your improved histogram method.
        See the readme for guidance. You may only use the the class environment
        and the libraries random and numba.

        This function will be run by TAs and graded by hand.

    Parameters
    ----------
    image: numpy.ndarray
        A numpy array of dimensions (HxWx3) and type np.uint8

    Returns
    -------
    numpy.ndarray
        A numpy array of dimensions (HxWx3) and type np.uint8
    """
    # TODO WRITE YOUR CODE HERE
    lab_image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2LAB)
    l = lab_image[:, :, 0]
    hist = np.zeros(Z_MAX-Z_MIN+1)
    for i in range(l.shape[0]):
        for j in range(l.shape[1]):
            hist[l[i][j]] += 1
    cdf = computeCumulativeDensity(hist)
    cdf = cdf / np.sum(hist)
    cdf *= np.argmax(hist)
    ncdf = np.zeros(shape=cdf.shape, dtype=cdf.dtype)
    ncdf = np.round(cv2.normalize(cdf, ncdf.astype(dtype=np.float64), alpha=0, beta=255, norm_type=cv2.NORM_MINMAX))
    for i in range(0, lab_image.shape[0]):
        for j in range(0, lab_image.shape[1]):
            lab_image[i, j, 0] = ncdf[lab_image[i, j, 0]]
    return cv2.cvtColor(lab_image, cv2.COLOR_LAB2BGR)

    raise NotImplementedError