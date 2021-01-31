import cv2
import numpy as np
from matplotlib import pyplot as plt


# Global Variables

# Base images
INPDIR = "./"

# Output directory
OUTDIR = "./"

# path to comparison images
COMPDIR = "./"

# Filenames for comparison
FILE_NAMES = np.sort(['fig8f_07_result.png', 'fig8_08_back_seam_result.png', 'fig8c_07_result.png',
              'fig8_08_backward_result.png', 'fig9_08_forward_result.png', 'fig5_07_result.png',
              'fig8_08_forward_result.png', 'fig8d_07_result.png', 'fig9_08_backward_result.png',
              'fig8_08_forward_seam_result.png'])

# Variable to set energy mode - {"backward" , "forward"}
ENERGY_FUNC = "backward"


######################
## MINIMUM FUNCTION ##
######################


# @params - M : cumulative energy map; 
#           {i,j}: position of current pixel; 
#           axis : {1 for vertical, 0 for horizontal}
# @returns - minimum energy value and the its index position.
def get_min(M, i, j, axis=1):
    if axis == 0:
        minidx = i
        minimum = M[i, j - 1]
        if i != 0 and minimum > M[i - 1, j - 1]:
            minimum = M[i - 1, j - 1]
            minidx = i - 1
        if i != M.shape[0] - 1 and minimum > M[i + 1, j - 1]:
            minimum = M[i + 1, j - 1]
            minidx = i + 1
        return minimum, minidx
    else:
        minidx = j
        minimum = M[i - 1, j]
        if j != 0 and minimum > M[i - 1, j - 1]:
            minimum = M[i - 1, j - 1]
            minidx = j - 1
        if j != M.shape[1] - 1 and minimum > M[i - 1, j + 1]:
            minimum = M[i - 1, j + 1]
            minidx = j + 1
        return minimum, minidx


# @params - img = input image; 
#           M = cumulative energy map; 
#           {i,j} = position of current pixel; 
#           axis = {1 for vertical, 0 for horizontal}
# @returns - minimum energy value
def get_min_cost(img, M, i, j, axis=1):
    row, col = img.shape
    if axis == 1:
        rside = int(img[i, (j + 1) % col])
        lside = int(img[i, j - 1])
        ctop = int(img[i - 1, j])

        mleft = int(M[i - 1, j - 1])
        mctop = int(M[i - 1, j])
        mright = int(M[i - 1, (j + 1) % col])

        CU = np.abs(rside - lside)
        CL = CU + np.abs(ctop - lside)
        CR = CU + np.abs(ctop - rside)

        return min(mleft + CL, mctop + CU, mright + CR)

    else:
        top = int(img[i - 1, j])
        bottom = int(img[(i + 1) % row, j])
        cleft = int(img[i, j - 1])

        mtop = int(M[i - 1, j - 1])
        mcleft = int(M[i, j - 1])
        mbottom = int(M[(i + 1) % row, j - 1])

        CU = np.abs(top - bottom)
        CT = CU + np.abs(cleft - top)
        CB = CU + np.abs(cleft - bottom)

        return min(mtop + CT, mcleft + CU, mbottom + CB)


######################
## ENERGY FUNCTIONS ##
######################


# @params - img = input image; 
#           axis = {1 for vertical, 0 for horizontal}
# @returns - Cumulative backward energy map
def backward_energy(img, axis=1):
    row, col, d = img.shape
    gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    gradx = cv2.Sobel(gray, cv2.CV_64F, 0, 1, borderType=cv2.BORDER_REFLECT)
    grady = cv2.Sobel(gray, cv2.CV_64F, 1, 0, borderType=cv2.BORDER_REFLECT)
    e1 = np.abs(gradx) + np.abs(grady)
    M = np.zeros((row, col))
    if axis == 1:
        M[0] = gray[0]
        for i in range(1, row):
            for j in range(0, col):
                M[i, j] = e1[i, j] + get_min(M, i, j, axis)[0]
        return M
    else:
        M[:, 0] = e1[:, 0]
        for j in range(1, col):
            for i in range(0, row):
                M[i, j] = e1[i, j] + get_min(M, i, j, axis)[0];
        return M


# @params - img = input image; 
#           axis = {1 for vertical, 0 for horizontal}
# @returns - Cumulative forward energy map
def forward_energy(img, axis=1):
    gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    row, col = gray.shape
    M = np.zeros((row, col))
    if axis == 1:
        M[0] = gray[0]
        for i in range(1, row):
            for j in range(0, col):
                M[i, j] = get_min_cost(gray, M, i, j, axis)
        return M
    else:
        M[:, 0] = gray[:, 0]
        for i in range(1, row):
            for j in range(0, col):
                M[i, j] = get_min_cost(gray, M, i, j, axis)
        return M


##########################
#  SEAM HELPER FUNCTIONS #
##########################


# @params - img = input image; 
#           seam = [indices for seam]; 
#           color = [Blue, Green, Red]; 
#           axis = {1 for vertical, 0 for horizontal}
# @returns - image marked with seam
def mark_seam(img, seam, color, axis=1):
    k = img.copy()
    r, c, d = img.shape
    if axis == 1:
        for i in range(r):
            k[i, seam[i]] = color
    else:
        for j in range(c):
            k[seam[j], j] = color
    return k


# @params - img = input image; 
#           seam = [indices for seam]; 
#           color = { -1 if average pixel intensity to be used, [Blue, Green, Red]}; 
#           axis = {1 for vertical, 0 for horizontal}
# @returns - image added with seam
def add_seam(img, seam, color, axis=1):
    r, c, d = img.shape
    if axis == 1:
        k = np.zeros((r, c + 1, d))
        for i in range(r):
            if color == -1:
                j = seam[i]
                if j == 0:
                    val = np.average(img[i, j:j + 2], axis=0)
                else:
                    val = np.average(img[i, j - 1:j + 2], axis=0)
                k[i] = np.insert(img[i], seam[i], val, axis=0)
            else:
                k[i] = np.insert(img[i], seam[i], color, axis=0)
    else:
        k = np.zeros((r + 1, c, d))
        for j in range(c):
            if color == -1:
                i = seam[j]
                if j == 0:
                    val = np.average(img[i:i + 2, j], axis=0)
                else:
                    val = np.average(img[i - 1:i + 2, j], axis=0)
                k[:, j] = np.insert(img[:, j], seam[j], val, axis=0)
            else:
                k[:, j] = np.insert(img[:, j], seam[j], color, axis=0)
    return k


# @params - img = input image; 
#           seam = [indices for seam];
#           axis = {1 for vertical, 0 for horizontal}
# @returns - image with removed seam
def remove_seam(img, seam, axis=1):
    r, c, d = img.shape
    if axis == 1:
        k = np.zeros((r, c - 1, d))
        for i in range(r):
            k[i] = np.delete(img[i], seam[i], axis=0)
        return k
    else:
        k = np.zeros((r - 1, c, d))
        for j in range(c):
            k[:, j] = np.delete(img[:, j], seam[j], axis=0)
    return k


# @params - M = cumulative energy map; 
#           idx = idx in the last column for which energy is minimum
# @returns - minimum energy horizontal seam
def get_horizontal_seam(M, idx):
    row, col = M.shape
    seamx = idx
    for j in range(1, col)[::-1]:
        idx = get_min(M, idx, j, axis=0)[1]
        seamx = np.append(seamx, idx)
    return seamx[::-1]


# @params - M = cumulative energy map; 
#           idx = idx in the last row for which energy is minimum
# @returns - minimum energy vertical seam
def get_vertical_seam(M, idx):
    row, col = M.shape
    seamx = idx
    for i in range(1, row)[::-1]:
        idx = get_min(M, i, idx, axis=1)[1]
        seamx = np.append(seamx, idx)
    return seamx[::-1]


# @params - energyMap = cumulative energy map; 
#           axis = {1 for vertical, 0 for horizontal}
# @returns - minimum energy seam along axis
def get_seam_along_axis(energyMap, axis=1):
    row, col = energyMap.shape
    if axis == 1:
        return get_vertical_seam(energyMap, np.argmin(energyMap[-1]))
    else:
        low_e = min(energyMap[:, -1:])
        idx = np.where(energyMap[:, -1:] == low_e)[0][0]
        return get_horizontal_seam(energyMap, idx)


# @params - img = input image; 
#           k = number of seams 
#           axis = {1 for vertical, 0 for horizontal}
# @returns - array of first k seams with minimum energy.
#          - img = image after reducing k seams.
def get_k_seams(img, k, axis=1):
    k_seams = []
    for n in range(k):
        energy_map = backward_energy if ENERGY_FUNC is "backward" else forward_energy
        M = energy_map(img)
        seam = get_seam_along_axis(M, axis)
        img = remove_seam(img, seam, axis)
        k_seams.append(seam)
    k_seams.reverse()
    return k_seams, img


# @params - img = input image; 
#           N = number of seams 
#           k_seams = array of N seams
#           color = { -1 : if average pixel intensity to be used, [Blue, Green, Red]}; 
#           axis = {1 for vertical, 0 for horizontal}
#           expand = {1 : if insert duplicate seams, 0 : if insert removed seams in reduced image}
# @returns - img = image after inserting seams in array k_seams.
def insert_seams(img, N, k_seams, color, axis=1, expand=1):
    for n in range(N):
        curr_seam = k_seams.pop()
        img = add_seam(img, curr_seam, color, axis)

        if expand == 1:
            for seam in k_seams:
                seam[np.where(curr_seam <= seam)] += 2

    return img.astype(np.uint8)


##################
#  API FUNCTIONS #
##################


# @params - img = input image;
#           n_seams = number of seams
#           axis = {1 for vertical, 0 for horizontal}
# @returns - img = image after removing seams
def reduce_image(img, n_seams, axis=1):
    for n in range(n_seams):
        energy_map = backward_energy if ENERGY_FUNC is "backward" else forward_energy
        M = energy_map(img, axis)
        seam = get_seam_along_axis(M, axis)
        img = remove_seam(img, seam, axis)
    return img.astype(np.uint8)


# @params - img = input image;
#           n_seams = number of seams
#           axis = {1 for vertical, 0 for horizontal}
# @returns - img = image after inserting removed seams with red color
def show_removed_seams(img, n_seams, axis=1):
    k_seams, im = get_k_seams(img, n_seams, axis)
    k_seams.reverse()
    return insert_seams(im, n_seams, k_seams, [0,0,255], axis, 0)


# @params - img = input image;
#           n_seams = number of seams
#           axis = {1 for vertical, 0 for horizontal}
# @returns - img = image after inserting seams with red color
def show_added_seams(img, n_seams, axis=1):
    k_seams, _ = get_k_seams(img, n_seams, axis)
    return insert_seams(img, n_seams, k_seams, [0,0,255], axis, 1)


# @params - img = input image;
#           n_seams = number of seams
#           axis = {1 for vertical, 0 for horizontal}
# @returns - img = image after inserting seams with average pixel intensities
def expand_image(img, n_seams, axis=1):
    k_seams, _ = get_k_seams(img, n_seams, axis)
    return insert_seams(img, n_seams, k_seams, -1, axis, 1)


########################
# COMPARISON FUNCTIONS #
########################


# @params - base_img : image provided for comparison
#           result_img : generated by this code
# @returns - similarity index ranging from 0 to 1
def compares_image_structure(base_img, result_img):
    base_img = cv2.cvtColor(base_img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    result_img = cv2.cvtColor(result_img.astype(np.uint8), cv2.COLOR_BGR2GRAY)

    grad1 = np.abs(cv2.Sobel(base_img, cv2.CV_64F, 0, 1, borderType=cv2.BORDER_REFLECT)) \
            + np.abs(cv2.Sobel(base_img, cv2.CV_64F, 1, 0, borderType=cv2.BORDER_REFLECT))
    grad2 = np.abs(cv2.Sobel(result_img, cv2.CV_64F, 0, 1, borderType=cv2.BORDER_REFLECT)) \
            + np.abs(cv2.Sobel(result_img, cv2.CV_64F, 1, 0, borderType=cv2.BORDER_REFLECT))

    cosine_similarity = np.sum(grad1 * grad2)
    cosine_similarity /= np.sqrt(np.sum(grad1 ** 2))
    cosine_similarity /= np.sqrt(np.sum(grad2 ** 2))

    return cosine_similarity


# @params - base_img : image provided for comparison
#           result_img : generated by this code
# @returns - similarity index ranging from 0 to 1
def compare_image_intensities(base_img, result_img):
    diff = 0
    ch = base_img.shape[2] if base_img.ndim == 3 else 1
    for i in range(ch):
        histogram1 = cv2.calcHist([base_img], [i], None, [256], [0, 256])
        histogram2 = cv2.calcHist([result_img], [i], None, [256], [0, 256])
        diff += cv2.compareHist(histogram1, histogram2, cv2.HISTCMP_CORREL)

    return diff / ch


# @params - names : filenames array
#           values : to be plotted
#           color : color fo bars
#           title : title of graph
#           xlabel, ylabel : labels on axes
# Saves figure in OUTDIR
# @returns - none
def plot_hbar(names, values, color, title, xlabel, ylabel):
    fig, sub_plt = plt.subplots(figsize=(10, 5))
    # Horizontal Bar Plot
    sub_plt.barh([name[3:-11] for name in names], values, color=color)

    # Remove axes splines
    for s in ['top', 'bottom', 'left', 'right']:
        sub_plt.spines[s].set_visible(False)

    # Remove x, y Ticks
    sub_plt.xaxis.set_ticks_position('none')
    sub_plt.yaxis.set_ticks_position('none')

    # Add padding between axes and labels
    sub_plt.xaxis.set_tick_params(pad=5)
    sub_plt.yaxis.set_tick_params(pad=5, labelsize=9)

    # # Add x, y gridlines
    sub_plt.grid(b=True, color='grey', linestyle='-.', linewidth=0.5, alpha=0.2)

    # Show top values
    sub_plt.invert_yaxis()

    # Add annotation to bars
    for i in sub_plt.patches:
        plt.text(i.get_width() + 0.005, i.get_y() + 0.5, str(round((i.get_width()), 2)), fontsize=10, fontweight='bold', color='black')

        # Add Plot Title
    sub_plt.set_title(title, loc='left', )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Add Text watermark
    fig.text(0.9, 0.15, 'agupta857', fontsize=6, color='grey', ha='right', va='bottom', alpha=0.7)

    # Save Plot
    plt.savefig(OUTDIR + title + ".png")


# Compares all the image results, draw bar graphs and print results on console.
def compare_image_with_baselines():
    intensity_match = []
    structure_match = []
    for filename in FILE_NAMES:
        base_img = cv2.imread(COMPDIR + filename, cv2.IMREAD_COLOR)
        result_img = cv2.imread(OUTDIR + filename, cv2.IMREAD_COLOR)
        r, c, _ = result_img.shape
        base_img = base_img[:r, :c]
        r, c, _ = base_img.shape
        result_img = result_img[:r, :c]
        h = compare_image_intensities(base_img, result_img)
        c = compares_image_structure(base_img, result_img)
        intensity_match = np.append(intensity_match, h)
        structure_match = np.append(structure_match, c)
    plot_hbar(FILE_NAMES, intensity_match, 'Green', title='intensities',
              xlabel='similarity index betwwen histograms', ylabel='images')
    plot_hbar(FILE_NAMES, structure_match, 'Maroon', title='structure', xlabel='cosine similarity',
              ylabel='images')

    return intensity_match, structure_match;


# Main function - Generate results & Uncomment to draw comparisons
if __name__ == '__main__':

    ENERGY_FUNC = "backward"
    axis = 1  # change to axis=0 for horizontal seam carving
    energy_type = backward_energy if ENERGY_FUNC is "backward" else forward_energy

    # waterfall fig5_07_base
    image = cv2.imread(INPDIR + "fig5_07_base.png", cv2.IMREAD_COLOR)
    n_seam = image.shape[0] // 2 if axis == 0 else image.shape[1] // 2
    cv2.imwrite(OUTDIR + "fig5_07_result.png", reduce_image(image.copy(), n_seam, axis))

    # dolphin fig8_07_base
    image = cv2.imread(INPDIR + "fig8_07_base.png", cv2.IMREAD_COLOR)
    n_seam = image.shape[0] // 2 if axis == 0 else image.shape[1] // 2
    cv2.imwrite(OUTDIR + "fig8c_07_result.png", show_added_seams(image.copy(), 119, axis))
    red_img = expand_image(image.copy(), 113, axis)
    cv2.imwrite(OUTDIR + "fig8d_07_result.png", red_img)
    cv2.imwrite(OUTDIR + "fig8f_07_result.png", expand_image(red_img.copy(), 121, axis))

    # bench fig8_08_base
    image = cv2.imread(INPDIR + "fig8_08_base.png", cv2.IMREAD_COLOR)
    n_seam = image.shape[0] // 2 if axis == 0 else image.shape[1] // 2
    cv2.imwrite(OUTDIR + "fig8_08_back_seam_result.png", show_removed_seams(image.copy(), n_seam, axis))
    cv2.imwrite(OUTDIR + "fig8_08_backward_result.png", reduce_image(image.copy(), n_seam, axis))

    ENERGY_FUNC = "forward"
    cv2.imwrite(OUTDIR + "fig8_08_forward_seam_result.png", show_removed_seams(image.copy(), n_seam, axis))
    cv2.imwrite(OUTDIR + "fig8_08_forward_result.png", reduce_image(image.copy(), n_seam, axis))

    # car fig9_08_base
    ENERGY_FUNC = "backward"
    image = cv2.imread(INPDIR + "fig9_08_base.png", cv2.IMREAD_COLOR)
    n_seam = image.shape[0] // 2 if axis == 0 else image.shape[1] // 2
    cv2.imwrite(OUTDIR + "fig9_08_backward_result.png", expand_image(image.copy(), n_seam, axis))

    ENERGY_FUNC = "forward"
    cv2.imwrite(OUTDIR + "fig9_08_forward_result.png", expand_image(image.copy(), n_seam, axis))

    # Finding Comparisons - Update COMPDIR | UNCOMMENT the following function call to save comparison graphs.
    # intensity_sim_idx, structure_sim_idx = compare_image_with_baselines()
    # for i in range(10): print(np.round(intensity_sim_idx[i],decimals=2), np.round(structure_sim_idx[i],decimals=2), FILE_NAMES[i])

