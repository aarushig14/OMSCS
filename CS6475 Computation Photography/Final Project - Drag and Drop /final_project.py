import cv2
import numpy as np
import optimisation as op

# change the names for the custom test images
SRC = "/Users/ag14/Desktop/result-finalprject/Aarushi_Gupta_Final/"
NAME = ["boat", "tiger", "flower", "baby"]

# use cv2.selectROI(image) to find coordinates
RECT = {
    "tiger": (352, 279, 248, 221),
    "flower": (85, 124, 230, 214),
    "baby": (15, 10, 275, 310),
    "boat": (180, 100, 244, 180)
}

# location for the object in target image - (col, row)
CENTER = {
    "flower": (220, 120),
    "baby": (220, 350),
    "tiger": (700, 500),
    "boat": (300, 900)
}

# user area selection encompassing coordinates in form of (col, row)
POLY = {
    "flower": np.array(
        [[90, 300], [70, 200], [100, 150], [130, 160], [140, 170], [150, 150], [180, 120], [210, 120], [220, 150],
         [270, 160], [300, 200], [320, 310], [310, 330], [250, 330], [200, 340], [140, 350], [120, 350]]),
    "baby": np.array(
        [[13, 320], [25, 250], [50, 210], [80, 140], [90, 60], [150, 10], [200, 15], [225, 10], [260, 50], [270, 120],
         [280, 220], [260, 300], [250, 325]]),
    "tiger": np.array(
        [[370, 300], [400, 290], [450, 285], [500, 300], [550, 310], [600, 295], [610, 310], [600, 320], [595, 440],
         [585, 450], [600, 500], [450, 490], [350, 500], [355, 450], [365, 350]]),
    "boat": np.array(
        [[200, 300], [180, 250], [180, 200], [180, 150], [200, 100], [270, 90], [300, 100], [400, 130], [450, 170],
         [450, 300], [300, 300]])
}

if __name__ == "__main__":

    # uncomment the image writing lines below to save the intermediate results.

    for ip_set in range(1,4):

        dir_name = SRC + "RESULT_SET_"+str(ip_set+1)

        src_name = dir_name+"/"+"input_1.jpg"
        dst_name = dir_name+"/"+"input_2.jpg"

        # print("Starting... ", src_name)

        ''' INPUT '''

        src = cv2.imread(src_name, cv2.IMREAD_COLOR)
        dst = cv2.imread(dst_name, cv2.IMREAD_COLOR)
        mask = np.zeros(src.shape[:2], src.dtype)

        ''' USER PARAMS '''

        rect = RECT.get(NAME[ip_set])
        center = CENTER.get(NAME[ip_set])
        poly = POLY.get(NAME[ip_set])

        ''' OBJECT OF INTEREST - GRAB CUT '''

        cv2.fillPoly(mask, [poly], 255)

        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)

        mask2, bg, fg = cv2.grabCut(src, mask.copy(), rect, bgdModel, fgdModel, 8, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask2 == 2) | (mask2 == 0), 0, 1).astype('uint8')
        img = src * mask2[:, :, np.newaxis]
        r, m2 = cv2.threshold(mask2, 0.5, 255, 0)

        # cv2.imwrite(os.path.join(OUT, src_name, "obj_mask.jpg"), m2)
        # cv2.imwrite(os.path.join(OUT, src_name, "usr_mask.jpg"), mask)
        # cv2.imwrite(os.path.join(OUT, src_name, "omega_band.jpg"), mask - m2)

        usr_contour = np.array(op.getBoundary(mask))
        obj_contour = np.array(op.getBoundary(m2))

        img = cv2.drawContours(src.copy(), usr_contour, -1, (255, 255, 255), 5)
        # cv2.imwrite(os.path.join(OUT, src_name, "user_boundary.jpg"), img)
        img = cv2.drawContours(src.copy(), obj_contour, -1, (255, 255, 255), 5)
        # cv2.imwrite(os.path.join(OUT, src_name, "object_boundary.jpg"), img)

        ''' POISSON IMAGE EDITING - RESULT WITH LIMITATION '''

        seamless = cv2.seamlessClone(src, dst, mask, center, cv2.NORMAL_CLONE)
        # cv2.imwrite(os.path.join(OUT, src_name, "before_optimisation.jpg"), seamless)

        ''' JUST TO PRODUCE IMAGE FOR SHORTEST CUT '''

        cut, d = op.getShortestCut(mask, m2, usr_contour, obj_contour)
        k = cv2.cvtColor((mask - m2), cv2.COLOR_GRAY2BGR)
        for n in cut[0]:
            k[n[1], n[0]] = (0, 0, 255)
        for n in cut[1]:
            k[n[1], n[0]] = (0, 255, 255)
        # cv2.imwrite(os.path.join(OUT, src_name, "shortest_cut.jpg"), k)

        ''' OPTIMISED BOUNDARY - RESULT OF THE PROJECT IMPLEMENTATION '''

        op_bound = op.optimiseBoundary(mask, m2, src, dst, center, usr_contour, obj_contour)

        optimum_mask = np.zeros(src.shape[:2], src.dtype)
        cv2.fillPoly(optimum_mask, [op_bound], 255)

        output = cv2.seamlessClone(src, dst, optimum_mask, center, cv2.NORMAL_CLONE)
        cv2.imwrite( dir_name+"/"+"result_1.jpg", output)

        ''' VARIANCE IN IMAGE INTENSITY ACROSS BOUNDARIES BEFORE AND AFTER OPTIMISATIONS '''

        boundaries = [usr_contour.reshape(usr_contour.shape[1], usr_contour.shape[3]), op_bound]
        for i in range(2):
            optimisedBoundary = boundaries[i]

            fs = np.array(optimisedBoundary)
            src_ctr = (src.shape[1] // 2, src.shape[0] // 2)
            ft = fs - src_ctr + center
            ft_contour = ft.reshape(1, ft.shape[0], 1, ft.shape[1])
            fs_contour = fs.reshape(1, fs.shape[0], 1, fs.shape[1])
            img = cv2.drawContours(src.copy(), fs_contour, -1, (0, 255, 255), 2)
            # cv2.imwrite(os.path.join(OUT, src_name, "optimised_boundary.jpg"), img)
            var = np.ones(src.shape) * 127
            for i in range(len(fs)):
                var[fs[i][1], fs[i][0]] = dst[ft[i][1], ft[i][0]] - dst[fs[i][1], fs[i][0]]
            var = (var - np.min(var)) * 255 / (np.max(var) - np.min(var))
            # cv2.imwrite(os.path.join(OUT, src_name, "variance" + str(i) + ".jpg"), var)