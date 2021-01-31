import cv2
import numpy as np
import heapq as hq

## Helper function ##

def getBoundary(mask):
    '''
    returns longest contour or the boundary for the black and white mask passed
    '''
    if mask.ndim == 3: mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mx = np.max(mask)
    mn = np.min(mask)
    ret, thresh = cv2.threshold(mask, (mx + mn) / 2, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        maxlength = 0
        longest = []
        for i in contours:
            if maxlength < len(i):
                maxlength = len(i)
                longest = [i]
        return longest
    return contours

## Shortest Cut ##

def path(pairs):
    '''
        automatically finds the shortest path between the pairs passed to return two list of pixels
        each list represents the pixels on either side of the cut between the pairs.
    '''
    xdiff, ydiff = pairs[1] - pairs[0]
    if (xdiff < 0 and ydiff < 0) or (xdiff > 0 and ydiff < 0):
        pairs = pairs[::-1]

    x1, y1 = pairs[0]
    x2, y2 = pairs[1]
    xdiff, ydiff = pairs[1] - pairs[0]
    start = []
    end = []
    if xdiff == 0:
        start = [[x1, y] for y in range(y1, y2 + 1)]
        end = [[x1 + 1, y] for y in range(y1, y2 + 1)]
    elif ydiff == 0:
        start = [[x, y1] for x in range(x1, x2 + 1)]
        end = [[x, y1 - 1] for x in range(x1, x2 + 1)]
    elif xdiff > 0 and ydiff > 0:
        start = [[x1 + i, y1 + i] for i in range(min(ydiff, xdiff) + 1)]
        end = [[x1 + i, y1 + i + 1] for i in range(min(ydiff, xdiff) + 1)]
        if ydiff < xdiff:
            last = start[-1]
            start += [[last[0] + i, last[1]] for i in range(1, x2 - last[0] + 1)]
            end += [[last[0] + i, last[1] + 1] for i in range(1, x2 - last[0] + 1)]
        elif xdiff < ydiff:
            last = start[-1]
            start += [[last[0], last[1] + i] for i in range(1, y2 - last[1] + 1)]
            end.remove(end[-1])
            end += [[last[0] - 1, last[1] + i] for i in range(1, y2 - last[1] + 1)]

    elif xdiff < 0 and ydiff > 0:
        xdiff *= -1
        start = [[x1 - i, y1 + i] for i in range(min(ydiff, xdiff) + 1)]
        end = [[x1 - i, y1 + i + 1] for i in range(min(ydiff, xdiff) + 1)]
        if ydiff < xdiff:
            last = start[-1]
            start += [[last[0] - i, last[1]] for i in range(1, last[0] - x2 + 1)]
            end += [[last[0] - i, last[1] + 1] for i in range(1, last[0] - x2 + 1)]
        elif xdiff < ydiff:
            last = start[-1]
            start += [[last[0], last[1] + i] for i in range(1, y2 - last[1] + 1)]
            end += [[last[0] + 1, last[1] + i] for i in range(1, y2 - last[1] + 1)]

    return start, end


def getShortestCut(usr_mask, obj_mask, usr_contour, obj_contour):
    '''
        uses a nested loop to find the shortest  L2-norm distance between the boundary points
        of both the boundary points passed.
    '''
    min_dist = np.max(usr_mask.shape)
    pairs = []

    usr_bound = usr_contour.reshape(usr_contour.shape[1], usr_contour.shape[3])
    obj_bound = obj_contour.reshape(obj_contour.shape[1], obj_contour.shape[3])

    for usr in usr_bound:
        for obj in obj_bound:
            dist = np.sum((usr - obj) ** 2) ** 0.5
            if min_dist >= dist:
                min_dist = dist
                pairs = [usr, obj]
    cut = path(pairs)
    rv = ([], [])
    for i in range(2):
        for n in cut[i]:
            if (usr_mask - obj_mask)[n[1], n[0]] != 0:
                rv[i].append(n)

    return rv, min_dist


## Optimisation Algorithm ##

def getBoundaryPixel(image, boundary):
    '''
        returns the array of pixel values in the image for the positions present in the boundary list.
    '''
    return np.array([image[pxl[1]][pxl[0]] for pxl in boundary])


def cost(q, src, dst, center, K):
    '''
        finds cost of each node according to the Eq 4 in the paper
        using L2 norm distance between the two pixel values.
    '''
    fs = src[q[1], q[0]]
    x, y = np.array(q) - (src.shape[1] // 2, src.shape[0] // 2) + center
    ft = dst[y, x]
    return np.sum(((ft - fs) - K) ** 2)


def getNeighbours(q, mask, cut):
    '''
        returns valid neighbours of the band pixel positions.
        valid neighbours consists of those which do not pass the CUT more than once
        and the pixel value in the mask should not be 0 meaning that the pixel is present in the
        omega band.
    '''
    x, y = q
    start = cut[0]
    end = cut[1]

    n = [x - 1, y], [x + 1, y], [x, y - 1], [x, y + 1]
    N = []
    if q in start:
        for i in n:
            if i not in start and i not in end and mask[i[1], i[0]] != 0:
                N.append(i)
    else:
        for i in n:
            if i not in start and mask[i[1], i[0]] != 0:
                N.append(i)
    return N


def getPath(parent, srcnode, dstnode):
    '''
        returns the path of the minimum energy boundary found by live wire algorithm.
        It backtracks from destination node to source node on either side of the cut to get a complete boundary.
    '''
    n = dstnode
    path = [n]
    while n != srcnode:
        n = parent[str(n)]
        path.append(n)
    return path[::-1]


def optimiseBoundary(usr_mask, obj_mask, src, dst, center, usr_contour, obj_contour):
    '''
        returns optimised boundary between the user selected and object of interest.

        This is derived from section 3.

        And Step 3 algorithm implements the Dijkstra from the Mortensen and Barrett 1995 work.
    '''
    usr_bound = usr_contour.reshape(usr_contour.shape[1], usr_contour.shape[3])
    obj_bound = obj_contour.reshape(obj_contour.shape[1], obj_contour.shape[3])

    mask = usr_mask - obj_mask
    cut, dist = getShortestCut(usr_mask, obj_mask, usr_contour, obj_contour)

    ''' step 1 : initialise to user boundary'''

    omega = usr_bound
    prevEnergy = -1
    count = 0

    while count != 2:

        '''step 2 : find K'''

        src_pixels = getBoundaryPixel(src, omega)
        src_ctr = ((np.max(usr_bound[:, 0]) + np.min(usr_bound[:, 0])) // 2,
                   (np.max(usr_bound[:, 1]) + np.min(usr_bound[:, 1])) // 2)
        dst_pixels = getBoundaryPixel(dst, omega - src_ctr + center)

        K = (np.sum((dst_pixels - src_pixels) ** 2)) ** 0.5
        K = np.divide(K, len(omega))
        # print("K: ", K, "count: ", count)

        '''
        step 3 : optimise the boundary with given K using Djikstra
                 Live Wire (2D DP Algorithm)
        '''

        min_path = []
        min_e = -1
        for i in range(len(cut[0])):

            s = cut[0][i]

            # Data Structure
            queue = []
            g = {}
            visited = []
            parent = {}

            # initialisation
            g[str(s)] = cost(s, src, dst, center, K)
            hq.heappush(queue, (g[str(s)], s))

            # Djikstra Algorithm
            while len(queue) > 0:
                q = qdist, qnode = hq.heappop(queue)
                visited.append(qnode)

                N = getNeighbours(qnode, mask, cut)

                for r in N:
                    str_r = str(r)
                    if r in visited:
                        continue

                    g[str(qnode)] = dist = qdist + cost(r, src, dst, center, K)

                    if ((g.get(str_r), r) in queue) and (dist < g[str_r]):
                        i = np.where(queue == (g[str_r], r))
                        queue[i] = queue[-1]
                        queue.pop()
                        hq.heapify(queue)
                    if (g.get(str_r), r) not in queue:
                        g[str_r] = dist
                        parent[str_r] = qnode
                        hq.heappush(queue, (dist, r))

            for j in range(len(cut[1])):

                optimisedBoundary = getPath(parent, cut[0][i], cut[1][j])
                e = 0
                for k in optimisedBoundary:
                    e += g[str(k)]

                if min_e == -1 or min_e > e:
                    min_e = e
                    min_path = optimisedBoundary

        '''step 4 : repeat until energy Eq 4 does not decrease for two consecutive times'''

        if min_e >= prevEnergy and prevEnergy != -1:
            count += 1
        else:
            count = 0

        omega = np.array(min_path)
        prevEnergy = min_e

    return omega
