CODE USAGE GUIDELINES
********************************

SCOPE :
A user-friendly system for seamless image composition, which we call drag-and-drop pasting. For Poisson image editing [Perez et al. 2003] to work well, the user must carefully draw a boundary on the source image to indicate the region of interest, such that salient structures in source and target images do not conflict with each other along the boundary. To make Poisson image editing more practical and easy to use, it implements a new objective function to compute an optimized boundary condition. A shortest closed-path algorithm is designed to search for the location of the boundary which will preserve the salient features of both the images.

GTBOX : https://gatech.box.com/s/98fiygrk0u9tb753qoegadn95eygm16b

INDEX:

1. Environment
2. File Location
3. Producing Results
4. APIs Provided

ENVIRONEMENT
********************************

The code works with class environment very well. Execute the following code to activate the environment.
>> conda activate CS6475

It imports following libraries: cv2, numpy, heapq and os. These all should be present in class env.

cv2 >> pip install opencv-python==3.2.0.8
numpy >> pip install numpy
matplotlib >> pip install matplotlib

Install pip :
>> curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
>> python get-pip.py
Ref: https://pip.pypa.io/en/stable/installing/

For more help consider the Assignment_0 at github.


FILE LOCATIONS
********************************

Some global variables are defined to provide the relative path for the images.

```
SRC: '/path/to/dir/with/images/‘’
NAME: [“list”, “of”,”names”,”for”,”input”,”set”]

```
Download the Test files from the GTBox link provided above.
Update the path of Root Directory “ Aarushi_Gupta_Final” to SRC variable.

To test on custom results there is an additional requirement to provide the user parameters specific to the images.

USER PARAMS : to identify the user params we use names stored in NAME. Names should match the order of result set in SRC folder.

RECT = use cv2.selectROI(src_image) to select the rectangle which surrounds the object of interest.
POLY = define some coordinates manually which on making a polygon will surround the area of interest for user. This doesn’t need to be careful selection just a wide selection which ensures a convex polygon. Note: try to avoid long straight lines as it will produce long cut and hence time will be consumed while calculating results.
CENTER = this is the location in the target image where the object is required to be pasted.

For Test files provided user params are defined in the files.


PRODUCING RESULTS
********************************

1. Uncomment the cv2.imwrite() statements to produce intermediate results.
2. Run >>> python final_project.py . This will execute the optimisation algorithm for all the result sets present in folder Aarushi_Gupta_Final

APIs PROVIDED
********************************

These are standalone APIs to generate intermediate results.

1. getBoundary(mask)
	- for any mask ( black and white) this will return the longest contour found by cv2.findCountour function.

2. getShortestCut(usr_mask, obj_mask, usr_contour, obj_contour)
	- this will return the shortest cut between the two boundaries. This cut will have two list: START for pixels to be used as source and END pixels to be used as destination while finding shorts path boundary.

3. optimiseBoundary(usr_mask, obj_mask, src, dst, center, usr_contour, obj_contour):
	- this is the main function of the project. It is responsible to use the live wire algorithm ( Dijkstra with O(N) time) to find the most optimum boundary which will produce seamless image composition unlike the poisson image editing which gives blurred seam on pasting the object if user doesn’t select the boundary carefully.
