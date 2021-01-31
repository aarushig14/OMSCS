CODE USAGE GUIDELINES
********************************

Video Presentation (2:59): https://youtu.be/xljw0Zr4u-Y
(Anyone with the link can view)

INDEX:

1. Environment
2. File Location
3. APIs Provided
4. Parameters (APIs)
5. Energy Function
6. Producing Results
7. Comparison
8. Some Other Helpful Functions
9. Tips: Alt Direction Seam Removal

ENVIRONEMENT
********************************

The code works with class environment very well. Execute the following code to activate the environment.
>> conda activate CS6475

It imports following libraries: cv2, numpy and matplotlib. These all should be present in class env.

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
INPDIR: '/path/to/source/images/'
OUTDIR: '/path/to/output/directory/'
COMPDIR: 'path/to/directory/with/comparison/images/'

```

Values currently set to current directory for all three variables ('./'). Remember to put a slash at the end of path to the directories.


APIs PROVIDED
********************************

These are standalone APIs to generate reduced and expanded images or just to retrieve added or removed seams marked in the images.

1. reduce_image(img, n_seams, axis=1)
	- reduces number of seams equal to n_seams along the axis passed in the parameter.

2. show_removed_seams(img, n_seams, axis=1)
	- show removed seams equal to n_seams in the original image along the axis passed in the parameter.

3. show_added_seams(img, n_seams, axis=1)
	- show added seams equal to n_seams in the original image along the axis passed in the parameter.

4. expand_image(img, n_seams, axis=1)
	- extends the image by adding number of seams equal to n_seams along the axis passed in the parameter.

PARAMETERS (APIs)
********************************

1. AXIS - {0 : horizontal direction, 1 : vertical direction}
	- By default all function takes axis = 1 as value.
2. IMG - Input image that needs to be manipulated.
3. n_seams - number of seams that needs to be added/removed from the input image.

ENERGY FUNCTION
********************************

A global variable ENERGY_FUNC is defined to establish the energy function to be used while executing any of the APIs. Set the value in small caps to specify the energy function to be used in the following statements until the value is changed again.
``` ENERGY_FUNC = 'backward' ``` for backward energy
``` ENERGY_FUNC = 'forward'  ``` for forward energy


PRODUCING RESULTS
********************************

Execute the code in the main function to produce results for all the 10 images.
In the end, uncomment the comparison function call statements. This will find the comparison metrics for each result, draw the bar graphs and print them on the console.
``` intensity_sim_idx, structure_sim_idx = compare_image_with_baselines() ```

Note: update the COMPDIR path before executing the code.

COMPARISON
********************************

Comparison call will produce two arrays first one for intensity similarity index and second one for the structural cosine similarity. It will also make a cal to plot_hbar method which will save both the metric bar graphs.

SOME OTHER HELPFUL FUNCTIONS
********************************

1. mark_seam(img, seam, color, axis=1) 
	- color size should match number of channels
	- will mark the seam in parameters on the input `img` along axis.

2. add_seam(img, seam, color, axis=1) 
        if color == -1 : will use the average pixel value of the neighbours.
                  else : any value passed.

3. remove_seam(img, seam, axis=1)
	- removes the provided seam from the image.

4. get_seam_along_axis(energyMap, axis=1)
	- returns minimum energy seam along axis for the energyMap provided.

5. get_k_seams(img, K, axis=1)
	- returns an array of K seams with minimum energy in the reverse order of removal.

6. insert_seams(img, N, k_seams, color, axis=1, expand=1)
	- inserts N seams provided in array k_seams.
	- color is an array of pixel intensity with BGR configuration. (eg, RED = [0,0,255])
	- expand tells if the image are being added to expand the image (value = 1 ) or else the function will insert the image without any shift in the indices.

TIPS: ALT DIRECTION SEAM REMOVAL
********************************


```
img = cv2.imread("/path/to/image/file", cv2.IMREAD_COLOR)
axis = 1
for i in range(n):
    axis = np.abs(1-axis) # alternative seam removal
    E = backward_energy(img, axis)
    seam = get_seam_along_axis(E, axis)
    img =  reduce_image(img, 1, axis)
```


