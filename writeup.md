##Advanced Lane Lines

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/undistort_output_2.png "Undistorted"
[image2]: ./test_images/undistorted_test1.jpg "Road Transformed"
[image3]: ./test_images/binary_test1.jpg "Binary Example"
[image4]: ./test_images/perspective_straight_lines1.jpg "Warp Example"
[image5]: ./test_images/perspective_test5.jpg "Output"
[image6]: ./examples/color_fit_lines.jpg "Fit Visual"
[image7]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

---
###README

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the 2nd-3rd code cell of the Jupyter notebook located in "./camera_calibration.ipynb".  

I start by obtained the reference code from 
[Udacity's Camera Calibration code](https://github.com/udacity/CarND-Camera-Calibration), and 
modified parameters so that the code run with this project (e.g. the number of chessboard corners, etc.). 
Then, the "object points", which assumes the image locate on xy plane (i.e. z=0), and the "image points", stores actual 
data points (x, y) on each images, are collected (in 2nd code cell).

After that, in the 3rd code cell the camera calibration and distortion coefficients are calculated 
using the `cv2.calibrateCamera()`, which uses `objpoints` and `imgpoints` as input variables.

The applied result of the distortion correction to the test image by the `cv2.undistort()` function
is as follow: 

![alt text][image1]

###Pipeline (single images)

####1. Provide an example of a distortion-corrected image.
The distortion-correction via the camera calibration is applied to the test images given in
"test_images" folder as well as the undistorted image results. 
Here the code for this step is found in the 5th cell of the Jupyter notebook in 
`image_generation.ipynb`. One of the test images' results is given below:
![alt text][image2]
Here, we can observe that the right-most car is partially cut due to image 
undistortion.

####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I used a combination of color and gradient thresholds to generate a binary image 
(the convert functions are found in the 2nd cell and thresholding steps in the 6th cell of Jupyter notebook
`image_generation.ipynb`, and the test results in "test_images" folder). 
As you may find in the code, I initially tried to use the magnitude and direction of 
the gradient but it turned out to generate more noise and after several trials I decided not to use
them. Here's an example of my output for this step.  
![alt text][image3]

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The codes for my perspective transform are found in the 7th (tuning) and 8th (example application) cell of Jupyter notebook
`image_generation.ipynb` (lines 28-55 in the cell), and the test results in "test_images" folder.
I used `cv2.getPerspectiveTransform()` function provided by openCV to obtain 
the transformation matrix, which requires the source (src) and destination (dst) points to
generate the matrix. I used a trapezoid to fit source points to the straight lines' image
with parameters: Bottom offset percentage (bottom_offset_pct),
Bottom trapezoid width percentage (bottom_width_pct),
Trapezoid height percentage (height_pct), 
and Upper trapezoid width percentage (upper_width_pct).
The source (src) and destination (dst) points of computed with the following formula:
```
src_upper_left = (img_size[0] * (.5 - upper_width_pct / 2), img_size[1] * (1 - height_pct - bottom_offset_pct))
src_upper_right = (img_size[0] * (.5 + upper_width_pct / 2), img_size[1] * (1 - height_pct - bottom_offset_pct))
src_bottom_right = (img_size[0] * (.5 + bottom_width_pct / 2), img_size[1] * (1 - bottom_offset_pct))
src_bottom_left = (img_size[0] * (.5 - bottom_width_pct / 2), img_size[1] * (1 - bottom_offset_pct))
dst_upper_left = (offset, 0)
dst_upper_right = (img_size[0]-offset, 0)
dst_bottom_right = (img_size[0]-offset, img_size[1])
dst_bottom_left = (offset, img_size[1]
```
This resulted in the parameters to generate bird-view images:

| Parameters       | Values (%)      |Description                        |
|:----------------:|:---------------:|:---------------------------------:|
| upper_width_pct  | 11.0            | Upper trapezoid width percentage  |
| bottom_offset_pct|  4.0            | Bottom offset percentage          |
| bottom_width_pct | 69.0            | Bottom trapezoid width percentage |
| height_pct       | 32.0            | Trapezoid height percentage       |
| offsett_pct      |img_size[0] * .18| Offset for destination points     |

![alt text][image4]

I further verified that my perspective transform was working properly in
the 8th (example application) cell of Jupyter notebook.
![alt text][image5]


####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image6]

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image7]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

