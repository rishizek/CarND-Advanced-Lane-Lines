##Advanced Lane Lines

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
[image2]: ./output_images/undistorted_test1.jpg "Road Transformed"
[image3]: ./output_images/binary_test1.jpg "Binary Example"
[image4]: ./output_images/perspective_straight_lines1.jpg "Warp Example"
[image5]: ./output_images/perspective_test5.jpg "Output"
[image6]: ./output_images/line_fit_test2.jpg "Fit Visual"
[image7]: ./output_images/result_test5.jpg "Output"
[video1]: ./output1_tracked.mp4 "Video"

---
####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the 2nd-3rd code cell of the Jupyter notebook located in "./camera_calibration.ipynb".  

I started by obtained the reference code from 
[Udacity's Camera Calibration code](https://github.com/udacity/CarND-Camera-Calibration), and 
modified parameters so that the code ran with this project (e.g. the number of chessboard corners, etc.). 
Then, the "object points", which assumes the image locate on xy plane (i.e. z=0), and the "image points", stores actual 
data points (x, y) on each images, are collected (in 2nd code cell).

After that, the camera calibration and distortion coefficients are calculated in the 3rd code cell 
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
where we can observe that the right-most car is partially cut due to the image 
undistortion. You can find other results in "output_images" folder.

####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I used a combination of color and gradient thresholds to generate a binary image 
(the convert functions are found in the 2nd cell, and thresholding steps in the 6th cell of Jupyter notebook
`image_generation.ipynb`, respectively, and the test results can be also found in "output_images" folder). 
As you may find in the code, I initially tried to use the magnitude and direction of 
the gradient but it turned out to generate more noise and after several trials I decided not to use
them. Here's an example of my output for this step.  
![alt text][image3]

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The codes for my perspective transform are found in the 7th (calibration by straight line) and 8th (test examples) cell of Jupyter notebook
`image_generation.ipynb` (lines 28-55 in the cell), and the test results in "test_images" folder.
I used `cv2.getPerspectiveTransform()` function provided by openCV to obtain 
the transformation matrix, which requires the source (src) and destination (dst) points to
generate the matrix. I used a trapezoid to fit source points to the straight lines' image
with parameters: bottom offset percentage (bottom_offset_pct),
bottom trapezoid width percentage (bottom_width_pct),
trapezoid height percentage (height_pct), 
and upper trapezoid width percentage (upper_width_pct).
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
| offsett_pct      |img_size[0] * .15| Offset for destination points     |

![alt text][image4]

I further verified that my perspective transform was working properly by applying it to test 
images in the 8th cell of Jupyter notebook (`image_generation.ipynb`).
![alt text][image5]


####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I used the sliding window method with convolution to find lane-line pixels
(lines 37-93 in `tracker.py` and lines 73-87 in 9th cell in `image_generation.ipynb`)
and fit their position by the 2nd order polynomial (lines 100-112 in 9th cell in `image_generation.ipynb`). 
At the sliding window method part, the starting position of the 
first window are determined by summing quarter bottom of binary 
image to estimate most reasonable starting points (lines 53-60 in `tracker.py`).
Also the final values of the line centers are averaged with past 15 results
to keep the markers stable (line 93 in `tracker.py`).

![alt text][image6]

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

[The radius of curvatur](http://www.intmath.com/applications-differentiation/8-radius-curvature.php)
 is computed by averaging the curvature of left and right lanes 
(lines 140-152 in 10th cell in `image_generation.ipynb`). Note that the naive calculation of 
the curvature generate the curvature in the pixel size, and not actual curvature of the road.
To resolve the problem, the "meter per pixel ratios for x and y direction" are determined and 
lane-line pixels were converted to the actual scale before the calculation of the curvature.

Similarly the position of the vehicle with respect to center was computed using the difference
between the center of the image and the center of left and right lanes for x direction at 
the bottom of image, with the "meter per pixel rations for x direction".

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 134-135 in 10th cell in `image_generation.ipynb`.
Here The inverse transformation from the destination to original source points, Minv, is used.
An example of my result on a test image is:

![alt text][image7]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

To generate the final video output, I gathered all `image_generation.ipynb` code to `video_gen.py` and 
refined the code to remove unnecessary code and reorganized code so the Tracker class works properly.

Here's a [link to my video result](./output1_tracked.mp4).


---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I found my lane detector failed to identify the correct lanes mainly in two situations.

First case happens when roads under sun light with shadows around it. Under this circumstance 
my binary image threshold with the color and gradient transformation, occasionally capture incorrect
pixes as lane lines. Currently, I am only using the x- and y-gradients of Sobel method and S-channel
of HLS as threshold, so I may be able to improve this issue if I further fine-tune the parameters 
and/or add other channels of HSF for the thresholds.

The other major problem my model confronted is when there is no lane at the video frame. Obviously
when there is no lane in the image it is impossible to locate the lane. To mitigate this issue, I 
averaged past 15 recoder of x-axis lane location.　Although the averaging method makes the detected 
lane locations fairly smooth, when the model incorrectly find fake lane, the lane line divert 
from the correct lane temporarily. This behavior could be avoided if we conduct sanity checks 
(lanes have similar curvature to before, right and left lanes are approximately similar distance away,
lanes are roughly parallel) and in case the sanity checks fail, we retain the previous lane positions 
from the frame.

Apply the above mentioned approach, I'm sure my model will be more robust.
