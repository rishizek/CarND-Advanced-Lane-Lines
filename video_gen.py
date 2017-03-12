from moviepy.editor import VideoFileClip
import numpy as np
import cv2
import pickle
from tracker import Tracker

# The code below is originally created in the lesson at Udacity self-driving car nano degree.
def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Calculate directional gradient
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    # Return the result
    return binary_output

def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Return the binary image
    return binary_output

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Calculate gradient direction
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction,
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output

# Define a function that thresholds the S-channel of HLS
# Use exclusive lower bound (>) and inclusive upper (<=)
def hls_select(img, thresh=(0, 255)):
    # 1) Convert to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    # 2) Apply a threshold to the S channel
    S = hls[:,:,2]
    binary_output = np.zeros_like(S)
    binary_output[(S > thresh[0]) & (S <= thresh[1])] = 1
    # 3) Return a binary image of threshold result
    return binary_output

def window_mask(width, height, img_ref, center,level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
    return output


def process_image(img):
    # undistort the image
    undistort = cv2.undistort(img, mtx, dist, None, mtx)

    # Choose a Sobel kernel size
    ksize = 3  # Choose a larger odd number to smooth gradient measurements

    # Apply each of the thresholding functions
    gradx = abs_sobel_thresh(undistort, orient='x', sobel_kernel=ksize, thresh=(20, 255))
    grady = abs_sobel_thresh(undistort, orient='y', sobel_kernel=ksize, thresh=(30, 255))
    # mag_binary = mag_thresh(undistort, sobel_kernel=ksize, mag_thresh=(30, 100))
    # dir_binary = dir_threshold(undistort, sobel_kernel=ksize, thresh=(0.7, 1.3)) #, thresh=(0, np.pi/2))
    c_binary = hls_select(undistort, thresh=(170, 220))

    # Pre-process image template
    preprocessImage = np.zeros_like(c_binary)
    # preprocessImage[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1)) | (c_binary == 1)] = 255
    preprocessImage[((gradx == 1) & (grady == 1)) | (c_binary == 1)] = 255

    # Perspective transformation
    # paramters
    upper_width_pct = .11  # Upper trapezoid width percentage
    bottom_offset_pct = .04  # Bottom offset percentage
    bottom_width_pct = .69  # Bottom trapezoid width percentage
    height_pct = .32  # Trapezoid height percentage
    # Grab the image shape
    img_size = (preprocessImage.shape[1], preprocessImage.shape[0])
    offset = img_size[0] * .15  # offset for destination points

    src_upper_left = (img_size[0] * (.5 - upper_width_pct / 2), img_size[1] * (1 - height_pct - bottom_offset_pct))
    src_upper_right = (img_size[0] * (.5 + upper_width_pct / 2), img_size[1] * (1 - height_pct - bottom_offset_pct))
    src_bottom_right = (img_size[0] * (.5 + bottom_width_pct / 2), img_size[1] * (1 - bottom_offset_pct))
    src_bottom_left = (img_size[0] * (.5 - bottom_width_pct / 2), img_size[1] * (1 - bottom_offset_pct))
    dst_upper_left = (offset, 0)
    dst_upper_right = (img_size[0] - offset, 0)
    dst_bottom_right = (img_size[0] - offset, img_size[1])
    dst_bottom_left = (offset, img_size[1])
    # For source points I'm grabbing the four trapezoid corners
    src = np.float32([src_upper_left, src_upper_right, src_bottom_right, src_bottom_left])
    # Destination points
    dst = np.float32([dst_upper_left, dst_upper_right, dst_bottom_right, dst_bottom_left])
    # cv2.getPerspectiveTransform() is used to get M, the transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    # The inverse transformation from the destination to original source points
    Minv = cv2.getPerspectiveTransform(dst, src)
    # cv2.warpPerspective() is used to warp the original image to a top-down view
    warped = cv2.warpPerspective(preprocessImage, M, img_size, flags=cv2.INTER_LINEAR)


    window_centroids = curve_centers.find_window_centroids(warped)

    # If we found any window centers
    if len(window_centroids) > 0:

        # Points used to draw all the left and right windows
        l_points = np.zeros_like(warped)
        r_points = np.zeros_like(warped)

        # Points used to find the left and right lanes
        leftx = []
        rightx = []

        # Go through each level and draw the windows
        for level in range(0, len(window_centroids)):
            # Add center value found in frame to the list of lane points per left and right
            leftx.append(window_centroids[level][0])
            rightx.append(window_centroids[level][1])

        output = np.array(cv2.merge((warped, warped, warped)),
                           np.uint8)  # making the original road pixels 3 color channels

    # If no window centers found, just display orginal road image
    else:
        output = np.array(cv2.merge((warped, warped, warped)), np.uint8)

    # Fit the lane boundaries to the left and right center poisitons found
    yvals = range(0, warped.shape[0])

    # Adopt values from mid point of each windows to fit line
    res_yvals = np.arange(warped.shape[0] - (window_height / 2), 0, -window_height)

    # Fit a second order polynomial to pixel positions in each lane line
    left_fit = np.polyfit(res_yvals, leftx, 2)
    left_fitx = left_fit[0] * np.array(yvals) * np.array(yvals) + left_fit[1] * np.array(yvals) + left_fit[2]
    left_fitx = np.array(left_fitx, np.int32)
    right_fit = np.polyfit(res_yvals, rightx, 2)
    right_fitx = right_fit[0] * np.array(yvals) * np.array(yvals) + right_fit[1] * np.array(yvals) + right_fit[2]
    right_fitx = np.array(right_fitx, np.int32)

    # Creating boundary for output lane to be used in cv2.fillPoly() function.
    # Left downward edge first and then right upward edge, and concatenate them.
    left_lane = \
        np.array(list(zip(np.concatenate((left_fitx - window_width / 2, left_fitx[::-1] + window_width / 2), axis=0),
                          np.concatenate((yvals, yvals[::-1]), axis=0))), np.int32)
    right_lane = \
        np.array(list(zip(np.concatenate((right_fitx - window_width / 2, right_fitx[::-1] + window_width / 2), axis=0),
                          np.concatenate((yvals, yvals[::-1]), axis=0))), np.int32)
    inner_lane = \
        np.array(list(zip(np.concatenate((left_fitx + window_width / 2, right_fitx[::-1] - window_width / 2), axis=0),
                          np.concatenate((yvals, yvals[::-1]), axis=0))), np.int32)

    road = np.zeros_like(output)
    road_bkg = np.zeros_like(img)
    cv2.fillPoly(road, [left_lane], color=[255, 0, 0])
    cv2.fillPoly(road, [right_lane], color=[0, 0, 255])
    cv2.fillPoly(road, [inner_lane], color=[0, 255, 0])
    cv2.fillPoly(road_bkg, [left_lane], color=[255, 255, 255])
    cv2.fillPoly(road_bkg, [right_lane], color=[255, 255, 255])

    road_warped = cv2.warpPerspective(road, Minv, img_size, flags=cv2.INTER_LINEAR)
    road_warped_bkg = cv2.warpPerspective(road_bkg, Minv, img_size, flags=cv2.INTER_LINEAR)

    base = cv2.addWeighted(undistort, 1, road_warped_bkg, -1.0, 0.0)
    output = cv2.addWeighted(base, 1, road_warped, 0.8, 0.0)

    ym_per_pix = curve_centers.ym_per_pix  # meters per pixel in y dimension
    xm_per_pix = curve_centers.xm_per_pix  # meters per pixel in x dimension

    # Caluculate the left and right lane curveture and average them
    left_curve_fit_cr = np.polyfit(np.array(res_yvals, np.float32) * ym_per_pix,
                                   np.array(leftx, np.float32) * xm_per_pix, 2)
    left_curverad = ((1 + (2 * left_curve_fit_cr[0] * yvals[-1] * ym_per_pix + left_curve_fit_cr[1]) ** 2) ** 1.5) \
                    / np.absolute(2 * left_curve_fit_cr[0])
    right_curve_fit_cr = np.polyfit(np.array(res_yvals, np.float32) * ym_per_pix,
                                    np.array(rightx, np.float32) * xm_per_pix, 2)
    right_curverad = ((1 + (2 * right_curve_fit_cr[0] * yvals[-1] * ym_per_pix + right_curve_fit_cr[1]) ** 2) ** 1.5) \
                     / np.absolute(2 * right_curve_fit_cr[0])
    curverad = (left_curverad + right_curverad) / 2

    left_fitx = left_fit[0] * np.array(yvals) * np.array(yvals) + left_fit[1] * np.array(yvals) + left_fit[2]
    left_fitx = np.array(left_fitx, np.int32)

    # Caluculate the offset of the car on the road
    camera_center = (left_fitx[-1] + right_fitx[-1]) / 2
    center_diff = (camera_center - warped.shape[1] / 2) * xm_per_pix
    side_pos = 'left'
    if center_diff <= 0:
        side_pos = 'right'

    # Draw the text showing curvature, offset, and speed
    cv2.putText(output, 'Radius of Curvature = ' + str(round(curverad, 3)) + '(m)', (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(output, 'Vehicle is ' + str(round(center_diff, 3)) + 'm ' + side_pos +
                ' of center', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return output


# Read in the saved objpoints and imgpoints
dist_pickle = pickle.load(open("camera_cal/calibration_pickle.p", "rb"))
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

output_vide = 'output1_tracked.mp4'
input_video = 'project_video.mp4'

window_width = 25
window_height = 80

# Set up the overall class to do all the tracking
curve_centers = Tracker(mywindow_width=window_width, mywindow_height=window_height, mymargin=25,
                        my_ym=10 / 720, my_xm=4 / 384, mysmooth_factor=15)

clip1 = VideoFileClip(input_video)
video_clip = clip1.fl_image(process_image)
video_clip.write_videofile(output_vide, audio=False)
