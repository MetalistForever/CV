import numpy as np
from skimage.transform import hough_line, hough_line_peaks
from skimage.transform import rotate
from skimage.feature import canny
from skimage.io import imread
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from scipy.stats import mode
import cv2
import argparse
from skimage.filters import gaussian
from scipy.ndimage import convolve

def get_parser():
    parser = argparse.ArgumentParser('rotate_and_remove_table')
    parser.add_argument('--input_name', '-i', required=True)
    parser.add_argument('--output_name', '-o', required=True)
    return parser

def skew_angle_hough_transform(image):
    # convert to edges
    edges = canny(image)
    # Hough transform between 0.1 - 180 degrees.
    tested_angles = np.deg2rad(np.arange(0.1, 180.0))
    h, theta, d = hough_line(edges, theta=tested_angles)
    # find line peaks and angles
    accum, angles, dists = hough_line_peaks(h, theta, d)
    # round the angles to 2 decimal places and find the most common angle.
    most_common_angle = mode(np.around(angles, decimals=2))[0]
    # convert the angle to degree for rotation.
    skew_angle = np.rad2deg(most_common_angle - np.pi/2)
    # print(skew_angle)
    return skew_angle, h, (accum, angles, dists)

def remove_table(image):
    # to get horizontal lines we use horizontal structure element
    horizontal = cv2.morphologyEx(rotated_image, cv2.MORPH_CLOSE, np.ones((1, 50)))
    # to get vertical lines we use vertical structure element
    vertical = cv2.morphologyEx(rotated_image, cv2.MORPH_CLOSE, np.ones((50, 1)))
    # the resulting image is the minimum of both images
    table = np.minimum(vertical, horizontal)
    ax[3].imshow(table, cmap='gray')
    ax[3].set_title('Extracted table')
    # inverted image for further extraction
    inv_image = cv2.bitwise_not(np.uint8(rotated_image * 255.))
    inv_table = cv2.bitwise_not(np.uint8(table * 255.))
    # added dilation operation to broaden the lines of table
    broad_table = cv2.dilate(inv_table, kernel=np.ones((3, 3)))
    # subtract the table
    float_diff = np.float64(inv_image) - np.float64(broad_table)
    inv_img_no_table = np.uint8(np.where(float_diff > 0, float_diff, 0))
    img_no_table = cv2.bitwise_not(inv_img_no_table)
    return img_no_table, table

args = get_parser().parse_args()
input_name = args.input_name
output_name = args.output_name

image = rgb2gray(imread(input_name))
# filter = np.array([[0.0, 0.0, 0.0], 
#                    [0.0, 1.0, 0.0],
#                    [0.0, 0.0, 0.0]])
# filter = gaussian(filter, sigma=3.0)

skew_angle, h_space, peaks = skew_angle_hough_transform(image)
# Code for highliting peaks
# h_peaks = h_space - convolve(h_space, weights=filter)
# h_peaks = np.where(h_peaks > 0, h_peaks, 0)
# print(h_space.sum())
# h_gauss = convolve(h_space, filter)
# print(h_gauss.max())
rotated_image = rotate(image, skew_angle, cval=1)

# print(peaks[0], peaks[1], peaks[2])

offset = np.round(np.sqrt(image.shape[0] ** 2 + image.shape[1] ** 2))
# Draw the image 
# fig, ax = plt.subplots(ncols=2, figsize=(12, 12))
# ax[0].imshow(h_space)
# ax[1].imshow(h_space)
# ax[1].scatter(peaks[1] * 180 / np.pi, peaks[2] + offset, c='r', s=5)
# plt.savefig('hough_space.jpg', format='jpg', bbox_inches='tight', dpi=600)
# plt.show()

img_no_table, table = remove_table(rotated_image)

fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(12,12))
ax[0, 0].imshow(image, cmap='gray')
ax[0, 0].set_title('Original image')
ax[0, 1].imshow(rotated_image, cmap='gray')
ax[0, 1].set_title('Rotated image')
ax[1, 0].imshow(table, cmap='gray')
ax[1, 0].set_title('Exctracted table')
ax[1, 1].imshow(img_no_table, cmap='gray')
ax[1, 1].set_title('Image with no table')

ax[0, 0].axis('off')
ax[0, 1].axis('off')
ax[1, 0].axis('off')
ax[1, 1].axis('off')
plt.savefig(output_name, format='jpg', bbox_inches='tight', dpi=600)
# plt.show()