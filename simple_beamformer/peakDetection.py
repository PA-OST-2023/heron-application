import numpy as np
import cv2 as cv
from math import degrees


def peak_detector(im):
    fig, ax = plt.subplots(2, 2)
    ret, im2 = cv.threshold(im, 50, 255, cv.THRESH_BINARY)
    im3 = cv.dilate(im2, None, iterations=2)

    contours, hier = cv.findContours(im3, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # draw a bounding box around each contour
    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        cv.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv.imshow("Contours", im)
    cv.waitKey()

    ax[0][0].imshow(im)
    ax[0][0].set_title("Base Im")

    ax[0][1].imshow(im2)
    ax[0][1].set_title("Thresholded")
    plt.show()
    pass


def peak_detector2(im, val_array=None, area_mask=None,sphere_factor=1.1, min_height=0, max_height=2**15, rel_max= 0.2, plot=False):
    if val_array is None:
        val_array = im
    struct_element = cv.getStructuringElement(cv.MORPH_ELLIPSE, (10, 10))
    im1 = cv.dilate(im, struct_element, iterations=2)
    maxmask = im == im1
    med = cv.medianBlur(im, ksize=31)

    maxmask = im == im1
    medmask = (
        im.astype(np.int16) >= med.astype(np.int16) + 10
    )  # peak property. avoiding unsigned difference.>>
    if area_mask is not None:
        mask = maxmask & medmask & area_mask
    else:
        mask = maxmask & medmask

    peaks_im = np.zeros_like(im,dtype=np.uint8)
    peaks_im[mask] = 255
    kernel = np.ones((3,3),np.uint8)
    peaks_im = cv.morphologyEx(peaks_im, cv.MORPH_CLOSE, kernel)
    output = cv.connectedComponentsWithStats(peaks_im, 8, cv.CV_32S)
    (numLabels, labels, stats, centroids) = output
    peaks = []
    theta = 0
    phi = 0
#     conversion_factor = np.pi/2 * sphere_factor / (im.shape[0]/2)
    conversion_factor = 1.8 / (im.shape[0]/2)

    max_value = np.max(val_array[area_mask])
    for i, center in enumerate(centroids[1:]):
        peak_val = np.max(val_array[labels==i+1])
        if peak_val < min_height or peak_val > max_height or peak_val < max_value * rel_max:
            continue
        peak_cart = center - np.array([im.shape[0]/2, im.shape[0]/2])
        theta = np.linalg.norm(peak_cart) * conversion_factor
        phi = np.arctan2(peak_cart[1], peak_cart[0])
        peak_norm = peak_cart * conversion_factor
        peak_t = np.array([peak_norm[0], peak_norm[1]])

        peaks.append(peak_t)

    if plot:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(2, 3)
        canvas = cv.cvtColor(im, cv.COLOR_GRAY2BGR)
        y, x = np.nonzero(mask)
        canvas[y, x] = (0, 0, 255)

        ax[0][0].imshow(im)
        ax[0][0].set_title("Base Im")

        ax[0][1].imshow(im1)
        ax[0][1].set_title("dilated")

        ax[0][2].imshow(med)
        ax[0][2].set_title("medianed")

        ax[1][0].imshow(mask)
        ax[1][0].set_title("mask")

        ax[1][1].imshow(canvas)
        ax[1][1].set_title("pikse")

        ax[1][2].imshow(peaks_im)
        ax[1][2].set_title("pikse")
        plt.show()

    return peaks


def arg_max_detector(x):
    return [np.unravel_index(np.argmax(x), x.shape)]

def arg_max_detector2(x, min_height=10):
    conversion_factor = 1.8 / (x.shape[0] / 2) 
    max_value_arg = np.unravel_index(np.argmax(x), x.shape)
    if x[max_value_arg] < min_height:
        return  []
    peak_cart = max_value_arg - np.array([x.shape[0] / 2, x.shape[0] / 2])
    peak_norm = peak_cart * conversion_factor
    peak_t = np.array([peak_norm[1], peak_norm[0]])
    return [peak_t]

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from numpy import cos, sin, pi
    import sys

    from sphere import create_sphere
    from scipy.interpolate import griddata

    sphere_size = 1500
    sphere_factor = 1.1
    sphere = create_sphere(
        sphere_size, sphere_factor
    )  # Make spher sligthly bigger for peak detection
    x = cos(sphere["phi"]) * sphere["theta"]
    y = sin(sphere["phi"]) * sphere["theta"]

    grid_x, grid_y = np.mgrid[-1.7:1.7:160j, -1.7:1.7:160j]
    mask = np.zeros_like(x)
    mask[:sphere_size] = 1
    grid = griddata((x, y), mask, (grid_x, grid_y), method="linear", fill_value=0.5).T
    mask = grid == 1
    plt.imshow(grid)
    plt.show()

    for i in range(80):
        im = cv.imread(f"./tmp/im{i}.png", cv.IMREAD_GRAYSCALE)  # 25
        peak_detector2(im, area_mask=mask, sphere_factor=sphere_factor, min_height=100, plot=True)
