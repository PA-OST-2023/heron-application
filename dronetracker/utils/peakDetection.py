import numpy as np
import cv2 as cv
from math import degrees


def peak_detector(im):
    fig, ax = plt.subplots(2, 2)
    ret, im2 = cv.threshold(im, 50, 255, cv.THRESH_BINARY)
    im3 = cv.dilate(im2, None, iterations=2)

    contours, hier = cv.findContours(im3, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    print("found", len(contours), "contours")
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


def peak_detector2(im, area_mask=None,sphere_factor=1.1, plot=False):
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
    conversion_factor = np.pi/2 * sphere_factor / 79.5
    print('<>'*30)
    for center in centroids[1:]:
        print("="*15)
        print(center)
        peak_cart = center - np.array([79.5, 79.5])
        print(peak_cart)
        theta = degrees(np.linalg.norm(peak_cart) * conversion_factor)
        phi = degrees(np.arctan2(-peak_cart[1], peak_cart[0]))
        print(f'{theta = }')
        print(f'{phi = }')

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

    return


def arg_max_detector(x):
    return [np.unravel_index(np.argmax(x), x.shape)]


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

    print(1)
    for i in range(80):
        im = cv.imread(f"./tmp/im{i}.png", cv.IMREAD_GRAYSCALE)  # 25
        print(im.dtype)
        peak_detector2(im, mask, sphere_factor, plot=True)
