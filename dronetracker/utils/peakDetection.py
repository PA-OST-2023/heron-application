import numpy as np
import cv2 as cv

def peak_detector(im):
    fig, ax = plt.subplots(2,2)
    ret, im2 = cv.threshold(im, 50, 255, cv.THRESH_BINARY)
    im3 = cv.dilate(im2, None, iterations=2)

    contours, hier = cv.findContours(im3, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    print('found', len(contours), 'contours')
# draw a bounding box around each contour
    for contour in contours:
        x,y,w,h = cv.boundingRect(contour)
        cv.rectangle(im, (x,y), (x+w,y+h), (255,0,0), 2)

    cv.imshow('Contours', im)
    cv.waitKey()

    ax[0][0].imshow(im)
    ax[0][0].set_title('Base Im')

    ax[0][1].imshow(im2)
    ax[0][1].set_title('Thresholded')
    plt.show()
    pass

def peak_detector2(im):
    fig, ax = plt.subplots(2,3)
    im1 = cv.dilate(im, None, iterations=2)
    maxmask = (im == im1)
    med = cv.medianBlur(im, ksize=23)

    maxmask = (im == im1)
    medmask = (im.astype(np.int16) >= med.astype(np.int16) + 20) # peak property. avoiding unsigned difference.>>

    mask = maxmask & medmask

    canvas = cv.cvtColor(im, cv.COLOR_GRAY2BGR)
    y,x = np.nonzero(mask)
    canvas[y,x] = (0,0,255)

    ax[0][0].imshow(im)
    ax[0][0].set_title('Base Im')

    ax[0][1].imshow(im1)
    ax[0][1].set_title('dilated')

    ax[0][2].imshow(med)
    ax[0][2].set_title('medianed')

    ax[1][0].imshow(mask)
    ax[1][0].set_title('mask')

    ax[1][1].imshow(canvas)
    ax[1][1].set_title('pikse')
    plt.show()


def arg_max_detector(x):
    return [np.unravel_index(np.argmax(x), x.shape)]


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    print(1)
    for i in range(80):
        im = cv.imread(f'./tmp/im{i}.png', cv.IMREAD_GRAYSCALE) #25
        print(im.dtype)
        peak_detector2(im)
