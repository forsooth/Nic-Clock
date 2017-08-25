import math
import time
from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sanddial import colors
from sanddial import err

MIN_AREA_P = 0.000025
OVERLAY_COLOR = (80, 0, 0)
SUCCESS_COLOR = (50, 200, 50)

BoundingBox = namedtuple('BoundingBox', 'bbt bbb bbl bbr')
Point = namedtuple('Point', 'x y')


# Find the midpoint between two points (average the X and Y)
def midpoint(ptA, ptB):
    return ((ptA.x + ptB.x) * 0.5, (ptA.y + ptB.y) * 0.5)


def dist(ptA, ptB):
    dx = ptA.x - ptB.x
    dy = ptA.y - ptB.y
    return math.sqrt(dx ** 2 + dy ** 2)


def threshold(img, boundaries):
    threshed = img.copy()
    for (lower, upper) in boundaries:
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")
        mask = cv2.inRange(threshed, lower, upper)
        threshed = cv2.bitwise_and(threshed, threshed, mask=mask)
    return threshed


def edge_detect(img):
    err.log("Taking blurred image edge detection")
    blurred = cv2.GaussianBlur(img.copy(), (21, 21), 0)

    # perform edge detection, then perform a dilation + erosion to
    # close gaps in between object edges
    err.log("Running Canny edge detection")
    edged = cv2.Canny(blurred.copy(), 50, 100)

    err.log("Dilating edges")
    dilated = cv2.dilate(edged.copy(), None, iterations=5)

    return dilated


def find_contours(img):
    # find contours in the edge map
    contours = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[1]

    return contours


def draw_bbox(img, dims, bbox):
    overlay = img.copy()
    width, height = dims
    cv2.rectangle(overlay, (0, 0), (width, bbox.bbt), OVERLAY_COLOR, -1)
    cv2.rectangle(overlay, (0, 0), (bbox.bbl, height), OVERLAY_COLOR, -1)
    cv2.rectangle(overlay, (bbox.bbr, 0), (width, height), OVERLAY_COLOR, -1)
    cv2.rectangle(overlay, (0, bbox.bbb), (width, height), OVERLAY_COLOR, -1)

    cv2.addWeighted(overlay, 0.25, img, 0.75, 0, img)


def oob(lftx, rgtx, topy, boty, bbox):
    # If the extreme points are not positioned in a way
    # that they could be our sand, mark them differently
    if lftx < bbox.bbl:
        return True
    if rgtx > bbox.bbr:
        return True
    if topy < bbox.bbt:
        return True
    if boty > bbox.bbb:
        return True
    return False


def find_sand(img, dilated, contours, dims, bbox):
    sand_p1 = dims
    sand_p2 = Point(0, 0)

    draw_bbox(img, dims, bbox)

    for contour in contours:
        # If the area contained in the contour is too small
        # (smaller than a 10px by 20px area), ignore it
        if cv2.contourArea(contour) < MIN_AREA_P * dims.x * dims.y:
            continue

        # Find the bounding box of this contour
        box = cv2.minAreaRect(contour)
        box = cv2.boxPoints(box)
        box = np.array(box, dtype="int")

        # get the extreme left/right/top/bottom points
        lftp = Point(*contour[contour[:, :, 0].argmin()][0])
        rgtp = Point(*contour[contour[:, :, 0].argmax()][0])
        topp = Point(*contour[contour[:, :, 1].argmin()][0])
        botp = Point(*contour[contour[:, :, 1].argmax()][0])

        # compute the Euclidean distance between the midpoints
        # to get the edge lengths
        boxh = botp.y - topp.y
        boxw = rgtp.x - lftp.x

        if oob(lftp.x, rgtp.x, topp.y, botp.y, bbox):
            continue
        else:
            if sand_p2.x < rgtp.x:
                sand_p2 = Point(rgtp.x, sand_p2.y)
            if sand_p1.x > lftp.x:
                sand_p1 = Point(lftp.x, sand_p1.y)
            if sand_p2.y < botp.y:
                sand_p2 = Point(sand_p2.x, botp.y)
            if sand_p1.x > topp.y:
                sand_p1 = Point(sand_p1.x, topp.y)
            print(colors.GREEN +
                  "Found sand at ({}, {})".format(lftp.x, topp.y))
            cv2.line(img, (int((lftp.x + rgtp.x) / 2), topp.y),
                     (int((lftp.x + rgtp.x) / 2), botp.y), SUCCESS_COLOR, 5)

        cv2.circle(img, lftp, 25, SUCCESS_COLOR, 5)
        cv2.circle(img, rgtp, 25, SUCCESS_COLOR, 5)
        cv2.circle(img, topp, 25, SUCCESS_COLOR, 5)
        cv2.circle(img, botp, 25, SUCCESS_COLOR, 5)

        cv2.drawContours(img, [box.astype("int")], -1, SUCCESS_COLOR, 2)

        cv2.drawContours(dilated, [box.astype("int")], -1, SUCCESS_COLOR, 2)

        # draw the object sizes on the image
        cv2.putText(img, "w: {:.1f}px".format(boxw),
                    (int(rgtp.x + 15), int(botp.y + 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, SUCCESS_COLOR, 2)

        cv2.putText(img, "h: {:.1f}px".format(boxh),
                    (int(rgtp.x + 15), int(botp.y)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, SUCCESS_COLOR, 2)

        cv2.putText(dilated, "w: {:.1f}px".format(boxw),
                    (int(rgtp.x + 15), int(botp.y + 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, SUCCESS_COLOR, 2)

        cv2.putText(dilated, "h: {:.1f}px".format(boxh),
                    (int(rgtp.x + 15), int(botp.y)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, SUCCESS_COLOR, 2)

    return img, dilated, sand_p2.x - sand_p1.x, sand_p2.y - sand_p1.y


class ImageProcessor():

    def __init__(self, width, height, channels):
        # Generate plot for drawing our figures
        plt.ion()
        self.__plt = plt.figure()

        self.dims = Point(width, height)

        self.input_img = None

        # This plot holds a color image; its array is w*h*c
        plt.subplot2grid((1, 2), (0, 0))
        emptyim = np.empty((width * height * channels),
                           dtype=np.uint8).reshape((height, width, channels))
        self.leftimg = plt.imshow(emptyim.copy(), animated=True)

        # This plot holds a grayscale image; its array is w*h
        plt.subplot2grid((1, 2), (0, 1))
        emptyim = np.empty((width * height),
                           dtype=np.uint8).reshape((height, width))
        self.rightimg = plt.imshow(emptyim.copy(), animated=True)

        self.sand_dims = Point(0, 0)

        bbt = int(height / 2 - 0.10 * height)
        bbb = int(height / 2 + 0.10 * height)
        bbl = int(width / 2 - 0.20 * width)
        bbr = int(width / 2 + 0.20 * width)

        # set bounding box for sand
        self.bbox = BoundingBox(bbt, bbb, bbl, bbr)

        plt.show()

    def load_img(self, img):
        self.input_img = img

    # Perform image analysis
    def analyze(self):
        img = self.input_img

        # Upper and lower bounds for colors used in thresholding
        boundaries = [([30, 60, 60], [100, 250, 250])]

        # Convert threshold boundary colors to printable hex codes
        lbound = '#' + ''.join(map(hex, boundaries[0][0])).replace('0x', '')
        ubound = '#' + ''.join(map(hex, boundaries[0][1])).replace('0x', '')
        err.log("Running threshold with lower limit {}, upper limit {}"
                .format(lbound, ubound))

        threshed = threshold(img, boundaries)
        edgemap = edge_detect(threshed)
        contours = find_contours(edgemap)
        img, edgemap, sandh, sandw = find_sand(img, edgemap, contours,
                                               self.dims, self.bbox)
        self.sand_dims = Point(sandw, sandh)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        print(colors.GREEN +
              "Found width and height of sand: {}px by {}px"
              .format(self.sand_dims.x, self.sand_dims.y))
        self.leftimg.set_data(img)
        self.rightimg.set_data(edgemap)

        plt.draw()
        plt.pause(0.1)
        time.sleep(0.1)

        if self.sand_dims.x == 0 or self.sand_dims.y == 0:
            return True
        return False


def main():
    # Initialize an image processor for the correct image dimensions
    imgproc = ImageProcessor(2268, 4032, 3)
    # Loop over the set of test images, displaying the output for each one
    for i in range(1, 28):
        # Prefix the test images with a '0' if they are one digit
        strnum = str(i)
        if len(strnum) == 1:
            strnum = '0' + strnum
        # Read the image from the file and begin our processing
        img = cv2.imread('../img/test' + strnum + '.jpg')
        imgproc.load_img(img)
        imgproc.analyze()
