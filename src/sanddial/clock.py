import math
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sanddial import colors
from sanddial import err


# Find the midpoint between two points (average the X and Y)
def midpoint(ptA, ptB):
    x1, y1 = ptA
    x2, y2 = ptB
    return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)


def dist(ptA, ptB):
    x1, y1 = ptA
    x2, y2 = ptB
    dx = x1 - x2
    dy = y1 - y2
    return math.sqrt(dx ** 2 + dy ** 2)


class BoundingBox():
    def __init__(self, bbt, bbb, bbl, bbr):
        self.bbt = bbt
        self.bbb = bbb
        self.bbl = bbl
        self.bbr = bbr


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


def draw_bbox(img, width, height, bbox):
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (width, bbox.bbt), (80, 0, 0), -1)
    cv2.rectangle(overlay, (0, 0), (bbox.bbl, height), (80, 0, 0), -1)
    cv2.rectangle(overlay, (bbox.bbr, 0), (width, height), (80, 0, 0), -1)
    cv2.rectangle(overlay, (0, bbox.bbb), (width, height), (80, 0, 0), -1)

    cv2.addWeighted(overlay, 0.25, img, 0.75, 0, img)


def find_sand(img, dilated, contours, width, height, minarea, bbox):
    sand_h = 0
    sand_w = 0

    sand_p1 = (width, height)
    sand_p2 = (0, 0)

    for contour in contours:
        # If the area contained in the contour is too small
        # (smaller than a 10px by 20px area), ignore it
        if cv2.contourArea(contour) < minarea:
            continue

        # Find the bounding box of this contour
        box = cv2.minAreaRect(contour)
        box = cv2.boxPoints(box)
        box = np.array(box, dtype="int")

        # get the extreme left/right/top/bottom points
        lftp = tuple(contour[contour[:, :, 0].argmin()][0])
        rgtp = tuple(contour[contour[:, :, 0].argmax()][0])
        topp = tuple(contour[contour[:, :, 1].argmin()][0])
        botp = tuple(contour[contour[:, :, 1].argmax()][0])

        lftx = lftp[0]
        rgtx = rgtp[0]
        topy = topp[1]
        boty = botp[1]

        # compute the Euclidean distance between the midpoints
        # to get the edge lengths
        boxh = boty - topy
        boxw = rgtx - lftx

        text_color = (50, 200, 50)
        sand_box = True

        topcolor = (50, 200, 50)
        botcolor = (50, 200, 50)
        lftcolor = (50, 200, 50)
        rgtcolor = (50, 200, 50)

        num_outside = 0

        # If the extreme points are not positioned in a way
        # that they could be our sand, mark them differently
        if lftx < bbox.bbl:
            text_color = (50, 50, 200)
            lftcolor = (50, 50, 200)
            sand_box = False
            num_outside += 1

        if rgtx > bbox.bbr:
            text_color = (50, 50, 200)
            rgtcolor = (50, 50, 200)
            sand_box = False
            num_outside += 1

        if topy < bbox.bbt:
            text_color = (50, 50, 200)
            topcolor = (50, 50, 200)
            sand_box = False
            num_outside += 1

        if boty > bbox.bbb:
            text_color = (50, 50, 200)
            botcolor = (50, 50, 200)
            sand_box = False
            num_outside += 1

        if num_outside > 1:
            continue

        if sand_box is True:
            x1, y1 = sand_p1
            x2, y2 = sand_p2
            if x2 < rgtx:
                x2 = rgtx
            if x1 > lftx:
                x1 = lftx
            if y2 < boty:
                y2 = boty
            if y1 > topy:
                y1 = topy
            sand_p1 = (x1, y1)
            sand_p2 = (x2, y2)
            print(colors.GREEN +
                  "Found sand at ({}, {})".format(lftx, topy))
            cv2.line(img, (int((lftx + rgtx) / 2), topy),
                     (int((lftx + rgtx) / 2), boty), text_color, 5)

        cv2.circle(img, lftp, 25, lftcolor, 5)
        cv2.circle(img, rgtp, 25, rgtcolor, 5)
        cv2.circle(img, topp, 25, topcolor, 5)
        cv2.circle(img, botp, 25, botcolor, 5)

        cv2.drawContours(img, [box.astype("int")], -1, text_color, 2)

        cv2.drawContours(dilated, [box.astype("int")], -1, text_color, 2)

        # draw the object sizes on the image
        cv2.putText(img, "w: {:.1f}px".format(boxw),
                    (int(rgtx + 15), int(boty + 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, text_color, 2)

        cv2.putText(img, "h: {:.1f}px".format(boxh),
                    (int(rgtx + 15), int(boty)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, text_color, 2)

        cv2.putText(dilated, "w: {:.1f}px".format(boxw),
                    (int(rgtx + 15), int(boty + 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, text_color[::-1], 2)

        cv2.putText(dilated, "h: {:.1f}px".format(boxh),
                    (int(rgtx + 15), int(boty)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, text_color[::-1], 2)

    sand_w = sand_p2[0] - sand_p1[0]
    sand_h = sand_p2[1] - sand_p1[1]
    return img, dilated, sand_w, sand_h


class Clock():

    def __init__(self, width, height, channels):
        # Generate plot for drawing our figures
        plt.ion()
        self.__plt = plt.figure()

        # This plot holds a color image; its array is w*h*c
        plt.subplot2grid((1, 2), (0, 0))
        emptyim = np.empty((width * height * channels),
                           dtype=np.uint8).reshape((height, width, channels))
        self.__leftim = plt.imshow(emptyim.copy(), animated=True)

        # This plot holds a grayscale image; its array is w*h
        plt.subplot2grid((1, 2), (0, 1))
        emptyim = np.empty((width * height),
                           dtype=np.uint8).reshape((height, width))
        self.__rightim = plt.imshow(emptyim.copy(), animated=True)

        self.width = width
        self.height = height
        self.channels = channels
        self.sand_w = 0
        self.sand_h = 0

        bbt = int(height / 2 - 0.10 * height)
        bbb = int(height / 2 + 0.10 * height)
        bbl = int(width / 2 - 0.20 * width)
        bbr = int(width / 2 + 0.20 * width)
        self.bbox = BoundingBox(bbt, bbb, bbl, bbr)
        # set bounding box for where sand must be

        self.start_time = 0
        self.hour = 0
        self.minute = 0

        self.__minarea = (width * height) * (0.005 * 0.005)

        self.img = None
        self.leftimg = None
        self.rightimg = None

        plt.show()

    def init_time(self):
        self.start_time = datetime.datetime.now()
        self.hour = self.start_time.hour
        self.minute = self.start_time.minute

    def load_img(self, img):
        self.img = img

    # Perform image analysis
    def tick(self):
        img = self.img

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
                                               self.width, self.height,
                                               self.__minarea,
                                               self.bbox)
        self.sand_h = sandh
        self.sand_w = sandw

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        self.leftimg = img
        self.rightimg = edgemap

    # Display image
    def tock(self):
        print(colors.GREEN +
              "Found width and height of sand: {}px by {}px"
              .format(self.sand_h, self.sand_w))
        self.__leftim.set_data(self.leftimg)
        self.__rightim.set_data(self.rightimg)
        plt.draw()

        plt.pause(0.1)
        time.sleep(0.1)

        if self.sand_h == 0 or self.sand_w == 0:
            return True
        return False


def main():
    # Initialize a clock for the correct image dimensions
    clock = Clock(2268, 4032, 3)
    # Loop over the set of test images, displaying the output for each one
    for i in range(1, 28):
        # Prefix the test images with a '0' if they are one digit
        strnum = str(i)
        if len(strnum) == 1:
            strnum = '0' + strnum
        # Read the image from the file and begin our processing
        img = cv2.imread('../img/test' + strnum + '.jpg')
        clock.load_img(img)
        clock.tick()
        clock.tock()
