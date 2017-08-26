"""This module performs the core image processing routine for the sanddial
clock. the expected main functionality is to load an image, then analyze it,
and repeat until desired to tell the time.
"""
import os
import math
from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sanddial import err

MIN_AREA_P = 0.000025
OVERLAY_COLOR = (80, 0, 0)
SUCCESS_COLOR = (50, 200, 50)
CHANNELS = 3

BoundingBox = namedtuple('BoundingBox', 'bbt bbb bbl bbr')
Point = namedtuple('Point', 'x y')


def midpoint(ptA, ptB):
    """Find the midpoint between two points, i.e. average their
    x and y coordinates.

    Args:
        ptA: the first of the two Points
        ptB: the second of the two Points

    Returns:
        The midpoint between the two points, with x and y as integers
    """
    return (int((ptA.x + ptB.x) * 0.5), int((ptA.y + ptB.y) * 0.5))


def dist(ptA, ptB):
    """Calculate the distance between two points, by taking the differences
    in their x and y coordinates and using the Pythagorean theorem.

    Args:
        ptA: the source Point
        ptB: the destination Point

    Returns: the distance between the two points as an integer
    """
    dx = ptA.x - ptB.x
    dy = ptA.y - ptB.y
    return int(math.sqrt(dx ** 2 + dy ** 2))


def threshold(img, boundaries):
    """Given an image and a list of boundaries, uses each pair of boundaries
    in the list to perform successive thresholdings, leaving all pixels not
    in the threshold range black.

    Args:
        img: the color OpenCV image on which to perform the thresholding
        boundaries: a list of tuples containing lower and upper bounds for
                    the thresholding in that order

    Returns: the thresholded image
    """
    threshed = img.copy()
    for (lower, upper) in boundaries:
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")
        mask = cv2.inRange(threshed, lower, upper)
        threshed = cv2.bitwise_and(threshed, threshed, mask=mask)
    return threshed


def edge_detect(img):
    """Given an image, performs an edge detection on that image.

    Args:
        img: the color OpenCV image on which edges are to be detected

    Returns: the grayscale edgemap image created from the edge detection
             process
    """

    err.log("Taking blurred image edge detection")
    blurred = cv2.GaussianBlur(img.copy(), (21, 21), 0)

    # perform edge detection, then perform a dilation + erosion to
    # close gaps in between object edges
    err.log("Running Canny edge detection")
    edged = cv2.Canny(blurred.copy(), 50, 100)

    err.log("Dilating edges")
    dilated = cv2.dilate(edged.copy(), None, iterations=1)

    return dilated


def find_contours(img):
    """Given an edge map, performs a Canny contour finding algorithm and
    returns the result.

    Args:
        img: the grayscale OpenCV edgemap image on which to perform the
             contour detection

    Returns:
        a list of OpenCV countours contained in the image
    """
    contours = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)[1]
    return contours


def draw_bbox(img, dims, bbox):
    """Given an image, the dimensions of that image, and a bounding box
    representing the four points of a rectangle within the bounds of that
    image, draw a gray overlay over the image everywhere except within the
    bounding box.

    Args:
        img: the color OpenCV image to overlay with the bounding box
        dims: the Point representing the width and height of the image
        bbox: the BoundingBox representing the found points of interest

    Returns: the new overlaid image.
    """

    overlay = img.copy()
    width, height = dims
    cv2.rectangle(overlay, (0, 0), (width, bbox.bbt), OVERLAY_COLOR, -1)
    cv2.rectangle(overlay, (0, 0), (bbox.bbl, height), OVERLAY_COLOR, -1)
    cv2.rectangle(overlay, (bbox.bbr, 0), (width, height), OVERLAY_COLOR, -1)
    cv2.rectangle(overlay, (0, bbox.bbb), (width, height), OVERLAY_COLOR, -1)

    cv2.addWeighted(overlay, 0.25, img, 0.75, 0, img)


def oob(lftx, rgtx, topy, boty, bbox):
    """Given four points of extremum and a bounding box, tell whether any
    of the points are out of bounds.

    Args:
        lftx: the x value of the leftmost point
        rgtx: the x value of the rightmost point
        topy: the y value of the topmost point
        boty: the y value of the bottommost point
        bbox: the BoundingBox determining which points should be considered
              'in bounds' and 'out of bounds'

    Returns: True if any points are out of bounds, and False otherwise
    """
    if lftx < bbox.bbl:
        return True
    if rgtx > bbox.bbr:
        return True
    if topy < bbox.bbt:
        return True
    if boty > bbox.bbb:
        return True
    return False


def find_objs(img, dilated, contours, dims, bbox):
    """Given an image, an edgemap of that image, a set of contours in that
    image over which we should iterate, the dimensions of that image, and
    a bounding box of interest, iterate over the countours and find any
    which fit in the bounding box. Return the width and height of any objects
    in the bounding box as determined by the extrema of contours in the region,
    as well as modified versions of the images.

    Args:
        img: the color OpenCV image on which to operate
        dilated: the grayscale OpenCV edgemap image from which the countours
                 were derived
        contours: a list of OpenCV contours found in the image
        dims: the dimensions of the images in question
        bbox: a BoundingBox representing an area of interest outside of which
              any contours should be ignored

    Returns: a tuple containing the following:
        a modified version of img with an overlay showing the bounding box
            and a highlight of any objects within the bounding box,
            as well as their extrema and dimensions (labeled on image)
        a modified version of dilated with the same modifications
        the width of any objects determined by taking the difference of the
            furthest right point of any contour in the bounding box and the
            furthest left point of any contour in the bounding box
        the height of any objects determined in the same way
    """
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
            err.success("Found sand at ({}, {})".format(lftp.x, topp.y))
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
    """This object defines an image processor which uses the above operations
    to detect sand in an image of an hourglass. It maintains state in the form
    of images, dimensions, etc. and is expected to be used for multiple runs
    over which the dimensions stay the same, e.g. a feed of images from a
    camera. Hence the expectation is a loop over which images are repeatedly
    added with load_image and then analyzed with analyze, after which the
    accessors for sand_width and sand_height may be called.
    """

    def __init__(self, width, height):
        """Generates a plot on which to output the image processing information
        as well as initializing blank images, setting simple class attributes,
        setting the expected values for the bounding box, and displaying
        the initial plot.

        Args:
            width: the width of the image, in pixels
            height: the height of the image, in pixels
        """

        # Generate plot for drawing our figures
        plt.ion()
        self.__plt = plt.figure()

        self.dims = Point(width, height)

        self.input_img = None

        # This plot holds a color image; its array is w*h*c
        plt.subplot2grid((1, 2), (0, 0))
        emptyim = np.empty((width * height * CHANNELS),
                           dtype=np.uint8).reshape((height, width, CHANNELS))
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
        """Load a new image into the image processor.

        Args:
            img: the color OpenCV image to load for processing.
        """
        self.input_img = img

    def analyze(self):
        """Perform a series of image processing operations on the most recently
        loaded image for the purpose of detecting the location of objects in
        a given range, in particular sand in an hourglass (though the module
        could certainly be ported to other uses with minor modifications to
        constants), and then setting the dimensions of those objects in pixels.

        Returns: False if no objects were detected in the bounding box, True
        otherwise.
        """

        img = self.input_img

        # Upper and lower bounds for colors used in thresholding
        boundaries = [([30, 60, 60], [100, 250, 250])]

        # Convert threshold boundary colors to printable hex codes
        lbound = '#' + ''.join(map(hex, boundaries[0][0])).replace('0x', '')
        ubound = '#' + ''.join(map(hex, boundaries[0][1])).replace('0x', '')
        err.log("Running threshold with lower limit {}, upper limit {}"
                .format(lbound, ubound))

        # perform some image processing operations
        threshed = threshold(img, boundaries)
        edgemap = edge_detect(threshed)
        contours = find_contours(edgemap)
        img, edgemap, sandh, sandw = find_objs(img, edgemap, contours,
                                               self.dims, self.bbox)
        # set the dimensions of the sand that we gained from finding objects
        # within our bounding box.
        self.sand_dims = Point(sandw, sandh)

        # OpenCV uses BGR, so convert to RGB for viewing
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        err.success("Found width and height of sand: {}px by {}px"
                    .format(self.sand_dims.x, self.sand_dims.y))

        # Reset the matplotlib axes' data, and redraw them
        self.leftimg.set_data(img)
        self.rightimg.set_data(edgemap)
        plt.draw()
        # This wait is necessary to allow the frames to update
        plt.pause(0.1)

        # Return true or false based on whether any sand was detected.
        if self.sand_dims.x <= 0 or self.sand_dims.y <= 0:
            return True
        return False


def test():
    """Perform a test run of the image processor suite of utilities on the
    expected test image files of an hourglass.
    """

    # Initialize an image processor for the correct image dimensions
    imgproc = ImageProcessor(2268, 4032)
    dir_path = os.path.dirname(os.path.realpath(__file__))

    # Loop over the set of test images, displaying the output for each one
    for i in range(1, 28):
        # Prefix the test images with a '0' if they are one digit
        strnum = str(i)
        if len(strnum) == 1:
            strnum = '0' + strnum
        # Read the image from the file and begin our processing
        img = cv2.imread(dir_path + '../../img/test' + strnum + '.jpg')
        imgproc.load_img(img)
        imgproc.analyze()
