import cv2
import numpy as np
from matplotlib import pyplot as plt
from imutils import contours
import imutils
import math
import colors
import err


# Find the midpoint between two points (average the X and Y)
def midpoint(ptA, ptB):
        return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

def dist(ptA, ptB):
        x1, y1 = ptA
        x2, y2 = ptB
        dx = x1 - x2
        dy = y1 - y2
        return math.sqrt(dx * dx + dy * dy)


def main(img_file):
        # Generate plot for drawing our figures
        fig = plt.figure()
        # Read the image fromm its file
        img = cv2.imread(img_file)
        # Grab original image dimensions
        height, width, channels = img.shape
        err.log("Original image width: {} height: {}".format(width, height))

        # Upper and lower bounds for colors used in thresholding
        boundaries = [
                ([0, 100, 100], [80, 250, 220])
        ]

        b0 = '#' + ''.join(map(lambda x: hex(x), boundaries[0][0])).replace('0x', '')
        b1 = '#' + ''.join(map(lambda x: hex(x), boundaries[0][1])).replace('0x', '')
        err.log("Running threshold with lower limit {}, upper limit {}".format(b0, b1))

        threshed = threshold(img, boundaries)

        ax1 = plt.subplot2grid((6, 4), (0, 0))
        plt.imshow(threshed)

        blurred, edged, dilated = edge_detect(threshed)

        ax2 = plt.subplot2grid((6, 4), (0, 1))
        plt.imshow(blurred)

        ax3 = plt.subplot2grid((6, 4), (0, 2))
        plt.imshow(edged)

        ax4 = plt.subplot2grid((6, 4), (0, 3))
        plt.imshow(dilated)

        cnts = find_contours(dilated.copy())

        img, dilated, sand_h, sand_w = draw_boxes(img, dilated.copy(), cnts, width, height)
        print(colors.t_green + "Found width and height of sand: {}px by {}px".format(sand_h, sand_w))

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax5 = plt.subplot2grid((6, 4), (2, 0), colspan=2, rowspan=5)
        plt.imshow(dilated)
        ax6 = plt.subplot2grid((6, 4), (2, 2), colspan=2, rowspan=5)
        plt.imshow(img)
        plt.show()


def threshold(img, boundaries):
        threshed = img.copy()
        for (lower, upper) in boundaries:
                lower = np.array(lower, dtype="uint8")
                upper = np.array(upper, dtype="uint8")
                mask = cv2.inRange(img, lower, upper)
                threshed = cv2.bitwise_and(img, img, mask=mask)
        return threshed


def edge_detect(img):
        err.log("Taking blurred image edge detection")
        blurred = cv2.GaussianBlur(img, (21, 21), 0)

        # perform edge detection, then perform a dilation + erosion to
        # close gaps in between object edges
        err.log("Running Canny edge detection")
        edged = cv2.Canny(blurred, 50, 100)

        err.log("Dilating edges")
        dilated = cv2.dilate(edged, None, iterations=5)

        return blurred, edged, dilated


def find_contours(img):
        # find contours in the edge map
        cnts = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]

        # sort the contours from left-to-right and initialize the
        # 'pixels per metric' calibration variable
        (cnts, _) = contours.sort_contours(cnts)

        return cnts


def draw_boxes(img, dilated, cnts, width, height):

        sand_h = -1
        sand_w = -1

        for c in cnts:
                # If the area contained in the contour is too small
                # (smaller than a 10px by 20px area), ignore it
                if cv2.contourArea(c) < (0.025 * height) * (0.025 * width):
                        continue

                # Find the bounding box of this contour
                box = cv2.minAreaRect(c)
                box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
                box = np.array(box, dtype="int")

                # Find midpoints of the bounding box edges
                (tl, tr, br, bl) = box
                (tltrX, tltrY) = midpoint(tl, tr)
                (blbrX, blbrY) = midpoint(bl, br)
                (tlblX, tlblY) = midpoint(tl, bl)
                (trbrX, trbrY) = midpoint(tr, br)
                 
                # compute the Euclidean distance between the midpoints
                # to get the edge lengths
                h = dist((tltrX, tltrY), (blbrX, blbrY))
                w = dist((tlblX, tlblY), (trbrX, trbrY))

                # get the extreme left/right/top/bottom points
                leftmost = tuple(c[c[:,:,0].argmin()][0])
                rightmost = tuple(c[c[:,:,0].argmax()][0])
                topmost = tuple(c[c[:,:,1].argmin()][0])
                bottommost = tuple(c[c[:,:,1].argmax()][0])

                text_color = (50, 200, 50)
                sand_box = True

                # If the extreme points are not positioned in a way
                # that they could be our sand, mark them differently
                x, y = leftmost
                if x > width / 2:
                        err.log("Detected false contour at ({}, {}); discarded for leftmost x value above {}".format(x, y, width / 2))
                        text_color = (50, 50, 200)
                        sand_box = False
                x, y = rightmost
                if x < width / 2:
                        err.log("Detected false contour at ({}, {}); discarded for rightmost x value below {}".format(x, y, width / 2))
                        text_color = (50, 50, 200)
                        sand_box = False
                x, y = topmost
                if y > height / 2 - .05 * height:
                        err.log("Detected false contour at ({}, {}); discarded for topmost y value above {}".format(x, y, height / 2 - 20))
                        text_color = (50, 50, 200)
                        sand_box = False
                if y < height / 2 - 35 * height:
                        err.log("Detected false contour at ({}, {}); discarded for topmost y value below {}".format(x, y, height / 2 - 250))
                        text_color = (50, 50, 200)
                        sand_box = False
                x, y = bottommost
                if y > height / 2 + 0.05 * height:
                        err.log("Detected false contour at ({}, {}); discarded for bottommost y value above {}".format(x, y, height / 2 + 30))
                        text_color = (50, 50, 200)
                        sand_box = False

                if sand_box == True:
                        sand_h = h
                        sand_w = w
                        print(colors.t_green + "Found sand at ({}, {})".format(x, y))

                cv2.drawContours(img, [box.astype("int")], -1, text_color, 2)
                cv2.drawContours(dilated, [box.astype("int")], -1, text_color, 2)

                # draw the object sizes on the image
                cv2.putText(img, "{:.1f}px".format(w),
                        (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, text_color, 2)
                cv2.putText(img, "{:.1f}px".format(h),
                        (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, text_color, 2)

                cv2.putText(dilated, "{:.1f}px".format(w),
                        (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, text_color, 2)
                cv2.putText(dilated, "{:.1f}px".format(h),
                        (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, text_color, 2)

        return img, dilated, sand_h, sand_w

main('img/test02.jpg')
