import numpy as np
import cv2
from scipy.spatial import distance as dist
import imutils
from imutils import perspective
import math
from collections import defaultdict

# img = cv.imread('Photos/board.jpg')

# cv.imshow('Board', img)


def computeImage(frame):
    pyr = cv2.pyrDown(frame, (frame.shape[1] / 2, frame.shape[0] / 2))
    pyr = cv2.pyrUp(pyr, (frame.shape[1], frame.shape[0]))
    gray = cv2.cvtColor(pyr, cv2.COLOR_BGR2GRAY)

    return gray


def computeImage2(frame):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # image = cv2.bilateralFilter(image,9,75,75)

    se = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    bg = cv2.morphologyEx(image, cv2.MORPH_DILATE, se)
    out_gray = cv2.divide(image, bg, scale=255)
    out_binary = cv2.threshold(out_gray, 0, 255, cv2.THRESH_OTSU)[1]

    blank = np.zeros((frame.shape[0], frame.shape[1], 1), dtype="uint8")

    contours1, hierarchy = cv2.findContours(out_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    cv2.drawContours(blank, contours1, -1, (255, 255, 255), 3)

    cv2.imshow('Frame3', blank)

    return out_binary


def getApproximatedCentrePoint(points):
    sumX = 0
    sumY = 0

    for point in points:
        x, y = point.ravel()
        sumX += x
        sumY += y

    medianX = int(sumX / (len(points)))
    medianY = int(sumY / (len(points)))

    return [medianX, medianY]


def distance(x1, y1, x2, y2):
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** (1 / 2)


def kClosest(points, target, K):
    pts = []
    n = len(points)
    d = []

    for i in range(n):
        d.append({
            "first": distance(points[i][0], points[i][1], target[0], target[1]),
            "second": i
        })

    d = sorted(d, key=lambda l: l["first"])

    for i in range(K):
        pt = []
        pt.append(points[d[i]["second"]][0])
        pt.append(points[d[i]["second"]][1])
        pts.append(pt)

    return pts


def order_points2(pts):
    # https://pyimagesearch.com/2016/03/21/ordering-coordinates-clockwise-with-python-and-opencv/
    xSorted = pts[np.argsort(pts[:, 0]), :]

    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]

    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost

    D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]

    return tl, tr, br, bl


def getInnerChilds(x, y):
    match (x, y):
        case (6, 6):
            return 5, 5
        case (6, 0):
            return 5, 1
        case (0, 6):
            return 1, 5
        case (0, 0):
            return 1, 1

    return -1, -1


def getInnerVertices(corners):
    blank = np.zeros((frame.shape[0], frame.shape[1], 1), dtype="uint8")
    for i, point in enumerate(corners):
        for j, point in enumerate(corners[:-1]):
            cv2.line(blank, (int(point.ravel()[0]), int(point.ravel()[1])),
                     (int(corners[i].ravel()[0]), int(corners[i].ravel()[1])),
                     (255, 255, 255), 3)
    contours, hierarchy = cv2.findContours(blank, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # big_contour = max(contours, key=cv2.contourArea)

    vertices = []
    for c in contours:
        approx = cv2.approxPolyDP(c, 0.009 * cv2.arcLength(c, True), True)
        n = approx.ravel()
        i = 0
        for j in n:
            if (i % 2 == 0):
                x = n[i]
                y = n[i + 1]

                vertices.append([x, y])

            i = i + 1


        (tl, tr, br, bl) = imutils.perspective.order_points(np.asarray(vertices))


    tl = kClosest(corners.tolist(), tl, 1)
    tr = kClosest(corners.tolist(), tr, 1)
    br = kClosest(corners.tolist(), br, 1)
    bl = kClosest(corners.tolist(), bl, 1)

    return (tl[0], tr[0], br[0], bl[0])


def getOuterVertices(cornersMatrix):
    parentTL = parentTR = parentBR = parentBL = -1

    for row in range(0, 7):
        for column in range(0, 7):
            x = cornersMatrix[row][column][0]
            y = cornersMatrix[row][column][1]
            if x == tl[0] and y == tl[1]:
                xChild, yChild = getInnerChilds(row, column)
                parentTL = ((2 * x - cornersMatrix[xChild][yChild][0]), (2 * y - cornersMatrix[xChild][yChild][1]))
            elif x == tr[0] and y == tr[1]:
                xChild, yChild = getInnerChilds(row, column)
                parentTR = ((2 * x - cornersMatrix[xChild][yChild][0]), (2 * y - cornersMatrix[xChild][yChild][1]))
            elif x == br[0] and y == br[1]:
                xChild, yChild = getInnerChilds(row, column)
                parentBR = ((2 * x - cornersMatrix[xChild][yChild][0]), (2 * y - cornersMatrix[xChild][yChild][1]))
            elif x == bl[0] and y == bl[1]:
                xChild, yChild = getInnerChilds(row, column)
                parentBL = ((2 * x - cornersMatrix[xChild][yChild][0]), (2 * y - cornersMatrix[xChild][yChild][1]))

    return parentTL, parentTR, parentBR, parentBL

def find(target, cornersMatrix):
    for row in range(0, 7):
        for column in range(0, 7):
            x = cornersMatrix[row][column][0]
            y = cornersMatrix[row][column][1]
            if x == target[0] and y == target[1]:
                return (row, column)
    return (None, None)


def createCornersMatrix(array):
    cornersMatrix_list = array.tolist()

    cornersMatrix = np.empty((7, 7), object)

    for idx in np.ndindex(7, 7):
        cornersMatrix[idx] = idx

    for row in range(0, 7):
        for column in range(0, 7):
            x = cornersMatrix_list[row][column][0]
            y = cornersMatrix_list[row][column][1]
            cornersMatrix[row][column] = (x, y)

    return cornersMatrix

def extend_line(p1, p2, distance=10000):
    diff = np.arctan2(p1[1] - p2[1], p1[0] - p2[0])
    p3_x = int(p1[0] + distance*np.cos(diff))
    p3_y = int(p1[1] + distance*np.sin(diff))
    p4_x = int(p1[0] - distance*np.cos(diff))
    p4_y = int(p1[1] - distance*np.sin(diff))
    return ((p3_x, p3_y), (p4_x, p4_y))

def getLines(frame, cornersMatrix, tl, tr, br ,bl, innerTL):

    horizonalLines = []
    verticalLines = []
    innerTL = (innerTL[0], innerTL[1])

    horizonalLines.append([tl, tr])
    verticalLines.append([tl, bl])
    horizonalLines.append([bl, br])
    verticalLines.append([br, tr])

    if cornersMatrix[6][0] == innerTL:
        verticalLines.append([tl, tr])
        horizonalLines.append([tl, bl])
        verticalLines.append([bl, br])
        horizonalLines.append([br, tr])

    cv2.line(frame, tl, tr, (255, 255, 255), 2)
    cv2.line(frame, tl, bl, (255, 255, 255), 2)
    cv2.line(frame, bl, br, (255, 255, 255), 2)
    cv2.line(frame, br, tr, (255, 255, 255), 2)

    for i in range(0, 7):
        pt1, pt2 = extend_line(cornersMatrix[i][0], cornersMatrix[i][6])
        pt3, pt4 = extend_line(cornersMatrix[0][i], cornersMatrix[6][i])
        cv2.line(frame, pt1, pt2, (255, 255, 255), 2)
        cv2.line(frame, pt3, pt4, (255, 255, 255), 2)

        horizonalLines.append([pt1, pt2])
        verticalLines.append([pt3, pt4])

    cv2.imshow('Frame2', frame)

    return horizonalLines, verticalLines

def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div

    return x, y


capture = cv2.VideoCapture(0)
nline = 7
ncol = 7
font = cv2.FONT_HERSHEY_COMPLEX

while True:
    isTrue, frame = capture.read()

    ## termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    imgFiltered = computeImage(frame)

    ret, corners = cv2.findChessboardCorners(imgFiltered, (nline, ncol), flags=cv2.CALIB_CB_ADAPTIVE_THRESH)

    if ret:

        corners2 = cv2.cornerSubPix(imgFiltered, corners, (11, 11), (-1, -1), criteria)
        corners2_int = corners2.astype(int)
        corners_final = corners2_int.reshape(49, 2)

        try:
            (tl, tr, br, bl) = getInnerVertices(corners_final)
            innerTL = tl
        except:
            pass

        cornersMatrix = createCornersMatrix(corners_final.reshape(7, 7, 2))

        try:
            (tl, tr, br, bl) = getOuterVertices(cornersMatrix)
            blank = np.zeros((frame.shape[0], frame.shape[1], 1), dtype="uint8")
            horizontalLines, verticalLines = getLines(blank, cornersMatrix, tl, tr, br, bl, innerTL)

            intersections = []
            for hLines in horizontalLines:
                for vLines in verticalLines:
                    try:
                        intersections.append(line_intersection(hLines, vLines))
                    except:
                        pass


            for point in intersections:
                print((int(point[0]), int(point[1])))
                cv2.circle(frame, (int(point[0]), int(point[1])), 5, (255, 0, 0), -1)



        except Exception as e:
            print(e)

        # for row in range(0, 7):
        #     for column in range(0, 7):
        #         x = cornersMatrix[row][column][0]
        #         y = cornersMatrix[row][column][1]
        #         cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)
        #         cv2.putText(frame, "(" + str(row) + "," + str(column) + ")",
        #                     (cornersMatrix[row][column][0], cornersMatrix[row][column][1]), font, 0.5, (0, 0, 0), 2,
        #                     cv2.LINE_AA)

    # cv2.imshow('imgFiltered', blank)
    cv2.imshow('Frame', frame)

    if (cv2.waitKey(20) & 0xFF == ord('d')):
        break

capture.release()
cv2.destroyAllWindows()

# cv.waitKey(0) #aspetta un tasto per l'imput
