import math
import numpy as np
import cv2

#img = cv.imread('Photos/board.jpg')

#cv.imshow('Board', img)


capture = cv2.VideoCapture("photos/test.mp4")


window_name = 'Square Detect'
title_trackbarMin = 'Min Area'
title_trackbarMax = 'Max Area'
# best is: 3500 - 14000
min_area = 3600
max_area = 20000

title_trackbarDelta = 'delta'


def on_trackbarMin(val):
    global min_area
    min_area = val


def on_trackbarMax(val):
    global max_area
    max_area = val


def computeImage(frame):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
    bg = cv2.morphologyEx(image, cv2.MORPH_DILATE, se)
    out_gray = cv2.divide(image, bg, scale=255)
    out_binary = cv2.threshold(out_gray, 0, 255, cv2.THRESH_OTSU)[1]
    return out_binary


def computeContours(frame):
    cnts = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    return cnts


def hasFourVertices(coords):
    if len(coords) != 4:
        return False
    else:
        return True


def detectBoardContourMode(image, contours):
    valid_contours = []
    for c in contours:
        area = cv2.contourArea(c)

        if area > min_area and area < max_area:

            approx = cv2.approxPolyDP(c, 0.10 * cv2.arcLength(c, True), True)  # al posto di 0.009 ho messo 0.10

            if hasFourVertices(approx):  # controllo che ci siano almeno 4 angoli
                valid_contours.append(approx)
                cv2.drawContours(image, [approx], 0, (255, 255, 255), 2)

                M = cv2.moments(approx)

                try:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    cv2.circle(image, (cX, cY), 3, (255, 255, 255), -1)
                except:
                    pass
    return valid_contours

def getMiddlePoint(points):
    sumX = 0
    sumY = 0
    for point in points:
        sumX += point[0]
        sumY += point[1]

    medianX = int(sumX / len(points))
    medianY = int(sumY / len(points))

    closestPoints = kClosest(points, [medianX, medianY], 4)

    centreOfSquare = []
    dist = 5
    while len(centreOfSquare) != 1:
        centreOfSquare = fuse(closestPoints, dist)
        dist += 5

    return centreOfSquare

# Calculate the Euclidean distance
# between two points
def distance(x1, y1, x2, y2):
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** (1 / 2)


# Function to calculate K closest points
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

def applySquareAreaFilter(image, contours):
    try:
        valid_points = []

        for pointsArray_list in contours:
            for pointsArray in pointsArray_list:
                for point in pointsArray:
                    valid_points.append([point[0], point[1]])

        centreOfImage = getMiddlePoint(valid_points)
        # CTC = Closest To Centre
        xCentre = int(centreOfImage[0][0])
        yCentre = int(centreOfImage[0][1])

        cv2.circle(image, (xCentre, yCentre), 5, (0, 0, 255), -1)

        pointsCTC = kClosest(valid_points, centreOfImage[0], 36)

        fusedPointsCTC = fuse(pointsCTC, 20) #da sistemare il 20

        fixedPointsCTC = kClosest(fusedPointsCTC, centreOfImage[0], 9)

        for point in fixedPointsCTC:
            x = int(point[0])
            y = int(point[1])
            cv2.circle(image, (x, y), 3, (0, 255, 255), -1)

        fixedPointsCTC = sorted(fixedPointsCTC , key=lambda k: k[1])
        list1, list2 = fixedPointsCTC[:3], fixedPointsCTC[6:]

        list1 = sorted(list1 , key=lambda k: k[0])
        list2 = sorted(list2 , key=lambda k: k[0])

        xA = int(4 * list1[2][0] - 3 * xCentre)
        yA = int(4 * list1[2][1] - 3 * yCentre)
        vertexA = (xA, yA)

        xB = int(4 * list1[0][0] - 3 * xCentre)
        yB = int(4 * list1[0][1] - 3 * yCentre)
        vertexB = (xB, yB)

        xC = int(5 * list2[0][0] - 4 * xCentre)
        yC = int(5 * list2[0][1] - 4 * yCentre)
        vertexC = (xC, yC)

        xD = int(5 * list2[2][0] - 4 * xCentre)
        yD = int(5 * list2[2][1] - 4 * yCentre)
        vertexD = (xD, yD)

        cv2.circle(image, vertexA, 5, (0, 0, 255), -1)
        cv2.circle(image, vertexB, 5, (0, 0, 255), -1)
        cv2.circle(image, vertexC, 5, (0, 0, 255), -1)
        cv2.circle(image, vertexD, 5, (0, 0, 255), -1)

        return fusedPointsCTC

    except Exception as e:
        print(e)

def dist2(p1, p2):
    return (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2

def fuse(points, d):
    ret = []
    d2 = d * d
    n = len(points)
    taken = [False] * n
    for i in range(n):
        if not taken[i]:
            count = 1
            point = [points[i][0], points[i][1]]
            taken[i] = True
            for j in range(i+1, n):
                if dist2(points[i], points[j]) < d2:
                    point[0] += points[j][0]
                    point[1] += points[j][1]
                    count+=1
                    taken[j] = True
            point[0] /= count
            point[1] /= count
            ret.append((point[0], point[1]))
    return ret


cv2.namedWindow(window_name)
cv2.createTrackbar(title_trackbarMin, window_name, min_area, max_area, on_trackbarMin)
cv2.createTrackbar(title_trackbarMax, window_name, min_area, max_area, on_trackbarMax)

while True:

    isTrue, image = capture.read()
    thresh = computeImage(image)

    contours = computeContours(thresh)

    my_img_1 = np.zeros((image.shape[0], image.shape[1], 1), dtype="uint8")

    valid_contours = detectBoardContourMode(my_img_1, contours)

    ciao = applySquareAreaFilter(image, valid_contours)

    # applySquareAreaFilter(my_img_1, valid_contours)

    cv2.imshow('thresh', my_img_1)
    cv2.imshow(window_name, image)

    if cv2.waitKey(20) & 0xFF == ord('d'):
        break


capture.release()
cv2.destroyAllWindows()
