import math
import numpy as np
import cv2


capture = cv2.VideoCapture(0)


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
    # image = cv2.bilateralFilter(image,9,75,75)

    se = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    bg = cv2.morphologyEx(image, cv2.MORPH_DILATE, se)
    out_gray = cv2.divide(image, bg, scale=255)
    out_binary = cv2.threshold(out_gray, 0, 255, cv2.THRESH_OTSU)[1]

    blank = np.zeros((frame.shape[0], frame.shape[1], 1), dtype="uint8")

    contours1, hierarchy = cv2.findContours(out_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    cv2.drawContours(blank, contours1, -1, (255, 255, 255), 3)

    cv2.imshow('Frame3', blank)

    return blank


def computeContours(frame):
    cnts = cv2.findContours(frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    return cnts


def hasFourVertices(coords):
    if len(coords) != 4:
        return False
    else:
        return True


def detectBoardSquares(image, contours, boardContour, isFilterWorking):
    valid_contours = []

    for c in contours:
        area = cv2.contourArea(c)

        if area > min_area and area < max_area:

            approx = cv2.approxPolyDP(c, 0.10 * cv2.arcLength(c, True), True)  # al posto di 0.009 ho messo 0.10

            if hasFourVertices(approx):  # controllo che ci siano almeno 4 angoli

                M = cv2.moments(approx)

                try:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])

                    if isFilterWorking:
                        if cv2.pointPolygonTest(boardContour[0], (cX, cY), False) == 1:
                            valid_contours.append(approx)
                            cv2.drawContours(image, [approx], 0, (255, 255, 255), 2)
                            cv2.circle(image, (cX, cY), 3, (255, 255, 255), -1)
                    else:
                        valid_contours.append(approx)
                        cv2.drawContours(image, [approx], 0, (255, 255, 255), 2)
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

    return (int(centreOfSquare[0][0]), int(centreOfSquare[0][1]))

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

def detect2x2(centreOfImage, valid_points):

    pointsCTC = kClosest(valid_points, centreOfImage, 36)

    fusedPointsCTC = fuse(pointsCTC, 20)  # da sistemare il 20

    fixedPointsCTC = kClosest(fusedPointsCTC, centreOfImage, 9)

    fixedSortedPointsCTC = sorted(fixedPointsCTC, key=lambda k: k[1])
    list1, list2 = fixedSortedPointsCTC[:3], fixedSortedPointsCTC[6:]

    list1 = sorted(list1 , key=lambda k: k[0])
    list2 = sorted(list2 , key=lambda k: k[0])

    vertexA = (int(list1[0][0]),int(list1[0][1]))
    vertexB = (int(list1[2][0]),int(list1[2][1]))
    vertexC = (int(list2[2][0]),int(list2[2][1]))
    vertexD = (int(list2[0][0]),int(list2[0][1]))

    return vertexA, vertexB, vertexC, vertexD

def detect4x4(valid_points, vertexA, vertexB, vertexC, vertexD):
    pointsCTS = []

    pointsCTS = pointsCTS + kClosest(valid_points, vertexA, 36)
    pointsCTS = pointsCTS + kClosest(valid_points, vertexB, 36)
    pointsCTS = pointsCTS + kClosest(valid_points, vertexC, 36)
    pointsCTS = pointsCTS + kClosest(valid_points, vertexD, 36)

    fusedPointsCTS = fuse(pointsCTS, 20)

    fusedPointsCTS = sorted(fusedPointsCTS, key=lambda k: k[1])
    list1, list2 = fusedPointsCTS[:5], fusedPointsCTS[20:]

    list1 = sorted(list1, key=lambda k: k[0])
    list2 = sorted(list2, key=lambda k: k[0])

    vertexA = (int(list1[0][0]), int(list1[0][1]))
    vertexB = (int(list1[4][0]), int(list1[4][1]))
    vertexC = (int(list2[4][0]), int(list2[4][1]))
    vertexD = (int(list2[0][0]), int(list2[0][1]))

    return vertexA, vertexB, vertexC, vertexD

def detect6x6(valid_points, vertexA, vertexB, vertexC, vertexD):
    pointsCTS = []

    pointsCTS = pointsCTS + kClosest(valid_points, vertexA, 36)
    pointsCTS = pointsCTS + kClosest(valid_points, vertexB, 36)
    pointsCTS = pointsCTS + kClosest(valid_points, vertexC, 36)
    pointsCTS = pointsCTS + kClosest(valid_points, vertexD, 36)

    fusedPointsCTS = fuse(pointsCTS, 20)

    fusedPointsCTS = sorted(fusedPointsCTS, key=lambda k: k[1])
    list1, list2 = fusedPointsCTS[:6], fusedPointsCTS[30:]

    list1 = sorted(list1, key=lambda k: k[0])
    list2 = sorted(list2, key=lambda k: k[0])

    vertexA = (int(list1[0][0]), int(list1[0][1]))
    vertexB = (int(list1[5][0]), int(list1[5][1]))
    vertexC = (int(list2[5][0]), int(list2[5][1]))
    vertexD = (int(list2[0][0]), int(list2[0][1]))

    return vertexA, vertexB, vertexC, vertexD

def detect8x8(valid_points, vertexA, vertexB, vertexC, vertexD):
    pointsCTS = []

    pointsCTS = pointsCTS + kClosest(valid_points, vertexA, 24) #perchè 24 dio merda
    pointsCTS = pointsCTS + kClosest(valid_points, vertexB, 24)
    pointsCTS = pointsCTS + kClosest(valid_points, vertexC, 24)
    pointsCTS = pointsCTS + kClosest(valid_points, vertexD, 24)

    fusedPointsCTS = fuse(pointsCTS, 20)

    # for point in fusedPointsCTS:
    #     x = int(point[0])
    #     y = int(point[1])
    #     cv2.circle(image, (x, y), 3, (0, 255, 255), -1)

    fusedPointsCTS = sorted(fusedPointsCTS, key=lambda k: k[1])
    list1, list2 = fusedPointsCTS[:6], fusedPointsCTS[30:]

    list1 = sorted(list1, key=lambda k: k[0])
    list2 = sorted(list2, key=lambda k: k[0])

    vertexA = (int(list1[0][0]), int(list1[0][1]))
    vertexB = (int(list1[5][0]), int(list1[5][1]))
    vertexC = (int(list2[5][0]), int(list2[5][1]))
    vertexD = (int(list2[0][0]), int(list2[0][1]))

    return vertexA, vertexB, vertexC, vertexD

def applySquareAreaFilter(image, contours):
    try:
        valid_points = []

        for pointsArray_list in contours:
            for pointsArray in pointsArray_list:
                for point in pointsArray:
                    valid_points.append([point[0], point[1]])

        centreOfImage = getMiddlePoint(valid_points)

        xCentre = centreOfImage[0]
        yCentre = centreOfImage[1]

        cv2.circle(image, centreOfImage, 5, (0, 0, 255), -1)

        vertexA, vertexB, vertexC, vertexD = detect2x2(centreOfImage, valid_points)

        vertexA, vertexB, vertexC, vertexD = detect4x4(valid_points, vertexA, vertexB, vertexC, vertexD)

        vertexA, vertexB, vertexC, vertexD = detect6x6(valid_points, vertexA, vertexB, vertexC, vertexD)

        vertexA, vertexB, vertexC, vertexD = detect8x8(valid_points, vertexA, vertexB, vertexC, vertexD)

        cv2.circle(image, vertexA, 5, (0, 0, 255), -1)
        cv2.circle(image, vertexB, 5, (0, 0, 255), -1)
        cv2.circle(image, vertexC, 5, (0, 0, 255), -1)
        cv2.circle(image, vertexD, 5, (0, 0, 255), -1)

        src = np.zeros((image.shape[0], image.shape[1], 1), dtype="uint8")

        cv2.line(src, vertexA, vertexB, (255), 3)
        cv2.line(src, vertexB, vertexC, (255), 3)
        cv2.line(src, vertexC, vertexD, (255), 3)
        cv2.line(src, vertexD, vertexA, (255), 3)

        boardContour, _ = cv2.findContours(src, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, boardContour, 0, (122, 255, 255), 2)
        return boardContour

    except Exception as e:
        # print(e)
        return [], 0

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

def isFilterWorking(image, valid_contours, boardContour):
    valid_contours_filtered = []
    try:
        count = 0

        for c in valid_contours:

            try:
                M = cv2.moments(c)

                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                if cv2.pointPolygonTest(boardContour[0], (cX,cY), False) == 1:

                    valid_contours_filtered.append(c)
                    count += 1
            except:
                pass

        if  60 <= count <= 64:
            return True, valid_contours_filtered
        else:
            return False, valid_contours_filtered
    except Exception as e:
        print(e)
        return False, valid_contours_filtered

def upgradeSquareAreaFilter(image, valid_contours):
    valid_points = []
    for pointsArray_list in valid_contours:
        for pointsArray in pointsArray_list:
            for point in pointsArray:
                valid_points.append([point[0], point[1]])


    fused = fuse(valid_points, 20)
    centreOfBoard = getMiddlePoint(fused)

    whiteSquare = np.zeros((image.shape[0], image.shape[1], 1), dtype="uint8")
    boardContours = np.zeros((image.shape[0], image.shape[1], 1), dtype="uint8")

    for i, point in enumerate(fused):
        cv2.circle(image, (int(point[0]), int(point[1])), 5, (51, 204, 51), -1)
        for j, point in enumerate(fused[:-1]):
            cv2.line(whiteSquare, (int(point[0]), int(point[1])), (int(fused[i][0]), int(fused[i][1])), (255, 255, 255),3)

    boardContour, _ = cv2.findContours(whiteSquare, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(image, boardContour, 0, (122, 0, 0), 2)

    return boardContour

cv2.namedWindow(window_name)
cv2.createTrackbar(title_trackbarMin, window_name, min_area, max_area, on_trackbarMin)
cv2.createTrackbar(title_trackbarMax, window_name, min_area, max_area, on_trackbarMax)
filterArea = 0
boardContour = []
isWorking = False
while True:

    isTrue, image = capture.read()

    thresh = computeImage(image)

    contours = computeContours(thresh)

    my_img_1 = np.zeros((image.shape[0], image.shape[1], 1), dtype="uint8")

    valid_contours = detectBoardSquares(my_img_1, contours, boardContour, isWorking)

    isWorking, valid_contours_filtered = isFilterWorking(image, valid_contours, boardContour)
    if(isWorking == False):
        boardContour = applySquareAreaFilter(image, valid_contours)
    else:
        boardContour = upgradeSquareAreaFilter(image, valid_contours_filtered)

    try:
            cv2.drawContours(image, [boardContour], 0, (122, 0, 0), 2)
    except:
        pass

    cv2.imshow('thresh', my_img_1)
    cv2.imshow(window_name, image)

    if cv2.waitKey(20) & 0xFF == ord('d'):
        break


capture.release()
cv2.destroyAllWindows()
