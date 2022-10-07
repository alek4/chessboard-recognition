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

def getAspectRatio(coords):

    for data in coords:
        sorted(data, key=lambda k: [k[1], k[0]])

    pt1 = coords[0][0]
    pt2 = coords[1][0]
    pt3 = coords[2][0]
    pt4 = coords[3][0]

    width = math.dist(pt1, pt2);
    height = math.dist(pt1, pt3);
    # sides = [math.dist(pt1, pt2), math.dist(pt2, pt3) , math.dist(pt3, pt4), math.dist(pt4, pt1)]
    ar = width / float(height)
    return ar

def digitalizeBoard():
    print("si")


cv2.namedWindow(window_name)
cv2.createTrackbar(title_trackbarMin, window_name, min_area, max_area, on_trackbarMin)
cv2.createTrackbar(title_trackbarMax, window_name, min_area, max_area, on_trackbarMax)



while True:


    isTrue, image = capture.read()
    thresh = computeImage(image)

    contours = computeContours(thresh)

    my_img_1 = np.zeros((image.shape[0], image.shape[1], 1), dtype="uint8")

    valid_contours = []
    for c in contours:
        area = cv2.contourArea(c)

        if area > min_area and area < max_area:

            approx = cv2.approxPolyDP(c, 0.10 * cv2.arcLength(c, True), True) #al posto di 0.009 ho messo 0.10

            if hasFourVertices(approx): #controllo che ci siano almeno 4 angoli
                ar = getAspectRatio(approx)

                valid_contours.append(approx)
                cv2.drawContours(my_img_1, [approx], 0, (255, 255, 255), 2)

                M = cv2.moments(approx)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.circle(my_img_1, (cX, cY), 3, (255, 255, 255), -1)
                    # image = cv2.putText(image, str(round(ar,2)), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0) , 2, cv2.LINE_AA)

    try:
        valid_points = []

        for pointsArray_list in valid_contours:
            for pointsArray in pointsArray_list:
                for point in pointsArray:
                    valid_points.append([point[0], point[1]])
        sumX = 0
        sumY = 0
        for point in valid_points:
            sumX += point[0]
            sumY += point[1]

        medianX = int(sumX / len(valid_points))
        medianY = int(sumY / len(valid_points))

        cv2.circle(my_img_1, (medianX, medianY), 200, (255, 255, 255), 2)

    except:
        pass


    # for i in rnge(0, 8):


    cv2.imshow('thresh', my_img_1)
    cv2.imshow(window_name, image)


    if cv2.waitKey(20) & 0xFF == ord('d'):
        break


capture.release()
cv2.destroyAllWindows()
