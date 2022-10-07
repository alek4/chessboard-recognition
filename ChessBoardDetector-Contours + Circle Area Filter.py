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

            approx = cv2.approxPolyDP(c, 0.10 * cv2.arcLength(c, True), True) #al posto di 0.009 ho messo 0.10

            if hasFourVertices(approx): #controllo che ci siano almeno 4 angoli
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

def applyCircleAreaFilter(image, contours):
    try:
        valid_points = []

        for pointsArray_list in contours:
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

        cv2.circle(image, (medianX, medianY), 200, (255, 255, 255), 2)

    except:
        pass



cv2.namedWindow(window_name)
cv2.createTrackbar(title_trackbarMin, window_name, min_area, max_area, on_trackbarMin)
cv2.createTrackbar(title_trackbarMax, window_name, min_area, max_area, on_trackbarMax)



while True:


    isTrue, image = capture.read()
    thresh = computeImage(image)

    contours = computeContours(thresh)

    my_img_1 = np.zeros((image.shape[0], image.shape[1], 1), dtype="uint8")

    valid_contours = detectBoardContourMode(my_img_1, contours)

    applyCircleAreaFilter(my_img_1, valid_contours)


    cv2.imshow('thresh', my_img_1)
    cv2.imshow(window_name, image)


    if cv2.waitKey(20) & 0xFF == ord('d'):
        break


capture.release()
cv2.destroyAllWindows()
