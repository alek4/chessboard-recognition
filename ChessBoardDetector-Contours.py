import math
import numpy as np
import cv2

#img = cv.imread('Photos/board.jpg')

#cv.imshow('Board', img)


capture = cv2.VideoCapture("photos/video_board.mp4")


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


def slice_per(source, step):
    return [source[step*i:step*i+step] for i in range(0,math.ceil(len(source)/step))]


def remove_every_nth(lst, n):
    del lst[n-1::n]
    return lst

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

def detectSidesLenght(coords):
    delta = 25
    pt1 = coords[0][0]
    pt2 = coords[1][0]
    pt3 = coords[2][0]
    pt4 = coords[3][0]

    sides = [math.dist(pt1, pt2), math.dist(pt2, pt3) , math.dist(pt3, pt4), math.dist(pt4, pt1)]

    return (math.isclose(sides[0], sides[1], abs_tol = delta)
            and math.isclose(sides[1], sides[2], abs_tol = delta)
            and math.isclose(sides[2], sides[3], abs_tol = delta)
            and math.isclose(sides[3], sides[0], abs_tol = delta))

cv2.namedWindow(window_name)
cv2.createTrackbar(title_trackbarMin, window_name, min_area, max_area, on_trackbarMin)
cv2.createTrackbar(title_trackbarMax, window_name, min_area, max_area, on_trackbarMax)


while True:

    isTrue, image = capture.read()
    thresh = computeImage(image)

    contours = computeContours(thresh)

    for c in contours:
        area = cv2.contourArea(c)

        if area > min_area and area < max_area:

            approx = cv2.approxPolyDP(c, 0.10 * cv2.arcLength(c, True), True) #al posto di 0.009 ho messo 0.10


            if hasFourVertices(approx): #controllo che ci siano almeno 4 angoli
                if detectSidesLenght(approx):
                    cv2.drawContours(image, [approx], 0, (0, 0, 255), 2)

            n = approx.ravel()
            i = 0


    cv2.imshow('thresh', thresh)
    img = image[100: 2000, 280: 1620]
    cv2.imshow(window_name, img)


    if cv2.waitKey(20) & 0xFF == ord('d'):
        break


capture.release()
cv2.destroyAllWindows()
