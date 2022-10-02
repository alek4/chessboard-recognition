import math

import numpy as np
import cv2

#img = cv.imread('Photos/board.jpg')

#cv.imshow('Board', img)


capture = cv2.VideoCapture("photos/video_board.mp4")

window_name = 'Square Detect'
title_trackbarMin = 'Min:'
title_trackbarMax = 'Max:'
# best is: 3500 - 14000
min_area = 1000
max_area = 20000


def drawWhiteSquare():
    # Find contours and filter using threshold area
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    for c in cnts:
        area = cv2.contourArea(c)
        if area > min_area and area < max_area:
            # x, y, w, h = cv2.boundingRect(c)
            # cv2.rectangle(image, (x, y), (x + w, y + h), (36, 255, 12), 2)

            approx = cv2.approxPolyDP(c, 0.009 * cv2.arcLength(c, True), True)

            cv2.drawContours(image, [approx], 0, (0, 0, 255), 2)

            n = approx.ravel()
            i = 0

            for j in n:
                if i % 2 == 0:
                    x = n[i]
                    y = n[i + 1]

                    # String containing the co-ordinates.
                    string = str(x) + " " + str(y)

                    cv2.circle(image, (x, y), 5, (255, 0, 0), -1)

                i = i + 1


def drawBlackSquare(detected_edges):
    # Find contours and filter using threshold area
    cnts = cv2.findContours(detected_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    for c in cnts:
        area = cv2.contourArea(c)
        if area > min_area and area < max_area:
            # x, y, w, h = cv2.boundingRect(c)
            # cv2.rectangle(image, (x, y), (x + w, y + h), (36, 255, 12), 2)

            approx = cv2.approxPolyDP(c, 0.009 * cv2.arcLength(c, True), True)

            cv2.drawContours(image, [approx], 0, (0, 0, 255), 2)

            n = approx.ravel()
            i = 0

            for j in n:
                if i % 2 == 0:
                    x = n[i]
                    y = n[i + 1]

                    # String containing the co-ordinates.
                    string = str(x) + " " + str(y)

                    cv2.circle(image, (x, y), 5, (255, 0, 0), -1)

                i = i + 1


def on_trackbarMin(val):
    global min_area
    min_area = val


def on_trackbarMax(val):
    global max_area
    max_area = val


cv2.namedWindow(window_name)
cv2.createTrackbar(title_trackbarMin, window_name, min_area, max_area, on_trackbarMin)
cv2.createTrackbar(title_trackbarMax, window_name, min_area, max_area, on_trackbarMax)

while True:
    # Load image, grayscale, median blur, sharpen image
    isTrue, image = capture.read()

    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    image = image[100: 2000, 280: 1620]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    kernel_size = 3
    blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    low_threshold = 80
    high_threshold = 80
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

    dilated = cv2.dilate(edges, (7, 7), iterations=7)

    edgesD = cv2.Canny(dilated, low_threshold, high_threshold)


    # invert = 255 - blur_gray
    # thresh = cv2.threshold(invert, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    # edgesB = cv2.Canny(thresh, low_threshold, high_threshold)



    # drawWhiteSquare()
    drawBlackSquare(dilated)

    cv2.imshow('edges', edges)
    cv2.imshow('edgesD', edgesD)
    cv2.imshow(window_name, image)


    if cv2.waitKey(20) & 0xFF == ord('d'):
        break


capture.release()
cv2.destroyAllWindows()
