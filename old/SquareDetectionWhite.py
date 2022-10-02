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
    # blur = cv2.medianBlur(gray, 5)
    # sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    # sharpen = cv2.filter2D(blur, -1, sharpen_kernel)
    #
    # # Threshold and morph close
    # thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    invert = 255 - opening

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

    # cv2.imshow('sharpen', sharpen)
    # cv2.imshow('close', close)
    cv2.imshow('thresh', thresh)
    cv2.imshow(window_name, image)
    # cv2.imshow("Invert", invert)


    if cv2.waitKey(20) & 0xFF == ord('d'):
        break


capture.release()
cv2.destroyAllWindows()
