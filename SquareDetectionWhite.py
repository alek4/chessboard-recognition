import math

import numpy as np
import cv2

#img = cv.imread('Photos/board.jpg')

#cv.imshow('Board', img)


capture = cv2.VideoCapture("photos/video_board.mp4")


window_name = 'Square Detect'
title_trackbarMin = 'Min'
title_trackbarMax = 'Max'
# best is: 3500 - 14000
min_area = 1000
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


cv2.namedWindow(window_name)
cv2.createTrackbar(title_trackbarMin, window_name, min_area, max_area, on_trackbarMin)
cv2.createTrackbar(title_trackbarMax, window_name, min_area, max_area, on_trackbarMax)

while True:

    isTrue, image = capture.read()
    thresh = computeImage(image)
    contours = computeContours(thresh)

    corners = []
    for c in contours:
        area = cv2.contourArea(c)

        if area > min_area and area < max_area:
            # x, y, w, h = cv2.boundingRect(c)
            # cv2.rectangle(image, (x, y), (x + w, y + h), (36, 255, 12), 2)

            approx = cv2.approxPolyDP(c, 0.009 * cv2.arcLength(c, True), True)

            # cv2.drawContours(image, [approx], 0, (0, 0, 255), 2)

            n = approx.ravel()
            i = 0

            for j in n:
                if i % 2 == 0:
                    x = n[i]
                    y = n[i + 1]

                    # String containing the co-ordinates.

                    # cv2.circle(image, (x, y), 5, (255, 0, 0), -1)
                    corners.append((x,y))

                i = i + 1

    if len(corners) > 0:

        corners = sorted(corners, key=lambda k: [k[1]])

        # corners = sorted(corners)

        # for idx, pt in enumerate(corners):
        #     font = cv2.FONT_HERSHEY_SIMPLEX
            # cv2.putText(image, str(idx), pt, font, 1, (255, 255, 255), 1, cv2.LINE_AA)
            # print(pt, idx)

        rows = slice_per(corners, 8)

        # del rows[1::2]
        # rows = remove_every_nth(rows, 4)

        for i, row in enumerate(rows):
            row = sorted(row)
            for j, pt in enumerate(row):
                cv2.circle(image, pt, 5, (255, 0, 0), -1)
                cv2.putText(image, str(j), pt, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)





    # cv2.imshow('sharpen', sharpen)
    # cv2.imshow('close', close)
    cv2.imshow('thresh', thresh)
    img = image[100: 2000, 280: 1620]
    cv2.imshow(window_name, img)

    # cv2.imshow("Invert", invert)


    if cv2.waitKey(20) & 0xFF == ord('d'):
        break


capture.release()
cv2.destroyAllWindows()
