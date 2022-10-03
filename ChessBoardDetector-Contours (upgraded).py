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
max_area = 50000

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
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

    se = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
    bg = cv2.morphologyEx(blur_gray, cv2.MORPH_DILATE, se)
    out_binary = cv2.threshold(bg, 0, 255, cv2.THRESH_OTSU)[1]
    cv2.imshow('bg', bg)
    low_threshold = 50
    high_threshold = 50
    edges = cv2.Canny(out_binary, low_threshold, high_threshold)

    kernel = np.ones((5, 5), np.uint8)
    dilation = cv2.dilate(edges, kernel, iterations=1)
    return dilation


def computeContours(frame):
    cnts = cv2.findContours(frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    return cnts


def hasFourVertices(coords):
    if len(coords) != 4:
        return False
    else:
        return True


cv2.namedWindow(window_name)
cv2.createTrackbar(title_trackbarMin, window_name, min_area, max_area, on_trackbarMin)
cv2.createTrackbar(title_trackbarMax, window_name, min_area, max_area, on_trackbarMax)


while True:

    isTrue, image = capture.read()
    edges = computeImage(image)

    contours = computeContours(edges)

    for c in contours:
        area = cv2.contourArea(c)

        if min_area < area < max_area:

            approx = cv2.approxPolyDP(c, 0.10 * cv2.arcLength(c, True), True)  # al posto di 0.009 ho messo 0.10

            if hasFourVertices(approx):  # controllo che ci siano almeno 4 angoli
                M = cv2.moments(approx)
                try:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    cv2.circle(image, (cX, cY), 7, (255, 255, 255), -1)
                except:
                    print("pisneo")
                    pass

                cv2.drawContours(image, [approx], 0, (0, 0, 255), 2)

            n = approx.ravel()
            i = 0

    cv2.imshow('edges', edges)
    # img = image[100: 2000, 280: 1620]
    # cv2.imshow("line_image", line_image)
    cv2.imshow(window_name, image)

    if cv2.waitKey(20) & 0xFF == ord('d'):
        break


capture.release()
cv2.destroyAllWindows()
