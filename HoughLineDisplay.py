import math

import numpy as np
import cv2 as cv

#img = cv.imread('Photos/board.jpg')

#cv.imshow('Board', img)


capture = cv.VideoCapture("photos/video_board.mp4")

max_lowThreshold = 100
window_name = 'Edge Map'
title_trackbar = 'Min Threshold:'
ratio = 3
kernel_size = 3
val = 40

while True:
    isTrue, frame = capture.read()

    src_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    low_threshold = val
    img_blur = cv.blur(src_gray, (3, 3))
    detected_edges = cv.Canny(img_blur, low_threshold, low_threshold * ratio, kernel_size)
    mask = detected_edges != 0
    dst = frame * (mask[:, :, None].astype(frame.dtype))

    dst_grayscale = cv.cvtColor(dst, cv.COLOR_BGR2GRAY)

    cdst = cv.cvtColor(dst_grayscale, cv.COLOR_GRAY2BGR)
    cdstP = np.copy(cdst)

    lines = cv.HoughLines(dst_grayscale, 1, np.pi / 180, 150, None, 0, 0)

    print(len(lines))

    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
            cv.line(cdst, pt1, pt2, (0, 0, 255), 3, cv.LINE_AA)

    linesP = cv.HoughLinesP(dst_grayscale, 1, np.pi / 180, 50, None, 50, 10)

    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv.LINE_AA)

    cv.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)
    cv.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)

    if cv.waitKey(20) & 0xFF == ord('d'):
        break




capture.release()
cv.destroyAllWindows()

#cv.waitKey(0) #aspetta un tasto per l'imput

