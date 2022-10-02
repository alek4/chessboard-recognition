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
val = 0
# def CannyThreshold(val):
#     low_threshold = val
#     img_blur = cv.blur(src_gray, (3,3))
#     detected_edges = cv.Canny(img_blur, low_threshold, low_threshold*ratio, kernel_size)
#     mask = detected_edges != 0
#     dst = frame * (mask[:,:,None].astype(frame.dtype))
#     cv.imshow(window_name, dst)


def on_trackbar(v):
    global val
    val = v


cv.namedWindow(window_name)
cv.createTrackbar(title_trackbar, window_name, 0, max_lowThreshold, on_trackbar)

while True:
    isTrue, frame = capture.read()

    src_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    low_threshold = val
    img_blur = cv.blur(src_gray, (3, 3))
    detected_edges = cv.Canny(img_blur, low_threshold, low_threshold * ratio, kernel_size)
    mask = detected_edges != 0
    dst = frame * (mask[:, :, None].astype(frame.dtype))
    cv.imshow(window_name, dst)

    if cv.waitKey(20) & 0xFF == ord('d'):
        break




capture.release()
cv.destroyAllWindows()

#cv.waitKey(0) #aspetta un tasto per l'imput