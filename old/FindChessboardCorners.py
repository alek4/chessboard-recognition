import numpy as np
import cv2 as cv

#img = cv.imread('Photos/board.jpg')

#cv.imshow('Board', img)


capture = cv.VideoCapture("Photos/video_board.mp4")
nline = 7
ncol = 7
while True:
    isTrue, frame = capture.read()
    ## termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    ## processing
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    thresh = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=1)
    invert = 255 - opening

    # Find the chessboard corners
    ret, corners = cv.findChessboardCorners(thresh, (nline, ncol), flags=cv.CALIB_CB_ADAPTIVE_THRESH )
    corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    print(corners2)

    # fnl = cv.drawChessboardCorners(frame, (7, 7), corners2, ret)
    # cv.imshow("fnl", fnl)

    for c in corners2:
        x = int(c.ravel()[0])
        y = int(c.ravel()[1])
        cv.circle(frame, (x,y), 5, (255, 0, 0), -1)

    cv.imshow('Thresh', thresh)
    cv.imshow('Frame', frame)

    if(cv.waitKey(20) & 0xFF==ord('d')):
        break




capture.release()
cv.destroyAllWindows()

#cv.waitKey(0) #aspetta un tasto per l'imput

