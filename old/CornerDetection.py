import numpy as np
import cv2 as cv

#img = cv.imread('Photos/board.jpg')

#cv.imshow('Board', img)


capture = cv.VideoCapture(0)
while True:
    isTrue, frame = capture.read()

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    thresh = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=1)
    invert = 255 - opening

    detected_edges = cv.Canny(thresh, 38, 38 * 3, 3)
    mask = detected_edges != 0
    dst = frame * (mask[:, :, None].astype(frame.dtype))

    dst_grayscale = cv.cvtColor(dst, cv.COLOR_BGR2GRAY)

    contours, hierarchies = cv.findContours(dst_grayscale, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # print(f'{len(contours)} contour(s) found')

    minArea, maxArea = 5000, 20000

    valid_cnts = []
    for c in contours:
        area = cv.contourArea(c)
        if area > minArea and area < maxArea:
            valid_cnts.append(c)

            approx = cv.approxPolyDP(c, 0.009 * cv.arcLength(c, True), True)

            cv.drawContours(frame, [approx], 0, (0, 0, 255), 2)

            n = approx.ravel()
            i = 0

            for j in n:
                if i % 2 == 0:
                    x = n[i]
                    y = n[i + 1]

                    # String containing the co-ordinates.
                    string = str(x) + " " + str(y)

                    cv.circle(frame, (x, y), 5, (255, 0, 0), -1)

                i = i + 1

            # draw centers
            M = cv.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv.circle(frame, (cX, cY), 5, (0, 0, 255), -1)

            # print(cX)


    # cv.drawContours(frame, valid_cnts, -1, (0, 0, 255), 2)

    cv.imshow('Video', frame)
    cv.imshow('Video Gray', thresh)
    cv.imshow("test", dst)



    #cv.imshow('Video R', frame_resized)

    if(cv.waitKey(20) & 0xFF==ord('d')):
        break




capture.release()
cv.destroyAllWindows()

#cv.waitKey(0) #aspetta un tasto per l'imput

