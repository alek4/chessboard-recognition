import numpy as np
import cv2
import imutils
from imutils import perspective
import PySimpleGUI as sg
from classes.Cell import Cell
import random


def computeImage(frame):
    # This function is used to compute the image to help FindChessBoardCorners find the chess board pattern
    # pyrDown and pyrUp are used for downscaling and upscaling the image, this helps a bit in removing noises
    # then we perform a gray color conversion which is required by FindChessBoardCorners
    pyr = cv2.pyrDown(frame, (frame.shape[1] / 2, frame.shape[0] / 2))
    pyr = cv2.pyrUp(pyr, (frame.shape[1], frame.shape[0]))
    gray = cv2.cvtColor(pyr, cv2.COLOR_BGR2GRAY)

    return gray


def distance(x1, y1, x2, y2):
    # A simple euclidean distance function, given 2 points's coordinates it returns the distance between them
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** (1 / 2)


def kClosest(points, target, K):
    # This function, given a list of points, returns the k points closest to a target point.
    pts = []
    n = len(points)
    d = []

    for i in range(n):
        d.append({
            "first": distance(points[i][0], points[i][1], target[0], target[1]),
            "second": i
        })

    d = sorted(d, key=lambda l: l["first"])

    for i in range(K):
        pt = []
        pt.append(points[d[i]["second"]][0])
        pt.append(points[d[i]["second"]][1])
        pts.append(pt)

    return pts


def getInnerVertices(corners):
    # This function aims to find the inner vertices of the board from the list of corners found by
    # FindChessBoardCorners. By "inner vertices" we mean the vertices of the 6x6 square inscribed in a 8x8 chess board
    # https://imgbox.com/nBAbWjw3

    # We start by connecting each single point of the list to the others with a white line.
    # By drawing all of these lines, we'll have a white rectangle as a result.
    # Then we ask to findContours function to detect the white rectangle's contour

    blank = np.zeros((frame.shape[0], frame.shape[1], 1), dtype="uint8")  # blank single channel image
    for i, point in enumerate(corners):
        for j, point in enumerate(corners[:-1]):
            cv2.line(blank, (int(point.ravel()[0]), int(point.ravel()[1])),
                     (int(corners[i].ravel()[0]), int(corners[i].ravel()[1])),
                     (255, 255, 255), 3)
    contours, hierarchy = cv2.findContours(blank, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Once we have the contour we can approximate it in order to find his 4 vertices:
    #  tl (top left), tr (top right), br (bottom right) and bl (bottom left)

    vertices = []
    for c in contours:
        approx = cv2.approxPolyDP(c, 0.009 * cv2.arcLength(c, True), True)
        n = approx.ravel()
        i = 0
        for _ in n:
            if (i % 2 == 0):
                x = n[i]
                y = n[i + 1]

                vertices.append([x, y])

            i = i + 1
    # Thanks to imutils's order_points function, we have our 4 innerVertices sorted in a clockwise order.
        (tl, tr, br, bl) = imutils.perspective.order_points(np.asarray(vertices))

    # Due to integer approximations we don't have the actual corners found by FindChessBoardCorners
    # So we have to find the closest actual points to our approximated vertices
    # this can be accomplished by kClosest function
    tl = kClosest(corners.tolist(), tl, 1)[0]
    tr = kClosest(corners.tolist(), tr, 1)[0]
    br = kClosest(corners.tolist(), br, 1)[0]
    bl = kClosest(corners.tolist(), bl, 1)[0]

    return tl, tr, br, bl



def create7x7CornersMatrix(array):
    # This function converts a numpy array of shape (49,2) into a 7x7 numpy matrix of tuples
    array = array.reshape(7, 7, 2)
    cornersMatrix_list = array.tolist()

    cornersMatrix = np.empty((7, 7), object)

    for row in range(0, 7):
        for column in range(0, 7):
            x = cornersMatrix_list[row][column][0]
            y = cornersMatrix_list[row][column][1]
            cornersMatrix[row][column] = (x, y)

    return cornersMatrix


def getChildVertices(x, y):
    # This function returns the fixed coordinates of the inner's child vertices. This is required cause of the fact that
    # FindChessBoardCorners sometimes sorts points by column or by row so the correct position in the matrix
    # depends on that
    # https://imgbox.com/LcGVFRLq
    match (x, y):
        case (6, 6):
            return 5, 5
        case (6, 0):
            return 5, 1
        case (0, 6):
            return 1, 5
        case (0, 0):
            return 1, 1

    return -1, -1


def getOuterVertices(cornersMatrix, tl, tr, br, bl):

    # Once we found our Inner and Child vertices, we are able to calculate their distance and replicate it in order to
    # find the related Parent vertices.
    # https://imgbox.com/PU4OPgyT

    #https://imgbox.com/PU4OPgyT
    parentTL = parentTR = parentBR = parentBL = -1

    for row in range(0, 7):
        for column in range(0, 7):
            x = cornersMatrix[row][column][0]
            y = cornersMatrix[row][column][1]
            if x == tl[0] and y == tl[1]:
                xChild, yChild = getChildVertices(row, column)
                parentTL = ((2 * x - cornersMatrix[xChild][yChild][0]), (2 * y - cornersMatrix[xChild][yChild][1]))
            elif x == tr[0] and y == tr[1]:
                xChild, yChild = getChildVertices(row, column)
                parentTR = ((2 * x - cornersMatrix[xChild][yChild][0]), (2 * y - cornersMatrix[xChild][yChild][1]))
            elif x == br[0] and y == br[1]:
                xChild, yChild = getChildVertices(row, column)
                parentBR = ((2 * x - cornersMatrix[xChild][yChild][0]), (2 * y - cornersMatrix[xChild][yChild][1]))
            elif x == bl[0] and y == bl[1]:
                xChild, yChild = getChildVertices(row, column)
                parentBL = ((2 * x - cornersMatrix[xChild][yChild][0]), (2 * y - cornersMatrix[xChild][yChild][1]))

    return parentTL, parentTR, parentBR, parentBL

def extend_line(p1, p2, distance=10000):
    # This function extends a line for a given distance
    diff = np.arctan2(p1[1] - p2[1], p1[0] - p2[0])
    p3_x = int(p1[0] + distance * np.cos(diff))
    p3_y = int(p1[1] + distance * np.sin(diff))
    p4_x = int(p1[0] - distance * np.cos(diff))
    p4_y = int(p1[1] - distance * np.sin(diff))
    return ((p3_x, p3_y), (p4_x, p4_y))


def getLines(frame, cornersMatrix, tl, tr, br, bl, innerTL):
    # Thanks to this function we are able to determine the vertical and horizontal lines of the chess board
    # It is necessary to divide them in order to find the intersections
    # innerTL is required as we need to know how FindChessBoardCorners sorted the found corners (by column or by row)
    horizonalLines = []
    verticalLines = []
    innerTL = (innerTL[0], innerTL[1])

    # here we divide the chess board's borders based on innerTL indexes in the cornersMatrix -> This solution sucks



    if cornersMatrix[6][0] == innerTL:
        verticalLines.append([tl, tr])
        horizonalLines.append([tl, bl])
        verticalLines.append([bl, br])
        horizonalLines.append([br, tr])
    else:
        horizonalLines.append([tl, tr])
        verticalLines.append([tl, bl])
        horizonalLines.append([bl, br])
        verticalLines.append([br, tr])


    # then we gather the other lines and we divide them
    for i in range(0, 7):
        pt1, pt2 = extend_line(cornersMatrix[i][0], cornersMatrix[i][6])
        pt3, pt4 = extend_line(cornersMatrix[0][i], cornersMatrix[6][i])
        cv2.line(frame, pt1, pt2, (255, 255, 255), 2)
        cv2.line(frame, pt3, pt4, (255, 255, 255), 2)

        horizonalLines.append([pt1, pt2])
        verticalLines.append([pt3, pt4])

    cv2.line(frame, tl, tr, (255, 255, 255), 2)
    cv2.line(frame, tl, bl, (255, 255, 255), 2)
    cv2.line(frame, bl, br, (255, 255, 255), 2)
    cv2.line(frame, br, tr, (255, 255, 255), 2)

    cv2.imshow('Frame2', frame)

    return horizonalLines, verticalLines


def line_intersection(line1, line2):
    # This function returns whether 2 lines intersect or not
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div

    return x, y


def warpImage(frame, tl, tr, br, bl):
    width, height = 400, 400

    srcPts = np.float32([[tl[0], tl[1]], [tr[0], tr[1]], [bl[0], bl[1]], [br[0], br[1]]])
    dstPts = np.float32([[0,0], [width, 0], [0, height], [width, height]])

    matrix = cv2.getPerspectiveTransform(srcPts, dstPts)
    output = cv2.warpPerspective(frame, matrix, (width, height))
    output = cv2.rotate(output, cv2.ROTATE_90_CLOCKWISE)
    output = cv2.flip(output, 1)

    return output
def topLeftToBottomRightSorter(points):

    sorted = []
    remaining = points

    for i in range(9):
        tl = min(remaining, key=lambda t: t[0] + t[1])
        tr = max(remaining, key=lambda t: t[0] - t[1])
        for point in remaining:
            d = np.cross(tr - tl, point - tl) / np.linalg.norm(tr - tl)
            if(d < 10):
                sorted.append(point.tolist())
                tmp = remaining.tolist()
                tmp.remove(point.tolist())
                remaining = np.array(tmp)

    print(len(sorted))
    return sorted

def saveBoardStatus(points):
    pointsMatrix = np.empty((9, 9), object)
    cellMatrix = []

    pointsSorted = topLeftToBottomRightSorter(points)

    for idx, p in enumerate(pointsSorted):
        row = idx // 9
        col = idx % 9
        pointsMatrix[row][col] = p

    for idx, row in enumerate(pointsMatrix):
        row = sorted(row, key=lambda k: k[0])
        pointsMatrix[idx] = row

    for row in pointsMatrix:
        for p in row:
            print(int(p[0]),",", int(p[1]), " " , end='')
        print()

    j = 0
    while j + 1 <= 8:
        i = 0
        row = []
        while i + 1 <= 8:
            coords = (chr(i + 65), j + 1)
            cell = Cell(pointsMatrix[i][j], pointsMatrix[i+1][j], pointsMatrix[i+1][j+1], pointsMatrix[i][j+1], coords)
            row.append(cell)
            i += 1
        j += 1
        cellMatrix.append(row)


    return cellMatrix



capture = cv2.VideoCapture(0)
nline = 7
ncol = 7
font = cv2.FONT_HERSHEY_COMPLEX
boardFound = False
board = []

sg.theme('Black')

# define the window layout
layout = [[sg.Text('Board detection', size=(40, 1), justification='center', font='Helvetica 20')],
          [sg.Image(filename='', key='image')],
           [sg.Button('Capture board', size=(13, 1), font='Any 14'),
            sg.Button('Exit', size=(10, 1), font='Helvetica 14'),
            sg.Button('Flip Coords', size=(13, 1), font='Helvetica 14', visible=boardFound),]]

# create the window and show it without the plot
window = sg.Window('Chess board detection system',
                   layout, location=(800, 400))



while True:
    isTrue, frame = capture.read()

    if not boardFound:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        imgFiltered = computeImage(frame)

        ret, corners = cv2.findChessboardCorners(imgFiltered, (nline, ncol), flags=cv2.CALIB_CB_ADAPTIVE_THRESH)

        if ret:

            corners = cv2.cornerSubPix(imgFiltered, corners, (11, 11), (-1, -1), criteria)
            corners_int = corners.astype(int)
            corners_final = corners_int.reshape(49, 2)

            try:
                (tl, tr, br, bl) = getInnerVertices(corners_final)
                innerTL = tl

                cornersMatrix = create7x7CornersMatrix(corners_final)

                (tl, tr, br, bl) = getOuterVertices(cornersMatrix, tl, tr, br, bl)
                blank = np.zeros((frame.shape[0], frame.shape[1], 1), dtype="uint8")
                horizontalLines, verticalLines = getLines(blank, cornersMatrix, tl, tr, br, bl, innerTL)

                intersections = []
                print(len(horizontalLines), len(verticalLines))
                for hLines in horizontalLines:
                    for vLines in verticalLines:
                        try:
                            intersections.append(line_intersection(hLines, vLines))
                        except:
                            pass
                intersections = np.array(intersections)

                warped = warpImage(frame, tl, tr, br, bl)

                for point in intersections.astype(int):
                    cv2.circle(frame, (point[0], point[1]), 5, (255, 0, 0), -1)

                    # for row in range(0, 7):
                    #     for column in range(0, 7):
                    #         x = cornersMatrix[row][column][0]
                    #         y = cornersMatrix[row][column][1]
                    #         cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)
                    #         cv2.putText(frame, "(" + str(row) + "," + str(column) + ")",
                    #                     (cornersMatrix[row][column][0], cornersMatrix[row][column][1]), font, 0.5, (0, 0, 0), 2,
                    #                     cv2.LINE_AA)
                for point in corners_final:
                    cv2.circle(frame, (point[0], point[1]), 7, (0, 255, 0))
            except Exception as e:
                print(e)
    else:
        for row in board:
            for i, cell in enumerate(row):
                cv2.putText(frame, str(cell.coords[0]) + str(cell.coords[1]), cell.center, font, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
                cv2.line(frame, cell.tl, cell.tr, (255, 0, 255), 2)
                cv2.line(frame, cell.tl, cell.bl, (255, 0, 255), 2)
                cv2.line(frame, cell.bl, cell.br, (255, 0, 255), 2)
                cv2.line(frame, cell.br, cell.tr, (255, 0, 255), 2)

        warped = warpImage(frame, board[0][0].tl, board[0][7].tr, board[7][7].br, board[7][0].bl)

    try:
        cv2.imshow('Warped', warped)
    except:
        pass

    event, values = window.read(timeout=20)
    imgbytes = cv2.imencode('.png', frame)[1].tobytes()  # ditto
    window['image'].update(data=imgbytes)

    if event == 'Exit' or event == sg.WIN_CLOSED:
        break

    if event == 'Capture board':
        board = saveBoardStatus(intersections)
        window['Flip Coords'].update(visible=True)
        boardFound = True

    if event == 'Flip Coords':
        for row in board:
            for cell in row:
                letter = chr(((72 + 1) - ord(cell.coords[0])) + 64)
                num = (8 + 1) - cell.coords[1]
                cell.coords = (letter, num)

    if (cv2.waitKey(20) & 0xFF == ord('d')):
        break

capture.release()
cv2.destroyAllWindows()

