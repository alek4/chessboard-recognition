import numpy as np
import cv2
import PySimpleGUI as sg
from classes.Cell import Cell
from classes.ChessBoardRecognition import ChessBoardRecognition
from classes.PiecesRecognition import PieceRecognition
import imutils
import math

def saveBoardStatus(points):
    pointsMatrix = np.empty((9, 9), object)
    cellMatrix = []

    pointsSorted = cr.topLeftToBottomRightSorter(points)

    for idx, p in enumerate(pointsSorted):
        row = idx // 9
        col = idx % 9
        pointsMatrix[row][col] = p

    for idx, row in enumerate(pointsMatrix):
        row = sorted(row, key=lambda k: k[0])
        pointsMatrix[idx] = row

    # for row in pointsMatrix:
    #     for p in row:
    #         # print(int(p[0]),",", int(p[1]), " " , end='')
    #     print()

    j = 0
    color = 0
    while j + 1 <= 8:
        i = 0
        row = []
        while i + 1 <= 8:
            coords = (chr(i + 65), j + 1)
            cell = Cell(pointsMatrix[i][j], pointsMatrix[i+1][j], pointsMatrix[i+1][j+1], pointsMatrix[i][j+1], coords, color)
            color = 1 - color
            row.append(cell)
            i += 1
        j += 1
        cellMatrix.append(row)
        color = 1 - color


    return cellMatrix


cr = ChessBoardRecognition()
pr = PieceRecognition()
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
            sg.Button('Flip Coords', size=(13, 1), font='Helvetica 14', visible=boardFound),
            sg.Button('Next Move', size=(13, 1), font='Helvetica 14', visible=boardFound),]]

# create the window and show it without the plot
window = sg.Window('Chess board detection system',
                   layout, location=(800, 400))



while True:
    isTrue, frame = capture.read()

    if not boardFound:
        intersections = cr.detect(frame, ncol, nline)

        for point in intersections.astype(int):
            cv2.circle(frame, (point[0], point[1]), 5, (255, 0, 0), -1)

    else:
        # for row in board:
        #     for i, cell in enumerate(row):
                # cv2.putText(frame, str(cell.coords[0]) + str(cell.coords[1]), cell.center, font, 0.2, (255, 0, 0), 1, cv2.LINE_AA)
                # cv2.putText(frame, str(cell.color), cell.center, font, 0.2, (255, 0, 0), 1, cv2.LINE_AA)
                # cv2.line(frame, cell.tl, cell.tr, (255, 0, 255), 2)
                # cv2.line(frame, cell.tl, cell.bl, (255, 0, 255), 2)
                # cv2.line(frame, cell.bl, cell.br, (255, 0, 255), 2)
                # cv2.line(frame, cell.br, cell.tr, (255, 0, 255), 2)

        warped = cr.warpImage(frame, board[0][0].tl, board[0][7].tr, board[7][7].br, board[7][0].bl)

        pr.getPieceAtCell(board[7][7], frame)

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
        window['Next Move'].update(visible=True)
        boardFound = True


        for row in board:
            for cell in row:
                cell.img = pr.getPieceAtCell(cell, frame)

    if event == 'Next Move':
        # for row in board:
        #     for cell in row:
        #         score = pr.calcDiff(cell, frame)
        #
        #         cv2.putText(frame, str(score), cell.center, font, 0.3, (0, 0, 255), 1, cv2.LINE_AA)
        #         cv2.imshow("scores", frame)
        score = pr.calcDiff(board[7][6], frame)
        # cv2.imshow("diff", diff)

        # print("SSIM: {}".format(score))

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

