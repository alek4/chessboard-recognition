import numpy as np
import cv2
from PIL import Image
from skimage.metrics import structural_similarity as compare_ssim
import math


class PieceRecognition:

    def __warpCoords(self, cell, frame):
        width, height = 200, 200

        srcPts = np.float32([[cell.tl[0], cell.tl[1]], [cell.tr[0], cell.tr[1]], [cell.bl[0], cell.bl[1]], [cell.br[0], cell.br[1]]])
        dstPts = np.float32([[0, 0], [width, 0], [0, height], [width, height]])

        matrix = cv2.getPerspectiveTransform(srcPts, dstPts)
        output = cv2.warpPerspective(frame, matrix, (width, height))
        output = cv2.rotate(output, cv2.ROTATE_90_CLOCKWISE)
        output = cv2.flip(output, 1)

        return output

    def getPieceAtCell(self, cell, frame):

        image = frame[cell.wtl[1]:cell.wbr[1], cell.wtl[0]:cell.wbr[0]]

        return image


    def changesDetection(self, old, frame, board):
        oldBlur = cv2.GaussianBlur(old, (5, 5), 0)
        newBlur = cv2.GaussianBlur(frame, (5, 5), 0)

        grayOld = cv2.cvtColor(oldBlur, cv2.COLOR_BGR2GRAY)
        grayNew = cv2.cvtColor(newBlur, cv2.COLOR_BGR2GRAY)

        (score, diff) = compare_ssim(grayOld, grayNew, full=True)

        threshold = 0.65
        diff = (diff * 255).astype("uint8")
        thresh = cv2.threshold(diff, threshold * 255, 255, cv2.THRESH_BINARY_INV)[1]

        mask = np.copy(thresh)

        # Esegue il flood fill dal punto (0, 0)
        cv2.floodFill(mask, None, (399, 399), 255)

        # Inverti la maschera per ottenere i buchi
        mask = cv2.bitwise_not(mask)

        # Unisci l'immagine threshold originale con la maschera dei buchi riempiti
        thresh = cv2.bitwise_or(thresh, mask)

        contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]

        before = old.copy()
        after = frame.copy()

        boundingBoxes = []


        for c in contours:
            area = cv2.contourArea(c)
            if area > 40:
                x, y, w, h = cv2.boundingRect(c)
                boundingBoxes.append(cv2.boundingRect(c))
                cv2.rectangle(before, (x, y), (x + w, y + h), (36, 255, 12), 2)
                cv2.rectangle(after, (x, y), (x + w, y + h), (36, 255, 12), 2)

        cv2.imshow('diff', diff)


        frame2 = frame.copy()

        for bb in boundingBoxes:
            bottomHalfBB = (bb[0], bb[1] + bb[3] // 2, bb[2], bb[3] // 2)
            cv2.rectangle(frame2, (bottomHalfBB[0], bottomHalfBB[1]),
                          (bottomHalfBB[0] + bottomHalfBB[2], bottomHalfBB[1] + bottomHalfBB[3]), (36, 255, 12), 2)

            centroid = (bottomHalfBB[2] // 2 + bottomHalfBB[0], bottomHalfBB[3] // 2 + bottomHalfBB[1])

            cv2.circle(frame2, centroid, 3, (255, 0, 0), -1)


        sorted_center_cluster_cells, frameDebug = self.cellsClustering(boundingBoxes, frame, board, thresh)

        cell1, cell2 = self.cellsFiltering(sorted_center_cluster_cells, board, frame, frameDebug)

        try:
            cv2.rectangle(frame2, cell1.wtl, cell1.wbr, (0, 0, 255), 2)
            cv2.rectangle(frame2, cell2.wtl, cell2.wbr, (0, 0, 255), 2)
        except:
            print("Nessun cambiamento rilevato")

        cv2.imshow("bbWarped", frame2)


    def deep_index(self, lst, w):
        return [(i, sub.index(w)) for (i, sub) in enumerate(lst) if w in sub]

    def cellsFiltering(self, cells, board, frame, frameDebug):


        chosenCells = [[None, 100], [None, 100]]


        for cell in cells:
            old = cv2.cvtColor(cell.img, cv2.COLOR_BGR2GRAY)
            new = cv2.cvtColor(self.getPieceAtCell(cell, frame), cv2.COLOR_BGR2GRAY)
            (cellScore, diff) = compare_ssim(old, new, full=True)
            cell.score = cellScore

            maxCellScoreIdx = chosenCells.index(max(chosenCells, key=lambda k: k[1]))

            if cell.score < chosenCells[maxCellScoreIdx][1]:
                deepIndex = self.deep_index(board, cell)[0]
                if deepIndex[1] < 7:
                    underCell = board[deepIndex[0]][deepIndex[1] + 1]
                    if underCell.score is not None:
                        if cell.score < underCell.score:
                            chosenCells[maxCellScoreIdx] = [cell, cell.score]
                    else:
                        chosenCells[maxCellScoreIdx] = [cell, cell.score]
                else:
                    chosenCells[maxCellScoreIdx] = [cell, cell.score]

            # for c in chosenCells:
                # if c[1] > cell.score:
                #
                #     deepIndex = self.deep_index(board, cell)[0]
                #     if deepIndex[1] < 7:
                #         underCell = board[deepIndex[0]][deepIndex[1]+1]
                #         if underCell.score == None:
                #             c[1] = cell.score
                #             c[0] = cell
                #
                #     break


            font = cv2.FONT_HERSHEY_COMPLEX
            cv2.putText(frameDebug, str(math.ceil(cellScore*100)), cell.calculateWCentroid(), font, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
            cv2.imshow("bbCircles", frameDebug)

        for cell in cells:
            cell.score = None

        return chosenCells[0][0], chosenCells[1][0]


    def cellsClustering(self, boundingBoxes, frame, board, thresh):
        def area(rect):
            return rect[2] * rect[3]

        boundingBoxes.sort(key=area)
        boundingBoxes.reverse()

        cellSide = frame.shape[1] // 8
        radius = cellSide + cellSide * .75
        centroids = []
        for bb in boundingBoxes:
            bottomHalfBB = (bb[0], bb[1] + bb[3] // 2, bb[2], bb[3] // 2)
            centroids.append((bottomHalfBB[2] // 2 + bottomHalfBB[0], bottomHalfBB[3] // 2 + bottomHalfBB[1]))
        frame2 = frame.copy()

        # cluster di bounding boxes
        clustersCentroids = []
        if len(centroids) != 0:

            while len(centroids) != 0:

                maxBB = centroids[0]
                centroids.remove(maxBB)
                toRemove = []

                for centroid in centroids:
                    dist = math.dist(maxBB, centroid)
                    if dist <= radius: # punto dentro al cerchio
                        toRemove.append(centroid)

                for centroid in toRemove:
                    centroids.remove(centroid)

                clustersCentroids.append(maxBB)

        for cc in clustersCentroids:
            cv2.circle(frame2, cc, int(radius), (255, 0, 0))
            cv2.circle(frame2, cc, 3, (255, 0, 0), -1)

        clusterCells = []
        # celle selezionate dai clusters
        for cc in clustersCentroids:
            for row in board:
                for cell in row:
                    center = cell.calculateWCentroid()
                    dist = math.dist(cc, center)
                    if dist <= radius:  # punto dentro al cerchio
                        cellThresh = thresh[cell.wtl[1]:cell.wbr[1], cell.wtl[0]:cell.wbr[0]]
                        result = np.all(np.isin(cellThresh, [0]))

                        print(result)

                        if not result:
                            cv2.rectangle(frame2, cell.wtl, cell.wbr, (36, 255, 12), 2)
                            # cv2.rectangle(thresh, cell.wtl, cell.wbr, (36, 255, 12), 2)
                            clusterCells.append(cell)

        sorted_center_cluster_cells = sorted(clusterCells, key=lambda x: x.calculateWCentroid()[1], reverse=True)

        cv2.imshow("bbCircles", frame2)
        cv2.imshow('thresh', thresh)

        return sorted_center_cluster_cells, frame2

    def getCenterOfImage(self, old, new):
        height, width, channels = new.shape
        upper_left = (width // 4, height // 4)
        bottom_right = (width * 3 // 4, height * 3 // 4)

        # cv2.rectangle(new, upper_left, bottom_right, (0, 255, 0), thickness=1)
        cropOld = old[upper_left[1]: bottom_right[1] + 1, upper_left[0]: bottom_right[0] + 1]
        cropNew = new[upper_left[1]: bottom_right[1] + 1, upper_left[0]: bottom_right[0] + 1]

        grayNew = cv2.cvtColor(cropNew, cv2.COLOR_BGR2GRAY)
        grayOld = cv2.cvtColor(cropOld, cv2.COLOR_BGR2GRAY)



        return grayOld, grayNew



