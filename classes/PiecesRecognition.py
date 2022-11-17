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

    def __filterImage(self, frame):
        sigma = 0.33
        v = np.median(frame)
        img = cv2.GaussianBlur(frame, (7, 7), 2)  # we use gaussian blur on the image to make it clear.
        lower = int(max(0, (1.0 - sigma) * v))  # we find the lower threshold.
        upper = int(min(255, (1.0 + sigma) * v))  # we find the higher threshold.
        img_edge = cv2.Canny(img, 50, 50)  # we use the canny function to edge canny the image.

        return img_edge

    def __removeShadow(self, img):
        rgb_planes = cv2.split(img)

        result_planes = []
        for plane in rgb_planes:
            dilated_img = cv2.dilate(plane, np.ones((7, 7), np.uint8))
            bg_img = cv2.medianBlur(dilated_img, 21)
            diff_img = 255 - cv2.absdiff(plane, bg_img)
            result_planes.append(diff_img)

        result = cv2.merge(result_planes)

        return result

    def getPieceAtCell(self, cell, frame):
        ot = self.__warpCoords(cell, frame)
        # image = frame[cell.tl[1]:cell.br[1], cell.tl[0]:cell.br[0]]
        cn = self.__filterImage(ot)
        # cv2.imshow("cell", cn)

        return ot

    def calcDiff(self, cell, frame):
        new = self.getPieceAtCell(cell, frame)
        old = cell.img

        height, width, channels = new.shape
        upper_left = (width // 4, height // 4)
        bottom_right = (width * 3 // 4, height * 3 // 4)

        # cv2.rectangle(new, upper_left, bottom_right, (0, 255, 0), thickness=1)
        cropOld = old[upper_left[1]: bottom_right[1] + 1, upper_left[0]: bottom_right[0] + 1]
        cropNew = new[upper_left[1]: bottom_right[1] + 1, upper_left[0]: bottom_right[0] + 1]

        grayNew = cv2.cvtColor(cropNew, cv2.COLOR_BGR2GRAY)
        grayOld = cv2.cvtColor(cropOld, cv2.COLOR_BGR2GRAY)
        (score, diff) = compare_ssim(grayNew, grayOld, full=True)

        cell.img = new
        # print(score)
        # cv2.imshow("ciao", cropOld)
        # cv2.imshow("ciao2", cropNew)

        return math.ceil(score * 100)
