import numpy as np
import cv2 as cv


def drawRectangle(frame, aa, bb, color=(0, 255, 0)):
    """
    Draw rectangle borders from aa and bb points (ie topLeft and bottomRight)
    :param frame: frame to draw
    :param aa: top left corner
    :param bb: bottpm right corner
    """
    pts = np.array([[aa[1], aa[0]],
                    [bb[1], aa[0]],
                    [bb[1], bb[0]],
                    [aa[1], bb[0]]], dtype=np.int32)
    cv.polylines(frame, [pts], True, color)


def drawUserRectangle(frame, topLeft, bottomRight):
    drawRectangle(frame, topLeft, bottomRight, color=(255, 0, 0))


def drawTracking(frame, resultTracking):
    drawRectangle(frame, resultTracking.aa, resultTracking.bb, color=(0, 255, 0))