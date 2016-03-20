#!/usr/bin/python
# -*- coding: utf-8 -*-

import cv2 as cv
import numpy as np
import argparse

rectUserTopLeft, rectUserBottomRight = (0, 0), (0, 0)
grabMouseMove = False
isAppPaused = False
canUserChangeRect = isAppPaused
nameWindow = 'image'


def userRectangleCallback(event, x, y, flags, param):
    """
    Handle user mouse events. Change the rectangle position.
    """
    global rectUserTopLeft, rectUserBottomRight, grabMouseMove

    if canUserChangeRect is False:
        return

    if event == cv.EVENT_LBUTTONDOWN:
        rectUserTopLeft = rectUserBottomRight = (x, y)
        grabMouseMove = True
    elif event == cv.EVENT_LBUTTONUP:
        grabMouseMove = False

    if event == cv.EVENT_MOUSEMOVE and grabMouseMove:
        rectUserBottomRight = (x, y)


def drawRectangle(frame, aa, bb, color=(0, 255, 0)):
    """
    Draw rectangle borders from aa and bb points (ie topLeft and bottomRight)
    :param frame: frame to draw
    :param aa: top left corner
    :param bb: bottpm right corner
    """
    pts = np.array([[aa[0], aa[1]],
                    [bb[0], aa[1]],
                    [bb[0], bb[1]],
                    [aa[0], bb[1]]], dtype=np.int32)
    cv.polylines(frame, [pts], True, color)


def drawUserRectangle(frame):
    drawRectangle(frame, rectUserTopLeft, rectUserBottomRight, color=(255,0,0))


def drawTracking(frame):
    """
    Draw the tracking result
    :param frame:
    :return:
    """
    pass

def track(frame):
    """
    Process tracking in color space.
    :param frame:
    :return:
    """
    pass

parser = argparse.ArgumentParser()
parser.add_argument('-v', '--video', type=str, help='Video filename to track. Use webcam by default')
args = parser.parse_args()

cv.namedWindow(nameWindow)
cv.setMouseCallback(nameWindow, userRectangleCallback)

capture = cv.VideoCapture(0 if args.video is None else args.video)
if not capture.isOpened():
    raise ValueError('Cannot capture video: wrong file name or no webcam present')

while True:
    isCaptured, frame = capture.read()

    while isAppPaused:
        # handle user rectangle selection
        frameCpy = frame.copy()
        drawUserRectangle(frameCpy)
        cv.imshow(nameWindow, frameCpy)

        if cv.waitKey(20) & 0xFF == ord('p'):
            isAppPaused = canUserChangeRect = False

    track(frame)
    drawTracking(frame)
    cv.imshow(nameWindow, frame)

    keyPressed = cv.waitKey(20) & 0xFF
    if keyPressed == ord('q'):
        break
    if keyPressed == ord('p'):
        isAppPaused = canUserChangeRect = True


capture.release()
cv.destroyAllWindows()

