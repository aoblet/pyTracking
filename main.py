#!/usr/bin/python
# -*- coding: utf-8 -*-

import cv2 as cv
import numpy as np
import argparse
from tracking import *
from draw import *

rectUserTopLeft, rectUserBottomRight = np.array([0, 0]), np.array([0, 0])
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
        rectUserTopLeft = rectUserBottomRight = np.array([y, x])
        grabMouseMove = True
    elif event == cv.EVENT_LBUTTONUP:
        grabMouseMove = False

    if event == cv.EVENT_MOUSEMOVE and grabMouseMove:
        rectUserBottomRight = np.array([y, x])

        # properly update rectangle bounds
        # Y
        if rectUserTopLeft[0] > rectUserBottomRight[0]:
            rectUserBottomRight[0], rectUserTopLeft[0] = rectUserTopLeft[0], rectUserBottomRight[0]

        # X
        if rectUserTopLeft[1] > rectUserBottomRight[1]:
            rectUserBottomRight[1], rectUserTopLeft[1] = rectUserTopLeft[1], rectUserBottomRight[1]


parser = argparse.ArgumentParser()
parser.add_argument('-v', '--video', type=str, help='Video filename to track. Use webcam by default')
args = parser.parse_args()
print args.video
isAppPaused = args.video is not None

cv.namedWindow(nameWindow)
cv.setMouseCallback(nameWindow, userRectangleCallback)

capture = cv.VideoCapture(0 if args.video is None else args.video)
if not capture.isOpened():
    raise ValueError('Cannot capture video: wrong file name or no webcam present')

tracked = None

while True:
    isCaptured, frame = capture.read()
    frameNp = np.asarray(frame)

    if tracked is not None:
        tracked = track(frame, tracked)
        drawTracking(frame, tracked)

    while isAppPaused:
        # handle user rectangle selection
        frameCpy = frame.copy()
        drawUserRectangle(frameCpy, rectUserTopLeft, rectUserBottomRight)
        cv.imshow(nameWindow, frameCpy)

        if cv.waitKey(20) & 0xFF == ord('p'):
            isAppPaused = canUserChangeRect = False
            X = extractFromAABB(np.asarray(frame), rectUserTopLeft, rectUserBottomRight)
            center = (rectUserTopLeft + rectUserBottomRight) * 0.5
            tracked = ResultTracking(rectUserTopLeft, rectUserBottomRight, center, hat_Qu(X, indexHistogram))

    cv.imshow(nameWindow, frame)
    keyPressed = cv.waitKey(20) & 0xFF
    if keyPressed == ord('q'):
        break
    if keyPressed == ord('p'):
        isAppPaused = canUserChangeRect = True


capture.release()
cv.destroyAllWindows()

