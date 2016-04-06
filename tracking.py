import numpy as np
from plot import *
import scipy.ndimage
from draw import *

indexHistogram = np.arange(0, 255)


class ResultTracking:
    def __init__(self, aa=None, bb=None, center=None, densityFunction=None):
        self.aa = aa if np.all(aa) is not None else np.array([0, 0])
        self.bb = bb if np.all(bb) is not None else np.array([0, 0])
        self.center = center if np.all(center) is not None else np.array([0, 0])
        self.densityFunction = densityFunction if np.all(densityFunction) is not None else np.array([])


def deltaKron(a):
    a = np.equal(a, 0).astype(int)
    return a


def k(x):
    d = 2 # dimension
    mask = np.less(x, 1).astype(int)
    a = 0.5 * (3.14**-1) * (d + 2) * (1-x) * mask
    return a

def getNormXY(X, Y):
    a = np.sqrt(X**2 + Y**2)
    return a


def getCenteredCoord(X, normalize=True):
    shapeY_half = int(X.shape[0] / 2)
    shapeX_half = int(X.shape[1] / 2)
    startY = -shapeY_half if (X.shape[0] % 2 == 0) else -shapeY_half - 1
    startX = -shapeX_half if (X.shape[1] % 2 == 0) else -shapeX_half - 1

    matX, matY = np.meshgrid(np.arange(startX, startX + X.shape[1]), np.arange(startY, startY + X.shape[0]))
    return np.array([matY/float(startY if normalize else 1), matX/float(startX if normalize else 1)])

def hat_Qu(X, u):
    XiCentered = getCenteredCoord(X)
    normXi = getNormXY(XiCentered[1], XiCentered[0])
    normWeight = k(normXi**2)

    # plotImageWithColorBar(normWeight, title="normWeight hatQU")
    # plotImageWithColorBar(normXi, title="normXI hatQU")
    # plotImageWithColorBar(XiCentered[0], title="Xi centered [0] hatQU")
    # plotImageWithColorBar(XiCentered[1], title="Xi Centered [1] hatQU")
    # pause()

    density = []
    for i in u:
        dKron = deltaKron(b(X)-i)
        density.append(C(XiCentered, normXi) * np.sum(normWeight * dKron))
    return np.array(density)

def hat_Pu(X, Ycenter, u, h):
    XiCentered = getCenteredCoord(X)
    normXi = getNormXY(XiCentered[0]/h, XiCentered[1]/h)
    normWeight = k(normXi**2)

    # plotImageWithColorBar(normWeight, title="normWeight hatQU")
    # plotImageWithColorBar(normXi, title="normXi hatQU")
    # pause()
    density = []
    for i in u:
        dKron = deltaKron(b(X)-i)
        density.append(Ch(XiCentered, normXi, h) * np.sum(normWeight * dKron))
    return np.array(density)

def Ch(X, normXi, h):
    a = 1.0 / (np.sum(k(normXi**2)))
    return a

def C(X, normXi):
    return 1.0 / np.sum(k(normXi**2))

def b(pixel):
    # BGR
    a = 0.2126*pixel[:, :, 2] + 0.7152*pixel[:, :, 1] + 0.0722*pixel[:, :, 0]
    return a.astype(int)


def extractFromAABB(X, aa, bb):
    return X[aa[0]:bb[0], aa[1]:bb[1], :]

def g(x):
    gradY, gradX = gradient(k(x))
    return -gradY, -gradX

def weight(X, hat_qu, hat_pu, u):
    res = np.zeros(X.shape[0:2])

    for i in u:
        res += deltaKron(b(X) - i) * np.sqrt(hat_qu[i]/(hat_pu[i]+0.001))
    return res
    # return gradient(res)

def colorToCoord(X):
    return np.array([np.arange(0, X.shape[0]), np.arange(0, X.shape[1])])

def gradient(X):
    im = X.astype(float)
    sobelFilter = np.array([-0.5, 0, 0.5])*-1  # times -1 for low to high gradient
    gradient_y = scipy.ndimage.convolve(im, sobelFilter[np.newaxis])
    gradient_x = scipy.ndimage.convolve(im, sobelFilter[np.newaxis].T)
    return gradient_y, gradient_x

def track(frame, previousTracking):
    """
    Process tracking in color space.
    """
    h = 1
    epsilon = 1
    frame = np.asarray(frame)
    maxIterations = 10000
    frameCpy = frame.copy()

    X = extractFromAABB(frame, previousTracking.aa, previousTracking.bb)

    hat_qu = previousTracking.densityFunction

    Y0 = previousTracking.center.copy()
    Y0_AA = previousTracking.aa.copy()
    Y0_BB = previousTracking.bb.copy()

    Y1_AA = previousTracking.aa.copy()
    Y1_BB = previousTracking.bb.copy()
    battaX = []
    battaY = []
    battaZ = []

    for i in range(0, maxIterations):
        Y0_color = extractFromAABB(frame, Y0_AA, Y0_BB)
        hatPU_Y0 = hat_Pu(Y0_color, Y0, indexHistogram, h)

        pY0 = np.sum(np.sqrt(hatPU_Y0 * hat_qu))

        weightX = weight(X, hat_qu, hatPU_Y0, indexHistogram)
        gradWeightY, gradWeightX = gradient(weightX)

        Xcoords = getCenteredCoord(X)

        norm_Y0_minus_X = getNormXY((Xcoords[0])/h, (Xcoords[1])/h)

        gradKernelX, gradKernelY  = g(norm_Y0_minus_X**2)

        wi_g_X = gradWeightX * gradKernelX
        wi_g_Y = gradWeightY * gradKernelY

        Y1_y = np.sum(Xcoords[0]*wi_g_Y) / np.sum(wi_g_Y)
        Y1_x = np.sum(Xcoords[1]*wi_g_X) / np.sum(wi_g_X)

        print("Y1_x %f" % Y1_x)
        print("Y1_y %f" % Y1_y)

        # Mean shift result
        Y1_AA[1] += Y1_x
        Y1_AA[0] += Y1_y

        Y1_BB[1] += Y1_x
        Y1_BB[0] += Y1_y
        Y1 = np.array([Y0[0] + Y1_y, Y0[1] + Y1_x])

        Y1_color = extractFromAABB(frame, Y1_AA, Y1_BB)
        hatPU_Y1 = hat_Pu(Y1_color, Y1, indexHistogram, h)

        pY1 = np.sum(np.sqrt(hatPU_Y1 * hat_qu))
        battaZ.append(pY1)
        battaX.append(Y1[1])
        battaY.append(Y1[0])

        print("pY1 %f " % pY1)
        print("pY0 %f " % pY0)

        while pY1 < pY0:
            Y1 = (Y0 + Y1) * 0.5
            Y1_AA = (Y0_AA + Y1_AA) * 0.5
            Y1_BB = (Y0_BB + Y1_BB) * 0.5

            Y1_color = extractFromAABB(frame, Y1_AA, Y1_BB)
            hatPU_Y1 = hat_Pu(Y1_color, Y1, indexHistogram, h)
            pY1 = np.sum(np.sqrt(hatPU_Y1 * hat_qu))

            battaZ.append(pY1)
            battaX.append(Y1[1])
            battaY.append(Y1[0])

            print("pY1 %f " % pY1)
            print("pY0 %f " % pY0)

            drawRectangle(frameCpy, Y1_AA, Y1_BB, color=(0, 0, 255))
            cv.imshow("res2", frameCpy)

        # If result found i.e distance between Y1 and Y0 less than a threshold, return tracking result
        if np.linalg.norm(Y1 - Y0) < epsilon:
            print("FOUND")
            # plot3D(battaX, battaY, battaZ, "battacha")
            # pause()
            hat_qu_Y1 = hat_Qu(Y1_color, indexHistogram)
            return ResultTracking(aa=Y1_AA, bb=Y1_BB, center=Y1, densityFunction=hat_qu_Y1)

        # Else candidate become the model by mean shift
        Y0 = Y1
        Y0_AA = Y1_AA.copy()
        Y0_BB = Y1_BB.copy()


    # Not found, return the previous tracking result
    # print("Not found")
    # return ResultTracking(aa=tmpAA, bb=tmpBB, center=Y1, densityFunction=hatPU_Y1)
    # return ResultTracking(aa=tmpAA, bb=tmpBB, center=Y1, densityFunction=hatPU_Y1)
    return ResultTracking(aa=Y1_AA, bb=Y1_BB, center=Y1, densityFunction=hatPU_Y1)


class Point:
    def __init__(self, y=None, x=None):
        self.y = y if y is not None else 0
        self.x = x if x is not None else 0
