import numpy as np
from plot import *
import scipy.ndimage
from draw import *

# RGB to luminance
indexesHistogram = np.arange(0, 256)


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
    # Epanechnikov kernel
    d = 2  # dimension 2
    mask = np.less(x, 1).astype(int)
    a = 0.5 * (3.14**-1) * (d + 2) * (1-x) * mask
    # a = 3.0/4.0 * (1-x) * mask
    # a = ((2*3.14)**-0.5) * np.exp(-0.5*x)
    # plotImageWithColorBar(a)
    # pause()
    return a


def getNormXY(X, Y):
    """
    :param X: Indexes in X
    :param Y: Indexes in Y
    :return: Euclidean norm of X and Y
    """
    return np.sqrt(X**2 + Y**2)


def g(x):
    gradY, gradX = gradient(k(x))
    # plotImageWithColorBar(-gradY, title='gradX')
    # plotImageWithColorBar(-gradX, title='gradY')
    # pause()
    return -gradY, -gradX


def getCenteredCoord(X, normalize=True):
    """
    :return: Y and X indexes centered at 0
    """
    shapeY_half = int(X.shape[0] / 2)
    shapeX_half = int(X.shape[1] / 2)
    startY = -shapeY_half if (X.shape[0] % 2 == 0) else -shapeY_half - 1
    startX = -shapeX_half if (X.shape[1] % 2 == 0) else -shapeX_half - 1

    matX, matY = np.meshgrid(np.arange(startX, startX + X.shape[1]), np.arange(startY, startY + X.shape[0]))
    normFactY = float(shapeY_half) if normalize else 1.0
    normFactX = float(shapeX_half) if normalize else 1.0

    return np.array([matY/normFactY, matX/normFactX])


def hat_Qu(X_gray, u):
    XiCentered = getCenteredCoord(X_gray)
    normXi = getNormXY(XiCentered[1], XiCentered[0])
    normWeight = k(normXi**2)
    C = 1.0/np.sum(normWeight)
    binHisto = binHistoLuminance(X_gray)

    density = []
    for i in u:
        dKron = deltaKron(b(X_gray)-i)
        density.append(C * np.sum(normWeight * dKron))
    return np.array(density)


def hat_Pu(X_gray, Ycenter, u, h):
    return hat_Qu(X_gray, u)
    # XiCentered = getCenteredCoord(X)
    # normXi = getNormXY(XiCentered[0]/h, XiCentered[1]/h)
    # normWeight = k(normXi**2)
    # Ch = 1.0/np.sum(normWeight)
    #
    # density = []
    # for i in u:
    #     dKron = deltaKron(b(X)-i)
    #     density.append(Ch * np.sum(normWeight * dKron))
    # return np.array(density)


def binHistoLuminance(X_gray):
    """
    :return: bin histogram of rgb luminance
    """
    gray_image_histo = X_gray.ravel()
    hist, bins = np.histogram(gray_image_histo, bins=255, range=(0, 255))
    return hist.astype(int)


def b(X_gray):
    return np.floor((X_gray/255.0*indexesHistogram.shape[0]))


def extractFromAABB(X, aa, bb, gray=False):
    data = X[aa[0]:bb[0], aa[1]:bb[1], :]
    return np.asarray(cv.cvtColor(data, cv.COLOR_BGR2GRAY)) if gray else data


def weight(X_gray, hat_qu, hat_pu, u):
    res = np.zeros(X_gray.shape[0:2])
    binHisto = binHistoLuminance(X_gray)

    for i in u:
        res += deltaKron(b(X_gray) - i) * (0 if hat_pu[i] == 0 else np.sqrt(hat_qu[i]/(hat_pu[i])))

    return res
    # return gradient(res)


def colorToCoord(X):
    return np.array([np.arange(0, X.shape[0]), np.arange(0, X.shape[1])])


def gradient(X):
    im = X.astype(float)
    sobelFilter = np.array([-0.5, 0, 0.5]) * -1 # times -1 for low to high gradient
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
    frameCpy = frame.copy()

    X_gray = extractFromAABB(frame, previousTracking.aa, previousTracking.bb, gray=True)
    hat_qu = previousTracking.densityFunction

    # Target model
    Y0 = previousTracking.center.copy()
    Y0_AA = previousTracking.aa.copy()
    Y0_BB = previousTracking.bb.copy()

    # Target candidate
    Y1 = Y0.copy()
    Y1_AA = previousTracking.aa.copy()
    Y1_BB = previousTracking.bb.copy()

    battaX = []
    battaY = []
    battaZ = []

    while True:
        Y0_color_gray = extractFromAABB(frame, Y0_AA, Y0_BB, gray=True)
        hatPU_Y0 = hat_Pu(Y0_color_gray, Y0, indexesHistogram, h)
        pY0 = np.sum(np.sqrt(hatPU_Y0 * hat_qu))

        weightX = weight(X_gray, hat_qu, hatPU_Y0, indexesHistogram)
        # plotImageWithColorBar(weightX, title="weight")
        # pause()
        weightXY, weightXX = gradient(weightX)

        Xcoords = getCenteredCoord(X_gray)

        norm_Y0_minus_X = getNormXY(Xcoords[0], Xcoords[1])

        gradKernelY, gradKernelX = g(norm_Y0_minus_X**2)
        wi_g_X = gradKernelX*weightX
        wi_g_Y = gradKernelY*weightX

        # plotImageWithColorBar(norm_Y0_minus_X, title="normY0")
        # plotImageWithColorBar(gradKernelX, title="gradKernelX")
        # plotImageWithColorBar(gradKernelY, title="gradKernelY")
        # pause()

        Y1_y = np.sum(Xcoords[0]*wi_g_Y) / np.sum(wi_g_Y)
        Y1_x = np.sum(Xcoords[1]*wi_g_X) / np.sum(wi_g_X)

        Y1_x = 0 if np.isnan(Y1_x) else Y1_x
        Y1_y = 0 if np.isnan(Y1_y) else Y1_y

        print("Mean shift y1.x : %f" % Y1_x)
        print("Mean shift y1.y : %f" % Y1_y)

        # Mean shift result
        # Y1_AA[1] += Y1_x
        # Y1_AA[0] += Y1_y
        # Y1_BB[1] += Y1_x
        # Y1_BB[0] += Y1_y
        # Y1[1] += Y1_x
        # Y1[0] += Y1_y

        Y1_AA[1] = Y0_AA[1] + Y1_x
        Y1_AA[0] = Y0_AA[0] + Y1_y
        Y1_BB[1] = Y0_BB[1] + Y1_x
        Y1_BB[0] = Y0_BB[0] + Y1_y
        Y1[1] = Y0[1] + (Y1_x*1)
        Y1[0] = Y0[0] + (Y1_y*1)

        drawRectangle(frameCpy, Y1_AA, Y1_BB, color=(255, 255, 255))
        cv.imshow("res2", frameCpy)

        Y1_color_gray = extractFromAABB(frame, Y1_AA, Y1_BB, gray=True)
        hatPU_Y1 = hat_Pu(Y1_color_gray, Y1, indexesHistogram, h)

        pY1 = np.sum(np.sqrt(hatPU_Y1 * hat_qu))
        battaZ.append(pY1)
        battaX.append(Y1[1])
        battaY.append(Y1[0])

        print("pY1 %f " % pY1)
        print("pY0 %f " % pY0)
        print("batta diff: %f" % (pY0 - pY1))

        # plotHistoCurve(hat_qu, title="model density")
        # pause()

        # plotHistoCurve(hat_qu, "previous frame")
        # plotHistoCurve(hatPU_Y0, "current frame")
        # pause()

        while pY1 < pY0 - 0.001:
            Y1 = (Y0 + Y1) * 0.5
            Y1_AA = (Y0_AA + Y1_AA) * 0.5
            Y1_BB = (Y0_BB + Y1_BB) * 0.5

            # print(Y1_AA)
            # print(Y1_BB)

            Y1_color_gray = extractFromAABB(frame, Y1_AA, Y1_BB, gray=True)
            hatPU_Y1 = hat_Pu(Y1_color_gray, Y1, indexesHistogram, h)
            pY1 = np.sum(np.sqrt(hatPU_Y1 * hat_qu))

            # print("while pY1 %f " % pY1)
            # print("while pY0 %f " % pY0)


            battaZ.append(pY1)
            battaX.append(Y1[1])
            battaY.append(Y1[0])

            drawRectangle(frameCpy, Y1_AA, Y1_BB, color=(0, 0, 255))
            cv.imshow("res2", frameCpy)
        print("py1 < py0 false")

        # If result found i.e distance between Y1 and Y0 less than a threshold, return tracking result
        if np.linalg.norm(Y1 - Y0) < epsilon or pY0 == pY1:
            print("FOUND")
            # cv.waitKey(0)
            # plot3D(battaX, battaY, battaZ, "battacha")
            # pause()
            drawRectangle(frameCpy, Y1_AA, Y1_BB, color=(255, 0, 0))
            cv.imshow("res2", frameCpy)
            hat_qu_Y1 = hat_Qu(Y1_color_gray, indexesHistogram)
            return ResultTracking(aa=Y1_AA, bb=Y1_BB, center=Y1, densityFunction=hat_qu_Y1)

        # Else candidate become the model by mean shift
        Y0 = Y1.copy()
        Y0_AA = Y1_AA.copy()
        Y0_BB = Y1_BB.copy()

