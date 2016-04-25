import numpy as np
from plot import *
import scipy.ndimage
from draw import *

# RGB to luminance
indexesHistogram = np.arange(0, 100)


class ResultTracking:
    def __init__(self, aa=None, bb=None, center=None, densityFunction=None):
        self.aa = aa if np.all(aa) is not None else np.array([0, 0])
        self.bb = bb if np.all(bb) is not None else np.array([0, 0])
        self.center = center if np.all(center) is not None else np.array([0, 0])


def deltaKron(a):
    a = np.equal(a, 0).astype(int)
    return a


def k(x):
    # Epanechnikov kernel
    d = 2  # dimension 2
    mask = np.less(x, 1).astype(int)
    a = 0.5 * (3.14) * (d + 2) * (1-x) * mask
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

    density = []
    for i in u:
        dKron = deltaKron(b(X_gray)-i)
        density.append(C * np.sum(normWeight * dKron))
    return np.array(density)


def hat_Pu(X_gray, Ycenter, u, hx, hy):
    XiCentered = getCenteredCoord(X_gray)
    normXi = getNormXY(XiCentered[1]/hx , XiCentered[0]/hy)
    normWeight = k(normXi**2)
    C = 1.0/np.sum(normWeight)

    density = []
    for i in u:
        dKron = deltaKron(b(X_gray)-i)
        density.append(C * np.sum(normWeight * dKron))
    return np.array(density)


def binHistoLuminance(X_gray):
    """
    :return: bin histogram of rgb luminance
    """
    gray_image_histo = X_gray.ravel()
    hist, bins = np.histogram(gray_image_histo, bins=255, range=(0, 255))
    return hist.astype(int)


def b(X_gray):
    # plotHistoCurve((X_gray/255.0*indexesHistogram.shape[0]))
    # plotHistoCurve((X_gray/255.0*1))
    # pause()
    return np.floor((X_gray/255.0*indexesHistogram.shape[0]))


def extractFromAABB(X, aa, bb, gray=False):
    data = X[aa[0]:bb[0], aa[1]:bb[1], :]
    return cv.cvtColor(data, cv.COLOR_BGR2GRAY) if gray else data


def weight(X_gray, hat_qu, hat_pu, u):
    res = np.zeros(X_gray.shape[0:2])

    for i in u:
        res += 0 if hat_pu[i] == 0 else deltaKron(b(X_gray) - i) * np.sqrt(hat_qu[i]/(hat_pu[i]))
    return res


def colorToCoord(X):
    return np.array([np.arange(0, X.shape[0]), np.arange(0, X.shape[1])])


def gradient(X):
    im = X.astype(float)
    sobelFilter = np.array([-0.5, 0, 0.5]) * -1 # times -1 for low to high gradient
    gradient_y = scipy.ndimage.convolve(im, sobelFilter[np.newaxis])
    gradient_x = scipy.ndimage.convolve(im, sobelFilter[np.newaxis].T)
    return gradient_y, gradient_x


def track(frame, previousTracking, modelDensity, captureWidth, captureHeight):
    """
    Process tracking in color space.
    """
    hx = np.abs(previousTracking.bb[1] - previousTracking.aa[1]) / 2
    hy = np.abs(previousTracking.bb[0] - previousTracking.aa[0]) / 2
    epsilon = 7
    frame = np.asarray(frame)
    frameCpy = frame.copy()

    X_gray = extractFromAABB(frame, previousTracking.aa, previousTracking.bb, gray=True)
    hat_qu = modelDensity

    # pause()
    # Used for g(||(y -x)/h||**2)
    Y0_minus_X_coords = getCenteredCoord(X_gray, False)
    norm_Y0_minus_X = getNormXY(Y0_minus_X_coords[1]/hx, Y0_minus_X_coords[0]/hy)
    gradKernelY, gradKernelX = g(norm_Y0_minus_X**2)

    # Target model
    Y0 = previousTracking.center.copy()
    Y0_AA = previousTracking.aa.copy()
    Y0_BB = previousTracking.bb.copy()

    # Target candidate
    Y1 = Y0.copy()
    Y1_AA = previousTracking.aa.copy()
    Y1_BB = previousTracking.bb.copy()
    nbIterBatta = 0

    while True:
        Y0_color_gray = extractFromAABB(frame, Y0_AA, Y0_BB, gray=True)
        hatPU_Y0 = hat_Pu(Y0_color_gray, Y0, indexesHistogram, hx, hy)
        pY0 = np.sum(np.sqrt(hatPU_Y0 * hat_qu))

        weightX = weight(X_gray, hat_qu, hatPU_Y0, indexesHistogram)
        wi_g_X = gradKernelX*weightX
        wi_g_Y = gradKernelY*weightX

        Y1_y = np.sum(Y0_minus_X_coords[0]*wi_g_Y) / np.sum(wi_g_Y)
        Y1_x = np.sum(Y0_minus_X_coords[1]*wi_g_X) / np.sum(wi_g_X)
        Y1_x = 0 if np.isnan(Y1_x) else Y1_x
        Y1_y = 0 if np.isnan(Y1_y) else Y1_y

        # print("Mean shift y1.x : %f" % Y1_x)
        # print("Mean shift y1.y : %f" % Y1_y)

        Y1_AA[1] = Y0_AA[1] + Y1_x
        Y1_AA[0] = Y0_AA[0] + Y1_y
        Y1_BB[1] = Y0_BB[1] + Y1_x
        Y1_BB[0] = Y0_BB[0] + Y1_y
        Y1[1] = Y0[1] + Y1_x
        Y1[0] = Y0[0] + Y1_y

        # Keep the MS inside the video resolution
        if Y1_AA[1] < 0 or Y1_BB[1] > captureWidth or Y1_AA[0] < 0 or Y1_BB[0] > captureHeight:
            print("Warning: mean shift outside the the capture dimension")
            return ResultTracking(aa=Y0_AA, bb=Y0_BB, center=Y0, densityFunction=hatPU_Y0)

        # drawRectangle(frameCpy, Y1_AA, Y1_BB, color=(255, 255, 255))
        # drawRectangle(frameCpy, previousTracking.aa, previousTracking.bb, color=(0, 0, 0))
        # cv.imshow("res2", frameCpy)

        Y1_color_gray = extractFromAABB(frame, Y1_AA, Y1_BB, gray=True)
        hatPU_Y1 = hat_Pu(Y1_color_gray, Y1, indexesHistogram, hx, hy)
        pY1 = np.sum(np.sqrt(hatPU_Y1 * hat_qu))

        # print("pY1 %f " % pY1)
        # print("pY0 %f " % pY0)
        # print("batta diff: %f" % (pY0 - pY1))

        # and np.linalg.norm(Y1 - Y0) > 0.00000001
        while pY1 < pY0 and Y1[0] != Y0[0] and Y1[1] != Y0[1]:
            Y1 = (Y0 + Y1) * 0.5
            Y1_AA = (Y0_AA + Y1_AA) * 0.5
            Y1_BB = (Y0_BB + Y1_BB) * 0.5

            Y1_color_gray = extractFromAABB(frame, Y1_AA, Y1_BB, gray=True)
            hatPU_Y1 = hat_Pu(Y1_color_gray, Y1, indexesHistogram, hx, hy)
            pY1 = np.sum(np.sqrt(hatPU_Y1 * hat_qu))
            # drawRectangle(frameCpy, Y1_AA, Y1_BB, color=(0, 0, msColorRect))
            # cv.imshow("res2", frameCpy)
            nbIterBatta += 1

        # If result found i.e distance between Y1 and Y0 less than a threshold, return tracking result
        if np.linalg.norm(Y1 - Y0) < epsilon:
            print("nbIterBatta %d" % nbIterBatta)
            # print("FOUND")
            # drawRectangle(frameCpy, Y1_AA, Y1_BB, color=(0, 255, 0))
            # cv.imshow("res2", frameCpy)
            # cv.waitKey(0)
            return ResultTracking(aa=Y1_AA, bb=Y1_BB, center=Y1, densityFunction=hat_qu)

        # Else candidate become the model by mean shift
        Y0 = Y1.copy()
        Y0_AA = Y1_AA.copy()
        Y0_BB = Y1_BB.copy()

