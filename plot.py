import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plotImageWithColorBar(I,cmap=plt.cm.Greys_r,title=''):
    """this function displays an image and a color bar"""
    fig = plt.figure(figsize=(8,7))
    ax=plt.subplot(1,1,1)
    plt.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0)
    implot=plt.imshow(I,cmap)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.colorbar(implot, use_gridspec=True)
    plt.title(title)

def plot3D(X, Y, Z, title=''):
    print X
    print Y
    print Z

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(X, Y, Z)
    plt.title(title)


def plotColorImage(I,title=''):
    """this function displays an image and a color bar"""
    fig = plt.figure(figsize=(8,7))
    ax=plt.subplot(1,1,1)
    plt.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0)
    implot=plt.imshow(I)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.colorbar(implot, use_gridspec=True)
    plt.title(title)


def plotHistoCurve(I, title=''):
    """this function displays an image and a color bar"""
    fig = plt.figure(figsize=(8,7))
    ax=plt.subplot(1,1,1)
    # plt.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0)
    plt.plot(I)
    plt.title(title)

def pause():
    """this function allows to refresh a figure and do a pause"""
    plt.draw()
    plt.show(block=True)

