from pylab import *
import skimage as ski
from skimage import data, io, filters, exposure, measure, feature
from skimage.filters import rank
from skimage import img_as_float, img_as_ubyte
from skimage.morphology import disk
import skimage.morphology as mp
from skimage import util
from skimage.color import rgb2hsv, hsv2rgb, rgb2gray, gray2rgb
from skimage.filters import *
from matplotlib import pylab as plt
from skimage.morphology import watershed
import scipy as sci
import numpy as np
from scipy import ndimage as ndi
from numpy import array
from skimage.measure import label
from skimage import data, util
from matplotlib import colors

io.use_plugin('matplotlib')


def wyswietl(checkpoint):
    rows = len(checkpoint)
    columns = 1
    fig = plt.figure(figsize=(10, rows * 10))
    ploty=[]
    for i in range(rows):
        ax = fig.add_subplot(rows, columns, i + 1)
        ploty.append(ax)
        io.imshow(checkpoint[i])
    return ploty


# zostawia kolor(w hsv) pomiędzy min i max i zmienia reszte na 0
def filter_colour(data, min, max, hsv):
    for array in data:
        for x in array:
            if x[hsv] < min or x[hsv] > max:
                x[0] = 0
                x[1] = 0
                x[2] = 0
    return data


# zmienia kolor(w hsv) pomiędzy min i max na 1, a reszte na zero
def filter_color_hard(data, min, max,hsv):
    output = []
    for array in data:
        temparray = []
        for x in array:
            if (x[hsv] < min or x[hsv] > max):
                temparray.append(0)
            else:
                temparray.append(1)
        output.append(temparray)
    return output


# tworzy maske jedynek na wodzie
def background_removal(data):
    checkpoint = []
    p1, p2 = np.percentile(data, (2, 95))
    data = exposure.rescale_intensity(data, in_range=(p1, p2))
    data = rgb2hsv(data)
    checkpoint.append(hsv2rgb(data))
    data = np.array(filter_color_hard(data, 0.4, 0.7,0))
    data = mp.dilation(data)
    checkpoint.append(data)
    contours = measure.find_contours(data, 0.2)

    return checkpoint, contours


# matematyczny argmin argmax
def arg_min_max(lista):
    # zwarca argumenty min i max danej listy
    max = min = lista[0]
    argmin = 0
    argmax = 0
    for k in range(len(lista)):
        if lista[k] < min:
            min = lista[k]
            argmin = k
        elif (lista[k] > max):
            max = lista[k]
            argmax = k
    return argmin, argmax


# tworzy maske zer w miejscu lądu
def outer_removal(img):
    img2 = img.copy()
    maxJ = len(img[0])
    for i in range(len(img)):
        for j in range(maxJ):
            if img[i][j] == 1:
                break
            img2[i][j] = 1
        for j in range(1, maxJ):
            if img[i][maxJ - j] == 1:
                break
            img2[i][maxJ - j] = 1
    return img2


# zostawia tylko wyspe (usuwa wode i stół)
def leave_only_island(img, mask):
    img2 = img.copy()
    for i in range(len(img)):
        for j in range(len(img[0])):
            if mask[i][j]:
                img2[i][j] = np.array([0, 0, 0])
    return img2


# zostawia tylko jeden kolor (niepoprawny grayscale)
def leave_only_one_color(img, color):
    if color == "red" or color == "r":
        color = 0
    elif color == "green" or color == "g":
        color = 1
    elif color == "blue" or color == "b":
        color = 2
    return np.array([[(x[color]) for x in array] for array in img])


# zwraca minimalną i maksymalną wartość z macierzy
def minimaxi(img):
    mini = 2
    maxi = -1
    for i in range(len(img)):
        for j in range(len(img[i])):
            if img[i][j] > maxi:
                maxi = img[i][j]
        if img[i][j] < mini:
            mini = img[i][j]
    return mini, maxi


# robi zakres od 0 to 1 z węższego
def ujednolic(img):
    mini, maxi = minimaxi(img)
    img = (img - mini) / (maxi - mini)
    return img


# przerabia na 0 i 1 zależnie od threshodu
def threshold(img, thr):
    # std = np.std(img)
    # var = np.var(img)
    # print(nazwa + " " + str(std) + " " + str(var))
    return img > thr


# Odwrotnie, bo czarny, a nie biały
def dilation(img):
    return mp.erosion(img)


# Odwrotnie, bo czarny, a nie biały
def erosion(img):
    return mp.dilation(img)

def erosion_loop(data,times):
    for i in range(times):
        data=mp.erosion(data)
    return data
def dilation_loop(data,times):
    for i in range(times):
        data=mp.dilation(data)
    return data

def kontury_do_srodkow(img,ax):
    contours = measure.find_contours(img, 0.5)

    for n, contour in enumerate(contours):
        print(len(contour))
        if (len(contour)>10):

            centroid = np.sum(contour, axis=0) / len(contour)
            q = np.random.uniform()
            c = colors.hsv_to_rgb([q, 1, 1])
            ax.plot(contour[:, 1], contour[:, 0], linewidth=3, color=c)
            if (q > 0.5):
                q -= 0.5
            else:
                q += 0.5
            c = colors.hsv_to_rgb([q, 1, 1])
            ax.plot(centroid[1], centroid[0], marker="o", color=c)


if __name__ == '__main__':
    start_time = time.time()
    data = io.imread('5-fried-16-times.jpg')
    data = img_as_float(data)
    checkpoint, contours = background_removal(data.copy())

    checkpoint.append(outer_removal(checkpoint[1]))
    checkpoint.append(leave_only_island(checkpoint[0], checkpoint[2]))

    checkpoint.append(hsv2rgb(np.array(filter_colour(rgb2hsv(checkpoint[len(checkpoint) - 1]), 0, 0.5,1))))
    checkpoint.append(hsv2rgb(np.array(filter_colour(rgb2hsv(checkpoint[len(checkpoint) - 1]), 0.1, 0.2, 0))))

    checkpoint.append(filters.sobel(rgb2gray(checkpoint[len(checkpoint) - 1])))

    checkpoint.append(dilation_loop(checkpoint[len(checkpoint) - 1],1))

    #checkpoint.append(ujednolic(leave_only_one_color(checkpoint[len(checkpoint) - 1].copy(), "red")))
    #checkpoint.append(threshold(checkpoint[len(checkpoint) - 1].copy(), 0.95))
    #checkpoint.append(dilation(checkpoint[5]))
    ploty = wyswietl(checkpoint)
    #kontury_do_srodkow(checkpoint[6],ploty[6])

    io.show()
    print(time.time() - start_time)
