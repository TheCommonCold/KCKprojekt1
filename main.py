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
    fig = plt.figure(figsize=(5, rows * 5))
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
def filter_colour_hard(data, min, max,hsv):
    output = []
    for array in data:
        temparray = []
        for x in array:
            if x[hsv] < min or x[hsv] > max:
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
    data = np.array(filter_colour_hard(data, 0.4, 0.7,0))
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

#daje numer największego contouru
def sort_contourow(contours):
    return sorted(contours,key=len)

def top_contoury(contours,ile):
    output=[]
    for i in range(len(contours)-ile,len(contours)):
        output.append(contours[i])
    return output

def centroid_z_konturu(contour,ax,colour):
    centroid = np.sum(contour, axis=0) / len(contour)
    q = np.random.uniform()
    c = colors.hsv_to_rgb([q, 1, 1])
    #ax.plot(contour[:, 1], contour[:, 0], linewidth=3, color=c)

    c = colors.hsv_to_rgb([colour, 1, 1])
    ax.plot(centroid[1], centroid[0], marker="o", color=c)

def kontury_do_srodkow(contours,ax,colour):
    for n, contour in enumerate(contours):
        if (len(contour)>10):
            centroid_z_konturu(contour,ax,colour)

def usuwanko_punktow(contour):
    test_dist=1
    delete_array=[];
    while(len(contour)>6):
        for x in range(len(contour)-1):
            for y in range(x,len(contour)-1):
                dist = math.sqrt((contour[y][0] - contour[x][0])**2 + (contour[y][1] - contour[x][1])**2)
                if(dist<test_dist):
                    delete_array.append(y)
        while(len(contour)-len(delete_array)<6):
            delete_array.pop()
        contour=np.delete(contour,delete_array ,0)
        print(len(contour))
        test_dist=test_dist+1

    return contour

def srednia_kanalu(data,rgb):
    sum=0
    n=0
    for array in data:
        for x in array:
            if x[0]!=0 and x[1]!=0 and x[2]!=0:
                sum = sum + x[rgb]
                n=n+1
    return sum/n


def findsheep(checkpoint,start):
    checkpoint.append(hsv2rgb(np.array(filter_colour(rgb2hsv(checkpoint[start]), 0.18, 0.25, 0))))
    checkpoint.append(hsv2rgb(np.array(filter_colour(rgb2hsv(checkpoint[len(checkpoint) - 1]), 0.5, 1, 1))))
    checkpoint.append(hsv2rgb(np.array(filter_colour(rgb2hsv(checkpoint[len(checkpoint) - 1]), 0.5, 1, 2))))
    checkpoint.append(erosion_loop(checkpoint[len(checkpoint) - 1], 1))
    checkpoint.append(dilation_loop(checkpoint[len(checkpoint) - 1], 8))
    checkpoint.append(threshold(rgb2gray(checkpoint[len(checkpoint) - 1]), 0.4))

def findforest(checkpoint,start):
    checkpoint.append(hsv2rgb(np.array(filter_colour(rgb2hsv(checkpoint[start]), 0.13, 0.18, 0))))
    checkpoint.append(hsv2rgb(np.array(filter_colour(rgb2hsv(checkpoint[len(checkpoint) - 1]), 0.4, 1, 1))))
    checkpoint.append(np.array(filter_colour(rgb2hsv(checkpoint[len(checkpoint) - 1]), 0, 0.4, 2)))
    checkpoint.append(erosion_loop(rgb2gray(checkpoint[len(checkpoint) - 1]), 1))
    checkpoint.append(dilation_loop(checkpoint[len(checkpoint) - 1], 8))
    checkpoint.append(threshold(checkpoint[len(checkpoint) - 1], 0.2))

def findclay(checkpoint,start):
    checkpoint.append(hsv2rgb(np.array(filter_colour(rgb2hsv(checkpoint[start]), 0.0, 0.08, 0))))
    checkpoint.append(hsv2rgb(np.array(filter_colour(rgb2hsv(checkpoint[len(checkpoint) - 1]), 0.5, 1, 1))))
    checkpoint.append(erosion_loop(rgb2gray(checkpoint[len(checkpoint) - 1]), 1))
    checkpoint.append(dilation_loop(rgb2gray(checkpoint[len(checkpoint) - 1]), 2))
    checkpoint.append(threshold(checkpoint[len(checkpoint) - 1], 0.2))

def findmountains(checkpoint, start):
    checkpoint.append(hsv2rgb(np.array(filter_colour(rgb2hsv(checkpoint[start]), 0.0, 0.1, 0))))
    checkpoint.append(hsv2rgb(np.array(filter_colour(rgb2hsv(checkpoint[len(checkpoint) - 1]), 0.0, 0.5, 1))))
    checkpoint.append(erosion_loop(rgb2gray(checkpoint[len(checkpoint) - 1]), 1))
    checkpoint.append(dilation_loop(rgb2gray(checkpoint[len(checkpoint) - 1]), 8))
    checkpoint.append(threshold(checkpoint[len(checkpoint) - 1], 0.15))

def findwheat(checkpoint, start):
    checkpoint.append(hsv2rgb(np.array(filter_colour(rgb2hsv(checkpoint[start]), 0.07, 0.13, 0))))
    checkpoint.append(hsv2rgb(np.array(filter_colour(rgb2hsv(checkpoint[len(checkpoint) - 1]), 0.6, 1, 1))))
    checkpoint.append(erosion_loop(rgb2gray(checkpoint[len(checkpoint) - 1]), 1))
    checkpoint.append(dilation_loop(rgb2gray(checkpoint[len(checkpoint) - 1]), 3))
    checkpoint.append(threshold(checkpoint[len(checkpoint) - 1], 0.15))

if __name__ == '__main__':
    start_time = time.time()
    data = io.imread('5-fried-16-times.jpg')
    data = img_as_float(data)
    checkpoint, contours = background_removal(data.copy())

    checkpoint.append(outer_removal(checkpoint[1]))
    checkpoint.append(leave_only_island(checkpoint[0], checkpoint[2]))

    #checkpoint.append(hsv2rgb(np.array(filter_colour(rgb2hsv(checkpoint[len(checkpoint) - 1]), 0, 0.5,1))))
    print(srednia_kanalu(checkpoint[3],0))
    print(srednia_kanalu(checkpoint[3], 1))
    print(srednia_kanalu(checkpoint[3], 2))


    findforest(checkpoint,3)
    contours1 = measure.find_contours(checkpoint[len(checkpoint) - 1], 0.5)
    findsheep(checkpoint,3)
    contours2 = measure.find_contours(checkpoint[len(checkpoint) - 1], 0.5)
    findclay(checkpoint,3)
    contours3 = measure.find_contours(checkpoint[len(checkpoint) - 1], 0.5)
    findmountains(checkpoint, 3)
    contours4 = measure.find_contours(checkpoint[len(checkpoint) - 1], 0.5)
    findwheat(checkpoint, 3)
    contours5 = measure.find_contours(checkpoint[len(checkpoint) - 1], 0.14)
    ploty = wyswietl(checkpoint)
    contours1 = sort_contourow(contours1)
    contours1 = top_contoury(contours1,4)
    kontury_do_srodkow(contours1, ploty[0], 0.25)
    contours2 = sort_contourow(contours2)
    contours2 = top_contoury(contours2, 4)
    kontury_do_srodkow(contours2, ploty[0], 0.4)
    contours3 = sort_contourow(contours3)
    contours3 = top_contoury(contours3, 3)
    kontury_do_srodkow(contours3, ploty[0], 0)
    contours4 = sort_contourow(contours4)
    contours4 = top_contoury(contours4, 3)
    kontury_do_srodkow(contours4, ploty[0], 0.8)
    contours5 = sort_contourow(contours5)
    contours5 = top_contoury(contours5, 5)
    kontury_do_srodkow(contours5, ploty[0], 0.7)

    #coords = measure.approximate_polygon(contours[najwiekszy_contour(contours)], tolerance=2)
    #coords=usuwanko_punktow(coords)

    io.show()
    print(time.time() - start_time)
