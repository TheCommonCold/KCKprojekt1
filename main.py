from __future__ import division
from pylab import *
import skimage as ski
from skimage import data, io, filters, exposure,measure,feature
from skimage.filters import rank
from skimage import img_as_float, img_as_ubyte
from skimage.morphology import disk
import skimage.morphology as mp
from skimage import util
from skimage.color import rgb2hsv, hsv2rgb, rgb2gray,gray2rgb
from skimage.filters import *
from matplotlib import pylab as plt
from skimage.morphology import watershed
import scipy as sci
import numpy as np
from scipy import ndimage as ndi
from numpy import array
from skimage.measure import label
from skimage import data, util
io.use_plugin('matplotlib')

def filter_colour(data,min,max):
    for array in data:
        for x in array:
            if (x[0]<min or x[0]>max):
                x[0]=0
                x[1]=0
                x[2]=0
    return data

def filter_color_hard(data,min,max):
    output=[]
    for array in data:
        temparray = []
        for x in array:
            if (x[0]<min or x[0]>max):
                temparray.append(0)
            else:
                temparray.append(1)
        output.append(temparray)
    return output

def background_removal(data):
    checkpoint=[]
    p1, p2 = np.percentile(data, (2, 95))
    data = exposure.rescale_intensity(data, in_range=(p1, p2))
    data=rgb2hsv(data)
    data = np.array([[(x[0],(x[1]+1)/2,x[2]) for x in array] for array in data])
    checkpoint.append(hsv2rgb(data))
    data=np.array(filter_color_hard(data,0.4,0.7))
    print(data.shape)
    data=mp.dilation(data)
    checkpoint.append(data)
    contours = measure.find_contours(data, 0.2)

    return checkpoint,contours

if __name__ == '__main__':
    data = io.imread('1.jpg')
    data = img_as_float(data)
    checkpoint,contours=background_removal(data)
    rows=len(checkpoint)
    columns=1
    fig = plt.figure(figsize=(10, len(checkpoint)*10))
    for i in range(1,len(checkpoint)+1):
        ax = fig.add_subplot(rows, columns, i)
        io.imshow(checkpoint[i-1])
        if(i==len(checkpoint)):
            for n, contour in enumerate(contours):
                ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
    io.show()