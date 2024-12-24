import numpy as np 
from matplotlib.image import imread
from PIL import Image
import pandas as pd 

input = imread('data/input/inputCropped.tif')
target = imread('data/target/targetCropped.tif')

print(input.shape)

tempList = []
print(input.shape[0])

for r in range(input.shape[0]):
    for c in range(input.shape[1]):
        tempList.append(input[r,c])

inputDataframe = pd.DataFrame(tempList)
print(inputDataframe)