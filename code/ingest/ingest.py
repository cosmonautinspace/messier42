import numpy as np 
from matplotlib.image import imread
from PIL import Image
import pandas as pd 

inputName = input('Enter the name of the input image(please include the file extension)>> ')
targetName = input('Enter the name of the target image(please include the file extension)>> ')
input = imread(f"data/images/input/{inputName}")
target = imread(f"data/images/target/{targetName}")

tempList = []
tempList2 = []

for r in range(input.shape[0]):
    for c in range(input.shape[1]):
        tempList.append(input[r,c])

for r in range(target.shape[0]):
    for c in range(target.shape[1]):
        tempList2.append(target[r,c])

inputDataframe = pd.DataFrame(tempList)
targetDataframe = pd.DataFrame(tempList2)


outputDataframe = pd.concat([inputDataframe,targetDataframe], axis=1)
outputDataframe.columns = ['rI','gI','bI','rT','gT','bT']
outputDataframe.to_csv('data/scrapeFromHere/pixelValues.csv', sep=',', index=False)

