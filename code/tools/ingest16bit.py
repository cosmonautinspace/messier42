import numpy as np 
import cv2 as cv
import pandas as pd 

inputName = input('Enter the name of the input image(please include the file extension)>> ')
targetName = input('Enter the name of the target image(please include the file extension)>> ')

'''For future reference
The -1 flag is necessary to load 16 bit images using opencv2'''
input = cv.imread(f"data/images/input/{inputName}",-1)
target = cv.imread(f"data/images/target/{targetName}",-1)


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


# "I" columns are the input variables for RGB values, respectively, taken from the unstretched image
# "T" columns are the target variables for RGB values, respectively, taken from the stretched image
outputDataframe.columns = ['rI','gI','bI','rT','gT','bT']
outputDataframe.to_csv('data/scrapeFromHere/16bitvalues.csv', sep=',', index=False)
