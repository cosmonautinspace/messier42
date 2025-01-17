import cv2 as cv 
from pybrain.tools.customxml.networkreader import NetworkReader
import math

'''This script is for the final adjustment of pixel values of a given unstrectched image'''

inputImage=input("Enter the name of the input image(placed in the input folder)")

image = cv.imread(f'data/images/input/{inputImage}',-1)
annRB = NetworkReader.readFrom("code/ann_PyBrain/modelRedBlue16bit.xml")
annG = NetworkReader.readFrom("code/ann_PyBrain/modelGreen16bit.xml")

for r in range(image.shape[0]):
    for c in range(image.shape[1]):
        pixelValue = image[r][c]
        rbM = math.ceil(annRB.activate([pixelValue[0],pixelValue[2]]))
        rG= math.ceil(annG.activate([pixelValue[1]]))
        image[r][c] = (pixelValue[0]*rbM, pixelValue[1]*rG, pixelValue[2]*rbM)


cv.imwrite(f'data/images/output/{inputImage}', image)