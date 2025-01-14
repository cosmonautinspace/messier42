from PIL import Image
from pybrain.tools.customxml.networkreader import NetworkReader
import math

'''This script is for the final adjustment of pixel values of a given unstrectched image'''

image = Image.open('data/images/input/inputCropped.tif')
annRB = NetworkReader.readFrom("code/ann_PyBrain/modelRedBlue.xml")
annG = NetworkReader.readFrom("code/ann_PyBrain/modelGreen.xml")

#just for testing purposes. real value will be calculated using the ann model
factor=30
for r in range(image.height):
    for c in range(image.width):
        pixelValue = image.getpixel((c,r))
        rbM = annRB.activate([pixelValue[0],pixelValue[2]])
        rG= annG.activate([pixelValue[1]])
        image.putpixel((c,r),(pixelValue[0]*math.ceil(rbM),pixelValue[1]*math.ceil(rG),pixelValue[2]*math.ceil(rbM)))

image.save('data/images/output/test2.png')