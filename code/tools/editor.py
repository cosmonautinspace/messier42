from PIL import Image
from pybrain.tools.customxml.networkreader import NetworkReader
import math
import pandas as pd
'''This script is for the final adjustment of pixel values of a given unstrectched image'''

#greenModel = input("Enter the name of the model to use for the Green Channel >> ")
#redModel = input("Enter the name of the model to use for the Red and Blue channels >> ")
#inputImage=input("Enter the name of the input image(placed in the input folder) >> ")



image = Image.open(f'data/photos/input/orion8bit.tif')
annRB = NetworkReader.readFrom(f"code/ann_PyBrain/currentSolutionRB.xml")
annG = NetworkReader.readFrom(f"code/ann_PyBrain/currentSolutionG.xml")

'''renormalization of values'''
normDf = pd.read_csv('data/cleanedData/normFactors.csv')
gMMean = normDf.gMMean[0]
gMSD = normDf.gMSD[0]
rbMMean = normDf.rbMMean[0]
rbMSD = normDf.rMSD[0]

rMean = normDf.rIMean[0]
gMean = normDf.gIMean[0]
bMean = normDf.bIMean[0]

rSD = normDf.rISD[0]
gSD = normDf.gISD[0]
bSD = normDf.bISD[0]


for r in range(image.height):
    for c in range(image.width):
        pixelValue = image.getpixel((c,r))
        rbM = annRB.activate([(pixelValue[0]-rMean)/rSD,(pixelValue[2]-bMean)/bSD])
        rG= annG.activate([(pixelValue[1]-gMean)/gSD])
        red = (pixelValue[0])*((rbM*rbMSD)+rbMMean)
        green = (pixelValue[1]*((rG*gMSD)+gMMean))
        blue = (pixelValue[2]*((rbM*rbMSD)+rbMMean))
        image.putpixel((c,r), (round(red[0]), round(green[0]), round(blue[0])))


image.save(f'data/photos/output/temp2S.tif')
