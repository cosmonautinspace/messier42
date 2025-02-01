import sys
sys.path.append("/tmp/codeBase/pybrain")
from PIL import Image
from pybrain.tools.customxml.networkreader import NetworkReader
import math
import pandas as pd
'''This script is for the final adjustment of pixel values of a given unstrectched image'''

#greenModel = input("Enter the name of the model to use for the Green Channel >> ")
#redModel = input("Enter the name of the model to use for the Red and Blue channels >> ")
#inputImage=input("Enter the name of the input image(placed in the input folder) >> ")



image = Image.open(f'/tmp/ai_system/activation_image.tif')
annG = NetworkReader.readFrom(f"/tmp/ai_system/currentSolution.xml")
activationdf = pd.read_csv(f'/tmp/ai_system/activation_data.csv')

'''renormalization of values'''
normDf = pd.read_csv('/tmp/ai_system/normFactors.csv')
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
        rbM = annG.activate([(pixelValue[0]-rMean)/rSD,(pixelValue[1]-gMean)/gSD,(pixelValue[2]-bMean)/bSD])
        red = (pixelValue[0])*((rbM[1]*rbMSD)+rbMMean)
        green = (pixelValue[1]*((rbM[0]*gMSD)+gMMean))
        blue = (pixelValue[2]*((rbM[1]*rbMSD)+rbMMean))
        image.putpixel((c,r), (round(red), round(green), round(blue)))


image.save(f'/tmp/ai_system/outputANN.tif')
