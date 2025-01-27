import pandas as pd 
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.structure import FeedForwardNetwork 
from pybrain.supervised import BackpropTrainer
from pybrain.structure.modules import SoftmaxLayer, LinearLayer, SigmoidLayer, TanhLayer, ReluLayer
#from pybrain.utilities import percentError
from pybrain.structure import FullConnection
from pybrain.tools.customxml.networkwriter import NetworkWriter
import random
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import statsmodels.api as sm


'''Dataset configuration:
RGB subpixel values as input variables (3)
RedBlueMultiplicationFactor and GreenMultiplicationFactor as the target variables (2)'''

data = pd.read_csv('data/extras/first_16bitvalues_cleanedDataWhole.csv')


'''Normalization of multiplication factors'''
gMMax = data["gM"].max()
rbMMax = data["rbM"].max()
for i in range(data["gM"].size):
    data["gM"].iloc[[i]] /= gMMax

for i in range(data["rbM"].size):
    data["rbM"].iloc[[i]] /= rbMMax

print('data read')
alldataG = SupervisedDataSet(inp=1,target=1)
alldataRB = SupervisedDataSet(inp=2,target=1)
for i in range(data['rI'].size):
    alldataG.addSample(inp=data.gI[i], target=data.gM[i])

for i in range(data['rI'].size):
    alldataRB.addSample(inp=[data.rI[i],data.bI[i]], target=[data.rbM[i]])
print('data prepped')

testG, trainG = alldataG.splitWithProportion(0.2)

testRB, trainRB = alldataRB.splitWithProportion(0.2)
print('data split')

'''Model configuration can be found below, need to optimize before final submission of the project....Just placeholder values for now, for testing 
purposes'''

hiddenLayers = 100
rounds = 1

'''
ann = FeedForwardNetwork()

inLayer = LinearLayer(inputLayers)
hiddenLayer = SoftmaxLayer(hiddenLayers)
outLayer = LinearLayer(outputLayers)
ann.addInputModule(inLayer)
ann.addModule(hiddenLayer)
ann.addOutputModule(outLayer)

inToHidden = FullConnection(inLayer,hiddenLayer)
hiddenToOut = FullConnection(hiddenLayer,outLayer)
ann.addConnection(inToHidden)
ann.addConnection(hiddenToOut)
ann.sortModules()
'''

annRB = FeedForwardNetwork()

inLayer = LinearLayer(2)
hiddenLayer = TanhLayer(hiddenLayers)
outLayer = LinearLayer(1)
annRB.addInputModule(inLayer)
annRB.addModule(hiddenLayer)
annRB.addOutputModule(outLayer)

inToHidden = FullConnection(inLayer,hiddenLayer)
hiddenToOut = FullConnection(hiddenLayer,outLayer)
annRB.addConnection(inToHidden)
annRB.addConnection(hiddenToOut)
annRB.sortModules()

annG = FeedForwardNetwork()

inLayer = LinearLayer(1)
hiddenLayer = TanhLayer(hiddenLayers)
outLayer = LinearLayer(1)
annG.addInputModule(inLayer)
annG.addModule(hiddenLayer)
annG.addOutputModule(outLayer)

inToHidden = FullConnection(inLayer,hiddenLayer)
hiddenToOut = FullConnection(hiddenLayer,outLayer)
annG.addConnection(inToHidden)
annG.addConnection(hiddenToOut)
annG.sortModules()
print('model built')

'''Training'''
'''
trainer = BackpropTrainer(ann, dataset=train, verbose=True, learningrate=0.00001)
for i in range(rounds):
    trainer.trainEpochs(1)
'''

trainer = BackpropTrainer(annRB, dataset=trainRB, verbose=True, learningrate=0.00001)
for i in range(rounds):
    trainer.trainEpochs(1)

trainer = BackpropTrainer(annG, dataset=trainG, verbose=True, learningrate=0.00001)
for i in range(rounds):
    trainer.trainEpochs(1)

NetworkWriter.writeToFile(annRB, "code/ann_PyBrain/modelRedBlue16bit.xml")
NetworkWriter.writeToFile(annG, "code/ann_PyBrain/modelGreen16bit.xml")

print(testRB)
print(annRB.activateOnDataset(testRB))

print(testG)
print(annG.activateOnDataset(testG))

print(annRB.activate([50,50]))