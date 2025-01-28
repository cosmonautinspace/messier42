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

data = pd.read_csv('data/cleanedData/train_data.csv')


print('data read')
alldataG = SupervisedDataSet(inp=1, target=1)
alldataRB = SupervisedDataSet(inp=2, target=1)
for i in range(data['rI'].size):
    alldataG.addSample(inp=data.gI[i], target=data.gM[i])
    alldataRB.addSample(inp=[data.rI[i], data.bI[i]], target=[data.rbM[i]])
print('data prepped')

testG, trainG = alldataG.splitWithProportion(0.2)

testRB, trainRB = alldataRB.splitWithProportion(0.2)
print('data split')

'''Model configuration can be found below, need to optimize before final submission of the project....Just placeholder values for now, for testing 
purposes'''

hiddenLayers = 100
rounds = 5

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

inLayerRB = LinearLayer(2)
hiddenLayerRB = SigmoidLayer(hiddenLayers)
outLayerRB = LinearLayer(1)
annRB.addInputModule(inLayerRB)
annRB.addModule(hiddenLayerRB)
annRB.addOutputModule(outLayerRB)
inToHiddenRB = FullConnection(inLayerRB, hiddenLayerRB)
hiddenToOutRB = FullConnection(hiddenLayerRB, outLayerRB)
annRB.addConnection(inToHiddenRB)
annRB.addConnection(hiddenToOutRB)
annRB.sortModules()

annG = FeedForwardNetwork()

inLayerG = LinearLayer(1)
hiddenLayerG = SigmoidLayer(hiddenLayers)
outLayerG = LinearLayer(1)
annG.addInputModule(inLayerG)
annG.addModule(hiddenLayerG)
annG.addOutputModule(outLayerG)
inToHiddenG = FullConnection(inLayerG, hiddenLayerG)
hiddenToOutG = FullConnection(hiddenLayerG, outLayerG)
annG.addConnection(inToHiddenG)
annG.addConnection(hiddenToOutG)
annG.sortModules()
print('model built')

'''Training

trainer = BackpropTrainer(ann, dataset=train, verbose=True, learningrate=0.00001)
for i in range(rounds):
    trainer.trainEpochs(1)
'''

trainerRB = BackpropTrainer(annRB, dataset=trainRB, verbose=True, learningrate=0.00001)
trainerG = BackpropTrainer(annG, dataset=trainG, verbose=True, learningrate=0.00001)

rb_train_losses, rb_test_losses = [], []
g_train_losses, g_test_losses = [], []

for i in range(rounds):
    rb_loss = trainerRB.train()
    rb_train_losses.append(rb_loss)
    rb_test_losses.append(trainerRB.testOnData(dataset=testRB))

    g_loss = trainerG.train()
    g_train_losses.append(g_loss)
    g_test_losses.append(trainerG.testOnData(dataset=testG))

NetworkWriter.writeToFile(annRB, f"code/ann_PyBrain/currentSolutionRB.xml")
NetworkWriter.writeToFile(annG, f"code/ann_PyBrain/currentSolutionG.xml")

np.savetxt("code/ann_PyBrain/rb_train_losses.csv", rb_train_losses, delimiter=",")
np.savetxt("code/ann_PyBrain/rb_test_losses.csv", rb_test_losses, delimiter=",")
np.savetxt("code/ann_PyBrain/g_train_losses.csv", g_train_losses, delimiter=",")
np.savetxt("code/ann_PyBrain/g_test_losses.csv", g_test_losses, delimiter=",")

