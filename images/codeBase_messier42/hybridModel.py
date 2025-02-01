import sys
sys.path.append("/tmp/codeBase/pybrain")
import pandas as pd 
from pybrain.datasets import SupervisedDataSet
from pybrain.structure import FeedForwardNetwork 
from pybrain.supervised import BackpropTrainer
from pybrain.structure.modules import SoftmaxLayer, LinearLayer, SigmoidLayer, TanhLayer, ReluLayer
#from pybrain.utilities import percentError
from pybrain.structure import FullConnection
from pybrain.tools.customxml.networkwriter import NetworkWriter
import numpy as np


'''Dataset configuration:
RGB subpixel values as input variables (3)
RedBlueMultiplicationFactor and GreenMultiplicationFactor as the target variables (2)'''

data = pd.read_csv('/tmp/ai_system/learningBase/train/train_data.csv')
testdf = pd.read_csv('/tmp/ai_system/learningBase/validation/test_data.csv')

print('data read')
alldata = SupervisedDataSet(inp=3, target=2)

for i in range(data['rI'].size):
    alldata.addSample(inp=[data.rI[i],data.gI[i],data.bI[i]], target=[data.gM[i], data.rbM[i]])

testData = SupervisedDataSet(inp=3, target=2)
for i in range(testdf['rI'].size):
    testData.addSample(inp=[data.rI[i],data.gI[i],data.bI[i]], target=[data.gM[i], data.rbM[i]])

print('data prepped')

train = alldata
test = testData
print('data split')


hiddenLayers = 250
rounds = 1

ann = FeedForwardNetwork()

inLayerG = LinearLayer(3)
hiddenLayerG = SigmoidLayer(hiddenLayers)
outLayerG = LinearLayer(2)
ann.addInputModule(inLayerG)
ann.addModule(hiddenLayerG)
ann.addOutputModule(outLayerG)
inToHiddenG = FullConnection(inLayerG, hiddenLayerG)
hiddenToOutG = FullConnection(hiddenLayerG, outLayerG)
ann.addConnection(inToHiddenG)
ann.addConnection(hiddenToOutG)
ann.sortModules()
print('model built')

'''Training'''

trainer = BackpropTrainer(ann, dataset=train, verbose=True, learningrate=0.00001)

train_losses, test_losses = [], []

for i in range(rounds):
    model_loss = trainer.train()
    train_losses.append(model_loss)
    test_losses.append(trainer.testOnData(dataset=test))

NetworkWriter.writeToFile(ann, f"/tmp/ai_system/currentSolution.xml")

np.savetxt("/tmp/ai_system/train_losses.csv", train_losses, delimiter=",")
np.savetxt("/tmp/ai_system/test_losses.csv", test_losses, delimiter=",")

