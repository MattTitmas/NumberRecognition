import numpy as np
import random
import sys


class NeuralNetwork(object):
    def __init__(Self):
        #784,512,256,128,10
        Self.InputLayers = 784
        Self.HiddenLayerOne = 512
        Self.HiddenLayerTwo = 256
        Self.HiddenLayerThree = 128
        Self.OutputLayer = 10
        Self.LearningRate = 0.4

        Self.Weights1 = np.random.randn(Self.InputLayers,Self.HiddenLayerOne)
        Self.Weights2 = np.random.randn(Self.HiddenLayerOne,Self.HiddenLayerTwo)
        Self.Weights3 = np.random.randn(Self.HiddenLayerTwo,Self.HiddenLayerThree)
        Self.Weights4 = np.random.randn(Self.HiddenLayerThree,Self.OutputLayer)


        Self.HiddenOneBias = np.random.rand(1, Self.HiddenLayerOne)
        Self.HiddenTwoBias = np.random.rand(1, Self.HiddenLayerTwo)
        Self.HiddenThreeBias = np.random.rand(1, Self.HiddenLayerThree)
        Self.OutputBias = np.random.rand(1, Self.OutputLayer)


    def ForwardPropogation(Self, Input):
        Self.DotProduct1 = np.dot(Input, Self.Weights1)
        Self.DotProduct1 += Self.HiddenOneBias
        Self.Activated1 = Self.Sigmoid(Self.DotProduct1)     # Hidden Layer One
        Self.DotProduct2 = np.dot(Self.Activated1, Self.Weights2)
        Self.DotProduct2 += Self.HiddenTwoBias
        Self.Activated2 = Self.Sigmoid(Self.DotProduct2)     # Hidden Layer Two
        Self.DotProduct3 = np.dot(Self.Activated2, Self.Weights3)
        Self.DotProduct3 += Self.HiddenThreeBias
        Self.Activated3 = Self.Sigmoid(Self.DotProduct3)     # Hidden Layer Three
        Self.DotProduct4 = np.dot(Self.Activated3, Self.Weights4)
        Self.DotProduct4 += Self.OutputBias
        Prediction = Self.Sigmoid(Self.DotProduct4)
        return Prediction                                     # Output Layer

    def BackwardPropogation(Self, X, Y, Prediction):
        Self.PredictionError = Y - Prediction
        Self.PredictionDelta = Self.PredictionError * Self.SigmoidPrime(Prediction)

        Self.HiddenThreeError = Self.PredictionDelta.dot(Self.Weights4.T)
        Self.HiddenThreeDelta = Self.HiddenThreeError * Self.SigmoidPrime(Self.Activated3)

        Self.HiddenTwoError = Self.HiddenThreeDelta.dot(Self.Weights3.T)
        Self.HiddenTwoDelta = Self.HiddenTwoError * Self.SigmoidPrime(Self.Activated2)

        Self.HiddenOneError = Self.HiddenTwoDelta.dot(Self.Weights2.T)
        Self.HiddenOneDelta = Self.HiddenOneError * Self.SigmoidPrime(Self.Activated1)

        
        
        Self.Weights1 += Self.LearningRate * (X.T.dot(Self.HiddenOneDelta))
        Self.Weights2 += Self.LearningRate * (Self.Activated1.T.dot(Self.HiddenTwoDelta))
        Self.Weights3 += Self.LearningRate * (Self.Activated2.T.dot(Self.HiddenThreeDelta))
        Self.Weights4 += Self.LearningRate * (Self.Activated3.T.dot(Self.PredictionDelta))

        Self.OutputBias += Self.PredictionDelta
        Self.HiddenOneBias += Self.HiddenOneDelta
        Self.HiddenTwoBias += Self.HiddenTwoDelta
        Self.HiddenThreeBias += Self.HiddenThreeDelta
    
    def Sigmoid(Self, S):
        return (1/(1+np.exp(-S)))

    def SigmoidPrime(Self, S):
        return (S*(1-S))

    def Train(Self,X,Y):
        Prediction = Self.ForwardPropogation(X)
        Self.BackwardPropogation(X,Y,Prediction)


    def MeanSquaredError(Self,Prediction, Actual):
         return sum(sum((Prediction - Actual)**2)/len(Prediction))




TrainingFile = open("train.csv","r")
TrainingLines = TrainingFile.readlines()
TrainingFile.close()




Labels = []
for x in range(1,42000):
    Temp = TrainingLines[x].split(",")
    Labels.append(Temp[0])


Pixels = []
for x in range(1,42000):
    Temp = TrainingLines[x].split(",")
    Temp = Temp[1:]
    Pixels.append(Temp)

for x in range(len(Pixels)):
    Pixels[x] = list((map(int, Pixels[x])))
    for y in range(len(Pixels[x])):
        if Pixels[x][y] != 0:
            Pixels[x][y] = 1


'''
Draw the numbers


for x in range(0,100):
    if int(Labels[x]) == 7:
        Pixel = Pixels[x]
        for x in range(0,28):
            String = ""
            for y in range(0,28):
                if Pixel[y + x*28] != 0:
                    String += "@"
                else:
                    String +=" "
            print(String)
        print("\n")

'''
'''
np.random.seed(0)
bot = NeuralNetwork()
bot.Train(np.array([1,1,1]),[1])
'''



AI = NeuralNetwork()
Epochs = 3
#==================#
# Training network #
#==================#
for x in range(Epochs):
    print("Epoch:",x+1)
    for x in range(len(Labels)):
        if int(Labels[x]) == 0:
            Wanted = [1,0,0,0,0,0,0,0,0,0]
        elif int(Labels[x]) == 1:
            Wanted = [0,1,0,0,0,0,0,0,0,0]
        elif int(Labels[x]) == 2:
            Wanted = [0,0,1,0,0,0,0,0,0,0]
        elif int(Labels[x]) == 3:
            Wanted = [0,0,0,1,0,0,0,0,0,0]
        elif int(Labels[x]) == 4:
            Wanted = [0,0,0,0,1,0,0,0,0,0]
        elif int(Labels[x]) == 5:
            Wanted = [0,0,0,0,0,1,0,0,0,0]
        elif int(Labels[x]) == 6:
            Wanted = [0,0,0,0,0,0,1,0,0,0]
        elif int(Labels[x]) == 7:
            Wanted = [0,0,0,0,0,0,0,1,0,0]
        elif int(Labels[x]) == 8:
            Wanted = [0,0,0,0,0,0,0,0,1,0]
        else:
            Wanted = [0,0,0,0,0,0,0,0,0,1]
        State = np.reshape(Pixels[x],[1,784])
        AI.Train(State,Wanted)
        if x % 1000 == 0:
            print(x)
    
np.save("AWeight1", AI.Weights1)
np.save("AWeight2", AI.Weights2)
np.save("AWeight3", AI.Weights3)
np.save("AWeight4", AI.Weights4)
np.save("AHiddenOneBias", AI.HiddenOneBias)
np.save("AHiddenTwoBias", AI.HiddenTwoBias)
np.save("AHiddenThreeBias", AI.HiddenThreeBias)
np.save("AOutputBias", AI.OutputBias)

Percent = 0
Total = 0
for x in range(100):
    Prediction = np.argmax(AI.ForwardPropogation(np.reshape(Pixels[x],[1,784])))
    if int(Labels[x]) == int(Prediction):
        Percent += 1
    Total += 1
print((Percent/Total)*100)
    


