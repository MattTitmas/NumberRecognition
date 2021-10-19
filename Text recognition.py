import numpy as np
import random
import sys


class NeuralNetwork(object):
    def __init__(Self): # Constructor Field
        #784,512,256,128,10
        Self.InputLayers = 784 # Number of pixels per number in the MNIST dataset
        Self.HiddenLayerOne = 512
        Self.HiddenLayerTwo = 256 # Slowly reduce the number of nodes per layer
        Self.HiddenLayerThree = 128
        Self.OutputLayer = 10 # Possible numbers 0-10
        Self.LearningRate = 0.4

        Self.Weights1 = np.random.randn(Self.InputLayers,Self.HiddenLayerOne) # Start by generating random weights (Side Effect: Every time you run the program you'll get different final weights)
        Self.Weights2 = np.random.randn(Self.HiddenLayerOne,Self.HiddenLayerTwo)
        Self.Weights3 = np.random.randn(Self.HiddenLayerTwo,Self.HiddenLayerThree)
        Self.Weights4 = np.random.randn(Self.HiddenLayerThree,Self.OutputLayer)


        Self.HiddenOneBias = np.random.rand(1, Self.HiddenLayerOne) # Randomly generate biases
        Self.HiddenTwoBias = np.random.rand(1, Self.HiddenLayerTwo)
        Self.HiddenThreeBias = np.random.rand(1, Self.HiddenLayerThree)
        Self.OutputBias = np.random.rand(1, Self.OutputLayer)


    def ForwardPropogation(Self, Input): # Propogate an input through the neural network to produce an output
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

    def BackwardPropogation(Self, X, Y, Prediction): # Propogate a result backwords through the neural network using gradient descent to find the Error Delta of each weight
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
    
    def Sigmoid(Self, S): # Sigmoid normalises values between 0 and 1
        return (1/(1+np.exp(-S)))

    def SigmoidPrime(Self, S): # Derivation of sigmoid, used for back propogation
        return (S*(1-S))

    def Train(Self,X,Y): # Trains the neural network
        Prediction = Self.ForwardPropogation(X)
        Self.BackwardPropogation(X,Y,Prediction)


    def MeanSquaredError(Self,Prediction, Actual): # Calculate overall error of Neural Network
         return sum(sum((Prediction - Actual)**2)/len(Prediction))




TrainingFile = open("train.csv","r")
TrainingLines = TrainingFile.readlines()
TrainingFile.close()




Labels = [] # Ensuring data is easily readable for the program
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



AI = NeuralNetwork()
Epochs = 3 # Can't be too high - risk overtraining the Neural Network
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
            print(x) # Can be commented out - allows the user to track how far into the dataset the program is
    
np.save("AWeight1", AI.Weights1)
np.save("AWeight2", AI.Weights2)
np.save("AWeight3", AI.Weights3)
np.save("AWeight4", AI.Weights4)
np.save("AHiddenOneBias", AI.HiddenOneBias)
np.save("AHiddenTwoBias", AI.HiddenTwoBias)
np.save("AHiddenThreeBias", AI.HiddenThreeBias)
np.save("AOutputBias", AI.OutputBias) # Save weights as numpy files

Percent = 0
Total = 0
for x in range(100):
    Prediction = np.argmax(AI.ForwardPropogation(np.reshape(Pixels[x],[1,784])))
    if int(Labels[x]) == int(Prediction):
        Percent += 1
    Total += 1
print((Percent/Total)*100) # Prints the percent of the dataset that is guessed correctly TODO - download test set and use that instead
   


