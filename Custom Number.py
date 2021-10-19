from PIL import Image
import numpy as np

class NeuralNetwork(object):
    def __init__(Self):
        Self.Weights1 = np.load("Weight1.npy")
        Self.Weights2 = np.load("Weight2.npy")
        Self.Weights3 = np.load("Weight3.npy")
        Self.Weights4 = np.load("Weight4.npy")

    def ForwardPropogation(Self, Input):
        Self.DotProduct1 = np.dot(Input, Self.Weights1)
        Self.Activated1 = Self.Sigmoid(Self.DotProduct1)     # Hidden Layer One
        Self.DotProduct2 = np.dot(Self.Activated1, Self.Weights2)
        Self.Activated2 = Self.Sigmoid(Self.DotProduct2)     # Hidden Layer Two
        Self.DotProduct3 = np.dot(Self.Activated2, Self.Weights3)
        Self.Activated3 = Self.Sigmoid(Self.DotProduct3)     # Hidden Layer Three
        Self.DotProduct4 = np.dot(Self.Activated3, Self.Weights4)
        Prediction = Self.Sigmoid(Self.DotProduct4)
        return Prediction                                     # Output Layer

    def Sigmoid(Self, S):
        return (1/(1+np.exp(-S)))

AI = NeuralNetwork()

def get_pixel(image, i, j):
    # Inside image bounds?
    width, height = image.size
    if i > width or j > height:
      return None
    pixel = image.getpixel((i, j))
    return pixel

NewPic = []    
Img = Image.open("Custom Number.png")
Pixels = Img.load
for i in range(28):
    String = ""
    Array = []
    for j in range(28):
        pixel = get_pixel(Img, j, i)
        if pixel[0] == 255:
            NewPixel = 0
        else:
            NewPixel = 1
        Array.append(NewPixel)
    if sum(Array) != 0:
        NewPic.append(Array)

Highest = 14
Lowest = 14
for x in range(len(NewPic)):
    for y in range(len(NewPic[x])):
        if NewPic[x][y] == 1 and Lowest > y:
            Lowest = y
        if NewPic[x][y] == 1 and Highest < y:
            Highest = y

Highest = 27-Highest
for x in range(len(NewPic)):
    del NewPic[x][:Lowest]
    NewPic[x] = NewPic[x][:-Highest]


Change = 1

while len(NewPic[1]) != 28:
    if Change == 1:
        for x in range(len(NewPic)):
            NewPic[x].append(0)
            Change = -1
    elif Change == -1:
        for x in range(len(NewPic)):
            NewPic[x].insert(0,0)
            Change = 1


while len(NewPic) != 28:
    if Change == 1:
        NewPic.insert(0,[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        Change = -1
    elif Change == -1:
        NewPic.append([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        Change = 1

        

for x in range(len(NewPic)):
    String = ""
    for y in range(len(NewPic[x])):
        if NewPic[x][y] == 0:
            String += " "
        else:
            String +="@"
    print(String)

FinalPic = []
for x in range(len(NewPic)):
    for y in range(len(NewPic[x])):
        FinalPic.append(NewPic[x][y])



Prediction = np.argmax(AI.ForwardPropogation(np.reshape(FinalPic,[1,784])))
print("Classified as:",Prediction)
