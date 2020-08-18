import numpy as np
import copy
import sys


class NeuralNetwork(object):
    def __init__(self, hiddenSize, inputSize, outputSize):
        # initiate layers
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.hiddenSize = hiddenSize

        layers = [self.inputSize] + self.hiddenSize + [self.outputSize]

        # initiate weights
        weights = []
        for i in range(len(layers)-1):
            w = 2*np.random.rand(layers[i], layers[i+1])-1
            weights.append(w)
        self.weights = weights

        # initiate bias
        bias = []
        for i in range(len(layers)-1):
            b = np.random.rand(layers[i+1])
            bias.append(b)
        self.bias = bias

        # initiate activations
        activations = []
        for i in range(len(layers)):
            a = np.zeros(layers[i])
            activations.append(a)
        self.activations = activations

        derivatives = []
        for i in range(len(layers) - 1):
            d = np.zeros((layers[i], layers[i + 1]))
            derivatives.append(d)
        self.derivatives = derivatives
        self.derivatives_old = copy.deepcopy(self.derivatives)

    def sigmoid(self, s, deriv=False):
        if (deriv == True):
            return s * (1-s)
        return 1/(1 + np.exp(-s))

    def feedForward(self, X):
        activations = X
        self.activations[0] = X
        for i, w in enumerate(self.weights):
            # calculate NN_input
            v = np.dot(activations, w)
            # calculate the activations
            b = self.bias[i]
            activations = self.sigmoid(v+b)
            self.activations[i+1] = activations
        return activations

    def backPropagate(self, error):
        for i in reversed(range(len(self.derivatives))):

            # get activation for previous layer
            activations = self.activations[i+1]

            # apply sigmoid derivative function
            delta = error * self.sigmoid(activations, deriv=True)
            

            # reshape delta as to have it as a 2d array
            delta_re = delta.reshape(delta.shape[0], -1).T

            # get activations for current layer
            current_activations = self.activations[i]

            # reshape activations as to have them as a 2d column matrix
            current_activations = current_activations.reshape(
                current_activations.shape[0], -1)

            # save derivative after applying matrix multiplication
            self.derivatives[i] = np.dot(current_activations, delta_re)
            #print(self.derivatives[i].shape)

            # backpropogate the next error
            error = np.dot(delta, self.weights[i].T)

    def train(self, X, Y, epochs, learning_rate,momentumRate):
        # now enter the training loop
        for i in range(epochs):
            sum_errors = 0
            
            # iterate through all the training data
            for j, input in enumerate(X):
                target = Y[j]

                # activate the network!
                output = self.feedForward(input)

                error = target - output
                #print(output, " - ", target)
                if i == 0 :
                    self.backPropagate(error)
                    self.derivatives_old = copy.deepcopy(self.derivatives)
                    print(self.derivatives_old[0]-self.derivatives[0])
                else:
                    self.derivatives_old = copy.deepcopy(self.derivatives)
                    self.backPropagate(error)
                # now perform gradient descent on the derivatives
                # (this will update the weights
                self.gradient_descent(learning_rate,momentumRate)

                # keep track of the MSE for reporting later
                sum_errors += self._mse(target, output)

            # Epoch complete, report the training error
            print("Error: {} at epoch {}".format(round(sum_errors / len(X) , 5), i+1))
        self.sum_all_err = sum_errors/len(X)
        print("Training complete! : ",sum_errors/len(X))
        print("=====")

    def gradient_descent(self, learningRate=1,momentumRate=1):
        # update the weights by stepping down the gradient
        for i in range(len(self.weights)):
            weights = self.weights[i]
            bias = self.bias[i]
            derivatives = self.derivatives[i]
            derivatives_old  = self.derivatives_old[i]
            delta = (derivatives * -learningRate) + ((derivatives-derivatives_old)*momentumRate)
            weights += delta
            delta = np.dot(delta.T,np.ones(delta.T.shape[1]))
            bias += delta
            

    def _mse(self, target, output):
        return np.average((target - output) ** 2)

def convert_output(max,min,data,flag = False):
    if flag == True:
        return  ( data*(max-min)) + min
    return  (data - min) / (max - min)

def convert_input(data):
    mean = data.mean(axis = 0)
    sd = data.std(axis = 0)
    return (data- mean)/ sd

def Preprocessing():

        # import data set
        with open("Flood_dataset.txt", "r") as f:
            content = f.readlines()
        del content[0:3]

        # split data set
        data = []
        for X in content:
            data.append(X.split())

        # convert data to list 
        output = [list(map(int, X[8:])) for X in data]
        input = [list(map(int, X[:8])) for X in data]

        input = np.array(input)
        output = np.array(output)
        #print(output)
        inputSize = input.shape[1]
        outputSize = output.shape[1]

        return input, output, inputSize, outputSize

def Preprocessing_Cross():
    # import data set
        with open("cross.pat", "r") as f:
            content = f.readlines()
        del content[0:3]

        # split data set
        output = []
        input = []
        for i,X in enumerate(content):
            if X[0] != 'p':
                if (i+1)%3 == 0:
                    a,b = X.split()
                    output.append([int(a),int(b)])
                else:
                    a,b = X.split()
                    input.append([float(a),float(b)])
        input = np.array(input)
        output = np.array(output)
        #print(input)
        inputSize = input.shape[1]
        outputSize = output.shape[1]

        return input, output, inputSize, outputSize
  

def cross_validations_split(dataset,output_dataset,folds):
    fold_size = int(dataset.shape[0] * folds/100)
    k = 0
    index = []
    for i in range(1,folds+1):
        if i < folds:
            index.append([k,i*fold_size])
        else:
            index.append([k,dataset.shape[0]])
        k = i*fold_size
    return index



model = 'B'
'''
while(True):
    print(' -- Please press one to training --> A or B -- ')
    print(' -- A : flood_dataset , B : cross_dataset -- ')
    print(' -- q : to exit -- ')
    model = input()
    print(type(model))
    if model == 'q' or model == 'Q':
        sys.exit()
    elif model == 'A' or model == 'a' or model == 'B' or model == 'b' or  model == 'q' :
        break
'''

print("What Size of Hidden layer Neural Network ?")
print(" -- Example : '4-2-2' --")
print(" -- Hidden layer have 3 layers and 4,2,2 nodes respectively -- ")
'''
hiddenSizeStr = input('Size of Hidden layer : ')
learningRate = input('Learning Rate : ')
learningRate = float(learningRate)
momentumRate = input('Momentum Rate : ')
momentumRate = float(momentumRate)
epochs = input('Epochs : ')
epochs = int(epochs)
hiddenSize = hiddenSizeStr.split("-")
hiddenSize = list(map(int, hiddenSize))
'''
learningRate = 0.8
momentumRate = 0.2
epochs = 1000
hiddenSize = [3]
sum_avg_train = 0
sum_avg_predict = 0

if model == 'A' or model == 'a':
    X, Y, inputSizeX, outputSizeY = Preprocessing()
    max,min = Y.max(),Y.min()
    y = convert_output(max,min,Y)
    x = convert_input(X)
    index_flood = cross_validations_split(x,y,10)
    NN_flood = NeuralNetwork(hiddenSize, inputSizeX, outputSizeY)
    for a,b in index_flood:
        inTest = np.concatenate((x[:a],x[b+1:]))
        outTest = np.concatenate((y[:a],y[b+1:]))
        NN_flood.train(inTest, outTest, epochs , learningRate,momentumRate)
        sum_avg_train += NN_flood.sum_all_err
        sum_avg_predict += np.sum(NN_flood._mse(NN_flood.feedForward(x[a:b,:]),y[a:b,:]),axis=0)
else:
    A, B, inputSizeA, outputSizeB = Preprocessing_Cross()
    index_cross = cross_validations_split(A,B,10)
    NN_cross = NeuralNetwork(hiddenSize, inputSizeA, outputSizeB)
    for a,b in index_cross:
        inTest = np.concatenate((A[:a],A[b+1:]))
        outTest = np.concatenate((B[:a],B[b+1:]))
        NN_cross.train(inTest, outTest, epochs , learningRate,momentumRate)
        sum_avg_train += NN_cross.sum_all_err
        sum_avg_predict += np.sum(NN_cross._mse(NN_cross.feedForward(A[a:b,:]),B[a:b,:]),axis=0)

predict = NN_cross.feedForward(A)
label_data = []
label_predict = []
for i in range(A.shape[0]):
    if predict[i][0] > predict[i][1]:
        label_predict.append(0)
    elif predict[i][0] <= predict[i][1]:
        label_predict.append(1)
    if B[i][0] > B[i][1]:
        label_data.append(0)
    elif B[i][0] <= B[i][1]:
        label_data.append(1)

a1 = 0
a2 = 0
b1 = 0
b2 = 0
for i in range(len(label_data)):
    if label_data[i] == 0:
        if label_predict[i] != label_data[i]:
            a2 += 1
        else:
            a1 += 1
    else:
        if label_predict[i] != label_data[i]:
            b1 += 1
        else:
            b2 += 1

print("== 0  ==== 1")
print("0 = ",a1," == ",a2)
print("1 = ",b1," == ",b2)
