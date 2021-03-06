import numpy as np
import time
import copy 


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
            w = np.random.rand(layers[i], layers[i+1])
            weights.append(w)
        self.weights = weights

        # initiate weights_t-1

        self.weights_befoe  = copy.deepcopy(self.weights)
        self.weights_next  = copy.deepcopy(self.weights)


        # initiate bias
        bias = []
        for i in range(len(layers)-1):
            b = np.random.rand(layers[i+1])
            bias.append(b)
        self.bias = bias        

        # initiate bias_t-1
        self.bias_before  = copy.deepcopy(self.bias)
        self.bias_next  = copy.deepcopy(self.bias)


        # initiate activations
        activations = []
        for i in range(len(layers)):
            a = np.zeros(layers[i])
            activations.append(a)
        self.activations = activations

        # initiate gradient_b
        derivatives_b = []
        for i in range(len(layers) - 1): 
            d = np.zeros(layers[i+1])
            derivatives_b.append(d)
        self.derivatives_b = derivatives_b

        # initiate gradient_w
        derivatives_w = []
        for i in range(len(layers) - 1):
            d = np.zeros((layers[i], layers[i + 1]))
            derivatives_w.append(d)
        self.derivatives_w = derivatives_w

        # initiate average_err
        self.average_err = 0

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
        for i in reversed(range(len(self.derivatives_w))):

            # get activation for previous layer
            activations = self.activations[i+1]

            # apply sigmoid derivative function
            delta = error * self.sigmoid(activations, deriv=True)

            # reshape delta as to have it as a 2d array
            delta_re = delta.reshape(delta.shape[0], -1).T

            self.derivatives_b[i] = copy.deepcopy(delta)

            # get activations for current layer
            current_activations = self.activations[i]

            # reshape activations as to have them as a 2d column matrix
            current_activations = current_activations.reshape(
                current_activations.shape[0], -1)

            # save derivative after applying matrix multiplication
            self.derivatives_w[i] = np.dot(current_activations, delta_re)


            # backpropogate the next error
            error = np.dot(delta, self.weights[i].T)

    def train(self, x, y, epochs, learning_rate,momentumRate):
        # now enter the training loop
        for i in range(epochs):
            sum_errors = 0

            # Random data
            data = np.concatenate([x,y] , axis=1)
            np.random.shuffle(data)
            Y = data[:,8:]
            X = data[:,:8]

            # iterate through all the training data
            for j, input in enumerate(X):
                target = Y[j]

                # activate the network!
                output = self.feedForward(input)

                error = target - output
                #print(output, " - ", target)
                self.backPropagate(error)
                # now perform gradient descent on the derivatives
                # (this will update the weights
                
                self.gradient_descent(learning_rate,momentumRate)

                # keep track of the MSE for reporting later
                sum_errors += self._mse(target, output)
          
            # Epoch complete, report the training error
            print("Error: {} at epoch {}".format(round(sum_errors / len(X) , 5), i+1))

        self.average_err = round(sum_errors / len(X) , 5)

        print("Training complete! : ",sum_errors/len(X))
        print("=====")

    def gradient_descent(self, learningRate=1,momentumRate=1):

        # update the weights by stepping down the gradient
        
        for i in range(len(self.weights)):

            weights = self.weights[i]
            weights_befoe = self.weights_befoe[i]
            weights_next = self.weights_next[i]

            bias = self.bias[i]
            bias_before = self.bias_before[i]
            bias_next = self.bias_next[i]

            derivatives_w = self.derivatives_w[i]

            derivatives_b = self.derivatives_b[i]

            weights_next += (derivatives_w * learningRate) + ((weights-weights_befoe)*momentumRate)

            bias_next += (derivatives_b * learningRate) + ((bias-bias_before)*momentumRate)


        self.weights_befoe = copy.deepcopy(self.weights)
        self.weights = copy.deepcopy(self.weights_next)

        self.bias_before = copy.deepcopy(self.bias)
        self.bias = copy.deepcopy(self.bias_next)
     
    def _mse(self, target, output):
        return np.average((target - output) ** 2)

def convert_output(NewMax,NewMin,OldMax,OldMin,OldValue,flag = "convert"):
    OldRange = (OldMax - OldMin)  
    NewRange = (NewMax - NewMin)  
    NewValue = (((OldValue - OldMin) * NewRange) / OldRange) + NewMin
        
    if flag == "convert":
        NewValue = 
    elif flag == "revert":
        NewValue = (OldValue - NewMin) * OldRange / ((OldValue - OldMin) * NewRange)
    else :
        print(" flag must only 'convert' or 'revert' ")
    return  NewValue

def convert_input(data):
    mean = data.mean(axis = 0)
    sd = data.std(axis = 0)
    return (data- mean)/ sd

def Preprocessing_Flood():

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
        data = [list(map(int, X[:9])) for X in data]

        input = np.array(input)
        output = np.array(output)
        max,min = output.max(),output.min()

        Y = convert_output(1,0,max,min,output,flag = "convert")
        X = convert_input(input)

        #data = np.concatenate([X,Y] , axis=1)

        inputSize = input.shape[1]
        outputSize = output.shape[1]

        return X,Y,inputSize, outputSize

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
        inputSize = input.shape[1]
        outputSize = output.shape[1]

        #Y = convert_output()

        return input, output, inputSize, outputSize
  
def cross_validations_split(shape,folds):
    fold_size = int(shape * folds/100)
    k = 0
    index = []
    for i in range(1,folds+1):
        if i < folds:
            index.append([k,i*fold_size])
        else:
            index.append([k,shape])
        k = i*fold_size
    return index


X,Y,inputSize,outputSize = Preprocessing_Flood()
hiddenSize = [2]
print(Y)
print(convert_output())

NN = NeuralNetwork(hiddenSize, inputSize, outputSize)

'''
for a,b in cross_validations_split(X.shape[0],10):
    inTest = np.concatenate((X[:a],X[b+1:]))
    outTest = np.concatenate((Y[:a],Y[b+1:]))
    NN.train(inTest, outTest, 1000 , 0.7  ,0.5)
'''