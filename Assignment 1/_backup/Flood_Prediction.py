import numpy as np
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
            activations = self.sigmoid(v)
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
                if i > 0 :
                    self.derivatives_old = copy.deepcopy(self.derivatives)
                self.backPropagate(error)
                # now perform gradient descent on the derivatives
                # (this will update the weights
                self.gradient_descent(learning_rate,momentumRate)

                # keep track of the MSE for reporting later
                sum_errors += self._mse(target, output)

            # Epoch complete, report the training error
            #print("Error: {} at epoch {}".format(round(sum_errors / len(X) , 5), i+1))
        self.sum_all_err = sum_errors/len(X)
        print("Training complete! : ",sum_errors/len(X),4)
        print("=====")

    def gradient_descent(self, learningRate=1,momentumRate=1):
        # update the weights by stepping down the gradient
        for i in range(len(self.weights)):
            weights = self.weights[i]
            derivatives = self.derivatives[i]
            derivatives_old  = self.derivatives_old[i]
            weights += (derivatives * learningRate) + (derivatives_old*momentumRate)

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


print("What Size of Hidden layer Neural Network ?")
print(" -- Example : '4-2-2' --")
print(" -- Hidden layer have 3 layers and 4,2,2 nodes respectively -- ")
#hiddenSizeStr = input('Size of Hidden layer : ')
#learningRate = input('Learning Rate : ')
#learningRate = float(learningRate)
#momentumRate = input('Momentum Rate : ')
#momentumRate = float(momentumRate)
#epochs = input('Epochs : ')
#epochs = int(epochs)

X, Y, inputSizeX, outputSizeY = Preprocessing()
max,min = Y.max(),Y.min()
y = convert_output(max,min,Y)
x = convert_input(X)

#hiddenSize = hiddenSizeStr.split("-")
#hiddenSize = list(map(int, hiddenSize))
index = cross_validations_split(x,y,10)

hiddenSize = [4,4,4]
learningRate_all = [0.1,0.2]
momentumRate = 0.5
epochs = 10
avg_predict = []
avg_traing = []

for learningRate in learningRate_all:
    NN = NeuralNetwork(hiddenSize, inputSizeX, outputSizeY)
    print(learningRate)
    sum_avg_train = 0
    sum_avg_predic = 0
    i = 1
    for a,b in index:
        print("---------------- 10-folds Cross validation No. : ", i ," ----------------",)
        inTest = np.concatenate((x[:a],x[b+1:]))
        outTest = np.concatenate((y[:a],y[b+1:]))
        NN.train(inTest, outTest, epochs, learningRate ,momentumRate)
        sum_avg_train += NN.sum_all_err
        
        print("Predict data : [",a,b,"]")
        #print(np.mean(NN.sum_all_err))
        #print(round(np.sum(NN._mse(NN.feedForward(x[a:b,:]),y[a:b,:]),axis=0),4))
        print(NN.sum_all_err)
        print("------------------------")
        sum_avg_predic += np.sum(NN._mse(NN.feedForward(x[a:b,:]),y[a:b,:]),axis=0)
        
        #print(sum_avg_predic)
        #print(np.sum(NN._mse(NN.feedForward(x[a:b,:]),y[a:b,:]),axis=0)) 
        i+=1
    avg_traing.append(sum_avg_train)
    avg_predict.append(sum_avg_predic)


#for i in range(len(learningRate_all)):
#    print(avg_predict/10)
#    print(avg_traing/10)
#    print("==================")
    

