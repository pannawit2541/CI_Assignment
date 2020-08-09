import numpy as np

class NeuralNetwork(object):
    def __init__(self, hiddenSize,inputSize,outputSize,learningRate):
        # initiate layers
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.hiddenSize = hiddenSize

        layers = [self.inputSize] + self.hiddenSize + [self.outputSize]
        

        # initiate weights
        weights = []
        for i in range(len(layers)-1):
            w = 2*np.random.rand(layers[i],layers[i+1])-1
            weights.append(w)
        self.weights = weights
        
       
        # initiate activations
        activations = []
        for i in range(len(layers)):
            a = np.zeros(layers[i])
            activations.append(a)
        self.activations = activations

            
    def sigmoid(self, s, deriv=False):
        if (deriv == True):
            return s * (1-s)
        return 1/(1 + np.exp(-s))

    def feedForward(self, X):
        activations = X
        self.activations[0] = X
        for i,w in enumerate(self.weights):
            # calculate NN_input
            v = np.dot(w.T,activations)
            # calculate the activations
            activations = self.sigmoid(v)
            self.activations[i+1] = activations
        return activations


def Preprocessing():
    with open("Flood_dataset.txt", "r") as f:
        content = f.readlines()
    del content[0:3]

    data = []
    for X in content:
        data.append(X.split())
    output = [list(map(int, X[8:])) for X in data]
    
    input = [list(map(int, X[:7])) for X in data]
    

    input = np.array(input)
    output = np.array(output)
    inputSize = input.shape[1]
    outputSize = output.shape[1]
    input = input.T
    output = output.T
    return input,output,inputSize,outputSize

X,Y,inputSize,outputSize = Preprocessing()

print("What Size of Hidden layer Neural Network ?")
print(" -- Example : '4-2-2' --")
print(" -- Hidden layer have 3 layers and 4,2,2 nodes respectively -- ")
#hiddenSizeStr = input('Size of Hidden layer : ')


hiddenSizeStr = '2'
hiddenSize = hiddenSizeStr.split("-")
hiddenSize = list(map(int, hiddenSize))
NN = NeuralNetwork(hiddenSize,inputSize,outputSize,0.5)

print(NN.feedForward(X[:,0]))



