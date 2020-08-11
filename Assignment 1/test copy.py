import numpy as np


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

    def train(self, X, Y, epochs, learning_rate):
        # now enter the training loop
        for i in range(epochs):
            sum_errors = 0

            # iterate through all the training data
            for j, input in enumerate(X):
                target = Y[j]

                # activate the network!
                output = self.feedForward(input)

                error = target - output

                self.backPropagate(error)

                # now perform gradient descent on the derivatives
                # (this will update the weights
                self.gradient_descent(learning_rate)

                # keep track of the MSE for reporting later
                sum_errors += self._mse(target, output)

            # Epoch complete, report the training error
            print("Error: {} at epoch {}".format(sum_errors / len(X), i+1))

        print("Training complete!")
        print("=====")

    def gradient_descent(self, learningRate=1):
        # update the weights by stepping down the gradient
        for i in range(len(self.weights)):
            weights = self.weights[i]
            derivatives = self.derivatives[i]
            weights += derivatives * learningRate

    def _mse(self, target, output):
        return np.average((target - output) ** 2)

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

        # Normalization by Standard score
        input = np.array(input)
        input_mean = input.mean(axis = 0)
        input_sd = input.std(axis = 0)
        
        output = np.array(output)
        output_mean = output.mean(axis = 0)
        output_sd = output.std(axis = 0)

        print((input[0,:]-input_mean)/input_sd)
        inputSize = input.shape[1]
        outputSize = output.shape[1]

        return input, output, inputSize, outputSize


#if __name__ == "__main__":

X, Y, inputSize, outputSize = Preprocessing()

print("What Size of Hidden layer Neural Network ?")
print(" -- Example : '4-2-2' --")
print(" -- Hidden layer have 3 layers and 4,2,2 nodes respectively -- ")
#hiddenSizeStr = input('Size of Hidden layer : ')


hiddenSizeStr = '2'
hiddenSize = hiddenSizeStr.split("-")
hiddenSize = list(map(int, hiddenSize))
#NN = NeuralNetwork(hiddenSize, inputSize, outputSize)

#NN.train(X, Y, 50, 0.1)