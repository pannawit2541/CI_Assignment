import numpy as np
import pandas as pd


class Particle_of_swarm(object):
    def __init__(self, hiddenSize, inputSize, outputSize):
        # initiate layers
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.hiddenSize = hiddenSize

        layers = [self.inputSize] + self.hiddenSize + [self.outputSize]

        # initiate positions
        positions = []
        for i in range(len(layers)-1):
            p = np.random.rand(layers[i], layers[i+1])
            positions.append(p)
        self.positions = positions
        self.positions_best = positions

        velocitys = []
        for i in range(len(layers) - 1):
            v = np.random.rand(layers[i], layers[i+1])
            velocitys.append(v)
        self.velocitys = velocitys

        self.pbest = -1

    def feedForward(self, X):
        Output_node = X
        for i, p in enumerate(self.positions):
     
            Output_node = np.dot(Output_node, p)

        return Output_node

    def object_funct(self, X, Y):

        sum_err = 0

        for j, input in enumerate(X):

            target = Y[j]
            output = self.feedForward(input)

            sum_err += self._mae(target, output)

        self.fx = (1/(sum_err+1))
        return self.fx

    def _mae(self, target, output):
        return np.average(abs(target - output))


def _readfile(file):

    # ----- Clean NaN Values -----
    df = pd.read_csv(file)
    df = df.fillna(method='ffill')

    # ----- Create Features -----
    X = df[['PT08.S1(CO)', 'PT08.S2(NMHC)', 'PT08.S3(NOx)',
            'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'RH', 'AH']].copy(deep=False)
    X.drop(X.tail(240).index, inplace=True)

    # ----- Create Desired outputs -----
    label = df[['C6H6(GT)']].copy(deep=False)

    Y_10Day = label.iloc[240:, :].reset_index(drop=True)
    Y_10Day.rename(columns={"C6H6(GT)": "C6H6(GT)_10"}, inplace=True)

    Y_5Day = label.iloc[120:, :].reset_index(drop=True)
    Y_5Day.drop(Y_5Day.tail(120).index, inplace=True)
    Y_5Day.rename(columns={"C6H6(GT)": "C6H6(GT)_5"}, inplace=True)

    Y = pd.concat([Y_5Day, Y_10Day], axis=1)

    Input = X.to_numpy()
    Output = Y.to_numpy()
    return Input, Output


Input, Output = _readfile('data/AirQualityUCI.csv')

particles = []
num_of_particle = 1

for i in range(0, num_of_particle):
    par = Particle_of_swarm([4], 8, 2)
    particles.append(par)

# print(particles[0].positions[0])

for i in range(10):
    print("===================== ",i)
    for p in particles:
        fx = p.object_funct(Input[0], Output[0])
        # check pbest

        if fx < p.pbest:
            p.pbest = fx
            p.positions_best = p.positions.copy()

        # update velocity
        for i in range(0, len(p.velocitys)):
            print("v",p.velocitys[i].shape)
            print("pb",p.positions_best[i].shape)
            print("p",p.positions[i].shape)
            p.velocitys[i] = p.velocitys[i] + 2.2 * \
                (p.positions_best[i]-p.positions[i])

        # update position

        p.positions = p.positions + p.velocitys


# print(particles[0].positions[0])
