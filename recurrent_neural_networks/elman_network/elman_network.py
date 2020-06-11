import numpy as np
import matplotlib.pyplot as plt
from .utils import create_xor_data
from tqdm import tqdm


class ElmanNetwork:
    def __init__(self, num_inputs, num_hidden, num_contextual, num_outputs):
        assert num_hidden >= num_contextual, 'Contextual units exceed hidden'
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_contextual = num_contextual
        self.num_outputs = num_outputs
        self.weights_input = np.random.normal(size=(self.num_hidden, self.num_inputs + self.num_contextual))
        self.bias_input = np.random.normal(size=(self.num_hidden, 1))
        self.weights_hidden = np.random.normal(size=(self.num_outputs, self.num_hidden))
        self.bias_hidden = np.random.normal(size=(self.num_outputs, 1))

    def __repr__(self):
        return f"ElmanNetwork(Inputs={self.num_inputs}, Hidden={self.num_hidden}, Contextual={self.num_contextual}, Outputs={self.num_outputs})"

    def sigmoid(self, a):
        return 1 / (1 + np.exp(-a))

    def forward_pass(self, X):
        self.H1 = self.sigmoid(np.dot(self.weights_input, X) + self.bias_input)
        self.O = self.sigmoid(np.dot(self.weights_hidden, self.H1) + self.bias_hidden)
        if self.O.shape == (1, 1):
            self.O = self.O[0][0]
        return self.O

    def backpropagation(self, X, y, learning_rate):
        error = self.O - y

        delta_output = error * self.O * (1 - self.O)
        self.weights_hidden -= learning_rate * np.dot(delta_output, self.H1.T)
        self.bias_hidden -= learning_rate * delta_output

        delta_input = np.dot(self.weights_hidden.T, delta_output) * self.H1 * (1 - self.H1)
        self.weights_input -= learning_rate * np.dot(delta_input, X.T)
        self.bias_input -= learning_rate * delta_input

    def train(self, inputs, outputs, learning_rate, passes):
        
        for _ in tqdm(range(passes)):
            X = 0.5 * np.ones((self.num_inputs + self.num_contextual, 1))
            for x, y in zip(inputs, outputs):
                x = x.reshape(self.num_inputs, 1)
                y = y.reshape(self.num_outputs, 1)
                X[:self.num_inputs] = x
                self.forward_pass(X)
                self.backpropagation(X, y, learning_rate)
                X[self.num_inputs:] = self.H1

    def predict(self, inputs, outputs, period=None):
        if period:
            squared_error = np.zeros((1, period))
        else:
            squared_error = np.zeros((1, len(outputs)))
        X = 0.5 * np.ones((self.num_inputs + self.num_contextual, 1))
        for i in range(len(outputs)):
            x = inputs[i].reshape(self.num_inputs, 1)
            y = outputs[i].reshape(self.num_outputs, 1)
            X[:self.num_inputs] = x
            self.forward_pass(X)
            X[self.num_inputs:] = self.H1
            squared_error[0, i] = ((self.O - y)**2)
        return squared_error



if __name__ == '__main__':
    np.random.seed(4321)
    X_train, y_train = create_xor_data(250)
    net = ElmanNetwork(1, 2, 2, 1)
    passes = 600
    net.train(X_train, y_train, 0.01, passes)
    # Test
    num_cycles = 1200
    cycles = np.zeros((num_cycles, 12))
    for i in tqdm(range(num_cycles)):
        X_test, y_test = create_xor_data(1)
        cycles[i, :] = net.predict(X_test, y_test)
    mean_cycles = np.mean(cycles, axis=0)
    plt.plot(range(1, 13), mean_cycles)
    plt.plot([2, 5, 8, 11], mean_cycles[[1, 4, 7, 10]], 'r.', markersize=10)
    plt.xlabel('Cycle', fontsize=12)
    plt.ylabel('Error', fontsize=12)
    plt.xlim([0, 13])
    plt.xticks(range(0,14))
    plt.ylim([0, 0.5])
    plt.show()
