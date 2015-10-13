
import numpy as npy

class neuralnetwork(object):
    def __init__(self, X, Y, parameters):
        """
        :param X: input vector
        :param Y: output vector
        :param parameters:
        """
        self.X = X
        self.Y = Y
        self.layers = len(parameters)

        # number of neurons without bias neurons in each layer.
        self.sizes = [layer[0] for layer in parameters]

        # activation functions for each layer.
        self.fs = [layer[1] for layer in parameters]

        # derivatives of activation functions for each layer.
        self.fprimes = [layer[2] for layer in parameters]

        # List of weight matrices taking the output of one layer to the input of the next.
        self.weights = []

        # Bias vector for each layer.
        self.biases = []

        # Input vector for each layer.
        self.inputs = []

        # Output vector for each layer.
        self.outputs = []

        # Vector of errors at each layer.
        self.errors = []

        # The estimated output
        self.estimates = []

        # create a neural network
        self.build_network()

    def build_network(self):
        # We initialize the weights randomly, and fill the other vectors with 1s.
        for layer in range(self.layers-1):
            n = self.sizes[layer]  # current layer
            m = self.sizes[layer+1]  # next layer
            self.weights.append(npy.random.normal(0, 1, (m, n)))
            self.biases.append(npy.random.normal(0, 1, (m, 1)))

            self.inputs.append(npy.zeros((n,1)))
            self.outputs.append(npy.zeros((n,1)))
            self.errors.append(npy.zeros((n, 1)))

        # There are only n-1 weight matrices, so we do the last case separately.
        n = self.sizes[-1]
        self.inputs.append(npy.zeros((n,1)))
        self.outputs.append(npy.zeros((n,1)))
        self.errors.append(npy.zeros((n, 1)))

    def feedforward(self, x):
        # Propagates the input from the input layer to the output layer.
        k = len(x)
        x.shape = (k, 1)
        self.inputs[0] = x
        self.outputs[0] = x
        for i in range(1, self.layers):
            self.inputs[i] = self.weights[i-1].dot(self.outputs[i-1])  # TODO: bias neurons can be added by adding "self.biases[i-1]" if necessary
            self.outputs[i] = self.fs[i](self.inputs[i])
        return self.outputs[-1]

    def update_weights(self,x,y):
        #Update the weight matrices for each layer based on a single input x and target y.
        output = self.feedforward(x)
        fp = self.fprimes[-1](self.outputs[-1])
        yy = npy.reshape(y, (2, 1))
        ee = output - yy
        self.errors[-1] = fp * ee

        n=self.layers-2
        for i in xrange(n,0,-1):
            self.errors[i] = self.fprimes[i](self.inputs[i]) * self.weights[i].T.dot(self.errors[i+1])
            # w_i_updated = w_i - \eta * E_j * O_i
            self.weights[i] = self.weights[i] - self.learning_rate * npy.outer(self.errors[i+1], self.outputs[i])
            # If bias neuron is required, the code line below can be uncommented.
            # self.biases[i] = self.biases[i] - self.learning_rate*self.errors[i+1]

        delta = self.learning_rate * npy.outer(self.errors[1], self.outputs[0])
        self.weights[0] = self.weights[0] - delta

        # If bias neuron is required, the code line below can be uncommented.
        # self.biases[0] = self.biases[0] - self.learning_rate*self.errors[1]

    def train(self, n_iter, learning_rate = 1):
        # update the weights after comparing each input in X with Y
        # repeat this process n_iter times.
        self.learning_rate = learning_rate
        number_of_rows_X = self.X.shape[0]
        for repeat in range(n_iter):
            temp = []
            for row in range(number_of_rows_X):
                x = self.X[row]
                y = self.Y[row]
                self.update_weights(x, y)
                temp.append(npy.reshape(self.outputs[-1], (1, 2)))
            self.estimates.append(temp)

    def predict_x(self, x):
        return self.feedforward(x)

    def predict(self, X):
        n = len(X)
        m = self.sizes[-1]
        ret = npy.ones((n,m))
        for i in range(len(X)):
            temp = self.feedforward(X[i])
            ret[i, :] = npy.reshape(temp, (1, 2))
        return ret