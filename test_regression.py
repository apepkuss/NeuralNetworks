
from emailclassifier import neuralnetwork
#compute logistic using precomputed exp.
from scipy.special import expit
import numpy as npy
import csv

def test(n_hidden_unites, rates, plots=False):

    with open('train2.csv', 'rb') as test1:
        reader = csv.reader(test1)
        data = [data for data in reader]

    train_data = npy.array(data, dtype=float)
    X = train_data[:, 0:5]
    Y = train_data[:, 5:]

    # draw a diagram for testing inputs
    # import matplotlib.pyplot as plt
    # plt.plot(X, Y, 'o')

    # create a neural network
    param = ((5, 0, 0), (n_hidden_unites, expit, logistic_prime), (2, expit, logistic_prime))

    predictions = []
    percentage = []
    for rate in rates:
        network = neuralnetwork(X, Y, param)  # create a neural network
        network.train(500, learning_rate=rate)  # train the network

    with open('train2.csv', 'rb') as test1:
        reader = csv.reader(test1)
        data = [data for data in reader]

    train_data = npy.array(data, dtype=float)
    X_predict = train_data[:, 0:5]
    Y_predict = train_data[:, 5:]

    prediction = network.predict(X_predict)
    predictions.append([rate, prediction])
    percentage.append((rate, test_classification(Y_predict, prediction)))

    print percentage

    # DEBUG
    #estimates = npy.array(network.estimates)
    #value = estimates[0, 0]
    #print value[0]

    # draw diagram to demonstrate the data
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1)
    #plt.plot([1,2,3,4], [1,4,9,16])
    #plt.show()
    if plots:
        ax.plot(X, Y, label='Sine', linewidth=2, color='black')
        for data in predictions:
            ax.plot(X, data[1], label="Learning Rate: "+str(data[0]))
        ax.legend()

def logistic(x):
    return 1.0/(1+npy.exp(-x))

def logistic_prime(x):
    ex=npy.exp(-x)
    return ex/(1+ex)**2

def test_classification(Y, prediction):
    correct = 0
    for i in range(prediction.shape[0]):
        y1, y2 = Y[i]
        e1, e2 = prediction[i]
        if (y1 > y2 and e1 > e2) or (y1 < y2 and e1 < e2):
            correct += 1

    return correct / (Y.shape[0] * 1.0) * 100

if __name__ == '__main__':
    test(5, [0.02], plots=True)

#Conclusion: It is quicker to choose a larger number of hidden nodes, a large number of iterations,
#and a small learning rate, than to experiment finding a good choice.