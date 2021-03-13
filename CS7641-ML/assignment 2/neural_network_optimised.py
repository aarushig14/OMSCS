import time
import pandas as pd
import sys
import os

import six

sys.modules['sklearn.externals.six'] = six

import mlrose_hiive as mlrose
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import numpy as np
import matplotlib.pyplot as plt

__PATH = "."
__ACTIVATION = 'tanh'
__MAX_ATTEMPTS = 100
__POP_SIZE = 500
__MUTATION_PROB = 0.2
__LEARNING_RATE = 1

if __name__ == '__main__':
    # create directories required
    try:
        os.mkdir(__PATH + "/RHC")
        os.mkdir(__PATH + "/SA")
        os.mkdir(__PATH + "/GA")
    except FileExistsError:
        pass
    except OSError as error:
        print("Error creating directory results.")

    # Load dataset wine quality
    filename = open('winequality.csv')

    data = pd.read_csv(filename)
    X, y = data.iloc[:, :-1], data.iloc[:, -1]
    y[y < 6] = 0
    y[y >= 6] = 1
    X, y = shuffle(X, y, random_state=26)

    # Split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    iterations = list([50, 100, 500, 1000, 2000, 3500, 5000])
    fit_time = []
    query_time = []
    accuracy = []
    loss = []
    print("Progress: ", end="")
    for i in range(len(iterations)):
        iters = iterations[i]
        print(i, end="")
        # print('iterations = ' + str(iters))

        # RHC - Initialize neural network object and fit object
        # print("Random Hill Climbing")
        nn_rhc = mlrose.NeuralNetwork(hidden_nodes=[30], activation=__ACTIVATION,
                                      algorithm='random_hill_climb', max_iters=iters, curve=True,
                                      bias=True, is_classifier=True, learning_rate=__LEARNING_RATE,
                                      early_stopping=True, clip_max=5, max_attempts=__MAX_ATTEMPTS,
                                      random_state=10)

        # SA - Initialize neural network object and fit object
        # print("Simulated Annealing")
        nn_sa = mlrose.NeuralNetwork(hidden_nodes=[30], activation=__ACTIVATION,
                                     algorithm='simulated_annealing', max_iters=iters,
                                     bias=True, is_classifier=True, learning_rate=__LEARNING_RATE,
                                     early_stopping=True, clip_max=5, max_attempts=__MAX_ATTEMPTS,
                                     random_state=10, curve=True, schedule=mlrose.GeomDecay(decay=0.1))

        # GA - Initialize neural network object and fit object
        # print("Gentic Algorithm")
        nn_ga = mlrose.NeuralNetwork(hidden_nodes=[30], activation=__ACTIVATION,
                                     algorithm='genetic_alg', max_iters=iters, curve=True,
                                     bias=True, is_classifier=True, learning_rate=__LEARNING_RATE,
                                     early_stopping=True, clip_max=5, max_attempts=__MAX_ATTEMPTS,
                                     random_state=10, mutation_prob=__MUTATION_PROB, pop_size=__POP_SIZE)

        models = [nn_rhc, nn_sa, nn_ga]

        fit_time.append([])
        query_time.append([])
        accuracy.append([])
        loss.append([])
        for m in models:
            print(".", end="")
            start = time.time()
            nn_fit_m = m.fit(X_train, y_train)
            loss[i].append(nn_fit_m.loss)
            curve = nn_fit_m.fitness_curve

            # Predict labels for train set and assess accuracy
            y_train_pred = m.predict(X_train)
            y_train_accuracy = accuracy_score(y_train, y_train_pred)
            fit_time[i].append(time.time() - start)

            # Predict labels for test set and assess accuracy
            start = time.time()
            y_test_pred = m.predict(X_test)
            y_test_accuracy = accuracy_score(y_test, y_test_pred)
            accuracy[i].append(y_test_accuracy)
            query_time[i].append(time.time() - start)

    fit_time = np.array(fit_time)
    query_time = np.array(query_time)
    accuracy = np.array(accuracy)
    loss = np.array(loss)

    rhc_df = {'Fit Time': fit_time[:, 0], 'Query Time': query_time[:, 0], 'Accuracy': accuracy[:, 0], 'Loss': loss[:, 0]}
    sa_df = {'Fit Time': fit_time[:, 1], 'Query Time': query_time[:, 1], 'Accuracy': accuracy[:, 1], 'Loss': loss[:, 1]}
    ga_df = {'Fit Time': fit_time[:, 2], 'Query Time': query_time[:, 2], 'Accuracy': accuracy[:, 2], 'Loss': loss[:, 2]}

    pd.DataFrame(data=rhc_df).to_csv('RHC/nn_rhc.csv')
    pd.DataFrame(data=sa_df).to_csv('SA/nn_sa.csv')
    pd.DataFrame(data=ga_df).to_csv('GA/nn_ga.csv')

    models = ['RHC', 'SA', 'GA']
    for i in range(len(models)):
        name = str(models[i])

        plt.figure(i)
        plt.title('TRAIN TIME')
        plt.xlabel('Iterations')
        plt.ylabel('Time')
        plt.xticks(np.arange(len(iterations)), [str(z) for z in iterations])
        plt.plot(fit_time[:, i], label=name)
        plt.legend()
        plt.savefig(name+'/nn_fittime.png')

        plt.figure(i+3)
        plt.title('ACCURACY / LOSS')
        plt.xlabel('Iterations')
        plt.ylabel('Accuracy')
        plt.xticks(np.arange(len(iterations)), [str(z) for z in iterations])
        plt.plot(accuracy[:, i], label='accuracy', color='green')
        plt.plot(loss[:, i], label='loss', color='orange')
        plt.legend()
        plt.savefig(name+'/nn_accuracy_loss.png')
