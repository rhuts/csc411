import numpy as np
import numpy.random as rnd
from sklearn.utils import shuffle
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from bonnerlib2 import dfContour

# disable warnings in sklearn
# needs to be above sklearn imports
# def warn(*args, **kwargs):
#     pass
# import warnings
# warnings.warn = warn

# import pickle
# import time



##########  QUESTION 1  ############

# Question 1(a)
'''
Generates two clusters of data.
The cluster for class 0 has N0 points, mean mu0 and covariance cov0.
The cluster for class 1 has N1 points, mean mu1 and covariance cov1.

Returns two arrays X and t, which are the data points and target values
respectively.
'''
def gen_data(mu0, mu1, cov0, cov1, N0, N1):

    # generate class 0 points
    cov_matrix_c0 = np.array([[1, cov0], [cov0, 1]])
    t_c0 = np.full(N0, 0)
    X_c0 = rnd.multivariate_normal(mu0, cov_matrix_c0, N0)

    # generate class 1 points
    cov_matrix_c1 = np.array([[1, cov1], [cov1, 1]])
    X_c1 = rnd.multivariate_normal(mu1, cov_matrix_c1, N1)
    t_c1 = np.full(N1, 1)

    # shuffle both classes
    X = np.concatenate((X_c0, X_c1))
    t = np.concatenate((t_c0, t_c1))
    X, t = shuffle(X, t)

    return X, t

# TODO
def predict():
    pass

# TODO
def getMetrics(clf):

    # weight matrix at each layer i
    print 'clf.coefs_'
    print [coef.shape for coef in clf.coefs_]
    print clf.coefs_

    # bias vector at each layer i
    print 'clf.intercepts_'
    print len(clf.intercepts_)
    print clf.intercepts_
    
    accuracy = 'TODO'
    precision = 'TODO'
    recall = 'TODO'
    return [accuracy, precision, recall]

def q1():
    print '\nQUESTION 1.\n-----------'

    # Question 1(a)
    N0_train, N1_train = 1000, 500
    N0_test, N1_test = 10000, 5000
    mu0, mu1 = (1, 1), (2, 2)
    cov0, cov1 = 0, 0.9
    X_train, t_train = gen_data(mu0, mu1, cov0, cov1, N0_train, N1_train)
    X_test, t_test = gen_data(mu0, mu1, cov0, cov1, N0_test, N1_test)




    # Question 1(b)

    # train a NN with one unit in the hidden layer
    clf = MLPClassifier(solver='sgd',
                        hidden_layer_sizes=1,
                        activation='logistic',
                        learning_rate_init=0.01,
                        tol=np.power(10, -8, dtype=float),
                        max_iter=1000)
    clf.fit(X_train, t_train)

    # TODO remove
    # print clf.score(X_test, t_test)

    # TODO this happens by default
    # The x and y axes should both extend from -3 to 6

    # TODO
    print '\nQuestion 1(b):'
    # print the accuracy, precision and recall of the neural net on the test data
    accuracy, precision, recall = getMetrics(clf)
    print '\taccuracy: {}'.format(accuracy)
    print '\tprecision: {}'.format(precision)
    print '\trecall: {}'.format(recall)

    # plot training data
    classToColor = np.array(['r', 'b'])
    plt.scatter(X_train[:, 0], X_train[:, 1], color=classToColor[t_train], s=2)
    plt.xlim(-3, 6); plt.ylim(-3, 6)
    plt.title('Question 1(b): Neural net with 1 hidden unit.')

    # draw decision boundary
    dfContour(clf)
    plt.show()




    # Question 1(c)

    # Train a neural net with two units in the hidden layer 12 times
    # make 12 plots of 12 different trainings, plot decision boundaries
    # plots in a 4 x 3 grid in a single figure
    fig, axs = plt.subplots(4, 3)
    plt.suptitle('Question 1(c): Neural nets with 2 hidden units.')
    

    for i in range(12):
        plt.sca(axs[i / 3, i % 3])
        plt.scatter(X_train[:, 0], X_train[:, 1], color=classToColor[t_train], s=2)

        clf = MLPClassifier(solver='sgd',
                            hidden_layer_sizes=2,
                            activation='logistic',
                            learning_rate_init=0.01,
                            tol=np.power(10, -8, dtype=float),
                            max_iter=1000)
        clf.fit(X_train, t_train)
        dfContour(clf)

    plt.show()




    # plot the training data and decision boundary of the NN with the highest test accuracy




# End of Q1 -------------------------------------------------------------------



# ------------------- Script for running the source file ---------------------\
q1()
# ------------------- End of script for running the source file --------------/