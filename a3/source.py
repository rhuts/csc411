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

'''
Sigmoid function.
'''
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# TODO
def predict():
    pass

# TODO
def getMetrics(clf, X_test, t_test, N0_test, N1_test):

    # print '\nclf.n_layers_'
    # print clf.n_layers_

    # weight matrix at each layer i
    # print '\nclf.coefs_'
    # print [coef.shape for coef in clf.coefs_]
    # print clf.coefs_

    # bias vector at each layer i
    # print '\nclf.intercepts_'
    # print len(clf.intercepts_)
    # print clf.intercepts_

    w1 = clf.coefs_[0]
    w2 = clf.coefs_[1]

    b1 = clf.intercepts_[0]
    b2 = clf.intercepts_[1]

    a = np.dot(X_test, w1) # ok

    b = a + b1 # ok

    c = sigmoid(b) # ok

    d = np.matmul(c, w2) # ok

    e = d + b2

    f = sigmoid(e)

    # True when model predicts class 1
    class1 = (f >= 0.5).reshape(f.size)

    accuracy = np.sum(np.equal(class1, t_test) == True) / float(t_test.size)

    
    # change to column vectors
    col_z = class1[:, np.newaxis]
    col_t = t_test[:, np.newaxis]

    # each row contains [prediction, target]
    table = np.concatenate((col_z, col_t), axis=1)

    # test and counts number of rows that have [1, 1] which is a True Positive
    TP = np.count_nonzero(np.all(table, axis=1))

    # print class1
    # print table
    # print 
    
    # test and counts number of rows that have [1, 0] which is a False Positive
    FP_table = np.zeros(table.shape)
    FP_table[:, 0] = 1
    FP = np.count_nonzero(np.all(np.equal(table, FP_table), axis=1))
    
    
    precision = TP / (float(TP) + FP)

    recall = TP / (float(N1_test))

    # print '\taccuracy: {}'.format(accuracy)
    # print '\tprecision: {}'.format(precision)
    # print '\trecall: {}'.format(recall)

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





    # TODO subroutine to lower line count to meet reqs
    # Question 1(b)

    # train a NN with one unit in the hidden layer
    clf = MLPClassifier(solver='sgd',
                        hidden_layer_sizes=(1,),
                        activation='logistic',
                        learning_rate_init=0.01,
                        tol=np.power(10, -8, dtype=float),
                        max_iter=1000)
    clf.fit(X_train, t_train)


    print '\nQuestion 1(b):'
    # print the accuracy, precision and recall of the neural net on the test data
    accuracy, precision, recall = getMetrics(clf, X_test, t_test, N0_test, N1_test)

    print '\t\tofficial score: {}'.format(clf.score(X_test, t_test)) # TODO remove this print
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






    # TODO subroutine to lower line count to meet reqs for below questions
    # Question 1(c)

    # 12 NNs with two units in the hidden layer
    # make 12 plots in 4x3 grid with decision boundaries
    fig, axs = plt.subplots(4, 3)
    plt.suptitle('Question 1(c): Neural nets with 2 hidden units.')
    
    best_clf = None
    acc_best = 0
    for i in range(12):
        plt.sca(axs[i / 3, i % 3])
        plt.scatter(X_train[:, 0], X_train[:, 1], color=classToColor[t_train], s=2)

        clf = MLPClassifier(solver='sgd',
                            hidden_layer_sizes=(2, ),
                            activation='logistic',
                            learning_rate_init=0.01,
                            tol=np.power(10, -8, dtype=float),
                            max_iter=1000)
        clf.fit(X_train, t_train)
        dfContour(clf)

        curr_acc, precision, recall = getMetrics(clf, X_test, t_test, N0_test, N1_test)

        if curr_acc > acc_best:
            # save new best clf
            best_clf = clf
            acc_best = curr_acc

    plt.show()
    
    print '\nQuestion 1(c):'

    # plot best model
    plt.scatter(X_train[:, 0], X_train[:, 1], color=classToColor[t_train], s=2)
    plt.xlim(-3, 6); plt.ylim(-3, 6)
    plt.title('Question 1(c): Best neural net with 2 hidden units.')
    dfContour(best_clf)

    plt.show()


    # print best model metrics
    accuracy, precision, recall = getMetrics(best_clf, X_test, t_test, N0_test, N1_test)
    print '\t\tofficial score: {}'.format(best_clf.score(X_test, t_test)) # TODO remove this print
    print '\taccuracy: {}'.format(accuracy)
    print '\tprecision: {}'.format(precision)
    print '\trecall: {}'.format(recall)






    # Question 1(d)
    # 12 NNs with three units in the hidden layer
    # make 12 plots in 4x3 grid with decision boundaries
    fig, axs = plt.subplots(4, 3)
    plt.suptitle('Question 1(d): Neural nets with 3 hidden units.')
    
    best_clf = None
    acc_best = 0
    for i in range(12):
        plt.sca(axs[i / 3, i % 3])
        plt.scatter(X_train[:, 0], X_train[:, 1], color=classToColor[t_train], s=2)

        clf = MLPClassifier(solver='sgd',
                            hidden_layer_sizes=(3, ),
                            activation='logistic',
                            learning_rate_init=0.01,
                            tol=np.power(10, -8, dtype=float),
                            max_iter=1000)
        clf.fit(X_train, t_train)
        dfContour(clf)

        curr_acc, precision, recall = getMetrics(clf, X_test, t_test, N0_test, N1_test)

        if curr_acc > acc_best:
            # save new best clf
            best_clf = clf
            acc_best = curr_acc

    plt.show()
    
    print '\nQuestion 1(d):'

    # plot best model
    plt.scatter(X_train[:, 0], X_train[:, 1], color=classToColor[t_train], s=2)
    plt.xlim(-3, 6); plt.ylim(-3, 6)
    plt.title('Question 1(d): Best neural net with 3 hidden units.')
    dfContour(best_clf)

    plt.show()


    # print best model metrics
    accuracy, precision, recall = getMetrics(best_clf, X_test, t_test, N0_test, N1_test)
    print '\t\tofficial score: {}'.format(best_clf.score(X_test, t_test)) # TODO remove this print
    print '\taccuracy: {}'.format(accuracy)
    print '\tprecision: {}'.format(precision)
    print '\trecall: {}'.format(recall)
    # TODO save print that has better accuracy than part (c)







    # Question 1(e)
    # 12 NNs with four units in the hidden layer
    # make 12 plots in 4x3 grid with decision boundaries
    fig, axs = plt.subplots(4, 3)
    plt.suptitle('Question 1(e): Neural nets with 4 hidden units.')
    
    best_clf = None
    acc_best = 0
    for i in range(12):
        plt.sca(axs[i / 3, i % 3])
        plt.scatter(X_train[:, 0], X_train[:, 1], color=classToColor[t_train], s=2)

        clf = MLPClassifier(solver='sgd',
                            hidden_layer_sizes=(4, ),
                            activation='logistic',
                            learning_rate_init=0.01,
                            tol=np.power(10, -8, dtype=float),
                            max_iter=1000)
        clf.fit(X_train, t_train)
        dfContour(clf)

        curr_acc, precision, recall = getMetrics(clf, X_test, t_test, N0_test, N1_test)

        if curr_acc > acc_best:
            # save new best clf
            best_clf = clf
            acc_best = curr_acc

    plt.show()
    
    print '\nQuestion 1(e):'

    # plot best model
    plt.scatter(X_train[:, 0], X_train[:, 1], color=classToColor[t_train], s=2)
    plt.xlim(-3, 6); plt.ylim(-3, 6)
    plt.title('Question 1(e): Best neural net with 4 hidden units.')
    dfContour(best_clf)

    plt.show()


    # print best model metrics
    accuracy, precision, recall = getMetrics(best_clf, X_test, t_test, N0_test, N1_test)
    print '\t\tofficial score: {}'.format(best_clf.score(X_test, t_test)) # TODO remove this print
    print '\taccuracy: {}'.format(accuracy)
    print '\tprecision: {}'.format(precision)
    print '\trecall: {}'.format(recall)
    # TODO save print that has about the same accuracy as part (d)




# End of Q1 -------------------------------------------------------------------



# ------------------- Script for running the source file ---------------------\
q1()
# ------------------- End of script for running the source file --------------/