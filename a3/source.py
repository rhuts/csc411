import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
from bonnerlib2 import dfContour
import pickle

# disable warnings in sklearn
# needs to be above sklearn imports
# def warn(*args, **kwargs):
#     pass
# import warnings
# warnings.warn = warn

from sklearn.utils import shuffle
from sklearn.neural_network import MLPClassifier

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

'''
Predicts the class with a specific threshold.
'''
def predict(clf, X, threshold=0.5):

    # learned parameters
    w1, w2 = clf.coefs_[0], clf.coefs_[1]
    b1, b2 = clf.intercepts_[0], clf.intercepts_[1]

    # get probabilities
    y = sigmoid(np.matmul(sigmoid(np.dot(X, w1) + b1), w2) + b2)

    # return predictions
    return (y >= threshold).reshape(y.size)


'''
Calculates accuracy, precision and recall.
'''
def getMetrics(clf, X_test, t_test, N0_test, N1_test):

    # TODO remove commentted test prints before submit
    # print '\nclf.n_layers_'
    # print clf.n_layers_

    # # weight matrix at each layer i
    # print '\nclf.coefs_'
    # print [coef.shape for coef in clf.coefs_]
    # print clf.coefs_

    # # bias vector at each layer i
    # print '\nclf.intercepts_'
    # print len(clf.intercepts_)
    # print clf.intercepts_

    y = predict(clf, X_test, threshold=0.5)

    # accuracy is % of correct classifications
    accuracy = np.sum(np.equal(y, t_test) == True) / float(t_test.size)
    
    # change to column vectors
    col_z = y[:, np.newaxis]
    col_t = t_test[:, np.newaxis]

    # each row contains [prediction, target]
    table = np.concatenate((col_z, col_t), axis=1)

    # test and counts number of rows that have [1, 1] which is a True Positive
    TP = np.count_nonzero(np.all(table, axis=1))

    # test and counts number of rows that have [1, 0] which is a False Positive
    FP_table = np.zeros(table.shape)
    FP_table[:, 0] = 1
    FP = np.count_nonzero(np.all(np.equal(table, FP_table), axis=1))
    
    
    precision = TP / (float(TP) + FP)

    recall = TP / (float(N1_test))

    return [accuracy, precision, recall]


'''
Trains 12 different Neural Networks.
Plots all 12 in a 4 x 3 grid.
Plots the model with the best accuracy separately
and prints its accuracy, precision and recall metrics.
'''
def bestOfTwelveNN(n_units, str_question, X_train, t_train, X_test, t_test, N0_test, N1_test):

    # make 12 plots in 4x3 grid with decision boundaries
    classToColor = np.array(['r', 'b'])
    fig, axs = plt.subplots(4, 3)
    plt.suptitle('Question {}: Neural nets with {} hidden units.'.format(str_question, n_units))
    
    best_clf, acc_best = None, 0
    for i in range(12):
        plt.sca(axs[i / 3, i % 3])
        plt.scatter(X_train[:, 0], X_train[:, 1], color=classToColor[t_train], s=2)

        clf = MLPClassifier(solver='sgd',
                            hidden_layer_sizes=(n_units, ),
                            activation='logistic',
                            learning_rate_init=0.01,
                            tol=np.power(10, -8, dtype=float),
                            max_iter=1000)
        clf.fit(X_train, t_train)
        dfContour(clf)

        curr_acc, precision, recall = getMetrics(clf, X_test, t_test, N0_test, N1_test)

        # update best clf
        if curr_acc > acc_best:
            best_clf, acc_best = clf, curr_acc

    plt.show()
    

    # plot best model
    plt.scatter(X_train[:, 0], X_train[:, 1], color=classToColor[t_train], s=2)
    plt.xlim(-3, 6); plt.ylim(-3, 6)
    plt.title('Question {}: Best neural net with {} hidden units.'.format(str_question, n_units))
    dfContour(best_clf)

    plt.show()


    # print best model metrics
    accuracy, precision, recall = getMetrics(best_clf, X_test, t_test, N0_test, N1_test)
    print '\nQuestion {}.'.format(str_question); print('-------------')
    print '\t\tofficial score: {}'.format(best_clf.score(X_test, t_test)) # TODO remove this print
    print '\taccuracy: {}'.format(accuracy); print '\tprecision: {}'.format(precision); print '\trecall: {}'.format(recall)

def q1():

    # Question 1(a)

    # generate training set and test set
    N0_train, N1_train = 1000, 500
    N0_test, N1_test = 10000, 5000
    mu0, mu1 = (1, 1), (2, 2)
    cov0, cov1 = 0, 0.9
    X_train, t_train = gen_data(mu0, mu1, cov0, cov1, N0_train, N1_train)
    X_test, t_test = gen_data(mu0, mu1, cov0, cov1, N0_test, N1_test)





    # Question 1(b)

    # train a NN with one unit in the hidden layer
    clf = MLPClassifier(solver='sgd',
                        hidden_layer_sizes=(1,),
                        activation='logistic',
                        learning_rate_init=0.01,
                        tol=np.power(10, -8, dtype=float),
                        max_iter=1000)
    clf.fit(X_train, t_train)


    accuracy, precision, recall = getMetrics(clf, X_test, t_test, N0_test, N1_test)

    # print the accuracy, precision and recall of the neural net on the test data
    print '\nQuestion 1(b).'; print('-------------')
    print '\t\tofficial score: {}'.format(clf.score(X_test, t_test)) # TODO remove this print
    print '\taccuracy: {}'.format(accuracy); print '\tprecision: {}'.format(precision); print '\trecall: {}'.format(recall)

    # plot training data
    classToColor = np.array(['r', 'b'])
    plt.scatter(X_train[:, 0], X_train[:, 1], color=classToColor[t_train], s=2)
    plt.xlim(-3, 6); plt.ylim(-3, 6); plt.title('Question 1(b): Neural net with 1 hidden unit.')

    # draw decision boundary
    dfContour(clf)
    plt.show()




    # Question 1(c)

    # 12 NNs with two units in the hidden layer
    bestOfTwelveNN(2, '1(c)', X_train, t_train, X_test, t_test, N0_test, N1_test)




    # Question 1(d)

    # 12 NNs with three units in the hidden layer
    # TODO save print that has better accuracy than part (c)
    bestOfTwelveNN(3, '1(d)', X_train, t_train, X_test, t_test, N0_test, N1_test)




    # Question 1(e)

    # 12 NNs with four units in the hidden layer
    # TODO save print that has about the same accuracy as part (d)
    bestOfTwelveNN(4, '1(e)', X_train, t_train, X_test, t_test, N0_test, N1_test)

# End of Q1 -------------------------------------------------------------------




##########  QUESTION 3  ############

def q3():
    
    # Question 3(a)

    # open train and test data
    with open('mnist.pickle','rb') as f:
        Xtrain, Ytrain, Xtest, Ytest = pickle.load(f)

    # Use the first 10,000 points of the MNIST training data as validation data,
    X_val, Y_val = Xtrain[:10000], Ytrain[:10000]
    # and use the next 10,000 points as the reduced training data
    X_train, Y_train = Xtrain[10000:20000], Ytrain[10000:20000]

    learning_rate = 1.0

    # /================= START EXPERIMENT =========================\
    # TODO remove experiment


    # # play around and find a good learning rate
    # clf = MLPClassifier(solver='sgd',
    #                         hidden_layer_sizes=(30, ),
    #                         activation='logistic',
    #                         batch_size=100,
    #                         learning_rate_init=learning_rate,
    #                         tol=np.power(10, -8, dtype=float),
    #                         max_iter=100,
    #                         verbose=True)
    # clf.fit(X_train, Y_train)
    # print 'clf.score(Xtest, Ytest)'
    # print clf.score(Xtest, Ytest)

    # \================= END EXPERIMENT =========================/




    # print '\nQuestion 3(a).'; print('-------------')

    # max_val_acc = 0
    # best_clf = None

    # # train the neural net 10 times
    # for i in range(10):
    #     clf = MLPClassifier(solver='sgd',
    #                         hidden_layer_sizes=(30, ),
    #                         activation='logistic',
    #                         batch_size=100,
    #                         learning_rate_init=learning_rate,
    #                         tol=np.power(10, -8, dtype=float),
    #                         max_iter=100,
    #                         verbose=False)
    #     clf.fit(X_train, Y_train)

    #     # Compute and print out the
    #     # validation accuracy of each trained net
    #     val_acc = clf.score(X_val, Y_val)
    #     print '\tvalidation accuracy of trained net {}: {}'.format(i + 1, val_acc)
    #     if val_acc > max_val_acc:
    #         max_val_acc = val_acc
    #         best_clf = clf

    # # Choose the trained net that has the maximum validation accuracy
    # # Print out its validation accuracy, test accuracy and cross entropy
    # # TODO calculate cross-entropy manually
    # print '\nmaximum validation accuracy: {}'.format(max_val_acc)
    # print 'maximum test accuracy: {}'.format(best_clf.score(Xtest, Ytest))


    # # print out the learning rate used
    # print 'learning rate used: {}'.format(learning_rate)


    # Question 3(b)
    
    print '\nQuestion 3(b).'; print('-------------')

    max_val_acc = 0
    best_clf = None

    # train the neural net 10 times
    for i in range(10):
        clf = MLPClassifier(solver='sgd',
                            hidden_layer_sizes=(30, ),
                            activation='logistic',
                            batch_size=10000,
                            learning_rate_init=learning_rate,
                            tol=np.power(10, -8, dtype=float),
                            max_iter=100,
                            verbose=False)
        clf.fit(X_train, Y_train)

        # Compute and print out the
        # validation accuracy of each trained net
        val_acc = clf.score(X_val, Y_val)
        print '\tvalidation accuracy of trained net {}: {}'.format(i + 1, val_acc)
        if val_acc > max_val_acc:
            max_val_acc = val_acc
            best_clf = clf

    # Choose the trained net that has the maximum validation accuracy
    # Print out its validation accuracy, test accuracy and cross entropy
    # TODO
    print '\nmaximum validation accuracy: {}'.format(max_val_acc)
    print 'maximum test accuracy: {}'.format(best_clf.score(Xtest, Ytest))

    # print out the learning rate used
    print 'learning rate used: {}'.format(learning_rate)

# End of Q3 -------------------------------------------------------------------


# ------------------- Script for running the source file ---------------------\
# q1()
# q2() Q2 is non-programming
q3()
# ------------------- End of script for running the source file --------------/