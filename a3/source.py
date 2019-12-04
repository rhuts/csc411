import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
from bonnerlib2 import dfContour
import pickle

# # disable warnings in sklearn
# # needs to be above sklearn imports
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
    bestOfTwelveNN(3, '1(d)', X_train, t_train, X_test, t_test, N0_test, N1_test)




    # Question 1(e)

    # 12 NNs with four units in the hidden layer
    bestOfTwelveNN(4, '1(e)', X_train, t_train, X_test, t_test, N0_test, N1_test)

# End of Q1 -------------------------------------------------------------------

'''
Used for playing around with learning rates.
Tune l_rate to change learning rate.
'''
def experiment_l_rate(X_train, Y_train, X_test, Y_test, l_rate):
    # play around and find a good learning rate
    clf = MLPClassifier(solver='sgd',
                            hidden_layer_sizes=(30, ),
                            activation='logistic',
                            batch_size=100,
                            learning_rate_init=l_rate,
                            tol=np.power(10, -8, dtype=float),
                            max_iter=10,
                            verbose=True)
    clf.fit(X_train, Y_train)
    print 'clf.score(Xtest, Ytest)'
    print clf.score(X_test, Y_test)


'''
TODO doc
'''
def bestOfTenNN(X_train, Y_train, X_val, Y_val, Xtest, Ytest, batch_size_, l_rate, max_iter_):
    
    max_val_acc = 0
    best_clf = None

    # train the neural net 10 times for max_iter_ iterations
    for i in range(10):
        clf = MLPClassifier(solver='sgd',
                            hidden_layer_sizes=(30, ),
                            activation='logistic',
                            batch_size=batch_size_,
                            learning_rate_init=l_rate,
                            tol=np.power(10, -8, dtype=float),
                            max_iter=max_iter_,
                            verbose=False)
        clf.fit(X_train, Y_train)

        # Compute and print out the
        # validation accuracy of each trained net
        # Choose the trained net that has the maximum validation accuracy
        val_acc = clf.score(X_val, Y_val)
        print '\tvalidation accuracy of trained net {}: {}'.format(i + 1, val_acc)
        if val_acc > max_val_acc:
            max_val_acc = val_acc
            best_clf = clf

    # convert Y_train to one-hot encoding
    n_classes = 10
    Y_train_onehot = np.eye(n_classes)[Y_train]

    # get prediction probabilities
    pred_prob = best_clf.predict_proba(X_train)

    # calculate cross-entropy
    cross_entropy = -np.sum(Y_train_onehot * np.log(pred_prob)) # TODO change all cross-entropies to test data

    # Print out its validation accuracy, test accuracy and cross entropy
    print '\n\tmaximum validation accuracy: {}'.format(max_val_acc)
    print '\ttest accuracy: {}'.format(best_clf.score(Xtest, Ytest))
    print '\tcross entropy: {}'.format(cross_entropy)

    # Print out the learning rate used
    print '\tlearning rate used: {}'.format(l_rate)

'''
This function is separate from bestOfTenNN() because
for loops are not permitted in Question 3(c)

TODO doc
'''
def trainNN(X_train, Y_train, X_val, Y_val, Xtest, Ytest, batch_size_, l_rate, max_iter_):
    
    # train NN for max_iter_ iterations
    clf = MLPClassifier(solver='sgd',
                            hidden_layer_sizes=(30, ),
                            activation='logistic',
                            batch_size=batch_size_,
                            learning_rate_init=l_rate,
                            tol=np.power(10, -8, dtype=float),
                            max_iter=max_iter_,
                            verbose=False)
    clf.fit(X_train, Y_train)
    

    # convert Y_train to one-hot encoding
    n_classes = 10
    Y_train_onehot = np.eye(n_classes)[Y_train]

    # get prediction probabilities
    pred_prob = clf.predict_proba(X_train)

    # calculate cross-entropy
    cross_entropy = -np.sum(Y_train_onehot * np.log(pred_prob))

    # Print the final training and test accuracies and cross entropy
    print '\ttraining accuracy (after 50 iterations): {}'.format(clf.score(X_train, Y_train))
    print '\ttest accuracy (after 50 iterations): {}'.format(clf.score(Xtest, Ytest))
    print '\tcross entropy (after 50 iterations): {}'.format(cross_entropy)

'''
TODO doc
'''
def softmax(y):
    exp_scores = np.exp(y)
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

'''
TODO doc
'''
def predict_probs(X, V, W, b1, b2):

    # get probabilities
    # y = sigmoid(np.matmul(sigmoid(np.dot(X, V) + b1), W) + b2)
    H = sigmoid(np.dot(X, V) + b1)
    O = softmax(np.matmul(H, W) + b2)

    # return probabilities
    # return y
    # exp_scores = np.exp(y)
    # return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return O, H


'''
TODO doc
'''
def get_score(X, T, params):
    
    # get prediction probabilities
    V, W, b1, b2 = params[0], params[1], params[2], params[3], 
    O, H = predict_probs(X, V, W, b1, b2)

    # convert probabilities to labels
    Y_hat = np.argmax(O, axis=1)

    # score is % of correct classifications
    return np.sum(np.equal(Y_hat, T) == True) / float(T.size)


'''
My implementation of Batch Gradient Descent.
TODO doc, optional params
Supplying a mini-batch of training data, max_iter_ value of 1, and initial_params
can be used for performing one iteration of mini-batch stochastic gradient descent
'''
def myBGD(X_train, Y_train, max_iter_, l_rate, init_params, n_hidden=30):

    # NOTES
    # - one hidden layer
    # - batch gradient descent
    # - one loop
    # - comment forward pass of training
    # - comment backward propagation
    # - Initialize the weight matrices 
    #   randomly using a standard Gaussian distribution (i.e., mean 0 and variance 1),
    # - initialize the bias terms to 0
    # - use the average gradient for weight updates
    # - 30 hidden units (variable for 3(f) )
    # - 100 iterations of gradient descent
    # - experiment to find a good learning rate
    # - test accuracy of around 85% after 100 iterations

    # optionally allow initial params for use in Question 3(e)+
    if init_params:
        V, W, b1, b2 = init_params[0], init_params[1], init_params[2], init_params[3]

    else:
        # init weight matrices randomly using standard Gaussian
        V = np.random.normal(0, 1, 784 * n_hidden).reshape((784, n_hidden)) # V is 784 x 30
        W = np.random.normal(0, 1, n_hidden * 10).reshape((n_hidden, 10))   # W is 30 x 10

        # init bias terms to 0
        b1 = np.zeros(n_hidden)   # b1 is 30 x 1
        b2 = np.zeros(10)   # b2 is 10 x 1

    # convert Y_train to one-hot encoding
    n_classes = 10
    Y_train_onehot = np.eye(n_classes)[Y_train]

    # perform max_iter_ iterations of gradient descent
    for i in range(max_iter_):

        ################ Forward Propagation ################
        O, H = predict_probs(X_train, V, W, b1, b2)

        ################ Back Propagation ################
        # O - T
        dCdZ = O - Y_train_onehot                       # 10000 x 10
        # H.T * (O - T)
        dCdW = np.dot(H.T, dCdZ)                        # 30 x 10
        # 1 * dC/dZ                     
        dCdw0 = np.dot(np.ones(dCdZ.shape[0]), dCdZ)    # 10 x 1
        # dC/dZ * W.T
        dCdH = np.dot(dCdZ, W.T)                        # 10000 x 30
        # H * (1 - H) * dC/dH                           # H.shape   # 10000 x 30
        dCdU = np.dot(dCdZ, W.T) * (1 - H)              # 10000 x 30
        # X.T * dC/dU
        dCdV = np.dot(X_train.T, dCdU)                  # 784 x 30
        # 1 * dC/dU
        dCdv0 = np.dot(np.ones(dCdU.shape[0]), dCdU)    # 30 x 1


        # use average gradient to update weight matricies

        # update weights
        N = X_train.shape[0]
        W = W - ((l_rate * dCdW) / N)
        V = V - ((l_rate * dCdV) / N)

        # update biases
        b2 = b2 - ((l_rate * dCdw0) / N)
        b1 = b1 - ((l_rate * dCdv0) / N)

        # return the learned params
        if (i == (max_iter_ - 1)):
            return [V, W, b1, b2]

'''
TODO doc
'''
def bestOfTenMyBGD(X_train, Y_train, X_val, Y_val, Xtest, Ytest):

    learning_rate = 1.0

    best_acc_val = 0
    best_params = []

    # train a neural net 10 times
    for i in range(2): # TODO change to 10
        # Compute and print out the validation accuracy of each trained net
        params = myBGD(X_train, Y_train, max_iter_=100, l_rate=learning_rate, init_params=[])
        acc_val = get_score(X_val, Y_val, params)
        print '\tvalidation accuracy of trained net {}: {}'.format(i + 1, acc_val)

        # Choose the trained net that has the maximum validation accuracy
        if (acc_val > best_acc_val):
            best_acc_val = acc_val
            best_params = params


    # convert Y_train to one-hot encoding
    n_classes = 10
    Y_test_onehot = np.eye(n_classes)[Ytest]

    # get prediction probabilities
    # pred_prob, H = best_clf.predict_proba(X_train)
    pred_prob, H = predict_probs(Xtest, best_params[0], best_params[1], best_params[2], best_params[3])

    # calculate cross-entropy
    cross_entropy = -np.sum(Y_test_onehot * np.log(pred_prob))

    # Print out its validation accuracy, test accuracy and cross entropy
    print '\n\tmaximum validation accuracy: {}'.format(best_acc_val)
    print '\ttest accuracy: {}'.format(get_score(Xtest, Ytest, best_params))
    print '\tcross entropy: {}'.format(cross_entropy)

    # Print out the learning rate used
    print '\tlearning rate used: {}'.format(learning_rate)




'''
TODO doc
'''
def mySGD(X_train, Y_train, Xtest, Ytest, l_rate, batch_size, n_epochs, verbose=False, n_hidden=30):


    # sweep across the entire shuffled data
    N = X_train.shape[0]

    # print 'N'
    # print N

    # get the number of mini-batches
    n_batches = N // batch_size
    if (N % batch_size) != 0:
        n_batches += 1

    params = []

    # for each sweep of the entire data (epoch)
    for i in range(n_epochs):

        ################ A NEW epoch is beginning ################

        # shuffle the training data randomly
        X_train, Y_train = shuffle(X_train, Y_train)

        # get indexes of each mini-batch
        for j in range(n_batches):

            start = j * batch_size
            end = min(start + batch_size, N)    # clamp to training data size

            ################ Mini-batch is created ################
            # using X_train[start:end]

            # switch to 100 hidden units for Question 3(f)
            if verbose and n_hidden != 30:
                updated_params = myBGD(X_train[start:end], Y_train[start:end], max_iter_=1, l_rate=l_rate, init_params=params, n_hidden=n_hidden)

            else:
                updated_params = myBGD(X_train[start:end], Y_train[start:end], max_iter_=1, l_rate=l_rate, init_params=params)

            params = updated_params
        
        # for Question 3(f)
        # print out the test accuracy after every ten epochs and the first
        if verbose and (((i + 1) % 10 == 0) or (i == 0)):
            acc_test = get_score(Xtest, Ytest, params)
            print '\ttest accuracy after epoch #{} is \t: {}'.format(i + 1, acc_test)

    return params


'''
TODO doc
'''
def bestOfTenMySGD(X_train, Y_train, X_val, Y_val, Xtest, Ytest, batch_size, n_epochs):

    learning_rate = 1.0

    best_acc_val = 0
    best_params = []

    # train a neural net 10 times
    for i in range(2): # TODO change to 10
        # Compute and print out the validation accuracy of each trained net
        params = mySGD(X_train, Y_train, Xtest, Ytest, l_rate=learning_rate, batch_size=batch_size, n_epochs=n_epochs)

        acc_val = get_score(X_val, Y_val, params)
        print '\tvalidation accuracy of trained net {}: {}'.format(i + 1, acc_val)

        # Choose the trained net that has the maximum validation accuracy
        if (acc_val > best_acc_val):
            best_acc_val = acc_val
            best_params = params


    # convert Y_train to one-hot encoding
    n_classes = 10
    Y_test_onehot = np.eye(n_classes)[Ytest]

    # get prediction probabilities
    # pred_prob, H = best_clf.predict_proba(X_train)
    pred_prob, H = predict_probs(Xtest, best_params[0], best_params[1], best_params[2], best_params[3])

    # calculate cross-entropy
    cross_entropy = -np.sum(Y_test_onehot * np.log(pred_prob))

    # Print out its validation accuracy, test accuracy and cross entropy
    print '\n\tmaximum validation accuracy: {}'.format(best_acc_val)
    print '\ttest accuracy: {}'.format(get_score(Xtest, Ytest, best_params))
    print '\tcross entropy: {}'.format(cross_entropy)

    # Print out the learning rate used
    print '\tlearning rate used: {}'.format(learning_rate)


##########  QUESTION 3  ############

def q3():
    
    # Question 3(a)

    # open train and test data
    with open('mnist.pickle','rb') as f:
        Xtrain, Ytrain, Xtest, Ytest = pickle.load(f)

    # Use the first 10,000 points of the MNIST training data as validation data,
    # and use the next 10,000 points as the reduced training data
    X_val, Y_val = Xtrain[:10000], Ytrain[:10000]
    X_train, Y_train = Xtrain[10000:20000], Ytrain[10000:20000]


    # playing around with learning rates
    # experiment_l_rate(X_train, Y_train, Xtest, Ytest, learning_rate)
    learning_rate = 1.0

    print '\nQuestion 3(a).'; print('-------------')
    # bestOfTenNN(X_train, Y_train, X_val, Y_val, Xtest, Ytest, batch_size_=100, l_rate=learning_rate, max_iter_=10)



    # Question 3(b)
    
    print '\nQuestion 3(b).'; print('-------------')
    # bestOfTenNN(X_train, Y_train, X_val, Y_val, Xtest, Ytest, batch_size_=10000, l_rate=learning_rate, max_iter_=10)
    # TODO 
    # explain why accuracy is much lower than in part (a) and is 75%-80%



    # Question 3(c)

    print '\nQuestion 3(c).'; print('-------------')
    # trainNN(X_train, Y_train, X_val, Y_val, Xtest, Ytest, batch_size_=10000, l_rate=learning_rate, max_iter_=50)
    # trainNN(X_train, Y_train, X_val, Y_val, Xtest, Ytest, batch_size_=10000, l_rate=learning_rate, max_iter_=200)
    

    # Question 3(d)

    # Batch Gradient Descent: implementation

    print '\nQuestion 3(d).'; print('-------------')
    # bestOfTenMyBGD(X_train, Y_train, X_val, Y_val, Xtest, Ytest)



    # Using the average gradient means that
    # the optimal learning rate does not change much when the
    # size of the training set changes (which is why the same learing rate worked in
    # parts (a) and (b)). Explain why this is
    # TODO explain

    # Question 3(e)

    # Stochastic Gradient Descent: implementation

    print '\nQuestion 3(e).'; print('-------------')
    bestOfTenMySGD(X_train, Y_train, X_val, Y_val, Xtest, Ytest, batch_size=100, n_epochs=100)

    # Question 3(f)

    print '\nQuestion 3(f).'; print('-------------')
    params = mySGD(Xtrain, Ytrain, Xtest, Ytest, l_rate=learning_rate, batch_size=100, n_epochs=100, verbose=True, n_hidden=100)

    # convert Ytest to one-hot encoding
    n_classes = 10
    Y_test_onehot = np.eye(n_classes)[Ytest]

    # get prediction probabilities
    # pred_prob, H = best_clf.predict_proba(X_train)
    pred_prob, H = predict_probs(Xtest, params[0], params[1], params[2], params[3])

    # calculate cross-entropy
    cross_entropy = -np.sum(Y_test_onehot * np.log(pred_prob))

    # Print out the final training accuracy, test accuracy and cross entropy
    print '\ttrain accuracy: {}'.format(get_score(Xtrain, Ytrain, params))
    print '\ttest accuracy: {}'.format(get_score(Xtest, Ytest, params))
    print '\tcross entropy: {}'.format(cross_entropy)



# End of Q3 -------------------------------------------------------------------


# ------------------- Script for running the source file ---------------------\
# q1()
# q2() Q2 is non-programming
q3()
# ------------------- End of script for running the source file --------------/