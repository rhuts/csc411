import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from bonnerlib2 import dfContour
import pickle
import time

# TODO docstrings for functions once done
# TODO proofs





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


def q1():

    # Question 1(b)
    N0, N1 = 10000, 5000
    mu0, mu1 = (1, 1), (2, 2)
    cov0, cov1 = 0, -0.9
    X, t = gen_data(mu0, mu1, cov0, cov1, N0, N1)


    # Question 1(c)
    classToColor = np.array(['r', 'b'])
    plt.scatter(X[:, 0], X[:, 1], color=classToColor[t], s=2)
    plt.xlim(-3, 6); plt.ylim(-3, 6)
    plt.title('Question 1(c): sample cluster data')
    plt.show()


# End of Q1 -------------------------------------------------------------------


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def probability(x, w, w0):
    return sigmoid(np.dot(x, w.T) + w0)

def predict(x, w, w0, threshold=0.5):
    return probability(x, w, w0) >= threshold

def get_metrics(X, t, N0, N1, logReg, threshold=0.5):

    # P(C = 1|x) = threshold
    # calculate 1500 x 1 class prediction
    z = predict(X, logReg.coef_, logReg.intercept_, threshold=threshold)
    z = z.reshape(z.size)

    # change to column vectors
    col_z = z[:, np.newaxis]
    col_t = t[:, np.newaxis]

    # each row contains [prediction, target]
    table = np.concatenate((col_z, col_t), axis=1)

    # test and counts number of rows that have [1, 1] which is a True Positive
    TP = np.count_nonzero(np.all(table, axis=1))

    # test and counts number of rows that have [1, 0] which is a False Positive
    FP_table = np.zeros(table.shape)
    FP_table[:, 0] = 1
    FP = np.count_nonzero(np.all(np.equal(table, FP_table), axis=1))

    print '\nP(C = 1|x) = ' + str(threshold)
    print '\tprecision'
    print '\t\t' + str(TP / (float(TP) + FP))
    print '\trecall P(C = 1|x) = ' + str(threshold)
    print '\t\t' + str(TP / (float(N1)))





##########  QUESTION 2  ############

def q2():
    # Question 2(a)
    # Use gen data to generate training data with 1000 points in class 0 and
    # 500 points in class 1. Use the same means and covariances as in Question 1(b).
    N0, N1 = 1000, 500
    mu0, mu1 = (1, 1), (2, 2)
    cov0, cov1 = 0, -0.9
    X, t = gen_data(mu0, mu1, cov0, cov1, N0, N1)


    # Question 2(b)
    logReg = LogisticRegression()
    print logReg
    logReg.fit(X, t)

    print '\nbias term w_0:'; print '\t' + str(logReg.intercept_)
    print '\nweight vector w:'; print '\t' + str(logReg.coef_)


    # Question 2(c)

    # first way - with score() method
    accuracy1 = logReg.score(X, t)
    print '\naccuracy1:'; print '\t' + str(accuracy1)

    # second way - for w and w_0
    # which is correct predictions / number of predictions

    # calculate 1500 x 1 class prediction
    z = predict(X, logReg.coef_, logReg.intercept_, threshold=0.5)

    # compare with target vector t
    accuracy2 = np.equal(z.reshape(z.size), t).mean()
    print '\naccuracy2:'; print '\t' + str(accuracy2)

    # difference of accuracies, should be zero
    accuracy_diff = accuracy1 - accuracy2
    print '\naccuracy difference (should be zero):'; print '\t' + str(accuracy_diff)


    # Question 2(d)
    # generate scatterplot of training data as in Q 1 c)
    classToColor = np.array(['r', 'b'])
    plt.scatter(X[:, 0], X[:, 1], color=classToColor[t], s=2)

    # draw the decision boundary as a black line ontop of the data
    w = logReg.coef_[0]
    # b = -logReg.intercept_[0] / w[1]
    # plt.plot(np.linspace(-3, 6), (-w[0] / w[1]) * np.linspace(-3, 6) + b, color='k')
    # plot(x, -(w1 * x + w0) / -w2)
    plt.plot(np.linspace(-3, 6), -(w[0] * np.linspace(-3, 6) + logReg.intercept_[0]) / w[1], color='k')

    # title the figure
    plt.xlim(-3, 6); plt.ylim(-3, 6); plt.title('Question 2(d): training data and decision boundary')
    plt.show() # TODO uncomment


    # Question 2(e)
    # Generate a scatter plot of the training data
    plt.scatter(X[:, 0], X[:, 1], color=classToColor[t], s=2)
 
    # draw three probability contours on top of the data: 

    # CS = plt.contour(xx, yy, Z, levels=[-3], colors='r', linestyles='solid')
    # plt.clabel(CS, fmt = '%2.1d', colors = 'k', fontsize=14) #contour line labels


    # z = predict(X, logReg.coef_, logReg.intercept_, threshold=0.5)
    # plt.contour(X[:, 0], X[:, 1], z)

    # P(C = 1|x) = 0.6
    # TODO write proof for solving sigmoid = 0.6 to find z=0.405465
    # TODO initial whiteboard attempt to prove might have wrong sign
    plt.plot(np.linspace(-3, 6), -(w[0] * np.linspace(-3, 6) + logReg.intercept_[0] - 0.405465) / w[1], color='b')

    # P(C = 1|x) = 0.5
    # TODO write proof for solving sigmoid = 0.5 to find z=0.0
    plt.plot(np.linspace(-3, 6), -(w[0] * np.linspace(-3, 6) + logReg.intercept_[0] - 0.0) / w[1], color='k')

    # P(C = 1|x) = 0.05
    # TODO write proof for solving sigmoid = 0.05 to find z=-2.94444
    plt.plot(np.linspace(-3, 6), -(w[0] * np.linspace(-3, 6) + logReg.intercept_[0] + 2.94444) / w[1], color='r')

    plt.xlim(-3, 6); plt.ylim(-3, 6); plt.title('Question 2(e): three contours')
    plt.show()


    # Question 2(f)
    # Use gen data to generate test data with 10,000 points in class 0 and
    # 5,000 points in class 1
    N0, N1 = 10000, 5000
    mu0, mu1 = (1, 1), (2, 2)
    cov0, cov1 = 0, -0.9
    X, t = gen_data(mu0, mu1, cov0, cov1, N0, N1)


    # Question 2(g)
    # Compute the precision and recall for each of the three contours in part (e)

    # P(C = 1|x) = 0.6
    get_metrics(X, t, N0, N1, logReg, threshold=0.6)

    # P(C = 1|x) = 0.5
    get_metrics(X, t, N0, N1, logReg, threshold=0.5)

    # P(C = 1|x) = 0.05
    get_metrics(X, t, N0, N1, logReg, threshold=0.05)

# End of Q2 -------------------------------------------------------------------




##########  QUESTION 4  ############

def gbclf_train_test(mu0, mu1, cov0, cov1, N0_train, N1_train, N0_test, N1_test, str_question):

    # generate train data from 2(a) and test data from 2(f)
    X_train, t_train = gen_data(mu0, mu1, cov0, cov1, N0_train, N1_train)
    X_test, t_test = gen_data(mu0, mu1, cov0, cov1, N0_test, N1_test)

    # train sklearn QuadraticDiscriminantAnalysis
    GBclf = QuadraticDiscriminantAnalysis()
    GBclf.fit(X_train, t_train)

    # compute and print out the accuracy of your classifier
    # with the test data from q2(f)
    accuracy = GBclf.score(X_test, t_test)
    print '\nAccuracy of Gaussian Bayes clf ' + str_question + ':'; print '\t' + str(accuracy)

    # plot the training data
    classToColor = np.array(['r', 'b'])
    plt.scatter(X_train[:, 0], X_train[:, 1], color=classToColor[t_train], s=2)

    # plot the decision boundary using dfContour
    dfContour(GBclf)
    plt.xlim(-3, 6); plt.ylim(-3, 6); plt.title('Question ' + str_question + ': Decision boundary and contours')
    plt.show()

def q4():

    # Question 4(a)
    gbclf_train_test((1, 1), (2, 2), 0, -0.9, 1000, 500, 10000, 5000, '4(a)')


    # Question 4(b)
    # part a) has three separate regions, two red and one blue.
    # Explain this result. Use diagrams in your explanation
    # TODO this ^


    # Question 4(c)
    # Generate new training and test data that is the same as
    # that of Question 2 except that cov1 = 0.9, instead of -0.9
    gbclf_train_test((1, 1), (2, 2), 0, 0.9, 1000, 500, 10000, 5000, '4(c)')


    # Question 4(d)
    # Repeat part (c), but in the training set, put 1,000 points in class 0
    # and 5,000 in class 1; and in the test set, 10,000 points in class 0
    # and 50,000 in class 1
    gbclf_train_test((1, 1), (2, 2), 0, 0.9, 1000, 5000, 10000, 50000, '4(d)')


    # Question 4(e)
    # TODO
    # this question is T.B.A.


# End of Q4 -------------------------------------------------------------------


# TODO docstring
def train_test_model(model, str_question, Xtrain, Ytrain, Xtest, Ytest):
    start = time.time()
    model.fit(Xtrain, Ytrain)
    end = time.time()
    duration = end - start
    print '\nTime required to fit the model {}: {}'.format(str_question, duration)

    # compute and print out the training and test accuracies
    accuracy_train = model.score(Xtrain, Ytrain)
    accuracy_test = model.score(Xtest, Ytest)
    print '\nAccuracy of classifier {}:'.format(str_question); print '\tTraining: ' + str(accuracy_train); print '\tTesting: ' + str(accuracy_test)




##########  QUESTION 5  ############

def q5():

    # open train and test data
    with open('mnist.pickle','rb') as f:
        Xtrain, Ytrain, Xtest, Ytest = pickle.load(f)


    # Question 5(a)
    # display 25 of the MNIST images at random (w/o replacement) in 5x5 subplot
    chosen_ind = rnd.randint(0, Xtrain.shape[0], 25)
    fig, axs = plt.subplots(5, 5)
    plt.suptitle('Question 5(a): 25 random MNIST images.')
    for i in range(len(chosen_ind)):
        axs[i / 5, i % 5].imshow(Xtrain[chosen_ind[i]].reshape((28, 28)), cmap='Greys', interpolation='nearest')
        axs[i / 5, i % 5].axis('off')

    plt.show()



    # Question 5(b)

    GBclf = QuadraticDiscriminantAnalysis()
    train_test_model(GBclf, '5(b)', Xtrain, Ytrain, Xtest, Ytest)




    # Question 5(c)

    GNBclf = GaussianNB()
    train_test_model(GNBclf, '5(c)', Xtrain, Ytrain, Xtest, Ytest)
   

    # Question 5(d)

    # add Gaussian noise
    sigma = 0.1
    noise = sigma * np.random.normal(size=np.shape(Xtrain))
    Xtrain = Xtrain + noise

    # repeat 5(a)
    fig, axs = plt.subplots(5, 5)
    plt.suptitle('Question 5(d): 25 random MNIST images after adding Gaussian noise.')
    for i in range(len(chosen_ind)):
        axs[i / 5, i % 5].imshow(Xtrain[chosen_ind[i]].reshape((28, 28)), cmap='Greys', interpolation='nearest')
        axs[i / 5, i % 5].axis('off')

    plt.show()

    # repeat 5(b)
    GBclf = QuadraticDiscriminantAnalysis()
    train_test_model(GBclf, '5(b) (with noisy data)', Xtrain, Ytrain, Xtest, Ytest)


    # repeat 5(c)
    GNBclf = GaussianNB()
    train_test_model(GNBclf, '5(c) (with noisy data)', Xtrain, Ytrain, Xtest, Ytest)


    # TODO non programming explanation of why adding noise improves accuracy
    # related to the inductive bias of Gaussian Bayes?
    # given a particular class membership, the probabilities of particular attributes
    # having particular values are independent of each other




# End of Q5 -------------------------------------------------------------------


# ------------------- Script for running the source file ---------------------\
# q1()
# q2()
# Question 3 is non-programming
# q4()
q5()
# ------------------- End of script for running the source file --------------/