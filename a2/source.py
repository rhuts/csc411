import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression


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
    print 'X'; print X
    print 't'; print t
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
    col_z = z[:, np.newaxis] # turn into column vector
    # print 'z'
    # print z
    # print z.shape

    col_t = t[:, np.newaxis] # turn into column vector
    # print 't'
    # print t
    # print t.shape

    table = np.concatenate((col_z, col_t), axis=1) # each row contains [prediction, target]
    # print 'table'
    # print table

    # test and counts number of rows that have [1, 1] which is a True Positive
    TP = np.count_nonzero(np.all(table, axis=1))

    # test and counts number of rows that have [1, 0] which is a False Positive
    FP_table = np.zeros(table.shape)
    FP_table[:, 0] = 1
    FP = np.count_nonzero(np.all(np.equal(table, FP_table), axis=1))

    # test and counts number of rows that have [0, 1] which is a False Negative
    # TODO replace TP + FN with N1
    # FN_table = np.zeros(table.shape)
    # FN_table[:, 1] = 1
    # FN = np.count_nonzero(np.all(np.equal(table, FN_table), axis=1))

    print '\nP(C = 1|x) = ' + str(threshold)
    print '\tprecision'
    print '\t\t' + str(TP / (float(TP) + FP))
    print '\trecall P(C = 1|x) = ' + str(threshold)
    print '\t\t' + str(TP / (float(N1)))

def q2():
    # Question 2(a)
    # Use gen data to generate training data with 1000 points in class 0 and
    # 500 points in class 1. Use the same means and covariances as in Question 1(b).
    N0, N1 = 1000, 500
    mu0, mu1 = (1, 1), (2, 2)
    cov0, cov1 = 0, -0.9
    X, t = gen_data(mu0, mu1, cov0, cov1, N0, N1)

    # FOR TESTING (below commented code)
    print 'X'; print X
    print 't'; print t

    # classToColor = np.array(['r', 'b'])
    # plt.scatter(X[:, 0], X[:, 1], color=classToColor[t], s=2)
    # plt.xlim(-3, 6); plt.ylim(-3, 6)
    # plt.title('Question 1(c): sample cluster data')
    # plt.show()


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


def q3():


# End of Q3 -------------------------------------------------------------------


# ------------------- Script for running the source file ---------------------\
# q1()
# q2()
q3()

# print '\n\nQuestion 4'
# print '----------'
# print '\nQuestion 4(b):'
# print '\t6 basis functions'
# q4b()

# ------------------- End of script for running the source file --------------/