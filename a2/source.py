import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt

# disable warnings in sklearn
# needs to be above sklearn imports
# def warn(*args, **kwargs):
#     pass
# import warnings
# warnings.warn = warn

from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from bonnerlib2 import dfContour
import pickle
import time



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

'''
Sigmoid function.
'''
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

'''
Probability function.
'''
def probability(x, w, w0):
    return sigmoid(np.dot(x, w.T) + w0)

'''
Predicts the class with a specific threshold.
'''
def predict(x, w, w0, threshold=0.5):
    return probability(x, w, w0) >= threshold


'''
Calculates precision and recall.
'''
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

    print '\tP(C = 1|x) = ' + str(threshold)
    print '\t\tprecision'; print '\t\t\t' + str(TP / (float(TP) + FP))
    print '\t\trecall'; print '\t\t\t' + str(TP / (float(N1)))





##########  QUESTION 2  ############

def q2():
    print '\nQUESTION 2.\n-----------'

    # Question 2(a)
    # Use gen data to generate training data with 1000 points in class 0 and
    # 500 points in class 1. Use the same means and covariances as in Question 1(b).
    N0, N1 = 1000, 500
    mu0, mu1 = (1, 1), (2, 2)
    cov0, cov1 = 0, -0.9
    X, t = gen_data(mu0, mu1, cov0, cov1, N0, N1)


    # Question 2(b)

    print '\nQuestion 2(b):'
    logReg = LogisticRegression()
    logReg.fit(X, t)

    print '\tbias term w_0:'; print '\t\t' + str(logReg.intercept_)
    print '\tweight vector w:'; print '\t\t' + str(logReg.coef_)


    # Question 2(c)

    print '\nQuestion 2(c):'
    # first way - with score() method
    accuracy1 = logReg.score(X, t)
    print '\taccuracy1:'; print '\t\t' + str(accuracy1)

    # second way - for w and w_0
    # which is correct predictions / number of predictions

    # calculate 1500 x 1 class prediction
    z = predict(X, logReg.coef_, logReg.intercept_, threshold=0.5)

    # compare with target vector t
    accuracy2 = np.equal(z.reshape(z.size), t).mean()
    print '\taccuracy2:'; print '\t\t' + str(accuracy2)

    # difference of accuracies, should be zero
    accuracy_diff = abs(accuracy1 - accuracy2)
    print '\taccuracy difference (should be zero):'; print '\t\t' + str(accuracy_diff)


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
    plt.show()


    # Question 2(e)
    # Generate a scatter plot of the training data
    plt.scatter(X[:, 0], X[:, 1], color=classToColor[t], s=2)
 
    # draw three probability contours on top of the data: 

    # P(C = 1|x) = 0.6
    plt.plot(np.linspace(-3, 6), (-logReg.intercept_[0] + 0.405465 - w[0] * np.linspace(-3, 6)) / w[1], color='b')

    # P(C = 1|x) = 0.5
    plt.plot(np.linspace(-3, 6), (-logReg.intercept_[0] - w[0] * np.linspace(-3, 6)) / w[1], color='k')

    # P(C = 1|x) = 0.05
    plt.plot(np.linspace(-3, 6), (-logReg.intercept_[0] -2.9444 - w[0] * np.linspace(-3, 6)) / w[1], color='r')

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
    print '\nQuestion 2(g):'

    # P(C = 1|x) = 0.6
    get_metrics(X, t, N0, N1, logReg, threshold=0.6)

    # P(C = 1|x) = 0.5
    get_metrics(X, t, N0, N1, logReg, threshold=0.5)

    # P(C = 1|x) = 0.05
    get_metrics(X, t, N0, N1, logReg, threshold=0.05)

# End of Q2 -------------------------------------------------------------------




##########  QUESTION 4  ############

'''
Trains a Gaussian Bayes classifier, prints its accuracy, and
displays the the decision boundaries and contours ontop of the training data.
'''
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
    print '\tAccuracy of Gaussian Bayes clf ' + str_question + ':'; print '\t\t' + str(accuracy)

    # plot the training data
    classToColor = np.array(['r', 'b'])
    plt.scatter(X_train[:, 0], X_train[:, 1], color=classToColor[t_train], s=2)

    # plot the decision boundary using dfContour
    dfContour(GBclf)
    # plt.xlim(-3, 6); plt.ylim(-3, 6); 
    plt.title('Question ' + str_question + ': Decision boundary and contours')
    plt.show()

def q4():

    print '\nQUESTION 4.\n-----------'

    # Question 4(a)

    print '\nQuestion 4(a):'
    gbclf_train_test((1, 1), (2, 2), 0, -0.9, 1000, 500, 10000, 5000, '4(a)')


    # Question 4(b)
    # this is part of the non-programming questions


    # Question 4(c)
    # Generate new training and test data that is the same as
    # that of Question 2 except that cov1 = 0.9, instead of -0.9
    print '\nQuestion 4(c):'
    gbclf_train_test((1, 1), (2, 2), 0, 0.9, 1000, 500, 10000, 5000, '4(c)')


    # Question 4(d)
    # Repeat part (c), but in the training set, put 1,000 points in class 0
    # and 5,000 in class 1; and in the test set, 10,000 points in class 0
    # and 50,000 in class 1
    print '\nQuestion 4(d):'
    gbclf_train_test((1, 1), (2, 2), 0, 0.9, 1000, 5000, 10000, 50000, '4(d)')


    # Question 4(e)
    print '\nQuestion 4(e):'
    print '\tI don\'t know'
    # write a Python function myGDA(Xtrain,Ttrain,Xtest,Ttest) that performs
    # Gaussian DiscriminantAnalysis for two classes


# End of Q4 -------------------------------------------------------------------


'''
Fits a given model to the training data, prints the time taken for fitting and
the training and testing accuracies of the model.

Returns the training and testing accuracy for use in Question 5(h)
'''
def train_test_model(model, str_question, Xtrain, Ytrain, Xtest, Ytest):
    start = time.time()
    model.fit(Xtrain, Ytrain)
    end = time.time()
    duration = end - start
    print '\tTime to fit model {}: {}'.format(str_question, duration)

    # compute and print out the training and test accuracies
    accuracy_train = model.score(Xtrain, Ytrain)
    accuracy_test = model.score(Xtest, Ytest)
    print '\t\tAccuracy of classifier {}:'.format(str_question); print '\t\t\tTraining: ' + str(accuracy_train); print '\t\t\tTesting: ' + str(accuracy_test)

    return accuracy_train, accuracy_test


##########  QUESTION 5  ############

'''
Evaluate Gaussian pdf for dataset X with mean and std_dev.
'''
def gaussian_pdf(X, mean, std_dev):
    exponent = np.exp(-((X - mean)**2 / (2 * std_dev**2 )))
    return (1 / (std_dev * (np.sqrt(2 * np.pi)))) * exponent

'''
Calculates accuracy between predicted and target values.
'''
def get_accuracy(y, t):
    return np.sum(np.equal(y, t)) / float(t.size)

'''
Performs Gaussian naive Bayes for multi-class classification for any number
of classes and data of any dimensionality.
Fits a Gaussian naive Bayes classifier to the training data

Returns both the training and test accuracies.
'''
def myGNB(Xtrain, Ttrain, Xtest, Ttest):

    # sort data by class
    XT_train = np.hstack((Xtrain, Ttrain.reshape((Ttrain.size, 1)))) # add class label as last column
    XT_train = XT_train[XT_train[:,-1].argsort()] # sort by last column, decreasing


    # get mean and std_dev of each class
    K = np.unique(XT_train[:, -1]).size # number of classes

    d = Xtrain.shape[1] # number of features

    mean_vectors = np.zeros((K, d))
    std_devs = np.zeros((K, d))

    probabilities_train = np.zeros((Xtrain.shape[0], K)) # data inputs x class probabilities
    probabilities_test = np.zeros((Xtest.shape[0], K)) # data inputs x class probabilities

    for c in range(K):

        # get boundaries for class in data
        start_c = np.where(XT_train[:, -1] == c)[0][0]
        if c == K-1:
            # last class ends at end of data
            end_c = XT_train.shape[0]
        else:
            # non-last class ends before next class
            end_c = np.where(XT_train[:, -1] == c+1)[0][0] # NOTE: use as exclusive bound
        
        chunk = XT_train[start_c:end_c, :-1] # select this class data without label column

        # compute mean vector for this class
        sums = np.sum(chunk, axis=0) # sum each column
        mean_vec = sums / float(chunk.shape[0]) # divide by num of observations in this class
        mean_vectors[c] = mean_vec # save mean vector for this class


        # compute std_dev
        diff_squared = np.square(chunk - mean_vectors[c])
        variance = np.sum(diff_squared, axis=0) / float(chunk.shape[0] - 1) # also divide by N - 1
        std_devs[c] = np.sqrt(variance)


        # input test data into gaussian pdf with mean and std_dev found above
        pdf_train = gaussian_pdf(Xtrain, mean_vectors[c], std_devs[c]) 
        pdf_test = gaussian_pdf(Xtest, mean_vectors[c], std_devs[c])

        # P(x_1 | C = 0) * P(x_2 | C = 0) * ...
        class_likelihood_train = np.prod(pdf_train, axis=1)
        class_likelihood_test = np.prod(pdf_test, axis=1)

        prior = chunk.shape[0] / float(XT_train.shape[0])
        probability_of_curr_class_per_row_train = class_likelihood_train * prior
        probability_of_curr_class_per_row_test = class_likelihood_test * prior
        probabilities_train[:, c] = probability_of_curr_class_per_row_train
        probabilities_test[:, c] = probability_of_curr_class_per_row_test
    
    # find the class with the largest probability in each row
    ytrain = np.argmax(probabilities_train, axis=1)
    ytest = np.argmax(probabilities_test, axis=1)

    return get_accuracy(ytrain, Ttrain), get_accuracy(ytest, Ttest)



def q5():
    print '\nQUESTION 5.\n-----------'

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
    # train full Gaussian Bayes classifier
    print '\nQuestion 5(b):'
    GBclf = QuadraticDiscriminantAnalysis()
    train_test_model(GBclf, '5(b)', Xtrain, Ytrain, Xtest, Ytest)




    # Question 5(c)
    # train Gaussian Naive Bayes classifier
    print '\nQuestion 5(c):'
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



    # Question 5(e)

    print '\nQuestion 5(e):'
    # repeat 5(b) with noise
    GBclf = QuadraticDiscriminantAnalysis()
    train_test_model(GBclf, '( full Bayes from 5(b) repeated with noisy data )', Xtrain, Ytrain, Xtest, Ytest); print ''


    # repeat 5(c) with noise
    GNBclf = GaussianNB()
    train_test_model(GNBclf, '( Naive Bayes from 5(c) repeated with noisy data )', Xtrain, Ytrain, Xtest, Ytest); print ''



    # Question 5(f)

    print '\nQuestion 5(f):'
    # repeat 5(e) using only the first 6000 elements of the noisy training data
    Xtrain_subset = Xtrain[0:6000, :]
    Ytrain_subset = Ytrain[0:6000]

    # full Bayes
    GBclf = QuadraticDiscriminantAnalysis()
    train_test_model(GBclf, '( full Bayes from 5(e) repeated with only 6000 elements )', Xtrain_subset, Ytrain_subset, Xtest, Ytest); print ''

    # Naive Bayes
    GNBclf = GaussianNB()
    accuracy5f_naive_train, accuracy5f_naive_test = train_test_model(GNBclf, '( Naive Bayes from 5(e) repeated with only 6000 elements )', Xtrain_subset, Ytrain_subset, Xtest, Ytest); print ''


    # Question 5(g)
    # display the mean vector for each digit that the Gaussian Naive Bayes 
    # classifier estimates
    fig, axs = plt.subplots(3, 4)
    plt.suptitle('Question 5(g): means for each digit class.')

    for i in range(len(GNBclf.theta_)):
        axs[i / 4, i % 4].imshow(GNBclf.theta_[i].reshape((28, 28)), cmap='Greys')
        axs[i / 4, i % 4].axis('off')
    axs[2, 2].axis('off')
    axs[2, 3].axis('off')
    plt.show()


    # Question 5(h)
    # Write a Python function myGNB(Xtrain,Ttrain,Xtest,Ttest) that performs
    # Gaussian naive Bayes for multi-class classification
    print '\nQuestion 5(h):'
    accuracy5h_train, accuracy5h_test = myGNB(Xtrain_subset, Ytrain_subset, Xtest, Ytest)

    # Print the training and test errors
    print '\tAccuracy of classifier:'; print '\t\tTraining: ' + str(accuracy5h_train); print '\t\tTesting: ' + str(accuracy5h_test)

    # Print the difference in the training and testing errors against 5(f)
    print '\tDifference in training and test accuracy vs 5(f):'; print '\t\tDiff. train (should be below 0.001): {}'.format(abs(accuracy5f_naive_train - accuracy5h_train)); print '\t\tDiff. test: {}'.format(abs(accuracy5f_naive_test - accuracy5h_test)); 







# End of Q5 -------------------------------------------------------------------


# ------------------- Script for running the source file ---------------------\
q1()
q2()
# Question 3 is non-programming
q4()
q5()
# ------------------- End of script for running the source file --------------/