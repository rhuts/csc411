import numpy as np
import numpy.random as rnd
import time
import pickle
import matplotlib.pyplot as plt
import sklearn.linear_model as lin
import sys, os

# ------------ globals ------------------
with open('data1.pickle', 'rb') as f:
    dataTrain, dataTest = pickle.load(f)
xMin = np.min(dataTrain[:, 0])
xMax = np.max(dataTrain[:, 0])
yMin = np.min(dataTrain[:, 1])
yMax = np.max(dataTrain[:, 1])
fig_num = 1
fig_title = 'title'
# ------------ end of globals -----------

def q1():
    # Question 1
    print '\n\nQuestion 1'
    print '----------'

    # Question 1(a)
    print '\nQuestion 1(a):'
    A = np.floor(10 * rnd.random((3, 4)))
    print A

    # Question 1(b)
    print '\nQuestion 1(b):'
    x = np.floor(10 * rnd.random((1, 4,)))
    print x

    # Question 1(c)
    print '\nQuestion 1(c):'
    B = A.reshape(6, 2)
    print B

    # Question 1(d)
    print '\nQuestion 1(d):'
    C = A + x
    print C

    # Question 1(e)
    print '\nQuestion 1(e):'
    y = x.reshape(4)
    print y

    # Question 1(f)
    print '\nQuestion 1(f):'
    A[0] = y
    print A

    # Question 1(g)
    print '\nQuestion 1(g):'
    A[1] = A[2] - y
    print A

    # Question 1(h)
    print '\nQuestion 1(h):'
    print A[:, :3]

    # Question 1(i)
    print '\nQuestion 1(i):'
    print A[:, 0::2]

    # Question 1(j)
    print '\nQuestion 1(j):'
    print np.min(A)

    # Question 1(k)
    print '\nQuestion 1(k):'
    print np.mean(A, axis=1)

    # Question 1(l)
    print '\nQuestion 1(l):'
    print np.cos(A)

    # Question 1(m)
    print '\nQuestion 1(m):'
    print np.square(np.sum(A, axis=0))

    # Question 1(n)
    print '\nQuestion 1(n):'
    print np.dot(A, np.transpose(A))

    # Question 1(o)
    print '\nQuestion 1(o):'
    print np.mean(np.dot(A, np.transpose(x)))

# End of Q1 -------------------------------------------------------------------

def q2():
    # Question 2    
    print '\n\nQuestion 2'
    print '----------'

    # Question 2(a)
    '''
    Computes AA^T + B, where A is an M x N matrix and B is an M x M matrix
    '''
    def myfun(A, B):
        i, j = np.shape(A)
        C = np.zeros((i, i))

        for row in range(i):
            for col in range(i):
                for n in range(j):
                    C[row, col] += A[row, n] * A[col, n]
                C[row, col] += B[row, col]
        return C

    # Question 2(b)
    # in non_programming_questions.pdf

    # Question 2(c)
    '''
    Measures execution speed of myfun(A, B) vs. numpy.matmul
    '''
    def mymeasure(M, N):
        print '\n\tRunning mymeasure({}, {})'.format(M, N)
        # create matricies
        A = 10 * rnd.random((M, N))
        B = 10 * rnd.random((M, M))

        # measure custom multiply
        start = time.time()
        C1 = myfun(A, B)
        end = time.time()
        duration = end - start
        print '\tExecution time of custom multiply: {}'.format(duration)

        # measure numpy matmul
        start = time.time()
        C2 = np.add(np.matmul(A, A.T), B)
        end = time.time()
        duration = end - start
        print '\tExecution time of numpy.matmul: {}'.format(duration)

        # compute magnitude of C1-C2 with numpy operations
        result = np.sum(np.absolute(np.subtract(C1, C2)))
        print '\tMagnitude of the difference matrix: {}'.format(result)
        

    # Question 2(d)
    print '\nQuestion 2(d):'
    mymeasure(200, 400)
    mymeasure(1000, 2000)

# End of Q2 -------------------------------------------------------------------

# Question 3
# All done in non_programming_questions.pdf

# End of Q3 -------------------------------------------------------------------

# Question 4(a)
'''
Computes and returns a feature matrix Z for data set x
'''
def feature_matrix(x, alpha, beta):

    Z = np.ones(shape=(x.shape[0], alpha.shape[0]))
    for i in range(x.shape[0]):
        Z[i] = x[i]
        Z[i] = 1/(1 + np.exp(beta * (alpha - Z[i]))) # 1 x M)
    return Z
    
# Question 4(b)
'''
Plots the non linear basis functions (sigmoid) defined by alpha and beta
'''
def plot_basis(alpha, beta):

    x = np.linspace(start=xMin, stop=xMax, num=1000)
    Z = feature_matrix(x, alpha, beta)
    plt.plot(Z)
    plt.title("Question 4(b): 6 basis functions"); plt.xlabel('x'); plt.ylabel('y')
    plt.show()

def q4b():
    alpha = np.linspace(start=xMin, stop=xMax, num=6)
    beta = 2
    plot_basis(alpha, beta)

'''
Returns the mean squared error of target values vector t and predicted values vector y
'''
def get_errors(t, y):
    return np.mean(np.square(np.subtract(t, y)))


# Question 4(c)
'''
Uses linear least squares to fit a function to the training data
using basis functions defined by alpha and beta.
Returns the weight vector, w, the bias term,
w0, and the errors, errtrain and errtest
'''
def my_fit(alpha, beta):

    x_train = dataTrain[:, 0]
    t_train = dataTrain[:, 1]

    # feature matrix Z computed in part (a) 
    Z = feature_matrix(x_train, alpha, beta)
    
    # augmented with a leading column of 1's (so it has one more column that Z)
    Zaug = np.hstack((np.ones(Z.shape[0]).reshape(Z.shape[0], 1), Z))
    # print np.around(Z, 3)
    # print np.around(Zaug, 3)

    # solve linear least squares
    w = np.linalg.lstsq(Zaug, t_train, rcond=None)[0] # w is a weight vector (w0, w1, ..., wM)

    # bias term
    w0 = w[0]
    # print 'biast term:'
    # print w0

    # weights
    weights = w[1:]
    # print 'rest of weights: '
    # print weights

    # errors
    err_train = get_errors(t_train, np.matmul(Z, weights) + w0)
    print '\terr_train'; print '\t\t' + str(err_train)

    x_test = dataTest[:, 0]
    t_test = dataTest[:, 1]
    err_test = get_errors(t_test, np.matmul(feature_matrix(x_test, alpha, beta), weights) + w0)
    print '\terr_test'; print '\t\t' + str(err_test)

    return np.array([w0, weights, err_train, err_test])



# Question 4(d)
'''
Plots the function y(x) defined by w, w0,
and using basis functions defined by alpha and beta, as a red curve.
Plots the training points as a blue scatterplot.
'''
def plotY(w, w0, alpha, beta):

    # compute feature matrix
    x = np.linspace(start=xMin, stop=xMax, num=1000)
    Z = feature_matrix(x, alpha, beta) # now have n x M

    # apply weights w
    y = np.matmul(Z, w) # now have n x 1 matrix

    # apply bias term w0
    y = y + w0

    # plots the function y(x) defined by equation (2)
    # plt.figure(fig_num); 
    plt.title(fig_title)
    plt.plot(x, y, color='r')

    # plot the training data
    x_train = dataTrain[:, 0]
    t_train = dataTrain[:, 1]
    plt.scatter(x_train, t_train, color='b')

    # set axis limits and display
    plt.ylim(-7.5, 12)
    plt.xlim(xMin - 0.2, xMax + 0.2)



# Question 4(e)
def q4e():
    global fig_num; global fig_title
    fig_num = 2
    fig_title = 'Question 4(e): the fitted function (5 basis functions)'
    alpha = np.linspace(start=xMin, stop=xMax, num=5)
    beta = 1

    fit = np.array(my_fit(alpha, beta))
    plotY(w=fit[1], w0=fit[0], alpha=alpha, beta=beta)
    plt.show()


# Question 4(f)
def q4f():
    global fig_num; global fig_title
    fig_num = 3
    fig_title = 'Question 4(f): the fitted function (12 basis functions)'
    alpha = np.linspace(start=xMin, stop=xMax, num=12)
    beta = 1
    fit = np.array(my_fit(alpha, beta))
    plotY(fit[1], fit[0], alpha=alpha, beta=beta)
    plt.show()


# Question 4(g)
def q4g():
    global fig_num; global fig_title
    fig_num = 4
    fig_title = 'Question 4(g): the fitted function (19 basis functions)'
    alpha = np.linspace(start=xMin, stop=xMax, num=19)
    beta = 1
    fit = np.array(my_fit(alpha, beta))
    plotY(fit[1], fit[0], alpha=alpha, beta=beta)
    plt.show()


# End of Q4 -------------------------------------------------------------------


# Question 5(a)
'''
Uses regularized least squares regression to fit a function to the
training data using basis functions defined by alpha and beta.
The coefficient of the regularization term is gamma.
Returns [w, w0, err_train, err_val]
'''
def myfit_reg(alpha, beta, gamma):

    # training data
    x_train = dataTrain[:, 0]; t_train = dataTrain[:, 1]

    # validation data
    x_val = dataVal[:, 0]; t_val = dataVal[:, 1]

    # feature matrix Z
    Z = feature_matrix(x_train, alpha, beta)
    
    # augmented with a leading column of 1's (so it has one more column that Z)
    Zaug = np.hstack((np.ones(Z.shape[0]).reshape(Z.shape[0], 1), Z))

    ridge = lin.Ridge(gamma)
    ridge.fit(Zaug, t_train)
    w = ridge.coef_
    w = w[1:]
    w0 = ridge.intercept_

    # compute the errors
    err_train = get_errors(t_train, np.matmul(Z, w) + w0)
    print '\terr_train'; print '\t\t' + str(err_train)

    err_val = get_errors(t_val, np.matmul(feature_matrix(x_val, alpha, beta), w) + w0)
    print '\terr_val'; print '\t\t' + str(err_val)

    return np.array([w, w0, err_train, err_val])



# Question 5(b)
def q5b():
    alpha = np.linspace(start=xMin, stop=xMax, num=19)
    beta = 1
    gamma = 10e-9

    fit = np.array(myfit_reg(alpha, beta, gamma))

    # def plotY(w, w0, alpha, beta)
    global fig_num; global fig_title
    fig_num = 6; fig_title = 'Question 5(b): the fitted function, 19 basis functions, gamma 10-9'
    plotY(w=fit[0], w0=fit[1], alpha=alpha, beta=beta)

    # show testing errors
    x_test = dataTest[:, 0]
    t_test = dataTest[:, 1]
    err_test = get_errors(t_test, np.matmul(feature_matrix(x_test, alpha, beta), fit[0]) + fit[1])
    print '\terr_test'; print '\t\t' + str(err_test)

    plt.show()

# Question 5(c)
def q5c():
    alpha = np.linspace(start=xMin, stop=xMax, num=19)
    beta = 1
    gamma = 0

    fit = np.array(myfit_reg(alpha, beta, gamma))

    # def plotY(w, w0, alpha, beta)
    global fig_num; fig_num = 7
    global fig_title; fig_title = 'Question 5(c): the fitted function, 19 basis functions, gamma 0'
    plotY(w=fit[0], w0=fit[1], alpha=alpha, beta=beta)

    # show testing errors
    x_test = dataTest[:, 0]
    t_test = dataTest[:, 1]
    err_test = get_errors(t_test, np.matmul(feature_matrix(x_test, alpha, beta), fit[0]) + fit[1])
    print '\terr_test'; print '\t\t' + str(err_test)

    plt.show()



# Question 5(d)
'''
Uses validation data to find the best value of the regularization coefficient gamma.
Alpha and beta specify the basis functions.
'''
def best_gamma(alpha, beta):
    fig, axs = plt.subplots(4, 4)

    train_errors = np.zeros(16)
    val_errors = np.zeros(16)
    gammas = np.zeros(16)
    weights = np.zeros((16, 19))
    bias_terms = np.zeros(16)

    # disable prints from myfit_reg()
    sys.stdout = open(os.devnull, 'w')

    i = 0
    for exponent in range(-26, 6, 2):

        fit = np.array(myfit_reg(alpha, beta, 10**exponent))

        # save results
        train_errors[i] = fit[2]
        val_errors[i] = fit[3]
        gammas[i] = 10**exponent
        weights[i] = fit[0]
        bias_terms[i] = fit[1]


        global fig_title; fig_title = ''; plt.suptitle('Question 5(d): best-fitting functions for log(gamma) = -26, -24, ..., 0, 2, 4')

        x = np.linspace(start=xMin, stop=xMax, num=1000)
        Z = feature_matrix(x, alpha, beta) # now have n x M

        # apply weights w
        y = np.matmul(Z, fit[0]) # now have n x 1 matrix

        # apply bias term w0
        y = y + fit[1]

        # plots the function y(x) defined by equation (2)
        plt.title(fig_title)
        axs[i / 4, i % 4].plot(x, y, color='r')

        # plot the training data
        axs[i / 4, i % 4].scatter(dataTrain[:, 0], dataTrain[:, 1], color='b')

        i += 1
    # re-enable prints
    sys.stdout = sys.__stdout__ 
    plt.show()

    # plot training and validation errors vs gamma
    plt.semilogx(gammas, train_errors, color='b')
    plt.semilogx(gammas, val_errors, color='r')
    plt.ylabel('error')
    plt.xlabel('gamma')
    plt.title('Question 5(d): training and validation error')
    plt.show()

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

    # plot weights for different values of gamma
    ax1.plot(weights[0])
    ax1.set_title('Question 5(d): weights for smallest gamma')

    optimal_gamma_idx = np.argmin(val_errors) # with lowest validation error
    ax2.plot(weights[optimal_gamma_idx])
    ax2.set_title('Question 5(d): optimal weights')

    ax3.plot(weights[15])
    ax3.set_title('Question 5(d): weights for largest gamma')
    plt.show()

    # plot the best fitting function
    plotY(weights[optimal_gamma_idx], bias_terms[optimal_gamma_idx], alpha=alpha, beta=beta)
    title = 'Question 5(d): best-fitting function(gamma = ' + str(gammas[optimal_gamma_idx]) + ')'
    plt.title(title)
    plt.show()

    print '\toptimal value of gamma: \n\t\t' + str(gammas[optimal_gamma_idx]) + '\n'
    print '\toptimal value of w_0: \n\t\t' + str(bias_terms[optimal_gamma_idx]) + '\n'

    # print training, validation, and test errors for optimal values of gamma and w
    x_train = dataTrain[:, 0]
    t_train = dataTrain[:, 1]
    err_train = get_errors(t_train, np.matmul(feature_matrix(x_train, alpha, beta), weights[optimal_gamma_idx]) + bias_terms[optimal_gamma_idx])
    print '\terr_train for the optimal values of gamma and w:'; print '\t\t' + str(err_train) + '\n'

    x_val = dataVal[:, 0]
    t_val = dataVal[:, 1]
    err_val = get_errors(t_val, np.matmul(feature_matrix(x_val, alpha, beta), weights[optimal_gamma_idx]) + bias_terms[optimal_gamma_idx])
    print '\terr_val for the optimal values of gamma and w:'; print '\t\t' + str(err_val) + '\n'

    x_test = dataTest[:, 0]
    t_test = dataTest[:, 1]
    err_test = get_errors(t_test, np.matmul(feature_matrix(x_test, alpha, beta), weights[optimal_gamma_idx]) + bias_terms[optimal_gamma_idx])
    print '\terr_test for the optimal values of gamma and w:'; print '\t\t' + str(err_test)

def q5d():
    alpha = np.linspace(start=xMin, stop=xMax, num=19)
    beta = 1
    best_gamma(alpha, beta)



# called for TESTING
def testQ5():

    # display the new data ------------ \\\\
    print 'dataTest'
    print dataTest

    print 'dataVal'
    print dataVal

    fig_num = 5
    fig_title = 'Question 5: dataTest(blue) and dataVal(green)'
    plt.figure(fig_num); plt.title(fig_title)
    x_test = dataTest[:, 0]
    t_test = dataTest[:, 1]
    plt.scatter(x_test, t_test, color='b')

    x_val = dataVal[:, 0]
    t_val = dataVal[:, 1]
    plt.scatter(x_val, t_val, color='g')
    plt.show()
    # end of display the new data ------ ////


# End of Q5 -------------------------------------------------------------------

# Question 6(c)
'''
Computes the gradients of the regularized loss function, where
Z is the feature matrix
T = t is the vector of target values
w is the weight vector
w0 is the bias term
gamma is the coefficient of the regularization term
Returns the formulas from 6(a) and 6(b) after computing
'''
def grad_reg(Z, T, w, w0, gamma):
    # N = Z.shape[0]
    # M = Z.shape[1] - 1
    # gradients = np.zeros(M + 1)
    # gradients_w0 = 

    # y = np.matmul(Z, w)
    # for m in range(M):
    #     # 2Z^T(y - t) + gamma 2 w
    #     gradients[m] = 2 * np.multiply(np.transpose(Z), np.subtract(y - t)) + 2 * gamma * w[m]

    #     # add w0
    #     # 21^T(y - t)
    #     gradients[m] =  np.hstack(2 * np.multiply(np.ones(N), np.subtract(y - t)), gradients[m])


    y = np.matmul(Z, w)
    
    # 2Z^T(y - t) + gamma 2 w
    a = 2 * np.matmul(np.transpose(Z), np.subtract(y, T))
 
    # print gamma * w
    second = 2 * gamma * w
    a = a + second

    # 21^T(y - t)
    b = 2 * np.matmul(np.ones(Z.shape[0]), np.subtract(y, T))
    
    return a, b


# Question 6(d)
'''
Uses gradient descent to fit a function to the training data
alpha and beta define the basis functions
gamma is the coefficient of the regularization term
lrate is the learning rate
'''
def myfit_grad_reg(alpha, beta, gamma, lrate):

    num_iter = 3000000
    num_plots = 9

    # train and test data
    x_train = dataTrain[:, 0]
    t_train = dataTrain[:, 1]
    Z_train = feature_matrix(x_train, alpha, beta)

    # test data
    x_test = dataTest[:, 0]
    t_test = dataTest[:, 1]
    Z_test = feature_matrix(x_test, alpha, beta)


    # Z = feature_matrix(x_train, alpha, beta) # N x M
    T = t_train

    # storage for gradiants
    plot_weights = np.zeros((num_plots, alpha.shape[0]))
    plot_bias_terms = np.zeros((num_plots, alpha.shape[0]))
    curr_plot = 0

    # storage for error
    train_errors = np.zeros(num_iter)
    test_errors = np.zeros(num_iter)

    # init w and w0 randomly
    w = np.random.randn(alpha.shape[0])
    w0 = np.random.randn(1) # [2.97318411]

    for i in range(num_iter):

        # compute the gradients
        grad_w, grad_w0 = grad_reg(Z_train, T, w, w0, gamma)
        

        # update w and w0
        w = w - lrate * grad_w
        w0 = w0 - lrate * grad_w

        # compute and record the training and testing error
        # y = np.matmul(Z, w)
        train_errors[i] = get_errors(t_train, np.matmul(Z_train, w))
        test_errors[i] = get_errors(t_test, np.matmul(Z_test, w))


        # if 6^0, 6^1, 6^2, ..., 6^8 th iteration
        if ((6**curr_plot) == i) and (curr_plot in np.arange(9)):

            # save this plot!
            # print 'plot index: ' + str(curr_plot)
            # print curr_plot
            # print w
            # print w0
            plot_weights[curr_plot] = w
            plot_bias_terms[curr_plot] = w0
            # print '\tplot index ' + str(curr_plot) + ' saved!'
            curr_plot += 1

        # TODO remove this testing condition
        # if curr_plot == 8:
        #     break

    # after 6^0, 6^1, 6^2, ..., 6^8 iterations
    # plot the fitted function at nine different time points:
    fig, axs = plt.subplots(3, 3)
    plt.suptitle('Question 6: fitted function as iterations increase')
    
    x = np.linspace(start=xMin, stop=xMax, num=1000)

    for i in range(num_plots):

        # apply weights w
        Z = feature_matrix(x, alpha, beta)
        y = np.matmul(Z, plot_weights[i]) # now have n x 1 matrix

        # apply bias term w0
        # print plot_bias_terms[i]
        # y = y + plot_bias_terms[i]

        # plot the function
        axs[i / 3, i % 3].plot(x, y, color='r')

        # plot the training data
        axs[i / 3, i % 3].scatter(dataTrain[:, 0], dataTrain[:, 1], color='b')
    plt.show()

    # plot the final fitted function
    plt.title('Question 6: fitted function')
    Z = feature_matrix(x, alpha, beta)
    y = np.matmul(Z, w)
    plt.plot(x, y, color='r')
    plt.scatter(dataTrain[:, 0], dataTrain[:, 1], color='b')
    plt.show()

    # plot the recorded training and test errors
    plt.title('Question 6: training and test error v.s. iterations')
    plt.ylabel('error')
    plt.xlabel('number of iterations')
    plt.plot(np.arange(num_iter), train_errors, color='b')
    plt.plot(np.arange(num_iter), test_errors, color='r')
    plt.show()

    # replot the recorded training and test errors using a log scale on the x axis
    plt.title('Question 6: training and test error v.s. iterations (log scale)')
    plt.semilogx(np.arange(num_iter), train_errors, color='b')
    plt.semilogx(np.arange(num_iter), test_errors, color='r')
    plt.show()

    # plot the last 1,000,000 training errors
    plt.title('Question 6: last 1,000,000 training errors')
    plt.plot(np.arange(1000000), train_errors[-1000000:], color='b')
    plt.show()

    # print the final training and test errors after 3,000,000 iterations
    print '\tfinal training error:'; print '\t\t' + str(train_errors[-1]) + '\n'
    print '\tfinal testing error:'; print '\t\t' + str(test_errors[-1]) + '\n'

    # refit the function using myfit_reg from Question 5(a) 
    # compute and print the training and test errors for this function
    # myfit_reg(alpha, beta, gamma)
    # Returns [w, w0, err_train, err_val]

    sys.stdout = open(os.devnull, 'w')      # disable prints
    fit = myfit_reg(alpha=alpha, beta=beta, gamma=0.0001)
    sys.stdout = sys.__stdout__             # re-enable prints

    train_error_reg = get_errors(t_train, np.matmul(Z_train, fit[0]))
    test_error_reg = get_errors(t_test, np.matmul(Z_test, fit[0]))
    print '\ttraining and test errors for myfit_reg:'
    print '\t\t' + str(train_error_reg) + '\n'
    print '\t\t' + str(test_error_reg) + '\n'

    # compute and print the difference in training errors for gradient descent and myfit_reg
    diff_train = abs(train_errors[-1] - train_error_reg)
    diff_test = abs(test_errors[-1] - test_error_reg)
    print '\tdifference in training errors for gradient descent and myfit_reg:'
    print '\t\t' + str(diff_train) + '\n'
    print '\tdifference in testing errors for gradient descent and myfit_reg:'
    print '\t\t' + str(diff_test) + '\n'


    # print the learning rate
    print '\tlearning rate:'
    print '\t\t' + str(lrate) + '\n'



# TESTING Question 6(d)
def q6d():

    alpha = np.linspace(start=xMin, stop=xMax, num=19)
    beta = 1
    gamma = 0.001
    lrate = 0.001

    myfit_grad_reg(alpha, beta, gamma, lrate)




# ------------------- Script for running the source file ---------------------\
q1()
q2()
# Question 3 is non programming

print '\n\nQuestion 4'
print '----------'
print '\nQuestion 4(b):'
print '\t6 basis functions'
q4b()

print '\nQuestion 4(e):'
print '\tFit a function to training data using 5 basis functions'
q4e()

print '\nQuestion 4(f):'
print '\tFit a function to training data using 12 basis functions'
q4f()

print '\nQuestion 4(g):'
print '\tFit a function to training data using 19 basis functions'
q4g()

with open('data2.pickle', 'rb') as f:
    dataVal, dataTest = pickle.load(f)

print '\n\nQuestion 5'
print '----------'
print '\nQuestion 5(b):'
print '\tFit a function to training data using gamma = 10^-9 and 19 basis functions'
q5b()

print '\nQuestion 5(c):'
print '\tFit a function to training data using gamma = 0 and 19 basis functions'
q5c()

print '\nQuestion 5(d):'
print '\tFind the best gamma value'
q5d()

print '\n\nQuestion 6'
print '----------'
print '\nQuestion 6(d):'
print '\tFit a function to data with least squares regression based on gradient descent'
q6d()

# ------------------- End of script for running the source file --------------/