import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import optimize as opt

# Function load data from file, return a data frame
def load_data(path, header):
    data_frame = pd.read_csv(path, header=header)
    return data_frame

# Activation function used to map any real value between 0 and 1
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Computes the weighted sum of inputs
def input_dot_product(theta, x):
    return np.dot(x,theta)

# Computes the hypothesis h(x)
def hypothesis(theta,x):
    return sigmoid(input_dot_product(theta,x))

# Cost function J(theta)
def cost_function(theta, x, y):
    m = x.shape[0]
    cost = -(1/m) * np.sum(y * np.log(hypothesis(theta, x)) + (1-y) * np.log(1-hypothesis(theta,x)))
    return cost

# Gradient of the cost function
def gradient(theta, x, y):
    m = x.shape[0]
    return (1/m) * np.dot(x.T, sigmoid(input_dot_product(theta,x)) - y)

# Function to minimize
def fit(x, y, theta):
    minimizer = opt.fmin_tnc(func=cost_function, x0=theta, fprime=gradient, args=(x, y.flatten()))
    return minimizer[0]

def main():
    # load data from file
    data = load_data('data/marks.txt', None)

    # X = feature values, all the columns except the last column
    X = data.iloc[:,:-1]

    # y = target values, last column of the data frame
    y = data.iloc[:,-1]
    
    # filter out the applicants that got admitted
    admitted = data.loc[y==1]

    # filter out the applicants that din't get admission
    not_admitted = data.loc[y==0]

    plt.scatter(admitted.iloc[:, 0], admitted.iloc[:, 1], s=10, label='Admitted')
    plt.scatter(not_admitted.iloc[:, 0], not_admitted.iloc[:, 1], s=10, label='Not Admitted')

    # data preparation
    #X = np.c_[np.ones((X.shape[0], 1)), X]
    X = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
    y = y[:, np.newaxis]
    theta = np.zeros((X.shape[1], 1))
    print('X.shape = {},\ny.shape = {},\ntheta.shape = {}'.format(X.shape,y.shape,theta.shape))
    
    parameters = fit(X, y, theta)
    print(parameters)

    ###TEST
    test_X = np.array([1, 100, 60])
    predict = hypothesis(parameters, test_X)
    print(predict)
    ####END TEST

    x_values = [np.min(X[:, 1] - 5), np.max(X[:, 2] + 5)]
    y_values = - (parameters[0] + np.dot(parameters[1], x_values)) / parameters[2]
    
    plt.plot(x_values, y_values, label='Decision Boundary')
    plt.xlabel('Marks in 1st Exam')
    plt.ylabel('Marks in 2nd Exam')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()