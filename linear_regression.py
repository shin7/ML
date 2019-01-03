# Implement Simple Linear Regression with 
# Gradient Descent Approach
#========================================
import random
import numpy as np
import sklearn
from sklearn.datasets.samples_generator import make_regression
import matplotlib.pyplot as plt

# Apply Gradient Descent to minimize error cost function
# Parameters
#   X, y: input data
#   alpha: learning rate
#   ep: convergence criteria
#   max_iter: the maximum iteration
def gradient_descent(X, y, alpha=0.0001, ep=0.0001, max_iter=100000):
    converged = False
    iter = 0
    m = X.shape[0]
    print('Total sample(m) = {}'.format(m))

    # initial theta
    t0 = np.random.random(1)
    t1 = np.random.random(1)

    # cost function, J(theta)
    J = sum([(t0 + t1*X[i] - y[i])**2 for i in range(m)])

    # Iterate Loop
    while not converged:
        # for each training sample, compute the gradient (d/d_theta j(theta))
        grad0 = 1.0/m * sum([(t0 + t1*X[i] - y[i]) for i in range(m)])
        grad1 = 1.0/m * sum([(t0 + t1*X[i] - y[i])*X[i] for i in range(m)])

        # update the theta_temp
        temp0 = t0 - alpha * grad0
        temp1 = t1 - alpha * grad1

        # update theta
        t0 = temp0
        t1 = temp1

        # mean squared error
        e = sum([(t0 + t1*X[i] - y[i])**2 for i in range(m)])

        if abs(J-e) <= ep:
            print('Converged, iterations: ', iter, '!!!')
            converged = True

        J = e # update error 
        iter += 1 # update iter
        
        if iter == max_iter:
            print('Max interactions exceeded!')
            converged = True
    
    return t0,t1

# Plot the regression line
# Parameter:
#   X,y: input data
#   theta: slope(theta[1]) and y-intercept(theta[0])
def plot_regression_line(X, y, theta, title='My Plot', xlabel='X', ylabel='Y'):
    # plotting the actual points as scatter plot
    plt.scatter(X, y, color='b', marker='o', s=20)

    # predicted response vector
    y_pred = theta[0] + theta[1] * X
    # plotting the regression line
    plt.plot(X, y_pred, color = "k")
        
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()
    
def main():
    # generate data
    X, y = make_regression(n_samples=100, n_features=1, n_informative=1, random_state=0, noise=35)
    print("X.shape = {}  \ny.shape = {}".format(X.shape, y.shape))
    
    alpha = 0.003
    ep = 0.001

    theta = gradient_descent(X, y, alpha, ep)
    print('theta0 = {} \ntheta1 = {}'.format(theta[0], theta[1]))

    plot_regression_line(X,y, theta)

if __name__ == "__main__":
    main()
