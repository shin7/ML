# Implement Simple Linear Regression with 
# Ordinary Least Square Method Approach
#========================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Estimate the coefficients(beta0, beta1)
def estimate_coefficients(x, y):
    # number of sample
    m = x.shape[0]
    
    # mean of x and y vector
    mean_x, mean_y = np.mean(x), np.mean(y)

    # calculating cross-deviation and deviation about x
    sum_cross_deviation_xy, sum_squared_deviations_x = 0, 0

    for i in range(m):
        sum_cross_deviation_xy += (x[i] - mean_x) * (y[i] - mean_y)
        sum_squared_deviations_x += (x[i] - mean_x) ** 2
    
    # calculating regression coefficients
    beta1 = sum_cross_deviation_xy / sum_squared_deviations_x
    beta0 = mean_y - beta1 * mean_x
    print('Coefficients: beta0 = {}, beta1 = {}'.format(beta0, beta1))
    return (beta0, beta1)

# Plot the regression line
def plot_regression_line(X, y, beta=None, title='My Plot', xlabel='X', ylabel='Y'):
    # plotting the actual points as scatter plot
    plt.scatter(X, y, color='b', marker='o', s=20, label='Scatter Plot')
    
    # predicted response vector
    y_pred = beta[0] + beta[1] * X
    
    # plotting the regression line
    plt.plot(X, y_pred, color = "k", label='Regression Line')
        
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.show()

def main():
    # read data from csv file
    data = pd.read_csv('data/headbrain.csv')
    print("data.shape = {}".format(data.shape))
    
    # load data to x and y
    x = data['Head Size(cm^3)'].values
    y = data['Brain Weight(grams)'].values
    print(x.shape)

    beta = estimate_coefficients(x,y)
    plot_regression_line(x,y,beta,xlabel='Head Size in cm3', ylabel='Brain Weight in grams')

if __name__ == "__main__":
    main()