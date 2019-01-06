import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import os

# Plot the regression line
def plot_regression_line(X, y, b0, b1, title='My Plot', xlabel='X', ylabel='Y'):
    # plotting the actual points as scatter plot
    plt.scatter(X, y, color='b', marker='o', s=20, label='Scatter Plot')
    
    # predicted response vector
    y_pred = b0 + b1 * X
    
    # plotting the regression line
    plt.plot(X, y_pred, color = "k", label='Regression Line')
        
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.show()

def main():
    # read data from csv file
    data = pd.read_csv(os.path.abspath('data/headbrain.csv'))
    print("data.shape = {}".format(data.shape))
    
    # load data to x and y
    X = data['Head Size(cm^3)'].values
    X = X.reshape(X.shape[0], 1)
    y = data['Brain Weight(grams)'].values

    # creating model
    model = LinearRegression()
    # fitting training data
    model.fit(X, y)
    # y prediction
    y_pred = model.predict(X)

    # coefficients
    print('Coefficients: beta0 = {}, beta1 = {}'.format(model.intercept_, model.coef_))

    # calculating RMSE and R2 score
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    r2_score = model.score(X, y)
    print('RMSE = {}'.format(rmse))
    print('R2_Score = {}'.format(r2_score))
    ###TEST
    test_pred = model.predict(np.matrix([3000]))
    print("Predicted value = {}".format(test_pred))
    ###END TEST
    plot_regression_line(X,y,model.intercept_,model.coef_,xlabel='Head Size in cm3', ylabel='Brain Weight in grams')

if __name__ == "__main__":
    main()