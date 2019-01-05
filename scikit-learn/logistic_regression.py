import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Function load data from file, return a data frame
def load_data(path, header):
    data_frame = pd.read_csv(path, header=header)
    return data_frame

def main():
    data = load_data(os.path.abspath('data/marks.txt'), None)
    # X = feature values, all the columns except the last column
    X = data.iloc[:,:-1]
    # y = target values, last column of the data frame
    y = data.iloc[:,-1]

    # data preparation
    X = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
    print('X.shape = {},\ny.shape = {}'.format(X.shape,y.shape))

    model = LogisticRegression(solver='lbfgs')
    model.fit(X, y)
    predicted_classes = model.predict(X)
    accuracy = accuracy_score(y,predicted_classes)
    parameters = model.coef_
    print('Parameters = {}\nAccuracy = {}'.format(parameters, accuracy))

    ###TEST
    test_X = np.matrix([1, 70, 60])
    pred = model.predict(test_X)
    if pred >= 0.5:
        print('Admitted; predict value = {}'.format(pred))
    else:
        print('Not Admitted; predict value = {}'.format(pred))
    ####END TEST

if __name__ == "__main__":
    main()