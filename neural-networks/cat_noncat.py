import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage

def load_dataset():
    train_dataset = h5py.File('data/cat_noncat/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels
    
    test_dataset = h5py.File('data/cat_noncat/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels
    
    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

def initialize_with_zeros(dim):
    '''
    This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.
    
    Argument:
    dim -- size of the w vector we want (or number of parameters in this case)
    
    Returns:
    w -- initialized vector of shape (dim, 1)
    b -- initialized scalar (corresponds to the bias)
    '''
    w = np.zeros(shape=(dim, 1))
    b = 0

    return w, b

def sigmoid(z):
    '''
    Compute the sigmoid of z
    
    Arguments:
    z -- A scalar or numpy array of any size.
    
    Return:
    s -- sigmoid(z)
    '''
    s = 1 / (1 + np.exp(-z))
    
    return s
    
def propagate(w, b, X, Y):
    '''
    Implement the cost function and its gradient for the propagation
    
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)
    
    Return:
    cost -- negative log-likelihood cost for logistic regression
    dw -- gradient of the loss with respect to w, thus same shape as w
    db -- gradient of the loss with respect to b, thus same shape as b
    '''
    m = X.shape[1]

    # FORWARD PROPAGATION (FROM X TO COST)
    A = sigmoid(np.dot(w.T, X) + b) # compute activation
    cost = -1 / m * np.sum(Y * np.log(A) + (1-Y) * np.log(1-A))

    # BACKWARD PROPAGATION (TO FIND GRAD)
    dw = (1/m) * np.dot(X,(A-Y).T)
    db = (1/m) * np.sum((A-Y))
    
    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    
    grads = {"dw": dw, "db": db}
    
    return grads, cost

def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    '''
    This function optimizes w and b by running a gradient descent algorithm
    
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- True to print the loss every 100 steps
    
    Returns:
    params -- dictionary containing the weights w and bias b
    grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
    costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.
    '''
    costs = []
    for i in range(num_iterations):
        # Cost and gradient calculation
        grads, cost = propagate(w, b, X, Y)

        # Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]

        # update rule
        w = w - learning_rate * dw
        b = b - learning_rate * db

        # Record the costs
        if(i % 100 == 0):
            costs.append(cost)
        
        # Print the cost every 100 training iterations
        if(print_cost and i % 100 == 0):
            print ("Cost after iteration %i: %f" %(i, cost))
            
    params = {"w": w, "b": b}
    grads = {"dw": dw, "db": db}
    
    return params, grads, costs

def predict(w, b, X):
    '''
    Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)
    
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    
    Returns:
    Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
    '''
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0], 1)

    # Compute vector "A" predicting the probabilities of a cat being present in the picture
    A = sigmoid(np.dot(w.T, X) + b)

    # Convert probabilities A[0,i] to actual predictions p[0,i]
    Y_prediction = np.round(A)

    assert(Y_prediction.shape == (1, m))

    return Y_prediction

def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
    '''
    Builds the logistic regression model
    
    Arguments:
    X_train -- training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
    Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
    X_test -- test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
    Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
    num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
    learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
    print_cost -- Set to true to print the cost every 100 iterations
    
    Returns:
    d -- dictionary containing information about the model.
    '''
    # initialize parameters with zeros
    w, b = initialize_with_zeros(X_train.shape[0])

    # Gradient descent
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)

    # Retrieve parameters w and b from dictionary "parameters"
    w = parameters["w"]
    b = parameters["b"]

    # Predict test/train set examples
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train" : Y_prediction_train,
         "w" : w,
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    
    return d

def test_prediction(w, b, num_px, classes, img_path):
    # Preprocess the image to fit your algorithm.
    image = np.array(ndimage.imread(img_path, flatten=False))
    my_image = scipy.misc.imresize(image, size=(num_px,num_px)).reshape((1, num_px * num_px * 3)).T
    my_predicted_image = predict(w, b, my_image)
    plt.figure()
    plt.imshow(image)
    plt.title('Prediction: {}'.format(classes[int(np.squeeze(my_predicted_image)),].decode("utf-8")))
    print ('y = {}, your algorithm predicts a "{}" picture.'.format(np.squeeze(my_predicted_image), classes[int(np.squeeze(my_predicted_image)),].decode("utf-8")))


def main():
    # Loading the data (cat/non-cat)
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
    
    # Example of a picture
    # index = 7
    # plt.figure()
    # plt.imshow(train_set_x_orig[index])
    # print ('y = {}, it\'s a "{}" picture.'.format(train_set_y[:, index], classes[np.squeeze(train_set_y[:, index])].decode('utf-8')))
    
    # Print dataset info
    m_train = train_set_x_orig.shape[0]
    m_test = test_set_x_orig.shape[0]
    num_px = train_set_x_orig.shape[1]
    
    print('Number of training examples: m_train = {}'.format(m_train))
    print('Number of testing examples: m_test = {}'.format(m_test))
    print('Height/Width of each image: num_px = {}'.format(num_px))
    print('Each image is of size: ({}, {}, 3)'.format(num_px, num_px))
    print('train_set_x shape: {}'.format(train_set_x_orig.shape))
    print('train_set_y shape: {}'.format(train_set_y.shape))
    print('test_set_x shape: {}'.format(test_set_x_orig.shape))
    print('test_set_y shape: {}'.format(test_set_y.shape))
    
    # Reshape the training and test examples
    train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
    test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
    
    print('train_set_x_flatten shape: {}'.format(train_set_x_flatten.shape))
    print('train_set_y shape: {}'.format(train_set_y.shape))
    print('test_set_x_flatten shape: {}'.format(test_set_x_flatten.shape))
    print('test_set_y shape: {}'.format(test_set_y.shape))

    # Standardize the dataset
    train_set_x = train_set_x_flatten / 255
    test_set_x = test_set_x_flatten / 255
    
    d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 1000, learning_rate = 0.005, print_cost = True)

    # index = 3
    # plt.figure()
    # plt.imshow(test_set_x[:,index].reshape((num_px, num_px, 3)))
    # print ('y = {}, you predicted that it is a "{}" picture.'.format(test_set_y[:,index], classes[int(d['Y_prediction_test'][:,index])].decode("utf-8")))

    # Plot learning curve (with costs)
    costs = np.squeeze(d['costs'])
    plt.figure()
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title('Learning rate = {}'.format(d['learning_rate']))

    ### TEST input new image
    test_prediction(d['w'], d['b'], num_px, classes, 'data/cat_noncat/cat.jpg')
    test_prediction(d['w'], d['b'], num_px, classes, 'data/img_align_celeba/000001.jpg')

    plt.show()

if __name__ == "__main__":
    main()