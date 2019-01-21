import os
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
from scipy.misc import imresize

def load_data():
    if not os.path.exists('data/img_align_celeba'):
        os.mkdir('data/img_align_celeba') # create directory
        
        # The data is shared on kadanze about Creative Applications of Deeplearning by Parag Mital.
        # This data contains pictures of celebrities.
        # Original source can be found here http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html.

        # perform 100 times
        for img_i in range(1, 101):
            f = '000%03d.jpg' % img_i
            url = 'https://s3.amazonaws.com/cadl/celeb-align/' + f
            print(url, end='\r')
            # download data from the url to new directory
            urllib.request.urlretrieve(url, os.path.join('data/img_align_celeba', f))

    else:
        print('data set already downloaded')

def read_data():
    return [os.path.join('data/img_align_celeba', file_i)
        for file_i in sorted(os.listdir('data/img_align_celeba'))
        if '.jpg' in file_i]

def img_crop_square(img):
    if(img.shape[0] > img.shape[1]):
        extra = (img.shape[0] - img.shape[1]) // 2
        crop = img[extra:-extra, :]
    elif(img.shape[1] > img.shape[0]):
        extra = (img.shape[1] - img.shape[0]) // 2
        crop = img[:, extra:-extra]
    else:
        crop = img
    return crop

# def montage(images):
#     """Draw all images as a montage separated by 1 pixel borders.
#     Parameters
#     ----------
#     images : numpy.ndarray
#         Input array to create montage of.
#         Array should be: batch x height x width x channels.
#     Returns
#     -------
#     m : numpy.ndarray
#         Montage image.
#     """
#     if isinstance(images, list):
#         images = np.array(images)
#     img_h = images.shape[1]
#     img_w = images.shape[2]
#     n_plots = int(np.ceil(np.sqrt(images.shape[0])))
#     if(len(images.shape)==4 and images.shape[3]==3):
#         m = np.ones((images.shape[1] * n_plots + n_plots + 1, images.shape[2] * n_plots + n_plots + 1, 3)) * 0.5
#     else:
#         m = np.ones((images.shape[1] * n_plots + n_plots + 1, images.shape[2] * n_plots + n_plots + 1)) * 0.5
#     for i in range(n_plots):
#         for j in range(n_plots):
#             this_filter = i * n_plots + j
#             if(this_filter < images.shape[0]):
#                 this_img = images[this_filter]
#                 m[1 + i + i * img_h:1 + i + (i + 1) * img_h, 1 + j + j * img_w:1 + j + (j + 1) * img_w] = this_img
#     return m

def initialize_with_zeros(dim):
    #Function takes in a parameter dim which is equal to no. of columns or pixels in the dataset
    w = np.zeros((1,dim))
    b = 0
    assert(w.shape == (1, dim)) #Assert statement ensures W and b has the required shape
    assert(isinstance(b, float) or isinstance(b, int))
    print('Initialized w and b: w.shape = {}, b = {}'.format(w.shape, b))
    return w, b

def sigmoid(z):
    # Activation function used to map any real value between 0 and 1
    return 1 / (1 + np.exp(-z))

def propagate(w, b, X, Y):
    """
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if male celebrity, 1 if female celebrity) of size (1, number of examples)
    
    Return:
    cost -- negative log-likelihood cost for logistic regression
    dw -- gradient of the loss with respect to w, thus same shape as w
    db -- gradient of the loss with respect to b, thus same shape as b
    
    """
    m = X.shape[1]

    # FORWARD PROPAGATION (FROM X TO COST)
    A = sigmoid(np.dot(w,X)+b) # compute sigmoid- np.dot is used for matrix multiplication
    cost = (-1/m)*(np.dot(Y,np.log(A.T)) + np.dot((1-Y),np.log((1-A).T))) # compute cost
    
    # BACKWARD PROPAGATION (TO FIND GRAD)
    dw = (1/m)*np.dot((A-Y),X.T)
    db = (1/m)*np.sum((A-Y))
    
    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost) #to make cost a scalar i.e a single value
    assert(cost.shape == ())
    
    grads = {"dw": dw, "db": db}
    return grads, cost

def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    """
    This function optimizes w and b by running a gradient descent algorithm
    
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if male celebrity, 1 if female celebrity) of size (1, number of examples)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- True to print the loss every 100 steps
    
    Returns:
    params -- dictionary containing the weights w and bias b
    costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.
    
    """
    costs = []
    
    for i in range(num_iterations): #This will iterate i from 0 till num_iterations-1
        
        # Cost and gradient calculation
        grads, cost = propagate(w, b, X, Y)
        
        # Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]
        
        # update rule
        w = w - learning_rate * dw
        b = b - learning_rate * db
        
        # Record the costs for every 100th iteration
        if i % 100 == 0:
            costs.append(cost)
            
        # Print the cost every 100 training examples
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
    
    # plot the cost
    plt.figure()
    plt.rcParams['figure.figsize'] = (10.0, 10.0)
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    params = {"w": w, "b": b}
    return params, costs

def predict(w, b, X):
    """
    Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)
    
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    
    Returns:
    Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
    
    """
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    #w = w.reshape(X.shape[0], 1)
     
    # Compute vector "A" predicting the probabilities of having a female celebrity in the picture
    A = sigmoid(np.dot(w,X)+b)
    Y_prediction = np.round(A)
    
    assert(Y_prediction.shape == (1, m))
    return Y_prediction

def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
    """
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
    
    """
    # initialize parameters with zeros
    m_train = X_train.shape[0]
    w, b = initialize_with_zeros(m_train)
    
    # Gradient descent
    parameters, costs = optimize(w, b, X_train, Y_train, num_iterations= num_iterations, learning_rate = learning_rate, print_cost = print_cost)
    
    # Retrieve parameters w and b from dictionary "parameters"
    w = parameters["w"]
    b = parameters["b"]
    
    # Predict test/train set examples
    Y_prediction_train = predict(w, b, X_train)
    Y_prediction_test = predict(w, b, X_test)
    
    # Print train/test Errors
    print("train accuracy: {} %".format(100*(1 - np.mean(np.abs(Y_prediction_train - Y_train)) )))
    print("test accuracy: {} %".format(100*(1 - np.mean(np.abs(Y_prediction_test - Y_test)) )))
    
    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train" : Y_prediction_train,
         "w" : w,
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    
    return d

def print_mislabeled_images(classes, X, y, p):
    """
    Plots images where predictions and truth were different.
    X -- dataset
    y -- true labels
    p -- predictions
    
    """
    a = p + y
    mislabeled_indices = np.asarray(np.where(a == 1))
    plt.figure(figsize=(15.0,5.0))
    num_images = len(mislabeled_indices[0])
    for i in range(num_images):
        index = mislabeled_indices[1][i]
        
        plt.subplot(2, num_images, i + 1)
        plt.imshow(X[:,index].reshape(64,64,3), interpolation='sinc')
        plt.axis('off')
        plt.rc('fontize=20')
        plt.title("Prediction: " + classes[int(p[0,index])] + " \n Class: " + classes[y[0,index]])

def main():
    load_data()
    files = read_data()
    y = np.array([
        1,1,0,1,1,1,0,0,1,1,
        1,0,0,1,0,0,1,1,1,0,
        0,1,0,1,0,1,1,1,1,0,
        1,0,0,1,1,0,0,0,1,1,
        0,1,1,1,1,1,1,0,0,0,
        0,0,0,1,0,0,1,1,1,0,
        0,1,1,0,0,1,0,0,0,0,
        1,0,1,1,1,0,1,1,0,0,
        0,0,1,1,1,1,1,1,1,0,
        0,1,1,1,1,1,1,1,1,1])
    y = y.reshape(1,y.shape[0])
    classes = np.array(['Male', 'Female'])
    y_train = y[:,:80]
    y_test = y[:,80:]

    imgs = []
    for file_i in files:
        img = plt.imread(file_i)
        cropped = img_crop_square(img)
        resized = imresize(cropped,(64,64))
        imgs.append(resized)

    #print(len(imgs))
    # plt.figure(figsize=(10,10))
    # plt.imshow(montage(imgs).astype(np.uint8))
    
    data = np.array(imgs)
    data = data/255
    
    train_x_orig = data[:80,:,:,:]
    test_x_orig = data[80:,:,:,:]
    
    train_x = train_x_orig.reshape(train_x_orig.shape[0],-1).T
    test_x = test_x_orig.reshape(test_x_orig.shape[0],-1).T
    
    m_train = train_x.shape[1]
    m_test = y_test.shape[1]
    num_px = train_x_orig.shape[1]
    
    print('Number of training examples: m_train = ' + str(m_train))
    print('Number of testing examples: m_test = ' + str(m_test))
    print('Height/Width of each image: num_px = ' + str(num_px))
    print('Each image is of size: (' + str(num_px) + ', ' + str(num_px) + ', 3)')
    print('train_x shape: ' + str(train_x_orig.shape))
    print('y_train shape: ' + str(y_train.shape))
    print('test_x shape: ' + str(test_x_orig.shape))
    print('y_test shape: ' + str(y_test.shape))
    print('train_x flatten shape: ' + str(train_x.shape))
    print('test_x flatten shape: ' + str(test_x.shape))
    
    d = model(train_x, y_train, test_x, y_test, num_iterations = 1000, learning_rate = 0.005, print_cost = True)
    print_mislabeled_images(classes, test_x, y_test, d["Y_prediction_test"])

    plt.show()

    ### TEST INPUT NEW IMAGE
    test_img = plt.imread(os.path.join('data','ishin.jpg'))
    plt.figure(figsize=(5.0,5.0))
    plt.imshow(test_img)
    test_img = img_crop_square(test_img)
    test_img = imresize(test_img,(64,64))
    test_img = test_img.reshape(1,-1).T
    test_label = classes[int(predict(d['w'], d['b'], test_img))]
    print('============> TEST input new image: {}'.format(test_label))
    plt.title('Prediction: {}'.format(test_label))
    plt.show()
    ### END TEST    

if __name__ == "__main__":
    main()