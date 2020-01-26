""" Neural Network - Single Perceptron regression """


import numpy as np


def model(X,w):
    """ Returns X * w """
    return np.matmul(X,w)


def RSS(E):
    """ Returns residual sum of squares """
    return float(0.5 * np.matmul(E.T,E))


def del_RSS(y,X,w):
    """ Returns gradient of RSS wrt w """
    return -np.matmul(X.T,y - np.matmul(X,w))


def train_model(X_train,y_train,w,epochs,learning_rate):
    """ Returns learnt weights and RSS (error) list for every epoch """
    error_RSS = []
    
    for i in range(1,epochs+1):
    
        w = w - learning_rate * del_RSS(y_train,X_train,w)
        
        f = model(X_train,w)
        rss = RSS(y_train - f)
        
        error_RSS.append(float(rss))
        
        print("Epoch:",i,"Error:",float(rss))
    
    return w,error_RSS



# loading data
from tensorflow.keras.datasets import boston_housing
(X_train,y_train),(X_test,y_test) = boston_housing.load_data()


# scaling 
from sklearn.preprocessing import MinMaxScaler

sc_X = MinMaxScaler(feature_range=(0,1))
X_train = sc_X.fit_transform(X_train)

sc_y = MinMaxScaler(feature_range=(0,1))
y_train = sc_y.fit_transform(y_train.reshape(-1,1))



# appending bias
m,n = X_train.shape
X_train = np.append(X_train,np.ones(shape=(m,1),dtype=float),axis=1)


# deleting unwanted columns
X_train = np.delete(X_train,[1,3],axis=1)
m,n = X_train.shape


# model - Single perceptron
w = np.random.uniform(low=0.0,high=1.0,size=(n,1))


# training the model
epochs = 500
learning_rate = 0.001

w,error_RSS = train_model(X_train,y_train,w,epochs,learning_rate)


# error plot 
import matplotlib.pyplot as plt
plt.plot(range(1,epochs+1),error_RSS)
plt.title("RSS VS Epochs")
plt.xlabel("Epochs")
plt.ylabel("Residual Sum of Squares")


# testing the model
X_test = sc_X.transform(X_test)
y_test = sc_y.transform(y_test.reshape(-1,1))

m,n = X_test.shape
X_test = np.append(X_test,np.ones(shape=(m,1),dtype=float),axis=1)
X_test = np.delete(X_test,[1,3],axis=1)


# prediction on test set
y_pred = model(X_test,w)
rss = float(RSS(y_test - y_pred))

print('Error on testing set:',rss)





