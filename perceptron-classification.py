
""" Binary classification using single perceptron """

import numpy as np
np.random.seed(0)

def model(X,w):
    """ Returns predicted values of classes (+1/-1) for each data point """
    
    return np.matmul(X,w)


def cost_func(y,X,w):
    """ Returns cost of classification with w weights vector """
    
    temp = y*np.matmul(X,w.ravel()).ravel()
    
    return -temp[temp < 0].sum()



def del_cost(y,X,w):
    """ Returns gradient of cost function evaluated at w """
    temp = y*(np.matmul(X,w.ravel()))
    
    y = np.copy(y[temp < 0])
    X = np.copy(X[temp < 0,:])
    
    return -np.sum(y.T * X.T,axis=1)



def model_train(X_train,y_train,w,epochs,learning_rate):
    """ Returns weights and accuracy list after training the model """
    accuracy = []
    
    for i in range(1,epochs+1):
        
        w = w - learning_rate * del_cost(y_train,X_train,w)
        
        
        f = model(X_train,w)
        f[f >= 0] = 1.0
        f[f < 0] = -1.0
        
        count = np.count_nonzero(f == y_train)
        
        print("Epochs:",i,"Accuracy:",count/X_train.shape[0])
        
        accuracy.append(count/X_train.shape[0])
        
        
    return w,accuracy
        


# importing dataset
import pandas as pd
data = pd.read_csv('pima-indians-diabetes.csv',header=None)

X = data.iloc[:,:-1].values
y = data.iloc[:,-1].values

# classes: +1 and -1
y[y == 0] = -1


# splitting into train and test cases
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)


# normalization
from sklearn.preprocessing import MinMaxScaler
sc_X = MinMaxScaler(feature_range=(0,1))
X_train = sc_X.fit_transform(X_train)


# adding bias
m,n = X_train.shape
X_train = np.append(X_train,np.ones(shape=(m,1)),axis=1)


# linear classifier
w = np.random.uniform(low=0.0,high=1.0,size=(n+1,))


# training
epochs = 500
learning_rate = 0.0001

w,accuracy = model_train(X_train,y_train,w,epochs,learning_rate)


# plot accuracy over epochs
import matplotlib.pyplot as plt
plt.plot(range(1,epochs+1),accuracy)
plt.title("Accuracy VS Epochs")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")


# testing model
X_test = sc_X.transform(X_test)
m,n = X_test.shape
X_test = np.append(X_test,np.ones(shape=(m,1),dtype=float),axis=1)


y_pred = model(X_test,w)
y_pred[y_pred >= 0] = 1
y_pred[y_pred < 0] = -1

count = np.count_nonzero(y_pred == y_test)
print('Accuracy on testing set:',count/X_test.shape[0])



