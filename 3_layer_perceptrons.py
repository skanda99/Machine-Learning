
""" 
Multilayer regression:
# of Layers - 1 input + 1 deep + 1 output
Activation functions - Relu in deep, Linear in output
Cost function - Residual Sum of Squares
"""

import numpy as np
np.random.seed(0)

def Relu(a):
    
    a[a < 0] = 0
    
    return a
    

def RSS(y_pred,y_train):
    
    rss = 0.5 * (y_pred - y_train)**2
#    rss = rss.sum()

    return rss


def der_Relu(a):
    
    a[a < 0] = 0
    a[a >= 0] = 1
    
    return a


def model_train(y_train,X_train,epochs,learning_rate,n2):
    
    n1,c = X_train.shape
    
    # weight matrices
    W1 = np.random.uniform(0,1,(n2,c+1)) * 0.001
    W2 = np.random.uniform(0,1,(1,n2+1)) * 0.001
    
    
    X_train = np.append(X_train,np.ones(shape=(n1,1)),axis=1)
    
    loss_list = []
    
    for epoch in range(1,epochs+1):
        
        del_W1_total = np.zeros(shape=(n2,c+1),dtype=float)
        del_W2_total = np.zeros(shape=(1,n2+1),dtype=float)
        
        for i in range(n1):
            
            
            # Forward phase
            
            z0 = np.copy(X_train[i,:])
            
            a1 = np.matmul(W1,z0)
            
            z1 = Relu(a1)               
            
            z1 = np.append(np.ones(shape=(1,1)),z1)
            
            a2 = np.matmul(W2,z1)
            
            z2 = a2
                  
            
            # Backward phase
            
            delta2 = -(y_train[i]-z2)
            
            del_W2 = delta2 * z1
            
            del_W2_total += del_W2
            
            delta1 = delta2 * W2[0,1:].T * der_Relu(a1)
            
            del_W1 = np.matmul(delta1.reshape((-1,1)),z0.T.reshape((1,-1)))
            
            del_W1_total += del_W1
            
        
        # updating weights
        W1 = W1 - learning_rate * del_W1_total
        
        W2 = W2 - learning_rate * del_W2_total
        
        
        # loss evaluation
        loss = 0
        
        for i in range(n1):
            
            
            # Forward phase
            
            z0 = np.copy(X_train[i,:])
            
            a1 = np.matmul(W1,z0)
            
            z1 = Relu(a1)
            
            z1 = np.append(np.ones(shape=(1,1)),z1)
            
            a2 = np.matmul(W2,z1)
            
            z2 = a2
            
            loss += RSS(z2,y_train[i])
        
        
        loss = float(loss)
        loss_list.append(loss)
        
        print('Epochs:',epoch,'Loss:',loss)
        
        
    return W1,W2,loss_list


def model_predict(X_test,y_test,W1,W2):
    
    n1,c = X_test.shape
    
    X_test = np.append(X_test,np.ones(shape=(n1,1)),axis=1)
    
    loss = 0
    pred = []
        
    for i in range(n1):
        
        
        # Forward phase
        
        z0 = np.copy(X_test[i,:])
        
        a1 = np.matmul(W1,z0)
        
        z1 = Relu(a1)
        
        z1 = np.append(np.ones(shape=(1,1)),z1)
        
        a2 = np.matmul(W2,z1)
        
        z2 = a2
        
        pred.append(z2)
        
        loss += RSS(z2,y_test[i])
    
    
    loss = float(loss)
    pred = np.array(pred)
    
    return pred,loss



"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Test Code
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

from keras.datasets import boston_housing

(x_train, y_train), (x_test, y_test) = boston_housing.load_data()


from sklearn.preprocessing import MinMaxScaler

sc_x = MinMaxScaler(feature_range=(0,1))
sc_y = MinMaxScaler(feature_range=(0,1))


x_train = sc_x.fit_transform(x_train)
y_train = sc_y.fit_transform(y_train.reshape(-1,1))


epochs = 500
learning_rate = 0.002
n2 = 13

W1,W2,loss_list = model_train(y_train,x_train,epochs,learning_rate,n2)


import matplotlib.pyplot as plt

plt.figure()
plt.plot(range(1,epochs+1),loss_list)
plt.title('Loss VS Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')

y_pred,loss = model_predict(x_train,y_train,W1,W2)

plt.figure()
plt.plot(range(404),y_pred,range(404),y_train)
plt.legend(['y_pred_train','y_train'])
plt.title('Training samples Predicted VS Actual Values')
plt.xlabel('samples')
plt.ylabel('values')


x_test = sc_x.transform(x_test)
y_test = sc_y.transform(y_test.reshape(-1,1))

y_pred,loss = model_predict(x_test,y_test,W1,W2)

plt.figure()
plt.plot(range(102),y_pred,range(102),y_test)
plt.legend(['y_pred_test','y_test'])
plt.title('Testing samples Predicted VS Actual Values')
plt.xlabel('samples')
plt.ylabel('values')


print('Loss: Test', loss)
