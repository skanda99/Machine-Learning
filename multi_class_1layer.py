""" 
Multiclass classification using Single layer of perceptrons.
activation function: softmax
cost function: cross-entropy
"""

import numpy as np
np.random.seed(0)

def model(X,W):
    
    G = np.matmul(X,W)
    
    E = np.exp(G)
    
    S = np.sum(E,axis=1)
    
    soft_max = E/np.reshape(S,(-1,1))
    
    return soft_max
    


def cross_entropy(y,X,W):
    
    G = np.matmul(X,W)
    
    E = np.exp(G)
    
    S = np.sum(E,axis=1)
    
    Z = np.sum(E*y,axis=1)
    
    L = np.log(Z/S)
    
    return -np.sum(L)/X.shape[0]



def del_cross_entropy(y_i,X_i,W):
    
    G = np.matmul(X_i,W)
    
    E = np.exp(G)
    
    S = np.sum(E,axis=1)
    
    Z = np.sum(G*y_i,axis=1)
    
    T_i = (S-Z)/S
    
    return -np.matmul(T_i.T,X_i).T / y_i.shape[0] 
    


def accuracy(y_t,y_p):
    
    return np.count_nonzero(np.argmax(y_t,axis=1) == np.argmax(y_p,axis=1)) / y_t.shape[0]



def model_train(X_train,y_train,W,epochs,learning_rate):
    
    
    accuracy_list = []
    loss_list = []
    
    for i in range(1,epochs+1):
        
        W_list = []
        for c in range(y_train.shape[1]):
            
            X_train_i = X_train[y_train[:,c] == 1,:]
            y_train_i = y_train[y_train[:,c] == 1,:]
            
            w_c = np.copy(W[:,c])
            w_c = w_c - learning_rate * del_cross_entropy(y_train_i,X_train_i,W)
            
            W_list.append(w_c)
            
        W = np.array(W_list).T
            
    
        loss = float(cross_entropy(y_train,X_train,W))
        
        y_pred = model(X_train,W)
        acc = accuracy(y_train,y_pred)      # define
        
        loss_list.append(loss)
        accuracy_list.append(acc)
        
        
        print("Epochs:",i,"Loss:",loss,"Accuracy:",acc)
        
    
    return W,accuracy_list,loss_list




""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""" Test Code Begins """
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


import pandas as pd
data = pd.read_csv('glass.csv',header=None)
data = data.iloc[:,1:]



from sklearn.preprocessing import LabelEncoder
le_y = LabelEncoder()
data.iloc[:,-1] = le_y.fit_transform(data.iloc[:,-1])


data = data.values
X = data[:,:-1]
y = data[:,-1]


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)


from sklearn.preprocessing import LabelBinarizer
lb_y = LabelBinarizer()
y_train = lb_y.fit_transform(y_train)


n,m = X_train.shape
c = y_train.shape[1]


from sklearn.preprocessing import MinMaxScaler
sc_X = MinMaxScaler(feature_range=(0,1))
X_train = sc_X.fit_transform(X_train)


import numpy as np
X_train = np.append(X_train,np.ones(shape=(n,1)),axis=1)


W = np.random.uniform(low=0.0,high=1.0,size=(m+1,c))


epochs = 250
learning_rate = 0.01

W,accuracy_list,loss_list = model_train(X_train,y_train,W,epochs,learning_rate)


import matplotlib.pyplot as plt
plt.plot(range(1,epochs+1),loss_list)
plt.plot(range(1,epochs+1),accuracy_list)
plt.legend(['Loss','Accuracy'])
plt.title('Metrics VS Epochs')
plt.xlabel('Epochs')
plt.ylabel('Metrics')



X_test = sc_X.transform(X_test)
X_test = np.append(X_test,np.ones(shape=(X_test.shape[0],1)),axis=1)
y_test = lb_y.transform(y_test)

print("Accuracy on testing set:",accuracy(y_test,model(X_test,W)))

