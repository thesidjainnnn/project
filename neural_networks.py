#importing libraries
import pandas as pd
import numpy as np
print('WITH SCRATCH\n\n')
#importing datasets
df_train=pd.read_csv('/Users/siddharthjain/Downloads/ds2_train.csv')
df_test=pd.read_csv('/Users/siddharthjain/Downloads/ds2_test.csv')
X_train=df_train[['x_1','x_2']]
y_train=df_train[['y']]
y_train=y_train.astype(int)
X_test=df_test[['x_1','x_2']]
y_test=df_test[['y']]
y_test=y_test.astype(int)


Z=np.zeros((800,2))
for i in range(800):
    Z[i,y_train.values[i]]=1

print(Z)


w_1= np.random.randn(3,2)
b_1=np.random.randn(3)
w_2=np.random.randn(3,2)
b_2=np.random.randn(2)

#fowrward propagation

def forward_prop(x,w_1,b_1,w_2,b_2):
    #first layer
    M=1/(1+np.exp((x.dot(w_1.T)+b_1)))
    #second layer
    A=M.dot(w_2)+b_2
    expA=np.exp(A)
    y=expA/ expA.sum(axis=1)
    y=y.dropna(how='all',axis=1)
    return y,M
    
#backpropagation
def diff_w_2(H,Z,Y):
    return H.T.dot(Z-Y)

def diff_w_1(X,H,Z,output,w_2):
    dZ=(Z-output).dot(w_2.T)*H*(1-H)
    return X.T.dot(dZ)

def diff_b_2(Z, Y):
    return (Z - Y).sum(axis=0)

def diff_b_1(Z, Y, w_2, H):
    return ((Z - Y). dot(w_2.T) * H * (1 - H)) .sum(axis=0)

#specifying learning rate

#hypertuning
learning_rate=np.arange(0.001,0.005,0.001)
n=np.arange(1000,1010,5)
acc_1=[]
acc_2=[]
for num in n:    
    for lr in learning_rate:
        for epoch in range (num):
            output,hidden=forward_prop(X_train,w_1,b_1,w_2,b_2)
            w_2+=  lr*diff_w_2(hidden,Z,output)
            b_2+= lr*diff_b_2(Z,output)
            w_1+= lr* diff_w_1(X_train,hidden,Z,output,w_2).T
            b_1+= lr*diff_b_1(Z,output,w_2,hidden) 

            #predictions
        # calculating probability
        hidden_output= 1/(1+np.exp(-X_test.dot(w_1.T)-b_1))
        outer_layer_output=hidden_output.dot(w_2)+b_2
        expA=np.exp(outer_layer_output)
        y_pred1=expA/(expA.sum(axis=1))
        y_pred1=y_pred1.dropna(how='all',axis=1)
        y_pred1=np.round(y_pred1)


        #calculating accuracy
        y_pred1_1=y_pred1[1]
        y_pred1_1=np.array(y_pred1_1)
        #print(y_pred1_1)

        acc_value1=[]
            
        for i in range(100):
            if(y_pred1_1[i]==y_test.values[i]):
                acc_value1.append(1)
            else:
                acc_value1.append(0)

        acc_score1=np.sum(acc_value1)
        acc_1.append((acc_score1)*100/len(y_test))
        
        #training dataset
        hidden_output= 1/(1+np.exp(-X_train.dot(w_1.T)-b_1))
        outer_layer_output=hidden_output.dot(w_2)+b_2
        expA=np.exp(outer_layer_output)
        y_pred2=expA/expA.sum(axis=1)
        y_pred2=y_pred2.dropna(how='all',axis=1)
        #print(y_pred2)
        y_pred2=np.round(y_pred2)


        y_pred2_1=y_pred2[1]
        y_pred2_1=np.array(y_pred2_1)
        #print(y_pred1_1)

        #calculating accuracy 
        acc_value2=[]
        for i in range(800):
            if(y_pred2_1[i]==y_train.values[i]):
                acc_value2.append(1)
            else:
                acc_value2.append(0)

        acc_score2=np.sum(acc_value2)
        acc_2.append((acc_score2)*100/len(y_train))



print(acc_2)
print('the max accuracy in train set made from scratch is ',max(acc_2),'%')

print(acc_1)
print('the max accuracy in test set made from scratch is ',max(acc_1),'%')


# calculating probability whether it will be 1 or 0
#now using sklearn and keras library
print('NOW WITH KERAS/SKLEARN')
# importing important libraries
import keras
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from math import sqrt

y_train=to_categorical(y_train)
y_test=to_categorical(y_test)

model=Sequential()
model.add(Dense(500,activation='relu',input_dim=2))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(2, activation='softmax'))
# adam is used as it automatically puts learning rate
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

model.fit(X_train,y_train,epochs=25)

pred_1= model.predict(X_train)
scores_1 = model.evaluate(X_train, y_train)
print('the accuracy of model in training set is ',scores_1[1]*100,'%')
print('the error in the model in training set is ',(1-scores_1[1])*100,'%')

pred_2= model.predict(X_test)
scores_2=model.evaluate(X_test,y_test)
print('the accuracy of model in testing set is ',scores_2[1]*100,'%')
print('the error in the model in testing set is ',(1-scores_2[1])*100,'%')


#applying hypertuning methods


# grid={'nb_epochs': np.arange(1,700,1)}
# nn_cv=GridSearchCV(model,grid,cv=10,scoring="accuracy")
# nn_cv.fit(X_train,y_train.ravel())
# print(nn_cv.best_score_)


