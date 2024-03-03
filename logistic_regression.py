#?""""""""""""""""""""""""
#importing necessary libraries
import pandas as pd
import numpy as np

#importing datasets 
df_train=pd.read_csv('/Users/siddharthjain/Downloads/ds1_train.csv')
X_train=df_train[['x_1','x_2']]
y_train=df_train[['y']]
df_test=pd.read_csv('/Users/siddharthjain/Downloads/ds1_test.csv')
X_test=df_test[['x_1','x_2']]
y_test=df_test[['y']]
epsilon=0.001

#defining sigmoid functi
def sigmoid(x):
    return 1/(1+np.exp(-x))

#defining x
def equation (x,w):
    z=np.dot(x,w) 
    return sigmoid(z)
# for simplicity i didn't involve b
#defining loss function
def loss_function(x,y,w):
    m,n=x.shape
    y_eqn= equation(x,w)
# i am using epsilon because it was showing error that log is becoming zero , so to solve that problem
    loss = y * np.log(y_eqn+epsilon) + (1 - y) * np.log(1 - y_eqn+epsilon)
    return -np.mean(loss)
    
#defining cost funtion
def gradient(x,y,w):
    m,n=x.shape
    y_eqn= equation(x,w)
    grad=np.dot(x.T,(y-y_eqn))
    return -grad/x.shape[0]

#defining gradient descent
def gradient_descent(x,y,number_iterations,lr):
    m,n=x.shape
    w=np.zeros(shape=(n,))
    error=[]
    
    for i in range(number_iterations):
        loss=loss_function(x,y,w)
        error.append(loss)
        grad=gradient(x,y,w)
        w=w-(lr*grad)
    
    return w,error  

#hypertuning the model via loops
acc_1=[]
acc_2=[]
learning_rate=np.arange(0.001,0.005,0.001)
num_iter= np.arange(1000,1200,5)


for number_iterations in num_iter:
    for lr in learning_rate:
        w,error=gradient_descent(X_train,y_train.values.ravel(),number_iterations,lr)
        pred_1= np.round(equation(X_test,w))
        acc_value1=[]
        for i in range(100):
            if(pred_1[i]==y_test.values[i]):
                acc_value1.append(1)
            else:
                acc_value1.append(0)

        acc_score1=np.sum(acc_value1)
        acc_1.append((acc_score1)*100/len(y_test))
        

        pred_2= np.round(equation(X_train,w))
        acc_value2=[]
        for i in range(800):
            if(pred_2[i]==y_train.values[i]):
                acc_value2.append(1)
            else:
                acc_value2.append(0)

        acc_score2=np.sum(acc_value2)
        acc_2.append((acc_score2)*100/len(y_train))




max_acc1=max(acc_1)
index_max_acc1=acc_1.index(max_acc1)
#print(index_max_acc1)
print('the accuracy of the model applied from scratch in testing set is', max_acc1 ,'%')


max_acc2=max(acc_2)
index_max_acc2=acc_1.index(max_acc1)
#print(index_max_acc2)
print('the accuracy of the model applied from scratch in training set is', max_acc2 ,'%')

print('\n\n\n\n')

# now applying logistic regression with scikit learn library

#y_train=y_train.values.ravel()
#y_test=y_test.values.ravel()

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
logreg=LogisticRegression()
logreg.fit(X_train,y_train.values.ravel())

pred_1s=logreg.predict(X_test)
score_1s=logreg.score(X_test,y_test.values)
acc_1s=score_1s*100
print('the accuracy of model applied through skicit lib in testing set is',acc_1s,'%')

pred_2s=logreg.predict(X_test)
score_2s=logreg.score(X_train,y_train.values)
acc_2s=score_2s*100
print('the accuracy of model applied through skicit lib in training set is',acc_2s,'%')

#hypertuning

grid={'C':np.logspace(1,10),'max_iter':np.arange(1000,1200,5)}
#use of c to prevent over fitting
sid=GridSearchCV(logreg,grid,cv=5)
sid.fit(X_train,y_train.values.ravel())
print(sid.best_params_)
#print(sid.predict(X_test))


