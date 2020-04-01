'''
Created on Mar 22, 2020

@author: Alvin
'''

if __name__ == '__main__':
    pass

import pandas as pd
import numpy as np

train = pd.read_csv('data/train.csv')

#feature scaling 
max_x = max(train['day no'])

#print (max_x)
x = train['day no'] / max_x
#x = train['day no']
y = train['questions']

# based on the lost function for linear regression
def compute_grad(parameter):
    grad = [0, 0]
    grad[0] = 2. * np.mean(parameter[0] + parameter[1] * x - y)
    grad[1] = 2. * np.mean(x * (parameter[0] + parameter[1] * x - y))
    return np.array(grad)

def update_parameter(parameter, alpha, grad):
    new_parameter = np.array(parameter) - alpha * grad
    return new_parameter

# Root Mean Square Error for linear regression 
def rmse(parameter):
    squared_err = (parameter[0] + parameter[1] * x - y) ** 2
    res = np.sqrt(np.mean(squared_err))
    return res


parameter = [1, 1]
alpha = 0.2
tol_L = 0.1

loss_before = 0
loss_current = 0

i = 0
while np.abs(loss_current - loss_before) > tol_L or i == 0:
    #print (grad)
    if i == 0:
        loss_before = rmse(parameter)
    else:
        loss_before = loss_current
    grad = compute_grad(parameter) 
    parameter = update_parameter(parameter, alpha, grad) 
    loss_current = rmse(parameter)
    i += 1
    print('Round %s Diff RMSE %s'%(i, abs(loss_current - loss_before)))

print('\nCoef after scaling: %s    \nIntercept %s'%(parameter[1], parameter[0]))

# to get Original Coef
parameter[1] = parameter[1] / max_x
print('\nOriginal Coef: %s   \nIntercept %s'%(parameter[1] , parameter[0]))
#print('Our Coef: %s   \nOur Intercept %s'%(parameter[1], parameter[0]))

res = loss_current
print('\nFinal root mean square error: %s'%res)


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(train[['day no']], train[['questions']])

#print (lr.coef_)

print('\nSklearn Coef: %s'%lr.coef_[0][0])
print('Sklearn Intercept: %s'%lr.intercept_[0])

x = train['day no']
sklearnres = rmse([lr.intercept_[0], lr.coef_[0][0]])
print('\nSklearn RMSE: %s'%sklearnres)


