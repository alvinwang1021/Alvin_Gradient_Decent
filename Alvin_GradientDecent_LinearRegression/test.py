import pandas as pd
import numpy as np

train = pd.read_csv('C:/Users/Alvin/Desktop/data/train.csv')

#print (train)

beta = [1, 1]
alpha = 0.2
tol_L = 0.1


max_x = max(train['day no'])
#print (max_x)
x = train['day no'] / max_x
#print (x)
y = train['questions']
#print (y)
                                                    

#a = np.array([[1], [3], [6]])
#b = np.array([1, 3, 6])
grad = [0, 0]
#print(beta[0])
#print(beta[1])

#print (a)
#print (1 + a)
#print (type(a))

#print (b)
#print (1 + b)
#print (type(b))
#print (beta[0] + beta[1] * x - y)
#print (2* np.mean(beta[0] + beta[1] * x - y))

grad[0] = 2. * np.mean(beta[0] + beta[1] * x - y)
grad[1] = 2. * np.mean(x * (beta[0] + beta[1] * x - y))

#print (grad)
#print (type(grad))
#print (np.array(grad))

grad = np.array(grad)
new_beta = np.array(beta) - alpha * grad
#print (new_beta)


squared_err = (beta[0] + beta[1] * x - y) ** 2
#print (squared_err)
#print (np.mean(squared_err))
res = np.sqrt(np.mean(squared_err))
#print (res)


import numpy as np
from sklearn.linear_model import LinearRegression
X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
y = np.dot(X, np.array([1, 2])) + 3
print (X)
print (y)

