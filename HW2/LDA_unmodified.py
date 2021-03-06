# -*- coding: utf-8 -*-
"""LDA.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/15n1JRiv8kSmmSKNqKoRE4Z0wnFzN_pgg
"""

#Generating random data
import numpy as np 
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


x_train = np.random.randn(20,2)





y_train = np.random.randint(1,3,(20,))
x_train = np.random.randn(20,2)+1.5*y_train.reshape(20,1) 
x_train = x_train-np.mean(x_train)

plt.plot(x_train[y_train==1,0], x_train[y_train==1,1], 'rx')
plt.plot(x_train[y_train==2,0], x_train[y_train==2,1], 'o')
plt.axis('equal')
plt.show()

#training  LDA


n = len(y_train)

mu1 = np.mean(x_train[y_train==1,:],axis=0) 
mu2 = np.mean(x_train[y_train==2,:],axis=0) 

Sigma = np.zeros((2,2))
for i in range(n):
  if y_train[i]==1:
    Sigma+= np.outer((x_train[i,:]-mu1),x_train[i,:]-mu1)/n
  else :
    Sigma+= np.outer((x_train[i,:]-mu2),x_train[i,:]-mu2)/n
q1 = sum(y_train==(1))/n
q2 = sum(y_train==(2))/n


g1 = multivariate_normal(mu1, Sigma)
g2 = multivariate_normal(mu2, Sigma)

'''
x = np.array([0.5,0.5])
if g1.pdf(x)*q1 > g2.pdf(x)*q2:
  print('Label is 1')
else:
  print('Label is 1')
'''


#Finding the decison boundry


from scipy.stats import multivariate_normal
x_0min = -5
x_0max = 5

x_1min = -5
x_1max = 5

Label1 = [[],[]]

Label2 = [[],[]]
x0 = np.linspace(x_0min,x_0max,150)
x1 = np.linspace(x_1min,x_1max,150)

for i in x0:
  for j in x1:
    if g1.pdf([i,j])*q1 >= g2.pdf([i,j])*q2:
      Label1[0].append(i)
      Label1[1].append(j)
    else:
      Label2[0].append(i)
      Label2[1].append(j)

plt.plot(Label1[0], Label1[1], 'rx')
plt.plot(Label2[0], Label2[1], 'o')
plt.tight_layout()
plt.show()

#Trainign QDA

n = len(y_train)

n1 = sum(y_train==1)
n2 = sum(y_train==2)
mu1 = np.mean(x_train[y_train==1,:],axis=0) 
mu2 = np.mean(x_train[y_train==2,:],axis=0) 

Sigma1 = np.zeros((2,2))
Sigma2 = np.zeros((2,2))
for i in range(n):
  if y_train[i]==1:
    Sigma1+= np.outer((x_train[i,:]-mu1),x_train[i,:]-mu1)/n1
  else :
    Sigma2+= np.outer((x_train[i,:]-mu2),x_train[i,:]-mu2)/n2
q1 = sum(y_train==(1))/n
q2 = sum(y_train==(2))/n


g1 = multivariate_normal(mu1, Sigma1)
g2 = multivariate_normal(mu2, Sigma2)

'''
x = np.array([0.5,0.5])
if g1.pdf(x)*q1 > g2.pdf(x)*q2:
  print('Label is 1')
else:
  print('Label is 1')
'''

#Finding the decison boundry
from scipy.stats import multivariate_normal
x_0min = -5
x_0max = 5

x_1min = -5
x_1max = 5

Label1 = [[],[]]

Label2 = [[],[]]
x0 = np.linspace(x_0min,x_0max,150)
x1 = np.linspace(x_1min,x_1max,150)

for i in x0:
  for j in x1:
    if g1.pdf([i,j])*q1 > g2.pdf([i,j])*q2:
      Label1[0].append(i)
      Label1[1].append(j)
    else:
      Label2[0].append(i)
      Label2[1].append(j)

plt.plot(Label1[0], Label1[1], 'rx')
plt.plot(Label2[0], Label2[1], 'o')
plt.tight_layout()
plt.show()