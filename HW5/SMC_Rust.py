import scipy.io
import numpy as np
from matplotlib import pyplot as plt

data = scipy.io.loadmat("mnist_49_3000.mat")
x = np.array(data["x"])
y = np.array(data["y"][0])
#y[y==-1] = 0

n=3000
n_train = 2000
n_test =  1000

# train test split
x_train = x[:,:n_train] #First 2000 columns of the 784 rows
x_test = x[:,n_train:]  #Last 1000 columns of the 784 rows
y_train = y[:n_train]
y_test = y[n_train:]   #Working


#Functions for calculating the gradient
def grad(w,b,x_train,y_train):
  c = 100
  n = 2000
  db = 0
  dw = np.array(w).reshape(-1,1)
  for i in range(2000):
    if y_train[i]*(w.T.dot(x_train[:,i].reshape(-1,1)) +b)<1:
      db += -(c*y[i])/n
      dw += -(c/n)*x_train[:,i].reshape(-1,1)*y_train[i]
  return dw,db


#Running the gradient descent
w = np.zeros((784,1))
b=0
for i in range(200):
  dw,db = grad(w,b,x_train,y_train)
  w-=0.01*dw
  b-=0.01*db

# Training Accuracy
y_pred=[]
for i in range(n_train):
  current_data = x_train[:,i].reshape(-1,1)
  y_pred.append(np.sign(w.T.dot(current_data)+b)[0][0])
y_pred = np.array(y_pred)
print("Training Accuracy: ",sum(y_pred==y_train)/n_train)

# Test Accuracy and Error
y_pred=[]
prob=[]
for i in range(n_test):
  current_data = x_test[:,i].reshape(-1,1)
  y_pred.append(np.sign(w.T.dot(current_data)+b)[0][0])
  prob.append((w.T.dot(current_data)+b)[0][0])
prob = np.array(prob)
y_pred = np.array(y_pred)
print("Prediction and test accuracy: ",sum(y_pred==y_test)/n_test)
print("Test Error: ",sum(y_pred!=y_test)/n_test)

#Grab top 5 most misclassified
temp=[]
prob = abs(prob)
for i in range(1000):
    if (y_pred[i]!=y_test[i]):
        temp.append((i, y_test[i],prob[i]))

temp = sorted(temp, key=lambda x: -x[2])

total=0
for i in range(2000):
    total+= max(0,1-(y_train[i]*(w.T.dot(x_train[:,i].reshape(-1,1)) +b)))
valueObj = min((pow((w),2))/2 + 100/2000*total)
print("Value of objective function at the optimum: ", valueObj)

for i in range(5):
    index = temp[i][0] #change the index to show different images
    image = x[:,index].reshape(28,28)
    plt.imshow(image, interpolation="nearest")
    if(temp[i][1]==1):
        plt.title("Actual: Nine  -  Predicted: Four")
    else:
        plt.title("Actual: Four  -  Predicted: Nine")
    plt.show()