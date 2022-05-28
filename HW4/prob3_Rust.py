import scipy.io
import numpy as np
from matplotlib import pyplot as plt

data = scipy.io.loadmat("mnist_49_3000.mat")
x = np.array(data["x"])
y = np.array(data["y"][0])
y[y==-1] = 0

n=3000
n_train = 2000
n_test =  1000

# train test split
x_train = x[:,:n_train] #First 2000 columns of the 784 rows
x_test = x[:,n_train:]  #Last 1000 columns of the 784 rows
y_train = y[:n_train]
y_test = y[n_train:]   #Working

#print("x_train: ",x_train)
#print("x_test: ",x_test)
#print("y_train: ",y_train)
#print("y_test: ", y_test)

#Functions for calculating the gradient
def sigmoid(theta,xi):
    return 1/(1+np.exp(-np.inner(theta,xi)))
def grad(theta,x_train,y_train):
    r = 10 # regularization parameter
    gradient = 2*r*theta
    hessian = 2*r*np.eye(x.shape[0])
    n = x_train.shape[1]
    for i in range(n):
        current_data = x_train[:,i]
        gradient += (current_data*(sigmoid(theta,current_data)-y_train[i]))
        hessian +=np.outer(current_data,current_data)*(sigmoid(theta,current_data))*(1-sigmoid(theta,current_data))
    return gradient, hessian


#Running the gradient descent
theta = np.zeros((x.shape[0],))
for i in range(5):
    print(i)
    g,h = grad(theta,x_train,y_train)
    print(sum(g))
    theta-= np.linalg.inv(h).dot(g)
print(sum(theta)/n_train)



# Training Accuracy
prob=[]
for i in range(n_train):
    current_data = x_train[:,i]
    prob.append(sigmoid(theta,current_data))
prob = np.array(prob)
y_pred = (prob>0.5)*1.0
print("Training Accuracy: ",sum(y_pred==y_train)/n_train)


#prediction and test accuracy
prob=[]
for i in range(n_test):
    current_data = x_test[:,i]
    prob.append(sigmoid(theta,current_data))
prob = np.array(prob)
y_pred = (prob>0.5)*1.0
print("Prediction and test accuracy: ",sum(y_pred==y_test)/n_test)
print("Test Error: ",sum(y_pred!=y_test)/n_test)

temp=[]
prob = abs(prob-.5)
for i in range(1000):
    if (y_pred[i]!=y_test[i]):
        temp.append((i, y_test[i],prob[i]))

temp = sorted(temp, key=lambda x: -x[2])
#print(temp)

for i in range(5):
    index = temp[i][0] #change the index to show different images
    image = x[:,index].reshape(28,28)
    plt.imshow(image, interpolation="nearest")
    if(temp[i][1]==1):
        plt.title("Actual: Nine  -  Predicted: Four")
    else:
        plt.title("Actual: Four  -  Predicted: Nine")
    plt.show()
#index = 0 #change the index to show different images
#image = x[:,index].reshape(28,28)
#plt.imshow(image, interpolation="nearest")
#plt.show()
### End of provided Code