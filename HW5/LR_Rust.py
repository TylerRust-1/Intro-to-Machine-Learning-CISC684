import numpy as np
from matplotlib import pyplot as plt

np.random.seed(0)
n=10
x_train = np.linspace(0,3,n)
y_train = 2.0*x_train + 1.0 + np.random.randn(n)

#Calculate the direct inverse
X = np.vstack([x_train, np.ones(len(x_train))]).T

#theta = [b;w]
theta = np.dot((np.dot(np.linalg.inv(np.dot(X.T,X)),X.T)),y_train)
print("OLS:") #Ordinary Least Squares
print(theta)
print("b: ", theta[0])
print("w: ", theta[1])

r=3
xBar = sum(x_train)/n
yBar = sum(y_train)/n   #Good

xTilde = x_train-xBar
yTilde = y_train-yBar

xTilde = np.array([xTilde]).T



w = np.dot(np.linalg.inv(np.dot(xTilde.T,xTilde)+(n*r)),(np.dot(xTilde.T,yTilde)))
b = yBar-np.dot(w.T,xBar)

print()
print("Ridge regression where lambda=3")
print("b: ", b)
print("w: ", w)


plt.plot(x_train, y_train, "o")
plt.plot(x_train, 2*x_train+1)
plt.plot(x_train, theta[1]*x_train+theta[0]) #OLS
plt.plot(x_train, w*x_train+b)

plt.legend(["data", "true_line", "OLS", "Ridge Regression"])
plt.show()

