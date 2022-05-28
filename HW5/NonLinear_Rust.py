import numpy as np
from matplotlib import pyplot as plt

np.random.seed(0)
n=10
m=8
x_train = np.linspace(0,3,n)
y_train = -x_train**2 + 2*x_train + 2 + 0.5*np.random.randn(n)
x_test = np.linspace(0,3,m)
y_test = -x_test**2 +2*x_test + 2 + 0.5*np.random.randn(m)


phi_x = np.column_stack((x_train, x_train**2, x_train**3, x_train**(1/2)))

phi_bar = sum(phi_x)/4
y_bar = sum(y_train)/10

phi_tilde = phi_x-phi_bar
y_tilde = y_train-y_bar

r=.1
w = np.dot(np.linalg.inv(np.dot(phi_tilde.T,phi_tilde)+(n*r*np.identity(4))),(np.dot(phi_tilde.T,y_tilde)))
b = y_bar- np.dot(w.T,phi_bar)

print("Ridge regression where lambda=0.1")
print("b: ", b)
print("w: ", w)

total = 0
for i in range(4):
    total+= pow((y_train[i]-np.dot(w.T,phi_x[i])-b),2)
print("Training Error: ",total/4)

total=0
for i in range(4,len(y_train)):
    total+=pow((y_train[i]-(w.T.dot(phi_x[i]))-b),2)
print("Test Error: ", total/(len(y_train)-4))

w_list =[]
b_list =[]
r_list =[]
for i in range(1,101):
    r=i*.001
    r_list.append(r)
    w = np.dot(np.linalg.inv(np.dot(phi_tilde.T,phi_tilde)+(n*r*np.identity(4))),(np.dot(phi_tilde.T,y_tilde)))
    b = y_bar- np.dot(w.T,phi_bar)
    w_list.append(w)
    b_list.append(b)

train_error_list =[]
test_error_list =[]
total=0
for i in range(100):
    #Training error
    total = 0
    w = w_list[i]
    b = b_list[i]
    for i in range(4):
        total+= pow((y_train[i]-np.dot(w.T,phi_x[i])-b),2)
    train_error_list.append(total/4)
    
    #Test Error
    total=0
    for i in range(4,len(y_train)):
        total+=pow((y_train[i]-(w.T.dot(phi_x[i]))-b),2)
    test_error_list.append(total/4)

plt.plot(x_train,y_train,"o")
plt.plot(x_test,y_test,"x")
plt.plot(x_train,-x_train**2 +2*x_train + 2)
#plt.plot(r_list, train_error_list)
#plt.plot(r_list,test_error_list)
plt.legend(["training samples","test samples","true line"])
plt.show()

fig = plt.figure()
ax1 = fig.add_subplot()
ax1.set_ylabel('Train Error')
ax1.set_xlabel('Lambda')
ax1.set_title('Training Error as a Function of Lambda')
plt.plot(r_list, train_error_list)
plt.show()

fig = plt.figure()
ax1 = fig.add_subplot()
ax1.set_ylabel('Test Error')
ax1.set_xlabel('Lambda')
ax1.set_title('Test Error as a Function of Lambda')
plt.plot(r_list,test_error_list)
plt.show()

#gonna try and find the optimal lambda value, might be wrong, idk.
res_lt = [ test_error_list[x] + train_error_list[x] for x in range (len (test_error_list))]
print("Optimal value for lambda: ",r_list[res_lt.index(min(res_lt))])