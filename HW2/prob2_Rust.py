import numpy as np
import urllib.request
from sklearn.model_selection import train_test_split
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import math

### Beginning of provided code
### Splits the data into training and test dataset
url =" http://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
raw_data = urllib.request.urlopen(url)
dataset = np.loadtxt(raw_data,delimiter=",")
x = dataset[:,0:-1]
y = dataset[:,-1]
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size = 0.30, random_state = 17)
### End of provided code

###TODO Train an LDA classifier for the problem of spam detection using the 
###     training data. Use the test data to report the test error.

n = len(y_train)

mu1 = np.mean(x_train[y_train==0,:],axis=0)  # Good
mu2 = np.mean(x_train[y_train==1,:],axis=0)  # Spam

Sigma = np.zeros((57,57))
for i in range(n):
  if y_train[i]==0:
    Sigma+= np.outer((x_train[i]-mu1),x_train[i]-mu1)/n # Good
  else :
    Sigma+= np.outer((x_train[i]-mu2),x_train[i]-mu2)/n # Spam
    
q1 = sum(y_train==(0))/n
q2 = sum(y_train==(1))/n

g1 = multivariate_normal(mu1, Sigma)
g2 = multivariate_normal(mu2, Sigma)

#g1.pdf([i])*q1 >= g2.pdf([i])*q2
predicted_spam=0
predicted_good=0
predicted_wrong=0
count=0
for i in x_test:
    if (g1.pdf([i])*q1 > g2.pdf([i])*q2):
        predicted_good+=1
        if(y_test[count]==1):
            predicted_wrong+=1
    else:
        predicted_spam+=1
        if(y_test[count]==0):
            predicted_wrong+=1
    count+=1
        
    


#for i in x_test:
    




test_total=0
real_spam=0
real_good=0

train_total=0
train_spam=0
train_good=0
for i in y_train:
    train_total+=1
    if i==0:
        train_good+=1
    else:
        train_spam+=1

for i in y_test:
    test_total=test_total+1
    if i==0:
        real_good+=1
    else:
        real_spam+=1
        
    

print("Total emails in train data: ",train_total)
print("Train Spam: ",train_spam)
print("Train Good: ",train_good)

print("Total emails in test data: ",test_total)
print("Actual Spam: ",real_spam)
print("Actual Good: ",real_good)

print("Predicted spam: ",predicted_spam)
print("Predicted good: ",predicted_good)
print("Predicted wrong: ",predicted_wrong)
print("Test error = ", predicted_wrong/test_total)

print(x_train)
print(y_train)