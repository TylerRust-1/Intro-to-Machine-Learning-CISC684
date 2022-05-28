import numpy as np
import urllib.request
from sklearn.model_selection import train_test_split
# My imports start here
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB




### Beginning of provided code
### Splits the data into training and test dataset
url =" http://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
raw_data = urllib.request.urlopen(url)
dataset = np.loadtxt(raw_data,delimiter=",")
x = dataset[:,0:-1]
m = np.median(x, axis = 0)
x = (x>m)*2+(x<=m)*1; # making the feature vectors binary
y = dataset[:,-1]
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size = 0.3, random_state = 17)
### End of provided code

###TODO Train Naive Bayes classifier for the problem of spam detection using the 
###     training data. Use the test data to report the test error.

gnb = GaussianNB()
y_pred = gnb.fit(x_train, y_train).predict(x_test)
print("Number of mislabeled points out of a total %d points : %d" % (x_test.shape[0], (y_test != y_pred).sum()))