
import numpy as np
import urllib.request
from sklearn.model_selection import train_test_split
url =" http://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
raw_data = urllib.request.urlopen(url)
dataset = np.loadtxt(raw_data,delimiter=",")
x = dataset[:,0:-1]
m = np.median(x, axis = 0)
x = (x>m)*2+(x<=m)*1; # making the feature vectors binary
y = dataset[:,-1]
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size = 0.3, random_state = 17)