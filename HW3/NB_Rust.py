import numpy as np
import urllib.request
from sklearn.model_selection import train_test_split

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
n = len(y_train)
#print(len(x_train))
#print(len(y_train))

#probability that i[n]=1 when y=0 # Not spam
n11=0
#probability that i[n]=2 when y=0 # Not spam
n12=0
#probability that i[n]=1 when y=1 # Spam
n21=0 
#probability that i[n]=2 when y=1 # Spam
n22=0

index=0

n11=[0 for i in range(0, 57)]
n12=[0 for i in range(0, 57)]
n21=[0 for i in range(0, 57)]
n22=[0 for i in range(0, 57)]
for i in x_train:
    count=0
    for j in i:
        if j==1 and y_train[index]==0:
            n11[count]=n11[count]+1
            
        elif j==2 and y_train[index]==0:
            n12[count]=n12[count]+1
            
        elif j==1 and y_train[index]==1:
            n21[count]=n21[count]+1
            
        elif j==2 and y_train[index]==1:
            n22[count]=n22[count]+1
        count+=1
    index+=1

# probabilities of each outcome
q1=sum(y_train==0) #Good
q2=sum(y_train==1) #Spam
#print("q1: ",q1, "q2: ",q2)



g11=[0 for i in range(0, 57)]
g12=[0 for i in range(0, 57)]
g21=[0 for i in range(0, 57)]
g22=[0 for i in range(0, 57)]

#Laplacian smoothing
for i in range(57):
    g11[i]=(n11[i]+1)/(q1+2) #g1(0)
    g12[i]=(n12[i]+1)/(q1+2) #g1(1)
    g21[i]=(n21[i]+1)/(q2+2) #g2(0)
    g22[i]=(n22[i]+1)/(q2+2) #g2(1)

#print("g-values: ",g11,g12,g21,g22)
#print(g11)
#print(g12)
#print(g21)
#print(g22)


#Applying classifier to test data
actual_spam=(sum(y_test==1))
actual_good=(sum(y_test==0))

predicted_spam=0
predicted_good=0
predicted_wrong=0
count=0

for i in x_test:
    productGood=q1
    productBad=q2
    #Summing good and bad
    temp=0
    for j in i:
        if j==1: 
            productGood=productGood*g11[temp] # Product of chance j==1 and not spam
            productBad=productBad*g21[temp] # Product of chance j==1 and spam
        else:
            productGood=productGood*g12[temp] # Product of chance j==2 and not spam
            productBad=productBad*g22[temp] # Product of chance j==2 and spam
        temp+=1

    if (productGood > productBad):
        predicted_good+=1
        if(y_test[count]==1): #if spam
            predicted_wrong+=1
    else:
        predicted_spam+=1
        if(y_test[count]==0): #if not spam
            predicted_wrong+=1
    count+=1

print("Actual Spam: ",actual_spam)
print("Actual Good: ",actual_good)
print("Predicted spam: ",predicted_spam)
print("Predicted good: ",predicted_good)
print("Predicted wrong: ",predicted_wrong)
print("Test error = ", predicted_wrong/len(y_test))
