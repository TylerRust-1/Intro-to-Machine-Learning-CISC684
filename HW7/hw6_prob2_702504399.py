import scipy.io
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

#np.reshape,np.linalg.eig, np.mean, np.diag.

data = scipy.io.loadmat("yalefaces.mat")
data = data["yalefaces"]
index = 2413 #2414 different faces
#plt.imshow(data[:,:,index])  #(48,42,2414)
#plt.show()


'''
#Given by professor in office hours
data = data.reshape(2016,2414)
mu = np.mean(data,axis=1)

S = np.zeros((2016, 2016))
for i in range(data.shape[1]):
    S+= np.outer((data[:,i]-mu),(data[:,i]-mu))/2414
'''
# My implementation of data reshaping
facevectors=[]
for i in range(len(data[0][0])):
    facevectors.append(np.reshape(data[:,:,i],2016))
facevectors=np.array(facevectors)
#print(facevectors)
#print(len(facevectors))  # number of faces
#print(len(facevectors[0]))  #length of vector
n=len(facevectors)
facevectors=facevectors.T

xbar=np.mean(facevectors, axis=1)
print(xbar)

S=np.cov(facevectors)/n
#print(S)


values, vectors = np.linalg.eig(S)

explained_variances = []
for i in range(len(values)):
    explained_variances.append(values[i] / np.sum(values))
    
#plt.plot(values)
#plt.show()
print(round(np.sum(explained_variances),1), "\n", explained_variances)

total=0
found=0
for i in range(len(explained_variances)):
    total+=explained_variances[i]
    if total>.95 and found==0:
        print("Principal components needed to represent 95% of variation: ",i+1) #+1 because it starts at 0

        print(f"Total dimensional reduction, {i}/{len(explained_variances)} =",round(1-((i+1)/len(explained_variances)),4))
        found=1
        
    if total>.99:
        print("Principal components needed to represent 99% of variation: ",i+1) #+1 because it starts at 0
        print(f"Total dimensional reduction, {i}/{len(explained_variances)} =",round(1-((i+1)/len(explained_variances)),4))
        break

idx = values.argsort()[::-1]   
values = values[idx]
vectors = vectors[:,idx]
columns = 5
rows = 4
fig = plt.figure(figsize=(48, 42))
for i in range(1, columns*rows+1):
    if i==1:
        img = xbar.reshape(48,42)
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)
    else:
        img = np.reshape(vectors[:,i-2],(48,42))
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)
plt.show()




