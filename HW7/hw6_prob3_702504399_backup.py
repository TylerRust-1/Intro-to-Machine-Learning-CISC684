import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data1 = [ ]
with open("data1.csv") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=",", quoting=csv.QUOTE_NONNUMERIC)
    for row in csv_reader:
        data1.append(row)
data1 = np.array(data1)

kmeans = KMeans(n_clusters=2)
kmeans = kmeans.fit(data1)
labels = kmeans.predict(data1)

x=data1[:,0]
y=data1[:,1]
Colors=["red", "blue"]
centroidColors=["black","green"]
for i in range(2): #Labels of the clusters 
    xL=[]
    yL=[]
    for k in range(len(data1)):
        if labels[k]==i: #Data points of each cluster 
            xL.append(x[k])
            yL.append(y[k])
    plt.scatter(xL,yL,c=Colors[i])
    
print(kmeans.cluster_centers_)
plt.scatter(kmeans.cluster_centers_[0], kmeans.cluster_centers_[0], s=50, c='black')
plt.scatter(kmeans.cluster_centers_[1], kmeans.cluster_centers_[1], s=50, c='green')

plt.show()



data2 = [ ]
with open("data2.csv") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=",", quoting=csv.QUOTE_NONNUMERIC)
    for row in csv_reader:
        data2.append(row)
data2 = np.array(data2)
kmeans = KMeans(n_clusters=2)
kmeans = kmeans.fit(data2)
labels = kmeans.predict(data2)

x=data2[:,0]
y=data2[:,1]
Colors=["red", "blue"]
centroidColors=["black","green"]
for i in range(2): #Labels of the clusters 
    xL=[]
    yL=[]
    for k in range(len(data1)):
        if labels[k]==i: #Data points of each cluster 
            xL.append(x[k])
            yL.append(y[k])
    plt.scatter(xL,yL,c=Colors[i])
    
plt.scatter(kmeans.cluster_centers_[0], kmeans.cluster_centers_[0], s=50, c='black')
plt.scatter(kmeans.cluster_centers_[1], kmeans.cluster_centers_[1], s=50, c='green')

plt.show()

