import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
random.seed(1)

data = [ ]
with open("data1.csv") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=",", quoting=csv.QUOTE_NONNUMERIC)
    for row in csv_reader:
        data.append(row)
data = np.array(data)
#plt.plot(data[:,0],data[:,1],"o")
#plt.show()
#print(data)


def distance(p1, p2):
    return(sum((p1 - p2)**2))**0.5

def closest_centroids(initial_centroids, data):
    assigned_centroid = []
    for i in data:
        dist=[]
        for j in initial_centroids:
            dist.append(distance(i, j))
        assigned_centroid.append(np.argmin(dist))
    return assigned_centroid

def new_centroids(clusters, data):
    new_centroids = []
    new_df = pd.concat([pd.DataFrame(data), pd.DataFrame(clusters, columns=['cluster'])],
                      axis=1)
    for c in set(new_df['cluster']):
        current_cluster = new_df[new_df['cluster'] == c][new_df.columns[:-1]]
        cluster_mean = current_cluster.mean(axis=0)
        new_centroids.append(cluster_mean)
    return new_centroids
    
    
init_centroids = random.sample(range(0, len(data)), 2)

centroids = []
for i in init_centroids:
    centroids.append(data[i])
centroids=np.array(centroids)


for i in range(5):
    get_centroids = closest_centroids(centroids, data)
    centroids = new_centroids(get_centroids, data)
    
    
x=data[:,0]
y=data[:,1]
Colors=["red", "blue"]
centroidColors=["black","green"]
xL=[]
yL=[]
wL=[]
zL=[]
sum_cluster1=0
sum_cluster2=0
for i in range(len(data)): #Labels of the clusters
    if (distance(data[i],centroids[0])<=distance(data[i],centroids[1])):
        xL.append(x[i])
        yL.append(y[i])
        sum_cluster1+=pow(distance(data[i],centroids[0]),2)
        #plt.scatter(data[i, 0], data[i, 1], alpha=0.1)
    else:
        wL.append(x[i])
        zL.append(y[i])
        sum_cluster2+=pow(distance(data[i],centroids[1]),2)
        #plt.scatter(data[i, 0], data[i, 1], alpha=0.1)
plt.scatter(xL,yL,c="red")
plt.scatter(wL,zL,c="blue")
    
plt.scatter(np.array(centroids)[0, 0], np.array(centroids)[0, 1], color='black')
plt.scatter(np.array(centroids)[1, 0], np.array(centroids)[1, 1], color='green')
#plt.scatter(data[:, 0], data[:, 1], alpha=0.1)

print("w1:",sum_cluster1)
print("w2:",sum_cluster2)
print("w_hat",sum_cluster1+sum_cluster2)
    

plt.show()



data2 = [ ]
with open("data2.csv") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=",", quoting=csv.QUOTE_NONNUMERIC)
    for row in csv_reader:
        data2.append(row)
data2 = np.array(data2)

init_centroids = random.sample(range(0, len(data2)), 2)

centroids = []
for i in init_centroids:
    centroids.append(data2[i])
centroids=np.array(centroids)


for i in range(10):
    get_centroids = closest_centroids(centroids, data2)
    centroids = new_centroids(get_centroids, data2)
    
    
x=data2[:,0]
y=data2[:,1]
Colors=["red", "blue"]
centroidColors=["black","green"]
xL=[]
yL=[]
wL=[]
zL=[]
for i in range(len(data2)): #Labels of the clusters
    if (distance(data2[i],centroids[0])<=distance(data2[i],centroids[1])):
        xL.append(x[i])
        yL.append(y[i])
    else:
        wL.append(x[i])
        zL.append(y[i])
plt.scatter(xL,yL,c="red")
plt.scatter(wL,zL,c="blue")
    
plt.scatter(np.array(centroids)[0, 0], np.array(centroids)[0, 1], color='black')
plt.scatter(np.array(centroids)[1, 0], np.array(centroids)[1, 1], color='green')
plt.show()

