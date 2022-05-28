import csv
import numpy as np
import matplotlib.pyplot as plt
import random
import math
from scipy.stats import multivariate_normal

random.seed(1)


with open("EM.csv") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=",", quoting=csv.QUOTE_NONNUMERIC)
    data = [ ]
    for row in csv_reader:
        data.append(row)
data = np.array(data)

n=len(data)


#INITIALIZATION OF PARAMETERS
K=3 #Number of Gaussian models in the mixture
phi=[1/K,1/K,1/K]
random_points=(random.sample(range(0, len(data)), 3)) #Gives three random indexes into data
mu=np.array([data[random_points[0]], data[random_points[1]], data[random_points[2]]])

S0 = np.cov(data.T)
S1 = np.cov(data.T)
S2 = np.cov(data.T)

w0=[0]*len(data)
w1=[0]*len(data)
w2=[0]*len(data)
for i in range(50):
    for j, d in enumerate(data):
        tot=0
        tot += phi[0] * multivariate_normal.pdf(d, mu[0], S0)
        tot += phi[1] * multivariate_normal.pdf(d, mu[1], S1)
        tot += phi[2] * multivariate_normal.pdf(d, mu[2], S2)
        w0[j] = (phi[0] * multivariate_normal.pdf(d, mu[0], S0)) / tot
        w1[j] = (phi[1] * multivariate_normal.pdf(d, mu[1], S1)) / tot
        w2[j] = (phi[2] * multivariate_normal.pdf(d, mu[2], S2)) / tot
    
    #Update Phi
    phi[0] = sum(w0)/n
    phi[1] = sum(w1)/n
    phi[2] = sum(w2)/n
    
    ##Update Mu
    mu[0] = np.dot(w0,data)/sum(w0)
    mu[1] = np.dot(w1,data)/sum(w1)
    mu[2] = np.dot(w2,data)/sum(w2)

    #Update Sigma
    sum0=0
    sum1=0
    sum2=0
    for j, d in enumerate(data):
        sum0 += w0[j] * np.outer(d-mu[0], (d-mu[0]))
        sum1 += w1[j] * np.outer(d-mu[1], (d-mu[1]))
        sum2 += w2[j] * np.outer(d-mu[2], (d-mu[2]))

    S0 = sum0/sum(w0)
    S1 = sum1/sum(w1)
    S2 = sum2/sum(w2)
            
print("Phi1: ", phi[0])
print("Phi2: ", phi[1])
print("Phi3: ", phi[2])

print("Mu1: ", mu[0])
print("Mu2: ", mu[1])
print("Mu3: ", mu[2])

print("Sigma1: ")
print(S0)
print("Sigma2: ")
print(S1)
print("Sigma3: ")
print(S2)  