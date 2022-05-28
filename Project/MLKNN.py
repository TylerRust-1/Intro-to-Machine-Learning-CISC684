import math

def euclidean_distance(row1, row2):
	distance = 0.0
	for i in range(len(row1)-1):
		distance += (row1[i] - row2[i])**2
	return math.sqrt(distance)


#this is where I split the code in the .ipynb notebook
############################################################################################

# Locate the most similar neighbors
def get_neighbors(train, test_row, num_neighbors):
	distances = list()
	for train_row in train:
		dist = euclidean_distance(train_row, test_row)
		distances.append((train_row, dist))
	distances.sort(key=lambda tup: tup[1])
	neighbors = list()
	for i in range(num_neighbors):
		neighbors.append(distances[i][0])
	return neighbors


############################################################################################

### - - Change this value to alter size of dataset - - ###
size = 1000
### - - - - - - - - - - - - - - - - - - - - - - - - -  ###

import random
from random import uniform
import numpy as np
import matplotlib.pyplot as plt

dataset = []
sep_val = size / 4

for i in range(size): 
   #points go in bottom left
   if (i <= sep_val): 
      #val1 = uniform(-5.0, 2.0)
      #val2 = uniform(-5.0, 2.0)
      val1 = uniform(-5.0, 4.0)
      val2 = uniform(-5.0, 4.0)
      label1 = 1
      label2 = 1
      dataset.append([val1, val2, [label1, label2]])
   #points go in top left
   elif (i > sep_val and i <= sep_val*2):
      #val1 = uniform(-5.0, 2.0)
      #val2 = uniform(4.0, 10.0)
      val1 = uniform(-5.0, 4.0)
      val2 = uniform(2.0, 10.0)
      label1 = 1
      label2 = 0
      dataset.append([val1, val2, [label1, label2]])
   #points go in bottom right
   elif (i > sep_val*2 and i <= sep_val*3):
      #val1 = uniform(4.0, 10.0)
      #val2 = uniform(-5.0, 2.0)
      val1 = uniform(2.0, 10.0)
      val2 = uniform(-5.0, 4.0)
      label1 = 0
      label2 = 1
      dataset.append([val1, val2, [label1, label2]])
   #points go in top right
   elif (i > sep_val*3 and i <= size):
      #val1 = uniform(4.0, 10.0)
      #val2 = uniform(4.0, 10.0)
      val1 = uniform(2.0, 10.0)
      val2 = uniform(2.0, 10.0)
      label1 = 0
      label2 = 0
      dataset.append([val1, val2, [label1, label2]])

  
data1 = np.array(dataset)
for i in range(len(data1)):
  if data1[i][2][0] == 1 and data1[i][2][1] == 1:
    plt.plot(data1[i][0],data1[i][1],'ro')
  elif data1[i][2][0] == 1 and data1[i][2][1] == 0:
    plt.plot(data1[i][0],data1[i][1],'rx')
  elif data1[i][2][0] == 0 and data1[i][2][1] == 1:
    plt.plot(data1[i][0],data1[i][1],'go')
  elif data1[i][2][0] == 0 and data1[i][2][1] == 0:
    plt.plot(data1[i][0],data1[i][1],'gx')

############################################################################################

#get unique lables
possibleLabels = list()
for i in range(len(dataset)):
  label = dataset[i][2]
  exists = False
  for j in range(len(possibleLabels)):
    if(label == possibleLabels[j]):
      exists = True
  if(exists == False):
    possibleLabels.append(label)
print(possibleLabels)


############################################################################################

#MULTILABEL KNN ALGORITHM
k = 5
#identify k nearest neighbors for xi in the dataset
neighbors_for_each_xi = list()
for i in range(len(dataset)):
  tmpRow = dataset.pop(i) 
  neighbors = get_neighbors(dataset,tmpRow,k)
  neighbors_for_each_xi.append(neighbors)
  dataset.insert(i, tmpRow)
  print("datapoint:", i)
  for neighbor in neighbors:
    print(neighbor)

############################################################################################

#find p(H_j) which represents the number of datapoints with label j for each label
#found by counting the number training examples associated with each label

s = 1 #laplacian smoothing
probHarr = list()
probNegHarr = list()
numLabels = 2
for i in range(len(possibleLabels)):
  count = 0
  for j in range(len(dataset)):
    if(dataset[j][2] == possibleLabels[i]):
      count += 1
  probH = (s+count)/((s*2)+len(dataset))
  probNegH = 1 - probH
  probHarr.append(probH)
  probNegHarr.append(probNegH)
for i in range(len(probHarr)):
  print('P('+str(possibleLabels[i])+') =',probHarr[i])

for j in range(len(probHarr)):
  print('P(-'+str(possibleLabels[j])+') =',probNegHarr[j])


############################################################################################

#maintain two frequency arrays κj and kjBar (kj is an 2d array [0-j][0-k])
#κj[r] counts the number of training examples which have label yj and have exactly r neighbors with label yj
#deltaX records the number of xi’s neighbors with label yj
kj = list([[0]*(k+1)]*(len(possibleLabels)+1))
kjBar = list([[0]*(k+1)]*(len(possibleLabels)+1))
for j in range(len(possibleLabels)):
  for i in range(len(dataset)):
    #number of training samples that have label yj
    deltaX = 0
    for r in range(k):
      if(neighbors_for_each_xi[i][r][2] == possibleLabels[j]):
        deltaX += 1
    if(dataset[i][2] == possibleLabels[j]):
      kj[j][deltaX] += 1
      #and have exactly r neighbors with yj
    if(dataset[i][2] != possibleLabels[j]):
      kjBar[j][deltaX] += 1
      #and have exactly r neighbors with yj
print('kj:',kj)
print('kjBar:',kjBar)

############################################################################################

### - - Change this value to alter size of testdataset - - ###
test_size = 250
### - - - - - - - - - - - - - - - - - - - - - - - - - - -  ###

import random
from random import uniform

test_dataset = []
test_labels = []

for i in range(test_size): 
   val1 = uniform(-5.0, 10.0)
   val2 = uniform(-5.0, 10.0)
   if (val1 >= -5 and val1 < 2 and val2 >= -5 and val2 < 2):
      label1 = 1
      label2 = 1
      test_dataset.append([val1, val2, [label1, label2]])
   #points go in top left
   elif (val1 > -5 and val1 <= 2 and val2 >= 4 and val2 < 10):
      label1 = 1
      label2 = 0
      test_dataset.append([val1, val2, [label1, label2]])
   #points go in bottom right
   elif (val1 >= 4 and val1 < 10 and val2 >= -5 and val2 < 2):
      label1 = 0
      label2 = 1
      test_dataset.append([val1, val2, [label1, label2]])
   #points go in top right
   elif (val1 > 4 and val1 <= 10 and val2 > 4 and val2 <= 10):
      label1 = 0
      label2 = 0
      test_dataset.append([val1, val2, [label1, label2]])

neighbors_for_each_xi_test = list()
for i in range(len(test_dataset)):
  neighbors = get_neighbors(dataset,test_dataset[i],k)
  neighbors_for_each_xi_test.append(neighbors)


############################################################################################

test_point_index = 6
print('datapoint:', test_dataset[test_point_index])
nearestNeighbors = get_neighbors(dataset, test_dataset[test_point_index], k)
print(nearestNeighbors)


############################################################################################

#calculate the number of neighbors with label j
C = list()
for j in range(len(possibleLabels)):
  count = 0
  for i in range(len(nearestNeighbors)):
    if(nearestNeighbors[i][2] == possibleLabels[j]):
        count += 1
  C.append(count)
#print(C)
for i in range(len(C)):
  print('count of neighbors with label '+str(possibleLabels[i])+':', C[i])



############################################################################################

#PREDICT LABEL (label set) OF NEW DATA POINT
possibleOutputLabels = list()
backUpPossibleOutputLabels = list()
for j in range(len(possibleLabels)):
  sumKj = 0
  sumKjBar = 0
  for r in range(len(kj[j])):
    sumKj += kj[j][r]
    sumKjBar += kjBar[j][r]
  probCjHj = (s + kj[j][C[j]]) / (s*(k+1) + sumKj)
  probHjCj = probHarr[j]*probCjHj
  probCjNegHj = (s + kjBar[j][C[j]]) / (s*(k+1) + sumKjBar)
  probNegHjCj = probNegHarr[j]*probCjNegHj
  print('label '+ str(possibleLabels[j]) + ':',probHjCj/probNegHjCj)
  if(probHjCj/probNegHjCj > 1):
    possibleOutputLabels.append(possibleLabels[j])
  elif(probHjCj/probNegHjCj == 1):
    possibleOutputLabels.append(possibleLabels[j])
  else:
    backUpPossibleOutputLabels.append([possibleLabels[j], probHjCj/probNegHjCj])
backUpPossibleOutputLabels.sort(key=lambda i:i[1], reverse=True)
if(possibleOutputLabels):
  print(possibleOutputLabels)
  newLabel = possibleOutputLabels[0]
else:
  print([backUpPossibleOutputLabels[0][0]])
  newLabel = backUpPossibleOutputLabels[0][0]



############################################################################################

data1 = np.array(dataset)
for i in range(len(data1)):
  if data1[i][2][0] == 1 and data1[i][2][1] == 1:
    plt.plot(data1[i][0],data1[i][1],'ro')
  elif data1[i][2][0] == 1 and data1[i][2][1] == 0:
    plt.plot(data1[i][0],data1[i][1],'rx')
  elif data1[i][2][0] == 0 and data1[i][2][1] == 1:
    plt.plot(data1[i][0],data1[i][1],'go')
  elif data1[i][2][0] == 0 and data1[i][2][1] == 0:
    plt.plot(data1[i][0],data1[i][1],'gx')
print('New Datapoint Location:', test_dataset[test_point_index])
if newLabel[0] == 1 and newLabel[1] == 1:
  plt.plot(test_dataset[test_point_index][0],test_dataset[test_point_index][1], 'bo')
elif newLabel[0] == 1 and newLabel[1] == 0:
  plt.plot(test_dataset[test_point_index][0],test_dataset[test_point_index][1], 'bx')
elif newLabel[0] == 0 and newLabel[1] == 1:
  plt.plot(test_dataset[test_point_index][0],test_dataset[test_point_index][1], 'yo')
elif newLabel[0] == 0 and newLabel[1] == 0:
  plt.plot(test_dataset[test_point_index][0],test_dataset[test_point_index][1], 'yx')



############################################################################################

#GET ACCURACY

sum = 0
for i in range(len(test_dataset)):
  possibleOutputLabels = list()
  backUpPossibleOutputLabels = list()
  C = list()
  for j in range(len(possibleLabels)):
    count = 0
    for m in range(len(neighbors_for_each_xi_test[i])):
      if(neighbors_for_each_xi_test[i][m][2] == possibleLabels[j]):
        count += 1
    C.append(count)
    sumKj = 0
    sumKjBar = 0
    for r in range(len(kj[j])):
      sumKj += kj[j][r]
      sumKjBar += kjBar[j][r]
    probCjHj = (s + kj[j][C[j]]) / (s*(k+1) + sumKj)
    probHjCj = probHarr[j]*probCjHj
    probCjNegHj = (s + kjBar[j][C[j]]) / (s*(k+1) + sumKjBar)
    probNegHjCj = probNegHarr[j]*probCjNegHj
    #print('label '+ str(possibleLabels[j]) + ':',probHjCj/probNegHjCj)
    if(probHjCj/probNegHjCj > 1):
      possibleOutputLabels.append(possibleLabels[j])
    elif(probHjCj/probNegHjCj == 1):
      possibleOutputLabels.append(possibleLabels[j])
    else:
      backUpPossibleOutputLabels.append([possibleLabels[j], probHjCj/probNegHjCj])
  #print(i,possibleOutputLabels)
  if(possibleOutputLabels):
    if(possibleOutputLabels[0] == test_dataset[i][2]):
      sum += 1
  else:
    if(backUpPossibleOutputLabels == test_dataset[i][2]):
      sum += 1

accuracy = (sum/len(test_dataset))*100
print('accuracy of MLKNN:', accuracy)


