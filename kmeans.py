import numpy as np
import pandas as pd
import math
import datetime
import os


train=pd.read_csv("SPECTF_New.csv")
train['Class'] = train['Class'].map({'Yes': 1, 'No': 0})


train_data=train.iloc[:,0:len(train.columns)-1].values
#print train_data
target = train["Class"].values


np.random.seed(10)

def distance(x,y):

	return np.sqrt(np.sum((x-y)**2))


def init_centroids(num_centroids,train_data):

	centroid1 = np.random.randint(0,len(train_data))
	centroid2 = np.random.randint(0,len(train_data))

	return centroid1,centroid2


def kmeans(num_centroids,train_data,target):

	c1,c2 = init_centroids(2,train_data)

	centroid1 = train_data[c1]
	centroid2 = train_data[c2]

	centroids = [centroid1,centroid2]

	count = 0

	while True:

		count += 1

		print "Iteration",count

		centroid1 = centroids[0]
		centroid2 = centroids[1]

		cluster1 = []
		cluster2 = []
		
		target_1 = []
		target_2 = []

		for i in range(len(train_data)):

			dis1 = distance(np.asarray(centroid1),np.asarray(train_data[i]))
			dis2 = distance(np.asarray(centroid2),np.asarray(train_data[i]))
			
			# print dis1,dis2

			if dis1 < dis2:
				cluster1.append(train_data[i])
				target_1.append(target[i])
			else:
				cluster2.append(train_data[i])
				target_2.append(target[i])
			

		new_centroid1 = np.mean(np.asarray(cluster1),axis=0)
		new_centroid2 = np.mean(np.asarray(cluster2),axis=0)
		# print centroids

		new_centroids = [new_centroid1 , new_centroid2]


		if np.array_equal(np.asarray(new_centroids),np.asarray(centroids)) :
			break

		else:
			centroids = new_centroids

	return centroids,cluster1,cluster2,target_1,target_2


def accuracy(cluster1,cluster2,target_1,target_2):

	count = 0

	for i in range(len(cluster1)):
		if target_1[i] == 0:
			count += 1

	for i in range(len(cluster2)):
		if target_2[i] == 1:
			count += 1

	return float(count)/(len(cluster1)+len(cluster2))


[centroids,cluster1,cluster2,target_1,target_2] = kmeans(2,train_data,target)

acc = accuracy(cluster1,cluster2,target_1,target_2)

print len(cluster1),len(target_1)

print 'Centroids',centroids
print 'Cluster1',len(cluster1)
print 'Cluster2',len(cluster2)

print "Accuracy:",100*acc
