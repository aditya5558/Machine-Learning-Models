import numpy as np
import pandas as pd
import math
import datetime
import os
from numpy import inf


train=pd.read_csv("SPECTF_New.csv")
train['Class'] = train['Class'].map({'Yes': 1, 'No': 0})


train_data=train.iloc[:,0:len(train.columns)-1].values
# print train_data
target = train["Class"].values

# train_data = np.array([2,3,4,5,6,7,8,9,10,11])

np.random.seed(10)

def distance(x,y):

	return np.sqrt(np.sum(np.power((x-y),2),axis=1))



def distance1(x,y):

	return np.sqrt(np.sum((x-y)**2))

def init_centroids(num_centroids,train_data):

	centroid1 = np.random.randint(0,len(train_data))
	centroid2 = np.random.randint(0,len(train_data))

	return centroid1,centroid2


def cmeans(num_centroids,train_data,m):


	c1,c2 = init_centroids(2,train_data)

	centroid1 = train_data[c1]
	centroid2 = train_data[c2]

	centroids = np.array([centroid1,centroid2])

	count = 0

	print "Centroids:",centroids

	while True:

		count += 1

		print "Iteration : ",count

		power = 2/(m-1)

		train_data_copy=np.reshape(train_data,(train_data.shape[0],train_data.shape[1],1))
		centroids_new=np.reshape(centroids,(centroids.shape[0],centroids.shape[1],1))

		# print train_data.shape
		# print centroids.shape

		difference = distance(train_data_copy.T,centroids_new)

		# print train_data.T.shape
		
		denom = np.sum(1/np.power(difference,power),axis=0,keepdims=True)


		member = 1/(np.power(difference,power)*denom)

		member[np.isnan(member)] = 1

		# print member
		member_sum = np.sum(np.power(member,m),axis=1,keepdims=True)

		# print np.sum(member,axis=0,keepdims=True)

		new_centroids = np.dot(np.power(member,m),train_data)/member_sum

		# print new_centroids-centroids

		# print new_centroids

		if np.array_equal(np.around(new_centroids,3),np.around(centroids,3)):
			print "Chosen Centroids : \n",new_centroids
			break
		else:
			centroids = new_centroids

	return new_centroids



def accuracy(centroids,target,train_data):

	centroid1 = centroids[0]
	centroid2 = centroids[1]

	cluster1 = []
	cluster2 = []
		
	target_1 = []
	target_2 = []

	for i in range(len(train_data)):

		dis1 = distance1(np.asarray(centroid1),np.asarray(train_data[i]))
		dis2 = distance1(np.asarray(centroid2),np.asarray(train_data[i]))
			
		# print dis1,dis2

		if dis1 < dis2:
			cluster1.append(train_data[i])
			target_1.append(target[i])
		else:
			cluster2.append(train_data[i])
			target_2.append(target[i])

	count = 0

	for i in range(len(cluster1)):
		if target_1[i] == 0:
			count += 1

	for i in range(len(cluster2)):
		if target_2[i] == 1:
			count += 1

	# print len(cluster1)

	return float(count)/(len(cluster1)+len(cluster2))




new_centroids = cmeans(2,train_data,2)
acc = accuracy(new_centroids,target,train_data)

print "Accuracy : ",acc*100
