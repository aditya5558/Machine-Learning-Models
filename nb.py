import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import math
import datetime
import os


train=pd.read_csv("SPECTF_New.csv")
train['Class'] = train['Class'].map({'Yes': 1, 'No': 0})


#train = shuffle(train)

train_data=train.iloc[:,0:len(train.columns)-1].values
#print train_data
target_glob = train["Class"].values



def splitDataset(dataset, n):
	
	trainSet = []
	testSet = []
	target_train=[]
	target_test=[]

	size = int(math.ceil(len(dataset) / 10))

	for i in range(0, 10):
		
		if i != n:
			trainSet.extend(dataset[i * size : i*size + size])
			target_train.extend(target_glob[i*size:i*size+size])
		else:
			for j in range(0, size):
				testSet.append(dataset[i * size + j])
				target_test.append(target_glob[i*size+j])
				

	trainSet.extend(dataset[9*size+size:])

	return [trainSet, testSet, target_train,target_test]


def norm(x,mean,sd):

	var=float(sd)**2
	denom=(2*np.pi*var)**.5
	num=math.exp(-(float(x)-float(mean))**2/(2*var))

	return num/denom

def mean_sd(train,target_train):

	mean=[[],[]]
	sd=[[],[]]

	class_1=np.nonzero(target_train)
	class_0=np.nonzero(1-np.array(target_train))
	

	att_1=train[class_1,:]
	att_0=train[class_0,:]

	att_0=att_0.reshape(att_0.shape[1],att_0.shape[2])
	att_1=att_1.reshape(att_1.shape[1],att_1.shape[2])


	mean[0]=np.mean(att_0,axis=0)
	mean[1]=np.mean(att_1,axis=0)

	sd[0]=np.std(att_0,axis=0)
	sd[1]=np.std(att_1,axis=0)

	#print sd[0],'\n',sd[1]


	return mean,sd


def naive_bayes(test_data,target_test,mean,sd):

	
	tp=0
	tn=0
	fp=0
	fn=0

	for i in range(len(test_data)):

		class_1=np.array(np.nonzero(target_glob))
		class_0=np.array(np.nonzero(1-target_glob))

		prob_1=float(class_1.shape[1])/len(train_data)
		prob_0=float(class_0.shape[1])/len(train_data)

		#print prob_0,'\n',prob_1

		for j in range(len(test_data[i])):

		 	prob_0*=norm(test_data[i][j],mean[0][j],sd[0][j])
		 	prob_1*=norm(test_data[i][j],mean[1][j],sd[1][j])

		#print prob_0,'\n',prob_1

		if prob_0 > prob_1:
			pred=0
		else:
			pred=1

		e=target_test[i]-pred

		if e==1:
			fn+=1
		elif e==-1:
			fp+=1
		elif pred==1 and e==0:
			tp+=1
		else:
			tn+=1

	print "Accuracy: ",100*float(tp+tn)/(tp+tn+fp+fn)
	# print tp,tn,fp,fn
	# print count/(len(test_data))

	return tp,tn,fp,fn


p=0
r=0
acc=0

for i in range(10):

	print "---------------------"
	print "Fold Number",i+1

	[train,test,target_train,target_test]=splitDataset(train_data,i)
	mean,sd=mean_sd(np.array(train),np.array(target_train))
	tp,tn,fp,fn=naive_bayes(test,target_test,mean,sd)

	p+=float(tp)/(tp+fp)
	r+=float(tp)/(tp+fn)
	acc+=float(tp+tn)/(tp+tn+fp+fn)


print "---------------------"
print "Mean Accuracy:",acc*10
print "Mean Precision:",p*10
print "Mean Recall:",r*10
print "---------------------"
