import numpy as np
import math
import pandas as pd
from sklearn.utils import shuffle
import copy
import os
import datetime

train=pd.read_csv("SPECTF_New.csv")


train['Class'] = train['Class'].map({'Yes': 0, 'No': 1})

#np.random.seed(10)
#np.random.shuffle(train)
#train = shuffle(train)


train_data=train.iloc[:,0:len(train.columns)-1].values


target_glob= train["Class"].values


#Model Parameters
weights=[]

for i in range(len(train.columns)-1):
	weights.append(1/(len(train.columns)+1))

#print(weights)

bias=1
l=0.5


ac=0
tp=0
tn=0
fp=0
fn=0


log_path = '.'
# if not os.path.exists(path):
#         os.makedirs(path)

f = open(os.path.join(log_path, 'simple_percep_spectf.log'), 'w')
def log(txt, do_print = 1):
    txt = str(datetime.datetime.now()) + ': ' + txt
    if do_print == 1:
        print(txt)
    f.write(txt + '\n')



def splitData(dataset, n):
	
	trainSet = []
	testSet = []
	target_train=[]
	target_test=[]

	
	size = int(math.ceil(len(dataset) / 10))
	#print(size)

	for i in range(0, 10):
		if i != n:
			trainSet.extend(dataset[i * size : i*size + size])
			target_train.extend(target_glob[i*size:i*size+size])
		else:
			for j in range(0, size):
				#temp = copy.deepcopy(dataset[i * size + j])
				testSet.append(dataset[i * size + j])
				target_test.append(target_glob[i*size+j])
				

	trainSet.extend(dataset[9*size+size:])

	return [trainSet, testSet, target_train,target_test]



#Perceptron-OR Model
def train_model(tr,target_train,weights,bias,l,n):
	
	print("Training Model...")

	for j in range(n):

		
		#print("Epoch:",j+1)
		count=0

		for i in range(len(tr)):

			y_cap=np.dot(weights,tr[i])+bias

			if y_cap > 0:
				y=1
			else:
				y=0

			e=target_train[i]-y
			err=e*l*tr[i]

			if e!=0:
				count+=1

			bias=bias+e*l
			weights=weights+err
		
		if count==0:
			break

	#print("Final Weights:",weights)
	return [weights,bias]


def test_model(te,target_test,weights,bias):

	print("Testing Model...")
	#print("Weights:",weights)
	global ac,tp,tn,fp,fn
	count=0
	#print(ac)

	for i in range(len(te)):

		y_cap=np.dot(weights,te[i])+bias

		if y_cap > 0:
			y=1
		else:
			y=0

		e=target_test[i]-y

		#print("Prediction:",y,"Target:",target_test[i])


		if e!=0:
			count+=1
		
		if e==1:
			fn+=1
		elif e==-1:
			fp+=1
		elif y==1 and e==0:
			tp+=1
		else:
			tn+=1


	t=(float)(len(te)-count)/len(te)*100
	#ac+=t

	#print("Wrong predictions:",count)

	print("Accuracy:",t)

#Model

for i in range(0,10):

	print("---------")
	print("k-fold CV Iteration:"+str(i+1))
	[tr,te,target_train,target_test]=splitData(train_data,i)
	#print(te[0])
	[wt,b]=train_model(tr,target_train,weights,bias,l,100)
	test_model(te,target_test,wt,b)

p=float(tp)/(tp+fp)

r=float(tp)/(tp+fn)

ac=float(tp+tn)/(tp+tn+fp+fn)

print("---------")
log("OR-Perceptron")
log("SPECTF Dataset")
log("Final Accuracy:"+str(ac*100))
log("Precision:"+str(p*100))
log("Recall:"+str(r*100))
