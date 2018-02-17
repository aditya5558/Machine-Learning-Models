import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import copy
import math
import datetime
import os

train=pd.read_csv("IRIS.csv")
train['class'] = train['class'].map({'Iris-setosa': 0, 'Iris-versicolor': 1})


train = shuffle(train)

train_data=train.iloc[:,0:len(train.columns)-1].values

target_glob = train["class"].values



h=5
n=len(train.columns)-1
l=0.1
th=0.5


wh = np.random.normal(1/(n*h+5),0.05,(n,h))
wo = np.random.normal(1/(h+1),0.05,(h,2))
bh=np.full((h),5)
bo=np.full((2),1)

# print(wh)
# print(wo)
# print(bh)
# print(bo)


log_path = '.'

f = open(os.path.join(log_path, 'mlp_spectf.log'), 'w')
def log(txt, do_print = 1):
    txt = str(datetime.datetime.now()) + ': ' + txt
    if do_print == 1:
        print(txt)
    f.write(txt + '\n')






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

def backprop(tr,actual_output,pred_o,pred_h,wh,wo,bh,bo,k):

	erro=(pred_o*(1-pred_o))*(actual_output-pred_o)
	errh=(pred_h*(1-pred_h))*(np.dot(wo,erro))


	delwo=l*np.dot(np.expand_dims(pred_h.T,axis=1),np.expand_dims(erro.T,axis=0))
	#print("hi",delwo)
	delbh=l*errh


	delwh=l*np.dot(np.expand_dims(tr[k].T,axis=1),np.expand_dims(errh.T,axis=0))
	#print(delwh)
	delbo=l*erro

	wo+=delwo
	wh+=delwh
	bh+=delbh
	bo+=delbo


	#print("Err hidden:",errh)

	#print("Err output",erro)

	return [wh,wo,bh,bo]

def forward(tr,bh,bo,wh,wo,target_train):


	for k in range(100):

		for i in range(len(tr)):

			inph=np.dot(tr[i],wh)+bh
			#print(inph)
			outh=1/(1+np.exp(-1*inph))
			#print(outh)

			inpo=np.dot(outh,wo)+bo
			#print(inpo)
			outo=1/(1+np.exp(-1*inpo))
			#print(outo)

			actual_output=[]

			#print(target_train[i])

			for x in range(0,2):
				if target_train[i] == x:	
					actual_output.append(1)
				else:
					actual_output.append(0)

			[wh,wo,bh,bo]=backprop(tr,actual_output,outo,outh,wh,wo,bh,bo,i)

			# print("Updated wh:")
			# print(wh)

			#print("Updated wo:")
			#print(wo)

			# print(bh,bo)

	# print(bh)
	# print(bo)

	return [wh,wo,bh,bo]

def test_model(te,target_test,wh,wo,bh,bo):
	
	count=0

	#print(wh)
	#print(wo)

	for i in range(len(te)):

		inph=np.dot(te[i],wh)+bh
		#print(inph)
		outh=1/(1+np.exp(-1*inph))
		#print(outh)

		inpo=np.dot(outh,wo)+bo
		#print(inpo)
		outo=1/(1+np.exp(-1*inpo))
		print(outo)


		if outo[1]>outo[0]:
			output=1
		else:
			output=0

		print(output," ",target_test[i])


		if(output==target_test[i]):
			count+=1

	print("Accuracy:",count/len(te))

	return count/len(te)


acc=0

for i in range(0,10):

	print("---------")
	print("k-fold CV Iteration:"+str(i+1))
	[tr,te,target_train,target_test]=splitDataset(train_data,i)
	#print(te)
	[wh,wo,bh,bo]=forward(tr,bh,bo,wh,wo,target_train)	
	acc+=test_model(te,target_test,wh,wo,bh,bo)

print("Final Accuracy:",acc*10)
log("Final Accuracy:"+str(acc*10))
