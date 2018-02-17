import csv
import pandas as pd
import numpy as np 


iris_filename = 'IRIS.csv'
#iris = pd.read_csv(iris_filename, sep=',', decimal='.',  names= ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'target'])
iris = pd.read_csv(iris_filename)
#print ((iris))

data = np.asarray(iris)

#print(data[0][-1])

#print (data.size)

for i in range (0,data.size//data[0].size):
	#print(i)
	data[i][-1] = (data[i][-1]=='Iris-setosa')

#print (data)

#
#####################################################################################################
#weight initialisation

weights = []

num_features = data[0].size-1

for i in range(0,num_features):
	weights.append(1.0/((num_features)+1))

weights = np.asarray(weights)

bias = np.float32(1.0)
LR = np.float32(0.1)
print (weights)

#np.random.seed(11)
#np.random.shuffle(data)
#####################################################################################################




for i in range (0,data.size//data[0].size):

	predicted_value = np.sign(np.dot(data[i][0:-1],weights)+bias)

	predicted_value=np.clip(predicted_value,0,1)

	print ('predicted value : ', predicted_value,' label : ',data[i][-1]+0)
	#print (data[i][-1])
	delta_weight = LR * (data[i][-1]-predicted_value) * data[i][0:-1]
	print (weights)
	weights = weights + delta_weight

