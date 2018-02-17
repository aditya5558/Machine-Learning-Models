import numpy as np
import pandas as pd
# from sklearn.utils import shuffle
import math
import datetime
import os


train=pd.read_csv("SPECTF_New.csv")
train['Class'] = train['Class'].map({'Yes': 1, 'No': 0})

np.random.seed(100)

#train = shuffle(train)

train_data=train.iloc[:,0:len(train.columns)-1].values
#print train_data
target_glob = train["Class"].values

max_acc=0
max_p=0
max_r=0


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

	#print "Accuracy: ",100*float(tp+tn)/(tp+tn+fp+fn)
	# print tp,tn,fp,fn
	# print count/(len(test_data))

	return tp,tn,fp,fn


def objective_func(train_data):


	p=0
	r=0
	acc=0

	for i in range(10):

		#print "---------------------"
		#print "Fold Number",i+1

		[train,test,target_train,target_test]=splitDataset(train_data,i)
		mean,sd=mean_sd(np.array(train),np.array(target_train))
		tp,tn,fp,fn=naive_bayes(test,target_test,mean,sd)

		try:
			p+=float(tp)/(tp+fp)
		except:
			p+=0
		try:
			r+=float(tp)/(tp+fn)
		except:
			r+=0

		acc+=float(tp+tn)/(tp+tn+fp+fn)


	# print "---------------------"
	# print "Mean Accuracy:",acc*10
	# print "Mean Precision:",p*10
	# print "Mean Recall:",r*10
	# print "---------------------"

	return acc*10,p*10,r*10



def init_population(min_chromosomes,max_chromosomes):

	# num_chromosomes=np.random.randint(min_chromosomes,max_chromosomes)
	num_chromosomes=30
	population=np.random.randint(2,size=(num_chromosomes,len(train.columns)-1))

	return num_chromosomes,population


def fitness_func(chromosome):

	# print train_data[:,chromosome==1].shape
	# print "hi",objective_func(train_data[:,chromosome==1])

	f,pr,re=objective_func(train_data[:,chromosome==1])
	
	#print f

	global max_acc,max_p,max_r,max_chromosome

	if f > max_acc:
		max_acc=f
		max_p=pr
		max_r=re
		max_chromosome=chromosome

	return objective_func(train_data[:,chromosome==1])


def selection(population):

	#print population

	fitness=[]
	ax=0
	for i in range(len(population)):
		x,y,z=fitness_func(population[i])
		if x>ax:
			ax=x

		fitness.append(x)

	print "Local Max Accuracy:",ax
	# print fitness

	fitness=np.asarray(fitness)
	total_fitness=np.sum(fitness)

	prob=[]
	cumulative_prob=[]

	for i in range(len(fitness)):

		prob.append(float(fitness[i])/total_fitness)
		if i==0:
			cumulative_prob.append(prob[i])
		else:
			cumulative_prob.append(prob[i]+cumulative_prob[i-1])

	# print prob,'\n',cumulative_prob	,'\n',total_fitness
	#print cumulative_prob
	duplicate_pop=np.array(population)

	# print "------------------"
	# print duplicate_pop

	for i in range(len(population)):

		r=np.random.random()
		#print r
		index=np.searchsorted(cumulative_prob,r)
		#print index

		population[i]=duplicate_pop[index]

	#print "------------------"

	#print population

	return population


def crossover(population,crossover_rate,num_chromosomes):

	limit=int(crossover_rate*len(population))

	# print limit,num_chromosomes

	if limit%2!=0:
		limit += 1

	chromosome_size=len(population[0])


	for i in range(limit/2):

		selected1=np.random.randint(0,num_chromosomes)
		selected2=np.random.randint(0,num_chromosomes)

		# print "selected",selected1, selected2
		
		crossover_point=np.random.randint(0,chromosome_size)
	

		# print "crossover point",crossover_point

		temp=np.array(population[selected1,crossover_point+1:])
		
		# print "selected 1",population[selected1]

		# print "selected 2",population[selected2]
		# print "temp",temp

		population[selected1,crossover_point+1:]=population[selected2,crossover_point+1:]
		population[selected2,crossover_point+1:]=temp

		# print "temp",temp
		# print "selected1",population[selected1]
		# print "selected2",population[selected2]
		# print "selected1",population[selected1]

	return population
	

def mutation(population,mutation_rate,num_chromosomes):

	limit=int(mutation_rate*len(population))

	chromosome_size=len(population[0])

	# print limit

	for i in range(limit):

		mutated_chromosome=np.random.randint(0,num_chromosomes)
		mutation_index=np.random.randint(0,chromosome_size)

		# print mutated_chromosome,mutation_index

		population[mutated_chromosome,mutation_index]=1-population[mutated_chromosome,mutation_index]

	return population


#### GA Algorithm ####


num,pop=init_population(40,100)


for i in range(100):

	print "-----------------"

	print "GA Iteration:",i+1
	
	pop=selection(pop)
	pop=crossover(pop,0.25,num)
	pop=mutation(pop,0.1,num)



print "-----------------"
print "Global Max Accuracy:",max_acc
print "Precision:",max_p
print "Recall:",max_r
print "Chromosome:",max_chromosome



