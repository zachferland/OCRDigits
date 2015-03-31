import data
from lshash import LSHash

# K Nearest Neigbors 
# Could explain format here
# p is a vector, training is an LSHash instance with all training examples indexed
def KNN(p, training, k):
	neighbors = training.query(p, num_results=k, distance_func="euclidean")
	return _majority(neighbors)

# Given closest neighbors, determines most likely label
def _majority(neighbors):
	counter = [0] * 10
 	for neigbor in neighbors:
		label = int(neigbor[0][1])
 		counter[label] += 1

 	return counter.index(max(counter))