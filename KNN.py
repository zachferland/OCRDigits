import data
import numpy

# K Nearest Neigbors 
# p is a 28*28 numPy matrix, training is array of tuples (p, label)
def KNN(p, training, k):
	neighbors = _neighbors(p, training, k)
	return _majority(neighbors)

# Determines the euclidean distance between two vectors
# Could extend KNN to add multiple distance functions
def _distance(p1, p2):
	return numpy.linalg.norm(p1 - p2)

# Returns the k closest neighbors of p in the training set
def _neighbors(p, training, k):
	distances = [_distance(p, training[i][0]) for i in xrange(len(training))]
	labels = [l[1] for l in training]
	points = zip(distances, labels)
	points.sort(key = lambda point: point[0])
	return points[:k]

# Given closest neighbors, determines most likely label
def _majority(neighbors):
  	counter = [0] * 10

 	for neigbor in neighbors:
 		label = neigbor[1]
 		counter[label] += 1

 	return counter.index(max(counter))