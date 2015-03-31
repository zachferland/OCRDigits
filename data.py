import idx2numpy
import numpy
from lshash import LSHash

# Converts data from MNIST data set http://yann.lecun.com/exdb/mnist/ to numpy arrays
# Uses idx2numpy https://github.com/ivanyu/idx2numpy
# To generalize, may want to turn each image into a single vector

folder = "data/"

# Returns training data in the form of tuple (image(28*28), label)
def getTraining(size):
	trainImages = idx2numpy.convert_from_file(folder + 'train-images-idx3-ubyte')
	trainLabels = idx2numpy.convert_from_file(folder + 'train-labels-idx1-ubyte')
	trainImages = trainImages.astype(float)
	size = min(trainImages.shape[0], size)
	return zip(trainImages[:size], trainLabels[:size])

# Returns testing data in the form of tuple (image(28*28), label)
def getTesting(size):
	testImages = idx2numpy.convert_from_file(folder + 't10k-images-idx3-ubyte')
	testLabels = idx2numpy.convert_from_file(folder + 't10k-labels-idx1-ubyte')
	testImages = testImages.astype(float)
	size = min(testImages.shape[0], size)
	return zip(testImages[:size], testLabels[:size])

# Returns training data in the form of tuple (image(1*784), label)
def getTrainingVectors(size):
	trainingData = getTraining(size)

	# Turn each into image into a single vector
	for i in xrange(size):
		trainingData[i] = (trainingData[i][0].flatten(), trainingData[i][1])

	return trainingData

# Returns testing data in the form of tuple (image(1*784), label)
def getTestingVectors(size):
	testingData = getTesting(size)

	# Turn each into image into a single vector
	for i in xrange(size):
		testingData[i] = (testingData[i][0].flatten(), testingData[i][1])

	return testingData

def getTrainingLSH(size):
	trainingData = getTrainingVectors(size)

	print "Building Locality Sensitive Hash Tables..."
	# This is very slow, and should be persisted

	# Hash all training examples
	# Choosing the the size of the hash and the number of queries, allows for a
	# tradeoff between, speed and accuracy
	lsh = LSHash(24, trainingData[0][0].size, num_hashtables = 18)

	for i in xrange(size):
		image = trainingData[i][0]
		label = trainingData[i][1]
		lsh.index(image, extra_data=label)

	return lsh