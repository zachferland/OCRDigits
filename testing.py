import sys
import KNN as classify
import LshKNN as classifyLSH
import optparse
import data
import time

parser = optparse.OptionParser()

parser.add_option('-t', '--trainingSize', dest="trainingSize")
parser.add_option('-s', '--testingSize', dest="testingSize")
parser.add_option('-l', '--lsh', dest="lsh")
parser.add_option('-k', '--k', dest="k")

options, args = parser.parse_args()

k = int(options.k)
trainSize = int(options.trainingSize)
testSize = int(options.testingSize)
lsh = int(options.lsh)

print "Loading training and testing data..."
if not lsh:
	trainingData = data.getTraining(trainSize)
	testingData = data.getTesting(testSize)
else:
	trainingData = data.getTrainingLSH(trainSize)
	testingData = data.getTestingVectors(testSize)

print "Data loaded and formatted"

print "Classifying " + str(testSize) + " images " + "on " + str(trainSize) + " samples..."
correct = 0
startTime = time.time()

if not lsh:
	print "Using K Nearest Neighbors with k = " + str(k) + " ..."
	for i in xrange(testSize):
		classification = classify.KNN(testingData[i][0], trainingData, k)
		if classification == testingData[i][1]: correct += 1 
		# print str(i), " Classification: ", str(classification), " Actual: ", str(testingData[i][1])
else:
	print "Using K Nearest Neighbors with LSH and k = " + str(k)
	for i in xrange(testSize):
		classification = classifyLSH.KNN(testingData[i][0], trainingData, k)
		if classification == testingData[i][1]: correct += 1 
		# print str(i), " Classification: ", str(classification), " Actual: ", str(testingData[i][1])

endTime = time.time()
totalTime = endTime - startTime
avgTime = float(totalTime) / float(testSize)
percentage = (float(correct) / float(testSize)) * 100

print "Result: " + str(correct) + " correct out of " + str(testSize) + " (" + "%.2f" % percentage + "%)"
print "Total Time: " + "%.2f" % totalTime + " Average Time per Classification: " + "%.3f" % avgTime
