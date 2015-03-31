import image
import sys
import KNN as classify
import LshKNN as classifyLSH
import optparse
import data

# Parse arguments
parser = optparse.OptionParser()
parser.add_option('-i', '--image', dest="image")
parser.add_option('-l', '--lsh', dest="lsh")
options, args = parser.parse_args()

img = options.image
lsh = int(options.lsh) #default this to 0
folder = "images/"
trainSize = 60000
result = ""
k = 3

print "Loading training data..."

if not lsh:
	trainingData = data.getTraining(trainSize)
else:
	trainingData = data.getTrainingLSH(trainSize)

print "Data loaded and formatted"

print "Segmenting Image...."

numbers = image.getNumbers(folder + img)

if not lsh:
	print "Classifying numbers using K Nearest Neighbors ..."
	for number in numbers:
		result += str(classify.KNN(number, trainingData, k)) + " "
else:
	print "Classifying numbers using K Nearest Neighbors with LSH ..."
	for number in numbers:
		result += str(classifyLSH.KNN(number.flatten(), trainingData, k)) + " "

print "Result: " + result