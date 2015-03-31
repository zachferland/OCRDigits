###################
##    README     ##
###################

Files: 
	data.py      # Gets training and testing data and formats it  
	image.py     # Processes an image, and does character segmentation
	KNN.py       # Implementation of KNN
	LshKNN.py    # Implementation of KNN with LSH
	OCR.py       # Runs OCR on and image and returns string
	testing.py   # Used for testing on the MNIST testing set


## To run files, the following packages are needed: (TODO makefile) ###############

OpenCV: $ brew tap homebrew/science
        $ brew install opencv

(may need to add to python path as well)
More Info: Without brew - http://opencv.org/

numPy: $ pip install numpy

idx2numpy: $ pip install idx2numpy

More Info: https://github.com/ivanyu/idx2numpy

LSHash: $ pip install lshash

More Info: https://github.com/kayzh/LSHash


## Testing ########################################################################


## To run KNN on MNIST test set  (Max trainingSize = 60000, Max testingSize = 10000)

$ python testing.py -k 3 --trainingSize 30000 --testingSize 500 --lsh 0

NOTE: each classification takes about 1400 miliseconds, so any large testSize will 
take considerable time.

## To run KNN with LSH on MNIST test set  

$ python testing.py -k 3 --trainingSize 30000 --testingSize 500 --lsh 1

NOTE: Building hash tables to run KNN with LSH takes considerable time, in most 
intances this would built once and then persisted.

## To Run OCR on an Image #########################################################

1) Place image in 'images' folder
2) $ python OCR.py -i IMAGENAME.JPG -l 0 

There exists example images to run it already

$ python OCR.py -i 1.JPG -l 0 

###################################################################################