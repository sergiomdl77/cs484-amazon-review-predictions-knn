from __future__ import print_function

from nltk.stem.snowball import SnowballStemmer

from nltk.corpus import stopwords
from nltk.sentiment import SentimentAnalyzer, SentimentIntensityAnalyzer
from nltk.sentiment.util import *

#*************************************************************************
#       INITIALIZING IMPORTANT VARIABLES (used to clean data)
#*************************************************************************

nltk.download('stopwords')
nltk.download('vader_lexicon')

#creating a pattern for regex
pattern = re.compile('[a-z]+')
stop_words = set(stopwords.words('english'))

#making stemmer appropriate for english language
stemmer = SnowballStemmer("english")
#**************************************************************************


#***************************************************************
#     important arrays that store reviews, labels
#       and dimension vectors for each review
#***************************************************************
trainReviews = []
trainLabels = []
trainReviewVectors = []

testReviews = []
testLabels = []
testReviewVectors = []

predictionLabels = []

significantWords = []
significantWordPolarities = []
distancesDict = {}

#***************************************************************


lowCaseCurLine = ''

# #openning training file for reading

#**************************************************************************
#   This function was takes a review (still in the form of bag of words
#   and extracts every word that has a significance for the sentiment
#   analysis of the review.  The criteria to select the word as a feature
#   is based on the polarity of such a word, for which the
#   SentimentIntensityAnalyzer from nltk package.
#**************************************************************************
def extract_significant_words(review):
    score = 0
    mySentAnalyzer = SentimentIntensityAnalyzer()
    for word in review:
        score = mySentAnalyzer.polarity_scores(word)["compound"]
        if score != 0:
            if word not in significantWords:
                significantWords.append(word)
                significantWordPolarities.append(score)


#**************************************************************************
#   This function receives one word from the list of words that have been
#   selected as significant for the sentiment analysis.  It also receives
#   the receives review, still in the form of a bag or words. This function
#   is in charge of counting and the number of times that word appears
#   in the review and returns the total count.
#***************************************************************************
def frequency_in_review(signifWord, index, review):
    frequency = 0

    for word in review:
        if word == signifWord:
            frequency += 1

#   This final lines of code for this function were added in an attempt to give
#   weight to the frequency according to the degree of polarity of the word (feature)
#   that we are accounting for in this review.

    frequency = frequency * significantWordPolarities[index]

    #   This line is to handle the blank reviews (blank lines in the data files)
    if len(review) != 0:            #   If the review is not a black line (meaning that the bag of
                                    #   words array that contains the tokens of this reviews is not empty)

        frequency = (frequency / len(review) ) * 1000 #   This formula is used to normalize the values

    return frequency


#***************************************************************************
#   This function receives a string with only 2 possible values ("train" or
#   "test") in order to know whether to read from either the train data file
#   or the test data file.
#   This function access the array of reviews (while they are still in the
#   form of bag of words) and obtains the frequency of every important word
#   stored in the significantWords array. This is done with the goal of
#   using those frequencies to create the dimension vector for each review.
#   All the vectors will be arrays of features.  This arrays are, in turn,
#   elements of an array of vectors called either trainReviewVectors or
#   testReviewVectors (accordingly).
#***************************************************************************
def vectorize_reviews(reviewType):

    count = 0
    if reviewType == "train":
        for review in trainReviews:  # vectorizing all reviews (one by one)

            vector = []
            for i in range( len(significantWords) ):
                vector.append(frequency_in_review(significantWords[i], i, review))

            trainReviewVectors.append(vector)
            count += 1
            print("vectorizing", count)

    else:

        for review in testReviews:  # vectorizing all reviews (one by one)

            vector = []
            for i in range ( len(significantWords) ):
                vector.append(frequency_in_review(significantWords[i], i, review))

            testReviewVectors.append(vector)
            count += 1
            print("vectorizing", count)


#********************************************************************************************
# reads the whole training file and cleans it from punctuation, white space, stop words.
# it also stems words with similar roots and stores each review as an array of words which
# will represent a more condensed and meaningful bag of words.
#********************************************************************************************
def clean_train_data():
    # openning training file for reading
    trainDataFile = open(r"train_file.dat", "r")

    curLine = trainDataFile.readline()  #reads first review
    while curLine:                      #starts reading all reviews (one by one)

        lowCaseCurLine = curLine.lower()  # getting rid of capital letters

        wordBag = pattern.findall(lowCaseCurLine)  # extract all words (disposing of spaces/symbols)
        reducedWordBag = []  # to store the word bag without the stop words
        for w in wordBag:
            if not w in stop_words:
                reducedWordBag.append(w)

        wordRoots = [stemmer.stem(word) for word in reducedWordBag]  # to store the word bag after stemming
        trainReviews.append(wordRoots)  # insert in a list of reviews destined for training the software

        extract_significant_words(wordRoots)   # searches and stores words with some polarity from current review

        if curLine[0] == '+':        # if this is a positive review
            trainLabels.append("+1")
        else:
            trainLabels.append("-1")

        curLine = trainDataFile.readline()

    trainDataFile.close()


#********************************************************************************************
# reads the whole test file and cleans it from punctuation, white space, stop words.
# it also stems words with similar roots and stores each review as an array of words which
# will represent a more condensed and meaningful bag of words.
#********************************************************************************************
def clean_test_data():
    # openning training file for reading
    testDataFile = open(r"test.dat", "r")

    curLine = testDataFile.readline()  #reads first review
    while curLine:                      #starts reading all reviews (one by one)

        lowCaseCurLine = curLine.lower()  # getting rid of capital letters

        wordBag = pattern.findall(lowCaseCurLine)  # extract all words (disposing of spaces/symbols)
        reducedWordBag = []  # to store the word bag without the stop words
        for w in wordBag:
            if not w in stop_words:
                reducedWordBag.append(w)

        wordRoots = [stemmer.stem(word) for word in reducedWordBag]  # to store the word bag after stemming
        testReviews.append(wordRoots)  # insert in a list of reviews destined for training the software

        curLine = testDataFile.readline()

    testDataFile.close()


#********************************************************************************************
# This function is only for practice purposes so that I can test the the project during its
# development. This is done by dividing the training data into some training data (80% of
# training data reviews) and some mock test data (20% of the traning data reviews)
#********************************************************************************************
def clean_data_80_20():
    # openning training file for reading
    trainDataFile = open(r"train_file.dat", "r")

    max_reviews = 10000

    reviewCount = 0

    curLine = trainDataFile.readline()  # reads first review
    while curLine and reviewCount < max_reviews:  # starts reading all reviews (one by one)

        lowCaseCurLine = curLine.lower()  # getting rid of capital letters

        wordBag = pattern.findall(lowCaseCurLine)  # extract all words (disposing of spaces/symbols)
        reducedWordBag = []  # to store the word bag without the stop words
        for w in wordBag:
            if not w in stop_words:
                reducedWordBag.append(w)

        wordRoots = [stemmer.stem(word) for word in reducedWordBag]  # to store the word bag after stemming

        if reviewCount % 5 != 0:    # to get 80% of reviews as mock train reviews

            trainReviews.append(wordRoots)  # insert in a list of reviews destined for training the software
            if curLine[0] == '+':  # if this is a positive review
                trainLabels.append("+1")
            else:
                trainLabels.append("-1")

        else:                       # to get 20% of reviews as mock test reviews (every 5 reviews)
            testReviews.append(wordRoots)  # insert in a list of reviews destined for training the software

            if curLine[0] == '+':  # if this is a positive review
                testLabels.append("+1")
            else:
                testLabels.append("-1")

        extract_significant_words(wordRoots)  # searches and stores words with some polarity from current review

        reviewCount += 1
        curLine = trainDataFile.readline()

    trainDataFile.close()


#**********************************************************************************
#   This function receives a vectorized test review (testVector) and utilizes the
#   Euclidean formula to calculate the distance betweeen the test review and each
#   one of the training reviews.  This Euclidean formula was slightly optimized
#   to improve the time data mining time by reducing the number of computations.
#   More specifically, if a certain feature has value of '0' in both vectors, then
#   the that feature is not computed, since that feature won't affect the result
#   of the Euclidean formula. Also, I eliminated the final part of the calculation
#   of the formula (the square root of the sum of all squared values) because the
#   proportions of the distances still be the same without this final step.
#*********************************************************************************
def produceDistances(testVector):
    for i in range(len(trainReviewVectors)):
        totalDist = 0

        # This is the implementation of the
        for j in range(len(testVector)):
            if (testVector[j] !=  0) or (trainReviewVectors[i][j] != 0):
                featureDist = testVector[j] - trainReviewVectors[i][j]
                totalDist += featureDist**2

#        totalDist = math.sqrt(totalDist)       # This line was commented in an
                                                # effort to reduce computations to
                                                # obtain the distance

        distancesDict.update( {i : totalDist} )

#*************************************************************************************
#   This block was added as an alternative approach for the produceDistances function
#   which differs from approach coded just about this commment in the following way:
#   It utilizes the spatial.distance.cosine function from the scipy package, instead
#   of the eucledian distance function
#   In the end, this approach didn't improve the accuracy, but it did increase the
#   time of processing for the data mining.
#
#def produceDistances(testVector):
#    for i in range(len(trainReviewVectors)):
#        distance =  spatial.distance.cosine(testVector, trainReviewVectors[i])

#        distancesDict.update( {i : distance} )
#*************************************************************************************


#*************************************************************************************
#   This function is the core of this Homework 1.  This function receives a K-factor
#   which determines how many nearest neighbors will be considered in the decision
#   of whether a review will be classified as positive (+1) or negative (-1).
#   The function calls the triggers the creation of an array of distances between a
#   test review (test) and all training reviews.  A global variable (distanceDict) of
#   the type dictionary will be filled with all distances by calling produceDistances.
#   A temporary distionary is created and filled with the kFactor smallest distances
#   and then each entry of this smallest distances (with their corresponding index to
#   the array of training reviews) will be accessed and the count of positive labels
#   versus the count of negative labels will determine the prediction of the current
#   test review. The prediction for each test is appended in an array of predictions
#   called (predictionLabels) which later on will be written on a file.
#**************************************************************************************
def knnClassification(kFactor):
    counter = 0

    for test in testReviewVectors:

        produceDistances(test)      #creates the array of distances between this test review and all training reviews

        #   This temporary dictionary is assigned the 5 shortest distances for the current test.
        nearestNeighbors = [(kFactor, distancesDict[kFactor]) for kFactor in sorted(distancesDict, key=lambda x:distancesDict[x])][:kFactor]

        positiveCount = 0
        negativeCount = 0
        mostSimilar = {k: v for k, v in nearestNeighbors}
        for k in mostSimilar:
            if trainLabels[k] == '+1':
                positiveCount += 1
            else:
                negativeCount += 1

        if positiveCount < negativeCount:
            predictionLabels.append("-1")
        else:
            predictionLabels.append("+1")

        counter += 1

    createPredictionsFile()     #once all predictions stored in predictionLabels array
                                #this function writes them into a file


#***********************************************************************************
#   This simple function reads the predictions stored in the global variable of
#   the array type (predictionLabels) and writes them all into a file
#   called "myResults.dat".
#***********************************************************************************
def createPredictionsFile():
    predictionsDataFile = open(r"format_new.dat", "w")

    for prediction in predictionLabels:
        predictionsDataFile.write(prediction+'\n')

    predictionsDataFile.close()

# def main():
    # clean_data_80_20()     # This line of code is for practice during development
clean_train_data()
clean_test_data()
vectorize_reviews(reviewType = "train")
vectorize_reviews(reviewType = "test")
knnClassification(kFactor = 29)



#***********************************************************************
#       This following blocks are just auxiliar functions
#       designed solely to print out important information
#       which was necessary during the development of this
#       software
#***********************************************************************

#************* Printing everything about significant words **************
#print(len(significantWords))
#mySentAnalyzer = SentimentIntensityAnalyzer()
#for word in significantWords:
#    print (word, mySentAnalyzer.polarity_scores(word)["compound"])
#*************************************************************************

#************** Printing training reviews/vectors/labels *****************
#for i in range(len(trainLabels)):
#    print(trainLabels[i], trainReviews[i])
#    print(trainReviewVectors[i])
#*************************************************************************

#************** Printing test reviews/vectors/labels *****************
#for i in range(len(testLabels)):
#    print(testLabels[i], testReviews[i])
#    print(testReviewVectors[i])
#*************************************************************************

#************** Printing prediction versus test Labels *******************
#
#correct = 0
#incorrect = 0
#accuracy = 0.0
#totalCount = 0
#for i in range( len(predictionLabels)):
#    totalCount += 1
#    print(predictionLabels[i], testLabels[i])
#    if predictionLabels[i] == testLabels[i]:
#        correct += 1
#    else:
#        incorrect += 1
#
#    accuracy = correct / (correct+incorrect)
#
#print("total ", totalCount, " correct: ",format(correct), " accuracy: ", format(accuracy))
#*************************************************************************


# if __name__ == '__main__':
#     main()




