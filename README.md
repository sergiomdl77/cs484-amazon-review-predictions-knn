# cs484-amazon-review-predictions-knn

Overview and Assignment Goals:
*************************************************
The objective of this assignment are the following:
1. Implement the Nearest Neighbor Classification Algorithm
2. Handle Text Data (Reviews of Amazon Baby Products)
3. Design and Engineer Features from Text Data.
4. Choose the Best Model i.e., Parameters of a Nearest Neighbor Selection, Features and Similarity
    Functions
    
Detailed Description:
A practical application in e-commerce applications is to infer sentiment (or polarity) from free form review text
submitted for range of products.

For the purposes of this assignment you have to implement a k-Nearest Neighbor Classifier to predict the
sentiment for 18506 reviews for baby products provided in the test file (test.data). Positive sentiment is
represented by a review rating and given by +1 and Negative Sentiment is represented by a review rating of -1.
In test.dat you are only provided the reviews but no ground truth rating which will be used for comparing your
predictions.

Training data consists of 18506 reviews as well and exists in file train_file.dat. Each row begins with the sentiment
score followed with a text of the rating.
