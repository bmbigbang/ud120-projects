#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
from sklearn.svm import SVC
model = SVC(kernel='rbf', C=10000)

t0 = time()
model.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"
t0 = time()
predictions = model.predict(features_test)
print "prediction time:", round(time()-t0, 3), "s"
accuracy = sum([1 if i == j else 0 for i, j in zip(predictions, labels_test)])/float(len(predictions))
print accuracy
print predictions[10], predictions[26], predictions[50]
#########################################################


