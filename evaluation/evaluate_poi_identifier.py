#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### your code goes here 

from sklearn import tree
model = tree.DecisionTreeClassifier()

from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.30, random_state=42)

model = tree.DecisionTreeClassifier()

model.fit(features_train, labels_train)
predictions = model.predict(features_test)

# accuracy
print sum([1 if i == j else 0 for i, j in zip(predictions, labels_test)])/float(len(predictions))

# POI number
print len([i for i, j in zip(predictions, labels_test) if j == 1])

# total people
print len(features_test)

# accuracy if predictions == 0 for all
print sum([1 if i == 0 else 0 for i in labels_test])/float(len(labels_test))

# total true positives:
print sum([1 if i == j == 1 else 0 for i, j in zip(predictions, labels_test)])

from sklearn.metrics import precision_score, recall_score
print precision_score(labels_test, predictions)
print recall_score(labels_test, predictions)

# dummy data practice
# predictions = [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1]
# true_labels = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]
# print sum([1 for i, j in zip(predictions, true_labels) if (i, j) == (1, 0)])
# print recall_score(true_labels, predictions)


