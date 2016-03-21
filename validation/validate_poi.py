#!/usr/bin/python


"""
    Starter code for the validation mini-project.
    The first step toward building your POI identifier!

    Start by loading/formatting the data

    After that, it's not our code anymore--it's yours!
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### first element is our labels, any added elements are predictor
### features. Keep this the same for the mini-project, but you'll
### have a different feature list when you do the final project.
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### it's all yours from here forward!  

from sklearn import tree
model = tree.DecisionTreeClassifier()

model.fit(features, labels)
predictions = model.predict(features)

accuracy = sum([1 if i == j else 0 for i, j in zip(predictions, labels)])/float(len(predictions))
print accuracy

from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.30, random_state=42)

model = tree.DecisionTreeClassifier()

model.fit(features_train, labels_train)
predictions = model.predict(features_test)

accuracy = sum([1 if i == j else 0 for i, j in zip(predictions, labels_test)])/float(len(predictions))
print accuracy
