#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
### available features: ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus',
#  'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options',
#  'other', 'long_term_incentive', 'restricted_stock', 'director_fees']
features_list = ['poi', 'salary', 'bonus', 'total_stock_value', 'expenses', 'long_term_incentive',
                 'deferral_payments', 'deferred_income']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
del data_dict['TOTAL']
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.

my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)
#print len(labels), len(features), len(features[0])

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

from sklearn.decomposition import PCA
pca = PCA(n_components=2).fit(features)
features_pca = pca.transform(features)
# print pca.explained_variance_ratio_

from sklearn.cross_validation import KFold
from sklearn.naive_bayes import GaussianNB # first algorithm
clf = GaussianNB()
from sklearn.neighbors import KNeighborsClassifier # second algorithm
from sklearn.grid_search import GridSearchCV
param_grid = {'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]}
clf2 = GridSearchCV(KNeighborsClassifier(algorithm="auto"), param_grid)
# for sklearn version 0.16 or prior, the class_weight parameter value is 'auto'


kf = KFold(len(features_pca), n_folds=10, random_state=13)
all = []
all2 = []
for train_indices, test_indices in kf:
    X_train = [features_pca[i] for i in train_indices]
    X_test = [features_pca[i] for i in test_indices]
    y_train = [labels[i] for i in train_indices]
    y_test = [labels[i] for i in test_indices]

    clf.fit(X_train, y_train)
    clf2.fit(X_train, y_train)
    pred = clf.predict(X_test)
    pred2 = clf2.predict(X_test)
    all.append(sum([1 if i == j else 0 for i, j in zip(pred, y_test)])/float(len(pred)))
    all2.append(sum([1 if i == j else 0 for i, j in zip(pred2, y_test)])/float(len(pred2)))

# testing and comparing the two methods
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features_pca, labels, test_size=0.3, random_state=42)
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
print "PCA accuracy using GaussianNB:", sum([1 if i == j else 0 for i, j in zip(pred, labels_test)])/float(len(pred))
print "KFold + PCA accuracy using GaussianNB:", sum(all)/len(all)
print "KFold + PCA accuracy using KNeighborsClassifier:", sum(all2)/len(all2)
print """an improvement in precision/recall is observed when PCA is used to combine the features
many features into two. However when KFold data test splitting method
is used, the accuracy becomes worst, showing that the number of anomalies are potentially very small such that
a randomised test set is bound to reduce the number of correctly predicted results.
This test was implemented as the number of data points are limited.
There appears to be no difference in using GaussianNB or KNeighborsClassifier however it if
n_neighbors=7 is not used as the arguement for the KN Classifier, the predictions become much worse.
Unfortunately the GridSearchCV method does not find this saddle point correctly.
This appears to be a pivotal number and due to the restriction in the number of data points."""

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
param_grid = {'n_components': [1, 2, 3]}
pca = GridSearchCV(PCA(n_components=2), param_grid)
# a SVC() search shows good precision, but very bad recall, unused for future
# sv = GridSearchCV(SVC(), {
#          'C': [1e3, 5e3, 1e4, 5e4, 1e5],
#           'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],
#           })
clf = Pipeline(steps=[('pca', pca), ('classifier', GaussianNB())])


# Example starting point. Try investigating other evaluation techniques!


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)