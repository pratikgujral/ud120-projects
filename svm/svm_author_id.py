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

clf = SVC(kernel='rbf') # By-default, C = 1.0
print "C = 1.0"
from time import time
t0 = time()

features_train = features_train[:len(features_train)/100] 
labels_train = labels_train[:len(labels_train)/100] 

clf.fit(features_train, labels_train)
print "Training Time SVM:", round(time()-t0,3), "seconds"

t1 = time()
pred = clf.predict(features_test)
print "Prediction Time SVM:", round(time()-t1,3), "seconds"

from sklearn.metrics import accuracy_score
print accuracy_score(pred, labels_test)
#########################################################

 # C = 10
clf = SVC(kernel='rbf', C = 10.)
print "C = 10.0"

t0 = time()

clf.fit(features_train, labels_train)
print "Training Time SVM:", round(time()-t0,3), "seconds"

t1 = time()
pred = clf.predict(features_test)
print "Prediction Time SVM:", round(time()-t1,3), "seconds"

from sklearn.metrics import accuracy_score
print accuracy_score(pred, labels_test)
#########################################################
 # C = 100
clf = SVC(kernel='rbf', C = 100.)
print "C = 100.0"

t0 = time()

clf.fit(features_train, labels_train)
print "Training Time SVM:", round(time()-t0,3), "seconds"

t1 = time()
pred = clf.predict(features_test)
print "Prediction Time SVM:", round(time()-t1,3), "seconds"

from sklearn.metrics import accuracy_score
print accuracy_score(pred, labels_test)
#########################################################
 # C = 1000
clf = SVC(kernel='rbf', C = 1000.)
print "C = 1000.0"

t0 = time()

clf.fit(features_train, labels_train)
print "Training Time SVM:", round(time()-t0,3), "seconds"

t1 = time()
pred = clf.predict(features_test)
print "Prediction Time SVM:", round(time()-t1,3), "seconds"

from sklearn.metrics import accuracy_score
print accuracy_score(pred, labels_test)
#########################################################
 # C = 10000
clf = SVC(kernel='rbf', C = 10000.)
print "C = 10000.0"

t0 = time()

clf.fit(features_train, labels_train)
print "Training Time SVM:", round(time()-t0,3), "seconds"

t1 = time()
pred = clf.predict(features_test)
print "Prediction Time SVM:", round(time()-t1,3), "seconds"

from sklearn.metrics import accuracy_score
print accuracy_score(pred, labels_test)