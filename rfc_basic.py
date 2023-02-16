#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 11:12:28 2021

@author: shirinjamshidi
"""

################################
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import numpy as np
import pandas as pd
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format
from sklearn import preprocessing
import feather
#from keras.models import model_from_json

######
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
#import keras
#from keras.models import Sequential
#from keras.layers import Dense, Dropout
#import keras.utils

##############Encoder Function####################
#You can change bins based on the range you decide
enc = preprocessing.LabelEncoder()

#bins = [0, 99.99999, 999.9999, 9999.9999, 1000000.0]
#labels = ['strong', 'medium', 'weak', 'inactive']

bins = [0, 9999.9999, 1000000.0]
labels = ['active', 'inactive']
###############################################################################
#Here add some constants
NEPOCHS = 100
FRAC_1 = 1
FRAC_SAMPLE = 1
NROWS_DRUGBANK = 10000
MWT_LOWER_BOUND = 100
MWT_UPPER_BOUND = 900
LOGP_LOWER_BOUND = -4
LOGP_UPPER_BOUND = 10
MIN_NUMBER_OBSERVATIONS_PER_TARGET = 25
###############################################################################
###########################ML Hyperparameters for shallow learning algorithms##
rf_basic_nestimators = 1000
rf_basic_min_samples_leaf = 10


#######################################-------------------
class FP:
  def __init__(self, fp):
        self.fp = fp
  def __str__(self):
      return self.fp.__str__()
  
final_set_featurized_filtered=pd.read_pickle("final_set_featurized_filtered.pickle")
###############Splitting the dataset into training and validation set################
X_vars = final_set_featurized_filtered[['xlogp', 'molecular_weight', 'hydrogen_bond_donors', 'hydrogen_bond_acceptors', 'HeavyAtomMolWt', 'MaxPartialCharge', 'MinPartialCharge', 'MaxAbsPartialCharge', 'MinAbsPartialCharge']]
y = final_set_featurized_filtered['gpcr_binding_encoded']
#####################################################################################
#Split data for training and testing purpose
#X_train, X_test, y_train, y_test = train_test_split(X_vars,y,test_size=0.3,stratify=y)
#MS: non stratified.. because ValueError: The least populated class in y has only 1 member, which is too few. The minimum number of groups for any class cannot be less than 2
X_train, X_test, y_train, y_test = train_test_split(X_vars,y,test_size=0.25)

#MS changing basic columns...
#X_basic_columns = X_test.iloc[:,1:5].columns


X_basic_columns = X_test.iloc[:,0:4].columns


del X_vars

coded_labels=feather.read_dataframe("coded_gpcr_list.feather")
scaler_1 = StandardScaler()
scaler_2 = StandardScaler()

#X_train.reset_index(inplace=True)
#X_train.columns = X_train.columns.astype(str)

#X_test.reset_index(inplace=True)
#X_test.columns = X_test.columns.astype(str)


###############################################################################
del final_set_featurized_filtered
######################Save some files################
#X_train_basic = scaler_1.fit_transform(X_train.iloc[:,2:6])
#X_test_basic = scaler_1.transform(X_test.iloc[:,2:6])
#y_train = y_train.iloc[:,1]
#y_test = y_test.iloc[:,1]
#MS
X_train_basic = scaler_1.fit_transform(X_train.iloc[:,0:4])
X_test_basic = scaler_1.transform(X_test.iloc[:,0:4])

NUMBER_TRAINING_EXAMPLES = y_train.size
NUMBER_TEST_EXAMPLES = y_test.size
##########Sklearn Models Initiation############################################
rfc_model_basic = RandomForestClassifier(min_samples_leaf=rf_basic_min_samples_leaf,bootstrap=True,oob_score=True,n_estimators=rf_basic_nestimators,n_jobs=-1)
#####################Models Training Using Basic Molecular Descriptors: 1- RandomForests#####################
rfc_basic = rfc_model_basic.fit(X_train_basic,  y_train.values.ravel())
rfc_basic_predictions = rfc_basic.predict(X_test_basic)

rfc_basic_classify_training_score = rfc_basic.score(X_train_basic,  y_train.values.ravel())
rfc_basic_classify_validation_score = rfc_basic.score(X_test_basic, y_test)
rfc_basic_classify_OOB_score = rfc_basic.oob_score_
rfc_basic_F1_score_micro = metrics.f1_score(y_test,rfc_basic_predictions,average="micro")
rfc_basic_F1_score_macro = metrics.f1_score(y_test,rfc_basic_predictions,average="macro")
rfc_basic_mcc_score = metrics.matthews_corrcoef(y_test,rfc_basic_predictions)

print ("The Training Score For The RandomForestClassifier Trained on Basic Molecular Descriptors Is: ", rfc_basic_classify_training_score)
print ("The Validation Score For The RandomForestClassifier Trained on Basic Molecular Descriptors Is: ", rfc_basic_classify_validation_score)
print ("The Out Of Bag Score For The RandomForestClassifier Trained on Basic Molecular Descriptors Is: ", rfc_basic.oob_score_)
print ("The Micro-F1 Score For The RandomForestClassifier Trained on Basic Molecular Descriptors Is: ", rfc_basic_F1_score_micro)
print ("The Macro-F1 Score For The RandomForestClassifier Trained on Basic Molecular Descriptors Is: ", rfc_basic_F1_score_macro)
print ("The Matthews correlation coefficient Score For The RandomForestClassifier Trained on Basic Molecular Descriptors Is: ", rfc_basic_mcc_score)
############################################################################################################################################
############################################################################################################################################
print ("This is the classification report for the RandomForestClassifier Trained on Basic Molecular Descriptors Is: ", \
       classification_report(y_test, rfc_basic_predictions))
print ("This is the features importance for the RandomForestClassifier Trained on Basic Molecular Descriptors Is: ", rfc_basic.feature_importances_)
rfc_basic_feature_importances = pd.Series(rfc_basic.feature_importances_, index=X_basic_columns)
#rfc_basic_feature_barplot = rfc_basic_feature_importances.nlargest(20).plot(kind='barh',fontsize=16)
#plt.savefig("rfc_basic_feature_barplot")
rfc_basic_feature_importances.to_csv("rfc_basic_feature_importances.csv")



######################free some memory################
del X_train_basic
del X_test_basic

###############################################################
####Save Models Scores
l3 = ["rfc_basic_classify_training_score", 
"rfc_basic_classify_validation_score",
"rfc_basic_F1_score_macro",
"rfc_basic_F1_score_micro",
"rfc_basic_mcc_score"
]


l4 = [rfc_basic_classify_training_score, 
rfc_basic_classify_validation_score,
rfc_basic_F1_score_macro,
rfc_basic_F1_score_micro,
rfc_basic_mcc_score
]

scoring_matrix = pd.DataFrame(l4,l3)
scoring_matrix.to_csv("scoring_matrix_full.csv")
###############################################################################
###############################################################################
l5 = ["rfc_basic_classify_training_score",
"rfc_basic_classify_validation_score",
"rfc_basic_F1_score_macro",
"rfc_basic_mcc_score"]


l6 = [rfc_basic_classify_training_score,
rfc_basic_classify_validation_score,
rfc_basic_F1_score_macro,
rfc_basic_mcc_score,
]


basic_molec_desc_df = pd.DataFrame(l6,l5).reset_index()

#######################################
###############################################################################
coded_labels.to_csv("coded_gpcr_list.csv")

print ("Mission Accomplished and prediction made")
