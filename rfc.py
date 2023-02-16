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
import pickle
#from keras.models import model_from_json

######
from rdkit.Chem import PandasTools
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

rf_rdkit_nestimators = 5000
rf_rdkit_min_samples_leaf = 3

rf_FP_nestimators = 1000
rf_FP_min_samples_leaf = 3
#############################################################################
xgb_basic_nestimators = 5000
xgb_basic_learning_rate = 0.001
xgb_basic_max_depth = 3

xgb_rdkit_nestimators = 5000
xgb_rdkit_learning_rate = 0.1
xgb_rdkit_max_depth = 5

xgb_FP_nestimators = 1000
xgb_FP_learning_rate = 0.1
xgb_FP_max_depth = 3
#############################################################################

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

#MS changing basic and rdkit columns...
#X_basic_columns = X_test.iloc[:,1:5].columns
#X_rdkit_columns = X_test.iloc[:,15:].columns

X_basic_columns = X_test.iloc[:,0:4].columns
X_rdkit_columns = X_test.iloc[:,4:].columns

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

##-----------------------------------------------------------------##

#gb_model_basic = xgb.XGBClassifier(learning_rate=xgb_basic_learning_rate,n_estimators=xgb_basic_nestimators,objective='multi:softprob',max_depth=xgb_basic_max_depth)
#####################Models Training Using Basic Molecular Descriptors: 2- XGB#####################
#xgb_basic = xgb_model_basic.fit(X_train_basic,  y_train.values.ravel())
#xgb_basic_predictions = xgb_basic.predict(X_test_basic)
#
#xgb_basic_classify_training_score = xgb_basic.score(X_train_basic,  y_train.values.ravel())
#xgb_basic_classify_validation_score = xgb_basic.score(X_test_basic, y_test)
#xgb_basic_F1_score_micro = metrics.f1_score(y_test,xgb_basic_predictions,average="micro")
#xgb_basic_F1_score_macro = metrics.f1_score(y_test,xgb_basic_predictions,average="macro")
#
#print ("The Training Score For XGB Trained on Basic Molecular Descriptors Is: ", xgb_basic_classify_training_score)
#print ("The Validation Score For XGB Trained on Basic Molecular Descriptors Is: ", xgb_basic_classify_validation_score)
#print ("The Micro-F1 Score For XGB Trained on Basic Molecular Descriptors Is: ", xgb_basic_F1_score_micro)
#print ("The Macro-F1 Score For XGB Trained on Basic Molecular Descriptors Is: ", xgb_basic_F1_score_macro)
############################################################################################################################################
############################################################################################################################################

############################################################################################################################################

#number_of_classes = y_train.nunique()

######################free some memory################
del X_train_basic
del X_test_basic



rfc_model_rdkit = RandomForestClassifier(min_samples_leaf=rf_rdkit_min_samples_leaf,bootstrap=True,oob_score=True,n_estimators=rf_rdkit_nestimators,n_jobs=-1)
###########################Load RDKIT Molecular Descriptors###########################
############################################################################################################################################
#X_train_rdkit = scaler_2.fit_transform(X_train.iloc[:,13:])
#X_test_rdkit = scaler_2.transform(X_test.iloc[:,13:])
X_train_rdkit = scaler_2.fit_transform(X_train.iloc[:,4:])
X_test_rdkit = scaler_2.transform(X_test.iloc[:,4:])
#####################Models Training Using RDKIT Molecular Descriptors: 6- RandomForests#####################
rfc_rdkit = rfc_model_rdkit.fit(X_train_rdkit,  y_train.values.ravel())
rfc_rdkit_predictions = rfc_rdkit.predict(X_test_rdkit)

rfc_rdkit_classify_training_score = rfc_rdkit.score(X_train_rdkit,  y_train.values.ravel())
rfc_rdkit_classify_validation_score = rfc_rdkit.score(X_test_rdkit, y_test)
rfc_rdkit_classify_OOB_score = rfc_rdkit.oob_score_
rfc_rdkit_F1_score_micro = metrics.f1_score(y_test,rfc_rdkit_predictions,average="micro")
rfc_rdkit_F1_score_macro = metrics.f1_score(y_test,rfc_rdkit_predictions,average="macro")
rfc_rdkit_mcc_score = metrics.matthews_corrcoef(y_test,rfc_rdkit_predictions)

print ("The Training Score For The RandomForestClassifier Trained on RDKIT Molecular Descriptors Is: ", rfc_rdkit_classify_training_score)
print ("The Validation Score For The RandomForestClassifier Trained on RDKIT Molecular Descriptors Is: ", rfc_rdkit_classify_validation_score)
print ("The Out Of Bag Score For The RandomForestClassifier Trained on RDKIT Molecular Descriptors Is: ", rfc_rdkit.oob_score_)
print ("The Micro-F1 Score For The RandomForestClassifier Trained on RDKIT Molecular Descriptors Is: ", rfc_rdkit_F1_score_micro)
print ("The Macro-F1 Score For The RandomForestClassifier Trained on RDKIT Molecular Descriptors Is: ", rfc_rdkit_F1_score_macro)
print ("The Matthews correlation coefficient Score For The RandomForestClassifier Trained on RDKIT Molecular Descriptors Is: ", rfc_rdkit_mcc_score)
############################################################################################################################################
############################################################################################################################################
print ("This is the classification report for the RandomForestClassifier Trained on RDKIT Molecular Descriptors Is: ", \
       classification_report(y_test, rfc_rdkit_predictions))
print ("This is the features importance for the RandomForestClassifier Trained on RDKIT Molecular Descriptors Is: ", rfc_rdkit.feature_importances_)

rfc_rdkit_feature_importances = pd.Series(rfc_rdkit.feature_importances_, index=X_rdkit_columns)
#rfc_rdkit_feature_barplot = rfc_rdkit_feature_importances.nlargest(20).plot(kind='barh',fontsize=16)
#plt.savefig("rfc_rdkit_feature_barplot")
rfc_rdkit_feature_importances.to_csv("rfc_rdkit_feature_importances.csv")

#xgb_model_rdkit = xgb.XGBClassifier(learning_rate=xgb_rdkit_learning_rate,n_estimators=xgb_rdkit_nestimators,objective='multi:softprob',max_depth=xgb_rdkit_max_depth)
#####################Models Training Using RDKIT Molecular Descriptors: 7- XGB#####################
#xgb_rdkit = xgb_model_rdkit.fit(X_train_rdkit,  y_train.values.ravel())
#xgb_rdkit_predictions = xgb_rdkit.predict(X_test_rdkit)
#
#xgb_rdkit_classify_training_score = xgb_rdkit.score(X_train_rdkit,  y_train.values.ravel())
#xgb_rdkit_classify_validation_score = xgb_rdkit.score(X_test_rdkit, y_test)
#xgb_rdkit_F1_score_micro = metrics.f1_score(y_test,xgb_rdkit_predictions,average="micro")
#xgb_rdkit_F1_score_macro = metrics.f1_score(y_test,xgb_rdkit_predictions,average="macro")
#
#print ("The Training Score For XGB Trained on RDKIT Molecular Descriptors Is: ", xgb_rdkit_classify_training_score)
#print ("The Validation Score For XGB Trained on RDKIT Molecular Descriptors Is: ", xgb_rdkit_classify_validation_score)
#print ("The Micro-F1 Score For XGB Trained on RDKIT Molecular Descriptors Is: ", xgb_rdkit_F1_score_micro)
#print ("The Macro-F1 Score For XGB Trained on RDKIT Molecular Descriptors Is: ", xgb_rdkit_F1_score_macro)
############################################################################################################################################
############################################################################################################################################

############################################################################################################################################

###########################DeepLearning Model Training On: 10- RDKIT Molecular Descriptors#########################
#number_of_classes = y_train.nunique()


######################free some memory################
del X_train_rdkit
del X_test_rdkit



################Load Molecular Fingerprints################################
X_train_FP = pd.DataFrame(np.load("X_train_FP.npy"))
X_test_FP = pd.DataFrame(np.load("X_test_FP.npy"))

rfc_model_FP = RandomForestClassifier(min_samples_leaf=rf_FP_min_samples_leaf,bootstrap=True,oob_score=True,n_estimators=rf_FP_nestimators,n_jobs=-1)
#####################Models Training Using RDKIT Molecular FingerPrints: 11- RandomForests#####################
rfc_FP = rfc_model_FP.fit(np.vstack(X_train_FP.iloc[:,0]),  y_train.values.ravel())

rfc_FP_predictions = rfc_FP.predict(np.vstack(X_test_FP.iloc[:,0]))

rfc_FP_classify_training_score = rfc_FP.score(np.vstack(X_train_FP.iloc[:,0]),  y_train.values.ravel())
rfc_FP_classify_validation_score = rfc_FP.score(np.vstack(X_test_FP.iloc[:,0]), y_test)
rfc_FP_classify_OOB_score = rfc_FP.oob_score_
rfc_FP_F1_score_micro = metrics.f1_score(y_test,rfc_FP_predictions,average="micro")
rfc_FP_F1_score_macro = metrics.f1_score(y_test,rfc_FP_predictions,average="macro")
rfc_FP_mcc_score = metrics.matthews_corrcoef(y_test,rfc_FP_predictions)

print ("The Training Score For The RandomForestClassifier Trained on RDKIT Molecular FingerPrints Is: ", rfc_FP_classify_training_score)
print ("The Validation Score For The RandomForestClassifier Trained on RDKIT Molecular FingerPrints Is: ", rfc_FP_classify_validation_score)
print ("The Out Of Bag Score For The RandomForestClassifier Trained on RDKIT Molecular FingerPrints Is: ", rfc_FP.oob_score_)
print ("The Micro-F1 Score For The RandomForestClassifier Trained on RDKIT Molecular FingerPrints Is: ", rfc_FP_F1_score_micro)
print ("The Macro-F1 Score For The RandomForestClassifier Trained on RDKIT Molecular FingerPrints Is: ", rfc_FP_F1_score_macro)
print ("The Matthews correlation coefficient Score For The RandomForestClassifier Trained on RDKIT Molecular FingerPrints Is: ", rfc_FP_mcc_score)
############################################################################################################################################
############################################################################################################################################
print ("This is the classification report for the RandomForestClassifier Trained on RDKIT Molecular FingerPrints Is: ", \
       classification_report(y_test, rfc_FP_predictions))
print ("This is the features importance for the RandomForestClassifier Trained on RDKIT Molecular FingerPrints Is: ", rfc_FP.feature_importances_)

rfc_FP_feature_importances = pd.Series(rfc_FP.feature_importances_)
#rfc_FP_feature_barplot = rfc_FP_feature_importances.nlargest(20).plot(kind='barh',fontsize=16)
#plt.savefig("rfc_FP_feature_barplot")
rfc_FP_feature_importances.to_csv("rfc_FP_feature_importances.csv")

#xgb_model_FP = xgb.XGBClassifier(learning_rate=xgb_FP_learning_rate,n_estimators=xgb_FP_nestimators,objective='multi:softprob',max_depth=xgb_FP_max_depth)
#####################Models Training Using RDKIT Molecular FingerPrints: 12- XGB#####################
#xgb_FP = xgb_model_FP.fit(np.vstack(X_train_FP.iloc[:,0]),  y_train.values.ravel())
#xgb_FP_predictions = xgb_FP.predict(np.vstack(X_test_FP.iloc[:,0]))
#
#xgb_FP_classify_training_score = xgb_FP.score(np.vstack(X_train_FP.iloc[:,0]),  y_train.values.ravel())
#xgb_FP_classify_validation_score = xgb_FP.score(np.vstack(X_test_FP.iloc[:,0]), y_test)
#xgb_FP_F1_score_micro = metrics.f1_score(y_test,xgb_FP_predictions,average="micro")
#xgb_FP_F1_score_macro = metrics.f1_score(y_test,xgb_FP_predictions,average="macro")
#
#print ("The Training Score For XGB Trained on RDKIT Molecular FingerPrints Is: ", xgb_FP_classify_training_score)
#print ("The Validation Score For XGB Trained on RDKIT Molecular FingerPrints Is: ", xgb_FP_classify_validation_score)
#print ("The Micro-F1 Score For XGB Trained on RDKIT Molecular FingerPrints Is: ", xgb_FP_F1_score_micro)
#print ("The Macro-F1 Score For XGB Trained on RDKIT Molecular FingerPrints Is: ", xgb_FP_F1_score_macro)
############################################################################################################################################
############################################################################################################################################

############################################################################################################################################
###########################DeepLearning Model Training On: 15- RDKIT Molecular FingerPrints#########################
#number_of_classes = y_train.nunique()


######################free some memory################
del X_train_FP
del X_test_FP

##############################################################################################################################
##############################################################################################################################
##############################################################################################################################
##Save statitical information
#l1 = [
#"UNIQUE_LIGANDS_AFTER_CLEANING",
#"MERGED_RECEPTORS_AFTER_CLEANING",
#"AVERAGE_MWT_AFTER_CLEANING",
#"MEDIAN_MWT_AFTER_CLEANING",
#"AVERAGE_XLOGP_AFTER_CLEANING",
#"MEDIAN_XLOGP_AFTER_CLEANING",
#"AVERAGE_HBDONORS_AFTER_CLEANING",
#"MEDIAN_HBDONORS_AFTER_CLEANING",
#"AVERAGE_HBACCEPTORS_AFTER_CLEANING",
#"MEDIAN_HBACCEPTORS_AFTER_CLEANING",
#"NUMBER_TRAINING_EXAMPLES",
#"NUMBER_TEST_EXAMPLES"]

#l2 = [
#UNIQUE_LIGANDS_AFTER_CLEANING,
#MERGED_RECEPTORS_AFTER_CLEANING,
#AVERAGE_MWT_AFTER_CLEANING,
#MEDIAN_MWT_AFTER_CLEANING,
#AVERAGE_XLOGP_AFTER_CLEANING,
#MEDIAN_XLOGP_AFTER_CLEANING,
#AVERAGE_HBDONORS_AFTER_CLEANING,
#MEDIAN_HBDONORS_AFTER_CLEANING,
#AVERAGE_HBACCEPTORS_AFTER_CLEANING,
#MEDIAN_HBACCEPTORS_AFTER_CLEANING,
#NUMBER_TRAINING_EXAMPLES,
#NUMBER_TEST_EXAMPLES]


#statistical_information = pd.DataFrame(l2,l1)
#statistical_information.to_csv("statistical_information.csv")

###############################################################
####Save Models Scores
l3 = ["rfc_basic_classify_training_score", 
"xgb_basic_classify_training_score",
"rfc_rdkit_classify_training_score", 
"xgb_rdkit_classify_training_score", 
"rfc_FP_classify_training_score",
"xgb_FP_classify_training_score",
"deep_learning_model_training_score_basic",
"deep_learning_model_training_score_rdkit",
"deep_learning_model_training_score_FP",
"rfc_basic_classify_validation_score",
"xgb_basic_classify_validation_score",
"rfc_rdkit_classify_validation_score",
"xgb_rdkit_classify_validation_score",
"rfc_FP_classify_validation_score",
"xgb_FP_classify_validation_score",
"deep_learning_model_validation_score_basic",
"deep_learning_model_validation_score_rdkit",
"deep_learning_model_validation_score_FP",
"rfc_basic_F1_score_macro",
"xgb_basic_F1_score_macro",
"rfc_rdkit_F1_score_macro",
"xgb_rdkit_F1_score_macro",
"rfc_FP_F1_score_macro", 
"xgb_FP_F1_score_macro",
"rfc_basic_F1_score_micro",
"xgb_basic_F1_score_micro",
"rfc_rdkit_F1_score_micro",
"xgb_rdkit_F1_score_micro",
"rfc_FP_F1_score_micro",
"xgb_FP_F1_score_micro",
"rfc_basic_mcc_score",
"rfc_rdkit_mcc_score", 
"rfc_FP_mcc_score",
]


l4 = [rfc_basic_classify_training_score, 
np.nan,
rfc_rdkit_classify_training_score, 
np.nan, 
rfc_FP_classify_training_score,
np.nan,
rfc_basic_classify_validation_score,
np.nan,
rfc_rdkit_classify_validation_score,
np.nan,
rfc_FP_classify_validation_score,
np.nan,
rfc_basic_F1_score_macro,
np.nan,
rfc_rdkit_F1_score_macro,
np.nan,
rfc_FP_F1_score_macro, 
np.nan,
rfc_basic_F1_score_micro,
np.nan,
rfc_rdkit_F1_score_micro,
np.nan,
rfc_FP_F1_score_micro,
np.nan,
rfc_basic_mcc_score, 
rfc_rdkit_mcc_score, 
rfc_FP_mcc_score,
]

scoring_matrix = pd.DataFrame(l4,l3)
scoring_matrix.to_csv("scoring_matrix_full.csv")
###############################################################################
###############################################################################
l5 = ["rfc_basic_classify_training_score",
"xgb_basic_classify_training_score",
"deep_learning_model_training_score_basic",
"rfc_basic_classify_validation_score",
"xgb_basic_classify_validation_score",
"deep_learning_model_validation_score_basic",
"rfc_basic_F1_score_macro",
"xgb_basic_F1_score_macro",
"rfc_basic_mcc_score", 
"xgb_basic_mcc_score"]


l6 = [rfc_basic_classify_training_score,
np.nan,
rfc_basic_classify_validation_score,
np.nan,
rfc_basic_F1_score_macro,
np.nan,
rfc_basic_mcc_score,
np.nan, 
]



l7 = ["rfc_rdkit_classify_training_score",
"xgb_rdkit_classify_training_score",
"deep_learning_model_training_score_rdkit",
"rfc_rdkit_classify_validation_score",
"xgb_rdkit_classify_validation_score",
"deep_learning_model_validation_score_rdkit",
"rfc_rdkit_F1_score_macro",
"xgb_rdkit_F1_score_macro",
"rfc_rdkit_mcc_score", 
"xgb_rdkit_mcc_score" 
]

l8 = [rfc_rdkit_classify_training_score,
np.nan,
rfc_rdkit_classify_validation_score,
np.nan,
rfc_rdkit_F1_score_macro,
np.nan,
rfc_rdkit_mcc_score
]


#l9 = ["rfc_FP_classify_training_score",
#"xgb_FP_classify_training_score",
#"deep_learning_model_training_score_FP",
#"rfc_FP_classify_validation_score",
#"xgb_FP_classify_validation_score",
#"deep_learning_model_validation_score_FP",
#"rfc_FP_F1_score_macro",
#"xgb_FP_F1_score_macro",
#"rfc_FP_mcc_score",
#]


l9 = ["rfc_FP_classify_training_score",
"xgb_FP_classify_training_score",
"deep_learning_model_training_score_FP",
"rfc_FP_classify_validation_score",
"xgb_FP_classify_validation_score",
"deep_learning_model_validation_score_FP",
"rfc_FP_F1_score_macro",
"xgb_FP_F1_score_macro",
"rfc_FP_mcc_score",
"xgb_FP_mcc_score"
]


l10 = [rfc_FP_classify_training_score,
np.nan,
rfc_FP_classify_validation_score,
np.nan,
rfc_FP_F1_score_macro,
np.nan,
rfc_FP_mcc_score,
np.nan,
]

basic_molec_desc_df = pd.DataFrame(l6,l5).reset_index()
rdkit_molec_desc_df = pd.DataFrame(l8,l7).reset_index()
rdkit_molec_FP_df = pd.DataFrame(l10,l9).reset_index()

scoring_matrix_concat = pd.concat([basic_molec_desc_df,rdkit_molec_desc_df,rdkit_molec_FP_df],axis=1,ignore_index=True)
scoring_matrix_concat.to_csv("scoring_matrix_concat.csv")


scoring_matrix_squeezed = scoring_matrix_concat.iloc[:,[1,3,5]]

scoring_matrix_squeezed.columns = ["Basic_Molecular_Descriptor","RDKIT_Molecular_Descriptor","RDKIT_Molecular_Fingerprints"]

index_list = ["rfc_classify_training_score", 
"xgb_classify_training_score", 
"deep_learning_model_training_score",
"rfc_classify_validation_score",
"xgb_classify_validation_score",
"deep_learning_model_validation_score",
"rfc_F1_score_macro", 
"xgb_F1_score_macro", 
"rfc_mcc_score", 
"xgb_mcc_score"
]

scoring_matrix_squeezed.index = index_list 

scoring_matrix_squeezed.to_csv("scoring_matrix_squeezed.csv")


############Save RDKIT descriptors Models###########
filename = 'rf_rdkit_classify_model.sav'
pickle.dump(rfc_rdkit, open(filename, 'wb'))

#filename = 'xgb_rdkit_classify_model.sav'
#pickle.dump(xgb_rdkit, open(filename, 'wb'))


#########Save FingerPrints Models
filename = 'rfc_rdkit_classify_fp.sav'
pickle.dump(rfc_FP, open(filename, 'wb'))

#filename = 'xgb_rdkit_classify_fp.sav'
#pickle.dump(xgb_FP, open(filename, 'wb'))

#------------------------------------------------------------
from deepchem.utils.typing import RDKitMol
from deepchem.feat.base_classes import MolecularFeaturizer

class MRDKitDescriptors(MolecularFeaturizer):
  def __init__(self, use_fragment=True, ipc_avg=True):
    self.use_fragment = use_fragment
    self.ipc_avg = ipc_avg
    self.descriptors = []
    self.descList = []

  def _featurize(self, datapoint: RDKitMol, **kwargs) -> np.ndarray:
    if 'mol' in kwargs:
      datapoint = kwargs.get("mol")
      raise DeprecationWarning(
          'Mol is being phased out as a parameter, please pass "datapoint" instead.'
      )
    # initialize
    if len(self.descList) == 0:
      try:
        from rdkit.Chem import Descriptors
        Descriptors.descList = [x for x in Descriptors.descList if (x[0]!="FpDensityMorgan1" and x[0]!="FpDensityMorgan2" and x[0]!="FpDensityMorgan3" and x[0]!="NumRadicalElectrons")]
        for descriptor, function in Descriptors.descList:
          if self.use_fragment is False and descriptor.startswith('fr_'):
            continue
          self.descriptors.append(descriptor)
          self.descList.append((descriptor, function))
      except ModuleNotFoundError:
        raise ImportError("This class requires RDKit to be installed.")

    # check initialization
    assert len(self.descriptors) == len(self.descList)

    features = []
    for desc_name, function in self.descList:
      if desc_name == 'Ipc' and self.ipc_avg:
        feature = function(datapoint, avg=True)
      else:
        feature = function(datapoint)
      features.append(feature)
    return np.asarray(features)

def featurize_smiles_df(df, featurizer, field, log_every_N=1000, verbose=True):
  """Featurize individual compounds in dataframe.
  Given a featurizer that operates on individual chemical compounds
  or macromolecules, compute & add features for that compound to the
  features dataframe
  """
  sample_elems = df[field].tolist()
  features = []
  from rdkit import Chem
  from rdkit.Chem import rdmolfiles
  from rdkit.Chem import rdmolops
  for ind, elem in enumerate(sample_elems):
    mol = Chem.MolFromSmiles(elem)
    # TODO (ytz) this is a bandage solution to reorder the atoms so
    # that they're always in the same canonical order. Presumably this
    # should be correctly implemented in the future for graph mols.
    if mol:
      new_order = rdmolfiles.CanonicalRankAtoms(mol)
      mol = rdmolops.RenumberAtoms(mol, new_order)
    if ind % log_every_N == 0:
      log("Featurizing sample %d" % ind, verbose)
    features.append(featurizer.featurize([mol]))
  valid_inds = np.array(
      [1 if elt.size > 0 else 0 for elt in features], dtype=bool)
  features = [elt for (is_valid, elt) in zip(valid_inds, features) if is_valid]
  return np.squeeze(np.array(features), axis=1), valid_inds


def log(string, verbose=True):
  """Print string if verbose."""
  if verbose:
    print(string)

from rdkit.Chem import AllChem
from rdkit import DataStructs

def addExplFP(df,molColumn):
    fpCache = []
    for mol in df[molColumn]:
        res = AllChem.GetMorganFingerprintAsBitVect(mol,4,nBits=2048)
        fpCache.append(res)
    arr = np.empty((len(df),), dtype=object)
    arr[:]=fpCache
    S =  pd.Series(arr,index=df.index,name='explFP')
    return df.join(pd.DataFrame(S))

def convertToNumpy(df,fpCol):
    fpCache = []
    for fp in df[fpCol]:
        res = np.zeros(len(fp),np.int32)
        DataStructs.ConvertToNumpyArray(fp,res)
        fpCache.append(res)
    '''
    it is necessary to constructs an empty object array in advance and fill that later,
    because directly initializing an array with the fingerprint would trigger the numpy
    type recognition and result in a array of integers that again would trigger pandas
    to construct a Series object per bit position
    '''
    arr = np.empty((len(df),), dtype=object)
    arr[:]=fpCache
    S =  pd.Series(arr,index=df.index,name='npFP')
    return df.join(pd.DataFrame(S))

#######################################
drug_bank_df = pd.read_csv("drug_bank_structures.csv",nrows=NROWS_DRUGBANK)
drug_bank_df.columns = drug_bank_df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
drug_bank_df_selected_cols = drug_bank_df[['drugbank_id','name','smiles','pubchem_substance_id','drug_groups']]
drug_bank_df_selected_cols.dropna(subset = ['smiles'],inplace=True)

rdkit_featurizer = MRDKitDescriptors()
drug_bank_df_selected_cols_feat = featurize_smiles_df(drug_bank_df_selected_cols,rdkit_featurizer,'smiles')
allowedDescriptors = [name[0] for name in rdkit_featurizer.descList]

drug_bank_df_selected_cols_feat_df = pd.DataFrame(drug_bank_df_selected_cols_feat[0],columns=allowedDescriptors)

drug_bank_df_selected_cols['tmp1'] = np.arange(len(drug_bank_df_selected_cols))
drug_bank_df_selected_cols_feat_df['tmp1'] = np.arange(len(drug_bank_df_selected_cols_feat_df))
drug_bank_df_selected_cols_featurized = pd.merge(drug_bank_df_selected_cols, drug_bank_df_selected_cols_feat_df, on=['tmp1'])
drug_bank_df_selected_cols_featurized.drop('tmp1', axis=1, inplace=True)
drug_bank_df_selected_cols_featurized.dropna(inplace=True)
mask7 = drug_bank_df_selected_cols_featurized.MolLogP.between(LOGP_LOWER_BOUND,LOGP_UPPER_BOUND)
mask8 = drug_bank_df_selected_cols_featurized.MolWt.between(MWT_LOWER_BOUND,MWT_UPPER_BOUND)
drug_bank_df_selected_cols_featurized_filtered = drug_bank_df_selected_cols_featurized[(mask7) & (mask8)]
PandasTools.AddMoleculeColumnToFrame(drug_bank_df_selected_cols_featurized_filtered,smilesCol='smiles',molCol='molecule',includeFingerprints=True)
drug_bank_df_selected_cols_featurized_filtered.dropna(inplace=True)

drug_bank_df_selected_cols_featurized_filtered_subset = drug_bank_df_selected_cols_featurized_filtered.iloc[:,0:5]
drug_bank_pred = drug_bank_df_selected_cols_featurized_filtered.iloc[:,5:116]
drug_bank_pred_fp = drug_bank_df_selected_cols_featurized_filtered.iloc[:,[0,1,2,3,116]]
drug_bank_pred_fp = addExplFP(drug_bank_pred_fp,'molecule')
drug_bank_pred_fp = convertToNumpy(drug_bank_pred_fp,'explFP')
X_drug_bank_fp_pred = drug_bank_pred_fp[['npFP']]
drug_bank_pred_scaled = scaler_2.transform(drug_bank_pred)
###############################################################################
###############################################################################

#Load the models
rf_rdkit_classify = pickle.load(open("rf_rdkit_classify_model.sav", 'rb'))
rf_rdkit_classify_prediction = rf_rdkit_classify.predict(drug_bank_pred_scaled)
rf_rdkit_classify_prediction_proba = rf_rdkit_classify.predict_proba(drug_bank_pred_scaled)
del rf_rdkit_classify
print ("rf_rdkit_classify prediction made")

##Load the models
#xgb_rdkit_classify = pickle.load(open("xgb_rdkit_classify_model.sav", 'rb'))
#xgb_rdkit_classify_prediction = xgb_rdkit_classify.predict(drug_bank_pred_scaled)
#xgb_rdkit_classify_prediction_proba = xgb_rdkit_classify.predict_proba(drug_bank_pred_scaled)
#del xgb_rdkit_classify
#print ("xgb_rdkit_classify prediction made")





# load json and create model




rfc_fp = pickle.load(open("rfc_rdkit_classify_fp.sav", 'rb'))
rfc_fp_prediction = rfc_fp.predict(np.vstack(X_drug_bank_fp_pred['npFP']))
rfc_fp_prediction_proba = rfc_fp.predict_proba(np.vstack(X_drug_bank_fp_pred['npFP']))
del rfc_fp 
print ("rfc_fp prediction made")

#xgb_fp = pickle.load(open("xgb_rdkit_classify_fp.sav", 'rb'))
#xgb_fp_prediction = xgb_fp.predict(np.vstack(X_drug_bank_fp_pred['npFP']))
#xgb_fp_prediction_proba = xgb_fp.predict_proba(np.vstack(X_drug_bank_fp_pred['npFP']))
#del xgb_fp 
#print ("xgb_fp prediction made")


# load json and create model


################################################################################

drug_bank_df_selected_cols_featurized_filtered_subset['prediction_class_rf_rdkit_classify_prediction'] = rf_rdkit_classify_prediction
drug_bank_df_selected_cols_featurized_filtered_subset['prediction_class_rf_rdkit_classify_prediction_proba'] = rf_rdkit_classify_prediction_proba.max()

#drug_bank_df_selected_cols_featurized_filtered_subset['prediction_class_xgb_rdkit_classify_prediction'] = xgb_rdkit_classify_prediction
#drug_bank_df_selected_cols_featurized_filtered_subset['prediction_class_xgb_rdkit_classify_prediction_proba'] = xgb_rdkit_classify_prediction_proba.max()

drug_bank_df_selected_cols_featurized_filtered_subset['prediction_class_rfc_fp_prediction'] = rfc_fp_prediction
drug_bank_df_selected_cols_featurized_filtered_subset['prediction_class_rfc_fp_prediction_proba'] = rfc_fp_prediction_proba.max()

#drug_bank_df_selected_cols_featurized_filtered_subset['prediction_class_xgb_fp_prediction'] = xgb_fp_prediction
#drug_bank_df_selected_cols_featurized_filtered_subset['prediction_class_xgb_fp_prediction_proba'] = xgb_fp_prediction_proba.max()

###############################################################################
###############################################################################
###############################################################################
coded_labels.to_csv("coded_gpcr_list.csv")
merged_predictions_fullcols = pd.merge(drug_bank_df_selected_cols_featurized_filtered_subset,coded_labels,how="inner", left_on="prediction_class_dl_model_fp_prediction", right_on="gpcr_binding_encoded")
merged_predictions_fullcols.drop_duplicates(inplace=True)

#merged_predictions = merged_predictions_fullcols.iloc[:,[0,1,2,3,4,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132]]
merged_predictions_fullcols.to_csv("final_drugbank_predictions_subset_drop_dup.csv")

print ("Mission Accomplished and prediction made")
