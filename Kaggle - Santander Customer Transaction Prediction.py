# -*- coding: utf-8 -*-
"""
Spyder Editor

Author @subbu narayanaswamy

This is a temporary script file.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

########################################################################################################################
############################################# IMPORTING DATA ###########################################################
########################################################################################################################
train = pd.read_csv("E:/Kaggle/Santander Customer Transaction Prediction/train.csv")
oov = pd.read_csv("E:/Kaggle/Santander Customer Transaction Prediction/test.csv")
oov1 = oov.drop(['ID_code'], axis=1)

## DATA EXPLORATION
type(train) ## Type of the dataset: Pandas DataFrame
train.shape ## rows X columns
train.dtypes ## Lists all variable names and data type
train.dtype ### ERROR: APPLICABLE ONLY FOR NUMPY ARRAYS
type(test)
test.shape
test.dtypes
train.head()
test.head()

## WRITING OUTPUT TO A EXCEL FILE
writer = pd.ExcelWriter('E:/Kaggle/Santander Customer Transaction Prediction/testxls.xlsx', engine = 'xlsxwriter') ## Creating an excel file and assigning it to 'writer'
test.dtypes.to_excel(writer) ## Writing data from 'variable_type' onto 'writer'
writer.save() ## Save the 'writer' file

## VARIABLE DISTRIBUTION
train.describe()
perc=[0.20,0.40,0.60,0.80]
include=['object','float','int']
train.describe(percentiles=perc, include=include) ## mean,std, min, max, 25%, 50%,75% for only select attributes

########################################################################################################################
############################################# FEATURE ENGINEERING ######################################################
########################################################################################################################

## Understand the correlation between X and Y



########################################################################################################################
################ SPLITTING DATASET INTO DEPENDENT VARIABLE AND OTHER PREDICTORS ########################################
########################################################################################################################

## CREATING Y DEPENDENT VARIABLE
y = train['target']
y1 = np.array(train['target'])

## CREATING X PREDICTORS
x = train.drop(['target','ID_code'], axis=1)
x1 = np.array(train.drop(['target','ID_code'], axis=1))

## CREATING TRAINING AND TESTING DATASET
x_train,x_test, y_train, y_test = train_test_split(x,y,test_size=0.4) ## training dataset is split into training & testing samples

## Qn: Why does train_test_split function convert numpy arrays into pandas series???
x1_train,x1_test, y1_train, y1_test = train_test_split(x,y,test_size=0.4) ## training dataset is split into training & testing samples

np.sum(y1,axis=0) ## 1 = 20,098; 0 = 179,902 10% target rate
np.sum(y1_train,axis=0) ## 12,047; 0 = 107,953 10% target rate
np.sum(y1_test,axis=0) ## 8,051; 0 = 71,949 10% target rate

print('training predictors:',x_train1.shape)
print('testing predictors:',x_test1.shape)
print('training y:',y1_train.shape)
print('testing y:',y1_test.shape)


########################################################################################################################
############################################## TRAINING THE MODEL ######################################################
########################################################################################################################

## RANDOM FOREST
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100)

clf = clf.fit(x1_train,y1_train)

## LIGHTGBM CLASSIFIER
import lightgbm as lgb
from lightgbm import LGBMClassifier

lgbm = LGBMClassifier(objective='binary', random_state=5)

lgbmfit = lgbm.fit(x1_train, y1_train)

## LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logregfit = logreg.fit(x1_train, y1_train)

## SCORING THE MODEL ON 40% TEST DATASET
##y1_pred = clf.predict(x1_test) ##np.array
##y1_pred = lgbmfit.predict(x1_test) ##np.array
y1_pred = logregfit.predict(x1_test) ##np.array


########################################################################################################################
############################################# MODEL EVALUATION #########################################################
########################################################################################################################

## SUMMING UP ALL PREDICTIONS
np.sum(y1_pred,axis=0)

## EXPORTING PREDICTIONS AS A CSV
np.savetxt("E:/Kaggle/Santander Customer Transaction Prediction/y1test.csv", y1_test, delimiter=",") ##Dump a NumPy array into a csv file

## CHECKING ACCURACY OF ACTUAL VS. PREDICTED IN THE TESTING DATASET
from sklearn import metrics
print("Accuracy:", metrics.accuracy_score(y1_test, y1_pred)) ## close to 90% accuracy

## CONFUSION MATRIX
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y1_test, y1_pred)
print(pd.DataFrame(confusion_matrix(y1_test, y1_pred, labels=[0,1]), index=['Truth: No Response', 'Truth: Response'], columns=['Predicted: No Response', 'Predicted: Response']))

## TP, TN, FP, FN
TP = cm[1][1]
TN = cm[0][0]
FP = cm[0][1]
FN = cm[1][0]

print ("True Positive:", TP)
print ("True Negative:", TN)
print ("False Positive:", FP)
print ("False Negative:", FN)

class_report = metrics.classification_report(y1_test, y1_pred)
class_report

## ROC CURVE
fpr, tpr, thresholds = metrics.roc_curve(y1_test, y1_pred)

%matplotlib inline
import matplotlib.pyplot as plt

plt.plot(fpr,tpr)
plt.rcParams['font.size'] = 12
plt.title('ROC curve for Santander Customer Transaction Prediction')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)

## AUC
print(metrics.roc_auc_score(y1_test,y1_pred)) ## 0.5 (pathetic)

## DEFINING A FUNCTION FOR THRESHOLDS -- USED MAINLY FOR LOGISTIC REGRESSION MODELS
def evaluate_threshold(threshold):
    print('Sensitivity:', tpr[thresholds > threshold][-1])
    print('Specificity:', 1 - fpr[thresholds > threshold][-1])

evaluate_threshold(0.1)


## Qn: What is the difference between dataframe / pandas series / numpy arrays????? VERY CONFUSING

########################################################################################################################
## SCORING THE MODEL ON OUT OF TIME SAMPLE TEST DATA -- NO PERFORMANCE AVAILABLE -- SUBMIT IN KAGGLE FOR EVALUATION ####
########################################################################################################################

##y_oov = clf.predict(oov1)  ##RF
##y_oov = lgbmfit.predict(oov1) ##LGBM
y_oov = logregfit.predict(oov1) ##Logistic Regression
y_oov.shape
type(y_oov)
type(oov)

## somehitng seems off - I only see 3 records flagged as target ???
np.savetxt("E:/Kaggle/Santander Customer Transaction Prediction/foo.csv", y_oov, delimiter=",") ##Dump a NumPy array into a csv file


oov['target'] = y_oov 

submission = oov[['ID_code','target']]
type(submission)
submission.shape

## WRITING OUTPUT TO A EXCEL FILE
writer = pd.ExcelWriter('E:/Kaggle/Santander Customer Transaction Prediction/submission_LR.xlsx', engine = 'xlsxwriter') ## Creating an excel file and assigning it to 'writer'
submission.to_excel(writer) ## Writing data from 'variable_type' onto 'writer'
writer.save() ## Save the 'writer' file