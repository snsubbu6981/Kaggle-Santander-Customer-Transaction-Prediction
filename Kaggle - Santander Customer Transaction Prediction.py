# -*- coding: utf-8 -*-
"""
Spyder Editor

Author @subbu narayanaswamy

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import scipy.stats as ss
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

total = trainData.isnull().sum().sort_values(ascending=False)
percent = (trainData.isnull().sum()/trainData.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])


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

## Information Value calculation

# import packages
import pandas as pd
import numpy as np
import pandas.core.algorithms as algos
from pandas import Series
import scipy.stats.stats as stats
import re
import traceback
import string

max_bin = 20
force_bin = 3

# define a binning function
def mono_bin(Y, X, n = max_bin):
    
    df1 = pd.DataFrame({"X": X, "Y": Y})
    justmiss = df1[['X','Y']][df1.X.isnull()]
    notmiss = df1[['X','Y']][df1.X.notnull()]
    r = 0
    while np.abs(r) < 1:
        try:
            d1 = pd.DataFrame({"X": notmiss.X, "Y": notmiss.Y, "Bucket": pd.qcut(notmiss.X, n)})
            d2 = d1.groupby('Bucket', as_index=True)
            r, p = stats.spearmanr(d2.mean().X, d2.mean().Y)
            n = n - 1 
        except Exception as e:
            n = n - 1

    if len(d2) == 1:
        n = force_bin         
        bins = algos.quantile(notmiss.X, np.linspace(0, 1, n))
        if len(np.unique(bins)) == 2:
            bins = np.insert(bins, 0, 1)
            bins[1] = bins[1]-(bins[1]/2)
        d1 = pd.DataFrame({"X": notmiss.X, "Y": notmiss.Y, "Bucket": pd.cut(notmiss.X, np.unique(bins),include_lowest=True)}) 
        d2 = d1.groupby('Bucket', as_index=True)
    
    d3 = pd.DataFrame({},index=[])
    d3["MIN_VALUE"] = d2.min().X
    d3["MAX_VALUE"] = d2.max().X
    d3["COUNT"] = d2.count().Y
    d3["EVENT"] = d2.sum().Y
    d3["NONEVENT"] = d2.count().Y - d2.sum().Y
    d3=d3.reset_index(drop=True)
    
    if len(justmiss.index) > 0:
        d4 = pd.DataFrame({'MIN_VALUE':np.nan},index=[0])
        d4["MAX_VALUE"] = np.nan
        d4["COUNT"] = justmiss.count().Y
        d4["EVENT"] = justmiss.sum().Y
        d4["NONEVENT"] = justmiss.count().Y - justmiss.sum().Y
        d3 = d3.append(d4,ignore_index=True)
    
    d3["EVENT_RATE"] = d3.EVENT/d3.COUNT
    d3["NON_EVENT_RATE"] = d3.NONEVENT/d3.COUNT
    d3["DIST_EVENT"] = d3.EVENT/d3.sum().EVENT
    d3["DIST_NON_EVENT"] = d3.NONEVENT/d3.sum().NONEVENT
    d3["WOE"] = np.log(d3.DIST_EVENT/d3.DIST_NON_EVENT)
    d3["IV"] = (d3.DIST_EVENT-d3.DIST_NON_EVENT)*np.log(d3.DIST_EVENT/d3.DIST_NON_EVENT)
    d3["VAR_NAME"] = "VAR"
    d3 = d3[['VAR_NAME','MIN_VALUE', 'MAX_VALUE', 'COUNT', 'EVENT', 'EVENT_RATE', 'NONEVENT', 'NON_EVENT_RATE', 'DIST_EVENT','DIST_NON_EVENT','WOE', 'IV']]       
    d3 = d3.replace([np.inf, -np.inf], 0)
    d3.IV = d3.IV.sum()
    
    return(d3)

def char_bin(Y, X):
        
    df1 = pd.DataFrame({"X": X, "Y": Y})
    justmiss = df1[['X','Y']][df1.X.isnull()]
    notmiss = df1[['X','Y']][df1.X.notnull()]    
    df2 = notmiss.groupby('X',as_index=True)
    
    d3 = pd.DataFrame({},index=[])
    d3["COUNT"] = df2.count().Y
    d3["MIN_VALUE"] = df2.sum().Y.index
    d3["MAX_VALUE"] = d3["MIN_VALUE"]
    d3["EVENT"] = df2.sum().Y
    d3["NONEVENT"] = df2.count().Y - df2.sum().Y
    
    if len(justmiss.index) > 0:
        d4 = pd.DataFrame({'MIN_VALUE':np.nan},index=[0])
        d4["MAX_VALUE"] = np.nan
        d4["COUNT"] = justmiss.count().Y
        d4["EVENT"] = justmiss.sum().Y
        d4["NONEVENT"] = justmiss.count().Y - justmiss.sum().Y
        d3 = d3.append(d4,ignore_index=True)
    
    d3["EVENT_RATE"] = d3.EVENT/d3.COUNT
    d3["NON_EVENT_RATE"] = d3.NONEVENT/d3.COUNT
    d3["DIST_EVENT"] = d3.EVENT/d3.sum().EVENT
    d3["DIST_NON_EVENT"] = d3.NONEVENT/d3.sum().NONEVENT
    d3["WOE"] = np.log(d3.DIST_EVENT/d3.DIST_NON_EVENT)
    d3["IV"] = (d3.DIST_EVENT-d3.DIST_NON_EVENT)*np.log(d3.DIST_EVENT/d3.DIST_NON_EVENT)
    d3["VAR_NAME"] = "VAR"
    d3 = d3[['VAR_NAME','MIN_VALUE', 'MAX_VALUE', 'COUNT', 'EVENT', 'EVENT_RATE', 'NONEVENT', 'NON_EVENT_RATE', 'DIST_EVENT','DIST_NON_EVENT','WOE', 'IV']]      
    d3 = d3.replace([np.inf, -np.inf], 0)
    d3.IV = d3.IV.sum()
    d3 = d3.reset_index(drop=True)
    
    return(d3)

def data_vars(df1, target):
    
    stack = traceback.extract_stack()
    filename, lineno, function_name, code = stack[-2]
    vars_name = re.compile(r'\((.*?)\).*$').search(code).groups()[0]
    final = (re.findall(r"[\w']+", vars_name))[-1]
    
    x = df1.dtypes.index
    count = -1
    
    for i in x:
        if i.upper() not in (final.upper()):
            if np.issubdtype(df1[i], np.number) and len(Series.unique(df1[i])) > 2:
                conv = mono_bin(target, df1[i])
                conv["VAR_NAME"] = i
            else:
                conv = char_bin(target, df1[i])
                conv["VAR_NAME"] = i            
                count = count + 1
                
            if count == 0:
                iv_df = conv
            else:
                iv_df = iv_df.append(conv,ignore_index=True)
    
    iv = pd.DataFrame({'IV':iv_df.groupby('VAR_NAME').IV.max()})
    iv = iv.reset_index()
    return(iv_df,iv)


final_iv, IV = data_vars(x1_test,y1_test)
final_iv
IV

IV.sort_values('IV')

IV.to_csv('E:/Kaggle/Santander Customer Transaction Prediction/IV.csv')


########################################################################################################################
################ SPLITTING DATASET INTO DEPENDENT VARIABLE AND OTHER PREDICTORS ########################################
########################################################################################################################

## CREATING Y DEPENDENT VARIABLE
y = train['target']
y1 = np.array(train['target'])

## CREATING X PREDICTORS
x = train.drop(['target','ID_code'], axis=1)
x1 = np.array(train.drop(['target','ID_code'], axis=1))

##Identifying predictive variables based on feature engineering
x1_iv = train[['var_139','var_146','var_26','var_12','var_53','var_6','var_174','var_166','var_110',
'var_148','var_34','var_76','var_149','var_2','var_21','var_81','var_40','var_190','var_169','var_165','var_1','var_115',
'var_198','var_13','var_67','var_99','var_118','var_184','var_192','var_22','var_133']]


## CREATING TRAINING AND TESTING DATASET
x_train,x_test, y_train, y_test = train_test_split(x,y,test_size=0.4) ## training dataset is split into training & testing samples

## Qn: Why does train_test_split function convert numpy arrays into pandas series???
x1_train,x1_test, y1_train, y1_test = train_test_split(x1_iv,y1,test_size=0.4) ## training dataset is split into training & testing samples


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x1_train = sc.fit_transform(x1_train)
x1_test = sc.transform(x1_test)


np.sum(y1,axis=0) ## 1 = 20,098; 0 = 179,902 10% target rate
np.sum(y1_train,axis=0) ## 12,047; 0 = 107,953 10% target rate
np.sum(y1_test,axis=0) ## 8,051; 0 = 71,949 10% target rate

print('training predictors:',x1_train.shape)
print('testing predictors:',x1_test.shape)
print('training y:',y1_train.shape)
print('testing y:',y1_test.shape)


########################################################################################################################
############################################## TRAINING THE MODEL ######################################################
########################################################################################################################

## RANDOM FOREST
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100)

clf = clf.fit(x1_train,y1_train)

## LIGHTGBM CLASSIFIER - Version 1
import lightgbm as lgb
from lightgbm import LGBMClassifier

lgbm = LGBMClassifier(objective='binary', random_state=5)
lgbmfit = lgbm.fit(x1_train, y1_train)

## LIGHTGBM CLASSIFIER - Version 2

## We need to convert our training data into LightGBM dataset format(this is mandatory for LightGBM training)
d_train = lgb.Dataset(x1_train, label=y1_train)

params = {}
params['learning_rate'] = 0.003
params['boosting_type'] = 'gbdt'
params['objective'] = 'binary'
params['metric'] = 'binary_logloss'
params['sub_feature'] = 0.5
params['num_leaves'] = 10
params['min_data'] = 50
params['max_depth'] = 10
lgbmfit1 = lgb.train(params, d_train, 500)

## LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logregfit = logreg.fit(x1_train, y1_train)


## SCORING THE MODEL ON 40% TEST DATASET
##y1_pred = clf.predict(x1_test) ##np.array
##y1_pred = lgbmfit.predict(x1_test) ##np.array
##y1_pred = logregfit.predict(x1_test) ##np.array
y1_pred = lgbmfit1.predict(x1_test) ##np.array ## Light GBM spits out probabilities instead of 0 or 1

np.savetxt("E:/Kaggle/Santander Customer Transaction Prediction/delete.csv", y1_pred, delimiter=",") ##Dump a NumPy array into a csv file

    #convert into binary values
    for i in range(0,79999):
        if y1_pred[i]>=.5:       # setting threshold to .05
            y1_pred[i]=1
        else:  
                y1_pred[i]=0

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