#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

import pandas as pd

#read the data
df= pd.read_csv("/Users/davidigbokwe/Documents/David's Files/MISM 6212/Project/Submissions/parkinsons.data")

#examine code
df.info() #data is not missing any values
df.describe()

##Get the features and labels
#features
features=df.drop(["status","name"],axis=1)

#labels is status column depicting Parkinson's or not
labels=df["status"]


####### EDA #######

#plot bar chat 
import seaborn as sns
import matplotlib.pyplot as plt
sns.countplot(df["status"]) 
plt.title("Healthy (0) vs Parkinson's (1)")
plt.show()

#lets look at count of labels
count = labels[labels==1].shape[0], labels[labels==0].shape[0]
print("count of 1, count of 0:",count)
"""count of 1, count of 0: (147, 48)"""

#plot heatmap to check correlation
plt.figure(figsize=(20,10))
sns.heatmap(df.corr(),annot=True,fmt=".2f",linewidths="2",cmap="Blues")
plt.show()

#plot highly correlated variables with the target: status
#Choosing top 5 correlations MDVP:Fo(Hz) (-.38),MDVP:Flo(Hz)(-.38),PPE(.53),spread2 (.45),spread1(.56)
plt.figure(figsize = (15,10))
sns.pairplot(df, vars=["MDVP:Fo(Hz)","MDVP:Flo(Hz)","PPE","spread2","spread1"],hue="status",palette="Dark2")
plt.show()



##################


#minmaxscaler 
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()

#split x and y
x=scaler.fit_transform(features)
y=labels

#train test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)

#get model
#conda install -c conda-forge xgboost
from xgboost import XGBClassifier
xgb=XGBClassifier()

#train
xgb.fit(x_train,y_train)


#Calculate the accuracy
y_pred=xgb.predict(x_test)

##evaluate the model now
#build confusion matrix
from sklearn.metrics import confusion_matrix, accuracy_score,recall_score,precision_score, f1_score
cmat = pd.DataFrame(confusion_matrix(y_test,y_pred, labels=[0,1]), index=["Actual:0", "Actual: 1"],columns=["pred:0","pred:1"])

print(cmat)

print("Accuracy is:", accuracy_score(y_test,y_pred))
print("Recall is:", recall_score(y_test,y_pred))
print("Precision is:", precision_score(y_test,y_pred)) 
print("F1 is:", f1_score(y_test,y_pred)) 
"""           pred:0  pred:1
Actual:0       13       0
Actual: 1       2      44
Accuracy is: 0.9661016949152542
Recall is: 0.9565217391304348
Precision is: 1.0
F1 is: 0.9777777777777777"""

######## check parameters #########

from pprint import pprint
# Look at parameters used by our current forest
print('Parameters currently in use:\n')
pprint(xgb.get_params())

"""Parameters currently in use:

{'base_score': 0.5,
 'booster': 'gbtree',
 'colsample_bylevel': 1,
 'colsample_bynode': 1,
 'colsample_bytree': 1,
 'gamma': 0,
 'gpu_id': -1,
 'importance_type': 'gain',
 'interaction_constraints': '',
 'learning_rate': 0.300000012,
 'max_delta_step': 0,
 'max_depth': 6,
 'min_child_weight': 1,
 'missing': nan,
 'monotone_constraints': '()',
 'n_estimators': 100,
 'n_jobs': 0,
 'num_parallel_tree': 1,
 'objective': 'binary:logistic',
 'random_state': 0,
 'reg_alpha': 0,
 'reg_lambda': 1,
 'scale_pos_weight': 1,
 'subsample': 1,
 'tree_method': 'exact',
 'validate_parameters': 1,
 'verbosity': None}"""



######## feature importance ########

import matplotlib.pyplot as plt

#Get variable importance
importance = xgb.feature_importances_

# plot feature importance
plt.figure(figsize = (10,6))
plt.bar([x for x in range(len(importance))], importance)
plt.show()

# summarize feature importance
for i,v in enumerate(importance):
	print("Feature: %0d, Score: %.5f" % (i,v))

print(sum(importance)) 
#1.0000000386498868



######## gridsearch ########

#lets imporve the model with gridsearch
param_grid = {'learning_rate': [0.03, 0.05, 0.07, 0.1], 'max_depth': [3, 5, 7, 9], 
              'objective': ['reg:logistic'], 'min_child_weight': [4, 5, 6], 'n_estimators': [100, 500, 1000, 2000]}

from sklearn.model_selection import GridSearchCV
grid=GridSearchCV(xgb,param_grid,verbose=3,scoring="f1", cv=10)


#train 
grid.fit(x_train,y_train)

## best parameters
grid.best_params_

#use parameters
xgb = XGBClassifier(learning_rate = 0.07, max_depth = 3, min_child_weight = 4, n_estimators = 100, objective = "reg:logistic")

#fit model
xgb.fit(x_test, y_test)


y_pred=xgb.predict(x_test)


#evaluate in matrix
cmat_grid = pd.DataFrame(confusion_matrix(y_test,y_pred, labels=[0,1]), index=["Actual:0", "Actual: 1"],columns=["pred:0","pred:1"])

print(cmat)

print("Accuracy is:", accuracy_score(y_test,y_pred))
print("Recall is:", recall_score(y_test,y_pred))
print("Precision is:", precision_score(y_test,y_pred)) 
print("F1 is:", f1_score(y_test,y_pred)) 
"""           pred:0  pred:1
Actual:0       12       1
Actual: 1       5      41
Accuracy is: 0.8983050847457628
Recall is: 0.8913043478260869
Precision is: 0.9761904761904762
F1 is: 0.9318181818181818"""


######## RF ########

#Random Forest
from sklearn.ensemble import RandomForestClassifier
rfc= RandomForestClassifier()
rfc.fit(x_train,y_train)


#Testing
y_pred = rfc.predict(x_test)
cmat_rf = pd.DataFrame(confusion_matrix(y_test,y_pred, labels=[0,1]), index=["Actual:0", "Actual: 1"],columns=["pred:0","pred:1"])

print(cmat_rf)
print("Accuracy is:", accuracy_score(y_test,y_pred))
print("Recall is:", recall_score(y_test,y_pred))
print("Precision is:", precision_score(y_test,y_pred))
print("F1 score is:" , f1_score(y_test, y_pred))
"""           pred:0  pred:1
Actual:0       11       2
Actual: 1       1      45
Accuracy is: 0.9491525423728814
Recall is: 0.9782608695652174
Precision is: 0.9574468085106383
F1 score is: 0.967741935483871"""



######## gridsearch ########


#grid search to optimize the forest
parameter_grid={"max_depth" : range(2,20), "min_samples_split": range(2,6), "n_estimators":[100, 300, 500, 1000]}

from sklearn.model_selection import GridSearchCV
grid= GridSearchCV(rfc, parameter_grid, verbose=3, scoring="f1",cv = 5)

#train
grid.fit(x_train, y_train)

#best params
grid.best_params_

#using parameters
rfc= RandomForestClassifier(max_depth=11, min_samples_split=5, n_estimators=100)
rfc.fit(x_train, y_train)


#Training
y_pred_train = rfc.predict(x_train)
print("Training F1 score is:" , f1_score(y_train, y_pred_train))

#Testing
y_pred = rfc.predict(x_test)
cmat_rf = pd.DataFrame(confusion_matrix(y_test,y_pred, labels=[0,1]), index=["Actual:0", "Actual: 1"],columns=["pred:0","pred:1"])

print(cmat_rf)
print("Accuracy is:", accuracy_score(y_test,y_pred))
print("Recall is:", recall_score(y_test,y_pred))
print("Precision is:", precision_score(y_test,y_pred))
print("F1 score is:" , f1_score(y_test, y_pred))
"""           pred:0  pred:1
Actual:0       10       3
Actual: 1       2      44
Accuracy is: 0.9152542372881356
Recall is: 0.9565217391304348
Precision is: 0.9361702127659575
F1 score is: 0.9462365591397849"""


######## logistic regression ########

#features: dropped highly correlated variables based on heatmap, cutt off 0.65 on heatmap
features=df[["MDVP:Fo(Hz)","MDVP:Fhi(Hz)","MDVP:Flo(Hz)","RPDE","DFA","spread2","D2"]]

#labels is status column depicting Parkinson's or not
labels=df["status"]

#minmaxscaler 
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()

#split x and y
x=scaler.fit_transform(features)
y=labels

#train test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)

# lets do the logistic regression
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression(solver="liblinear")
logmodel.fit(x_train,y_train)


## make predictions
y_pred = logmodel.predict(x_test)

#get the probabilities
y_pred_proba = logmodel.predict_proba(x_test)

##get the coeff and intercept
logmodel.coef_ #-1.25210172, -0.60309446, -0.93269892,  0.40827134,  0.80577288, 1.79005749,  1.69643565
logmodel.intercept_ #-0.22300314

#evaluate the model
from sklearn.metrics import f1_score
f1_score(y_test,y_pred)


from sklearn.metrics import confusion_matrix, accuracy_score,recall_score,precision_score
cmat_log = pd.DataFrame(confusion_matrix(y_test,y_pred, labels=[0,1]), index=["Actual:0", "Actual: 1"],columns=["pred:0","pred:1"])

print(cmat_log)
print("Accuracy is:", accuracy_score(y_test,y_pred))
print("Recall is:", recall_score(y_test,y_pred))
print("Precision is:", precision_score(y_test,y_pred)) 
print("F1 is:", f1_score(y_test,y_pred)) 

"""            pred:0  pred:1
Actual:0        9       4
Actual: 1       0      46
Accuracy is: 0.9322033898305084
Recall is: 1.0
Precision is: 0.92
F1 is: 0.9583333333333334"""

"""Parameters currently in use:

{'C': 1.0,
 'class_weight': None,
 'dual': False,
 'fit_intercept': True,
 'intercept_scaling': 1,
 'l1_ratio': None,
 'max_iter': 100,
 'multi_class': 'auto',
 'n_jobs': None,
 'penalty': 'l2',
 'random_state': None,
 'solver': 'liblinear',
 'tol': 0.0001,
 'verbose': 0,
 'warm_start': False}"""

######## feature importance ########


# get importance
import matplotlib.pyplot as plt
importance = logmodel.coef_[0]
# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
plt.bar([x for x in range(len(importance))], importance)
plt.show()

"""spread2 and D2 most important features"""

######## gridsearch ########


#paramters
param_grid={'C':[0.01,0.1,1,10,100, 1000], 'dual':[True, False], "penalty":['none', 'l1', 'l2', 'elasticnet'],"solver":["lbfgs"], "max_iter":[1000,3000, 5000,7000,10000]}

#verbose will show the output
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(logmodel,param_grid,verbose=3,scoring="f1")

#fit the grid
grid.fit(x_train,y_train)

## best parameters
grid.best_params_

### see if model improves with these parameters
logmodel = LogisticRegression(C=10,penalty="l2",solver="lbfgs", max_iter=1000, dual=False)
logmodel.fit(x_train,y_train)


#make predictions
y_pred = logmodel.predict(x_test)


from sklearn.metrics import confusion_matrix, accuracy_score,recall_score,precision_score
cmat_log = pd.DataFrame(confusion_matrix(y_test,y_pred, labels=[0,1]), index=["Actual:0", "Actual: 1"],columns=["pred:0","pred:1"])

print(cmat_log)
print("Accuracy is:", accuracy_score(y_test,y_pred))
print("Recall is:", recall_score(y_test,y_pred))
print("Precision is:", precision_score(y_test,y_pred)) 
print("F1 is:", f1_score(y_test,y_pred)) 
"""            pred:0  pred:1
Actual:0        9       4
Actual: 1       2      44
Accuracy is: 0.8983050847457628
Recall is: 0.9565217391304348
Precision is: 0.9166666666666666
F1 is: 0.9361702127659574"""



