# coding: utf-8

import pandas as pd
import scipy as sc
import numpy as np

df = pd.read_csv("train.csv",header=0)
df.head()
df['TARGET'].describe()


from sklearn.cross_validation import train_test_split
train, test = train_test_split(df, test_size=0.30, random_state=1)
train.to_csv("TrainData.csv", sep=',')
test.to_csv("TestData.csv", sep=',')


train = pd.read_csv("TrainData.csv",header=0)
train.shape

test = pd.read_csv("TestData.csv",header=0)
test.shape

test.drop(test.columns[[0, 1]], axis=1, inplace=True)
test.shape
ytest = test.iloc[:,369].values
xtest = test.iloc[:,0:369].values


# explore the data
train.describe()
train.drop(train.columns[[0, 1]], axis=1, inplace=True)
train.head()

# creat X and y 
y = train.iloc[:,369].values
x = train.iloc[:,0:369].values


import scipy.stats
# look over the y value
scipy.stats.describe(y)

# seprate the data to train and test
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=1)

scipy.stats.describe(y_test)


# KNN model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
neighbors = np.linspace(2,50,48).astype(int)
KNN = KNeighborsClassifier(n_jobs=-1)
auc0 = list()
auc1 = list()

for K in neighbors:
    KNN.n_neighbors = K
    KNN.fit(X_train,y_train)
    y_pred = KNN.predict_proba(X_test)
    auc_0 = roc_auc_score(y_test,y_pred[:,0])
    auc_1 = roc_auc_score(y_test,y_pred[:,1])
    auc0.append(auc_0)
    auc1.append(auc_1)
    print('K is:{0}  AUC 0 is: {1} AUC 1 {2} '.format(K,auc_0,auc_1))
## The best model for KNN is K=41

from sklearn.neighbors import KNeighborsClassifier
KNN_clf = KNeighborsClassifier(n_neighbors = 41,n_jobs=-1)
KNN_clf.fit(X_train,y_train)
KNN_pred = KNN_clf.predict_proba(X_test)
KNN_AUC = roc_auc_score(y_test,KNN_pred[:,1])
KNN_AUC


#Deep Neural Networks model
import skflow
from sklearn import datasets, metrics
from sklearn.metrics import roc_auc_score

units = np.linspace(5,50,10).astype(int)*10
DNNauc0 = list()
DNNauc1 = list()

for u in units:
    classifier = skflow.TensorFlowDNNClassifier(hidden_units=[100, u, 100], n_classes=2)
    classifier.fit(X_train,y_train)
    y_pred = classifier.predict_proba(X_test)
    auc_0 = roc_auc_score(y_test,y_pred[:,0])
    auc_1 = roc_auc_score(y_test,y_pred[:,1])
    DNNauc0.append(auc_0)
    DNNauc1.append(auc_1)
    print('hidden unite is:{0}  AUC 0 is: {1} AUC 1 {2} '.format(u,auc_0,auc_1))

DNN_clf = skflow.TensorFlowDNNClassifier(hidden_units=[100, 250, 100], n_classes=2)
DNN_clf.fit(X_train,y_train)
DNN_pred = DNN_clf.predict_proba(X_test)
DNN_auc = roc_auc_score(y_test,DNN_pred[:,1])
DNN_auc

#  C is Inverse of regularization strength; must be a positive float. Like in support 
# vector machines, smaller values specify stronger regularization.



from sklearn import linear_model

logistic_classifier = linear_model.LogisticRegression(penalty='l1',tol = 0.001)
C = np.linspace(0,1,20)
C=C[1:]
Lg_auc0 = list()
lg_auc1 = list()

for c in C:
    logistic_classifier.C = c
    logistic_classifier.fit(X_train,y_train)
    pred_prob = logistic_classifier.predict_proba(X_test)
    lgauc_0 = roc_auc_score(y_test,pred_prob[:,0])
    lgauc_1 = roc_auc_score(y_test,pred_prob[:,1])
    Lg_auc0.append(lgauc_0)
    lg_auc1.append(lgauc_1)
    print('C is:{0}  AUC 0 is: {1} AUC 1 {2} '.format(c,lgauc_0,lgauc_1))

logistic_clf = linear_model.LogisticRegression(C = 0.7894736842105263,penalty='l1',                                                      tol = 0.001)
logistic_clf.fit(X_train,y_train)
LG_pred = logistic_clf.predict_proba(X_test)
LG_AUC = roc_auc_score(y_test,LG_pred[:,1])
LG_AUC

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
fpr[0], tpr[0], _ = roc_curve(y_test, KNN_pred[:,1])
roc_auc[0] = roc_auc_score(y_test,KNN_pred[:,1])
fpr[1], tpr[1], _ = roc_curve(y_test,DNN_pred[:,1])
roc_auc[1] = roc_auc_score(y_test,DNN_pred[:,1])
fpr[2], tpr[2], _ = roc_curve(y_test,LG_pred[:,1])
roc_auc[2] = roc_auc_score(y_test,LG_pred[:,1])


plt.figure()
#plt.plot(fpr[0], tpr[0], label='ROC curve for KNN (area = %0.2f)' % roc_auc[0])
#plt.plot(fpr[1], tpr[1], label='ROC curve for DNN (area = %0.2f)' % roc_auc[1])
plt.plot(fpr[2], tpr[2], label='ROC curve for \n LogisticRegression (area = %0.2f)' % roc_auc[2])
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic ')
plt.legend(loc="lower right")
plt.show()


# Now for the test data
KNN_pred_test = KNN_clf.predict_proba(xtest)
DNN_pred_test = DNN_clf.predict_proba(xtest)
LG_pred_test = logistic_clf.predict_proba(xtest)

fpr_test = dict()
tpr_test = dict()
roc_auc_test = dict()
fpr_test[0], tpr_test[0], _ = roc_curve(ytest, KNN_pred_test[:,1])
roc_auc_test[0] = roc_auc_score(ytest,KNN_pred_test[:,1])
fpr_test[1], tpr_test[1], _ = roc_curve(ytest,DNN_pred_test[:,1])
roc_auc_test[1] = roc_auc_score(ytest,DNN_pred_test[:,1])
fpr_test[2], tpr_test[2], _ = roc_curve(ytest,LG_pred_test[:,1])
roc_auc_test[2] = roc_auc_score(ytest,LG_pred_test[:,1])

plt.figure()
plt.plot(fpr_test[0], tpr_test[0], label='ROC curve for KNN (area = %0.2f)' % roc_auc_test[0])
plt.plot(fpr_test[1], tpr_test[1], label='ROC curve for DNN (area = %0.2f)' % roc_auc_test[1])
plt.plot(fpr_test[2], tpr_test[2], label='ROC curve for \n LogisticRegression (area = %0.2f)' % roc_auc_test[2])
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic for test data ')
plt.legend(loc="lower right")
plt.show()