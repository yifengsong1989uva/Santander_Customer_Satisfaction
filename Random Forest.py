##################################################################
###Random Forest Model for Santander Customer Satisfaction dataset
##################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score



#################################################
### Tune parameters using the Entire training Set
#################################################
satisfaction = pd.read_csv("train.csv", sep = ",")
df = pd.DataFrame(satisfaction)
df["TARGET"]=df["TARGET"].astype(str)

#Not use ID, training/test splitting     
X = df.iloc[:,1:370]
y = df.iloc[:,370]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=0)

#find the best guess of max_depth for the trees that improve the test set accuracy
accuracy=[]
for i in range(2,26):
    rf=RandomForestClassifier(n_estimators=200, criterion="entropy", max_depth=i,random_state=0)
    rf.fit(X_train, y_train)
    y_pred=rf.predict_proba(X_test)[:,1]
    accuracy.append(roc_auc_score(y_test.astype("int"),y_pred))
plt.plot(range(2,26),accuracy)
plt.xlabel("Maximum Depth")
plt.ylabel("AUC")
#max_depth=21 is the best so far

#find the best guess of number of trees that will give a high test set accuracy
accuracy=[]
for i in range(20,360,20):
    rf=RandomForestClassifier(n_estimators=i, criterion="entropy", max_depth=21,random_state=0)
    rf.fit(X_train, y_train)
    y_pred=rf.predict_proba(X_test)[:,1]
    accuracy.append(roc_auc_score(y_test.astype("int"),y_pred))
plt.plot(range(20,360,20),accuracy)
plt.xlabel("Number of trees")
plt.ylabel("AUC")
#n_estimators=320 is selected, the auc is 0.8316

#try to change of max_features parameters and see if a non-default value gives a
#better performance
accuracy=[]
for i in [9,14,19,24,29,39,49,69,99,129]:
    rf=RandomForestClassifier(n_estimators=320, criterion="entropy", max_depth=21,random_state=0,max_features=i)
    rf.fit(X_train, y_train)
    y_pred=rf.predict_proba(X_test)[:,1]
    accuracy.append(roc_auc_score(y_test.astype("int"),y_pred))
plt.plot([9,14,19,24,29,39,49,69,99,129],accuracy)
plt.xlabel("Max_Features")
plt.ylabel("AUC")
#max_features=19 indeed the best auc

#There, max_features=19, n_estimators=320, max_depth=21 are finally selected



############################################################
###Make predictions using the selected training and test set
############################################################
train_data = pd.read_csv("TrainData.csv", sep = ",")
df = pd.DataFrame(train_data)
df["TARGET"]=df["TARGET"].astype(str)
test_data = pd.read_csv("TestData.csv",sep=",")
df_test=pd.DataFrame(test_data)
df_test["TARGET"]=df_test["TARGET"].astype(str)
X1 = df.iloc[:,2:371]
y1 = df.iloc[:,371]
X2 = df_test.iloc[:,2:371]
y2 = df_test.iloc[:,371]

rf=RandomForestClassifier(n_estimators=320, criterion="entropy", max_depth=21,random_state=0)
rf.fit(X1, y1)
y_pred=rf.predict_proba(X2)[:,1]
roc_auc_score(y2.astype("int"),y_pred)
#test set auc score: 0.822

# visualize the feature importances (with the 10 most important features)
plt.bar(left=np.arange(10), height=np.array(sorted(rf.feature_importances_))[::-1][:10], width=0.7)
plt.xticks(np.arange(10) + 0.7/2, X1.dtypes.index[np.array(rf.feature_importances_).argsort()[::-1][:10]],rotation=90)

#save the predicted probabilities for the test set into a csv file
np.savetxt("rf_pred.csv", y_pred, delimiter=",")
