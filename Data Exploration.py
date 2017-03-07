import pandas as pd
import numpy as np
import seaborn as sns
sns.set(style="white",color_codes=True)
from matplotlib import pyplot as plt

import re
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
pd.set_option('display.precision', 5)


import os
os.chdir("~/SantanderCustSatisf")
train = pd.read_csv("data/train.csv")
train.describe(include="all")
train.info()
train.head()
train.shape
df = pd.DataFrame(train.TARGET.value_counts())
df["Percent"] = 100*df['TARGET']/train.shape[0]
df

train.var3.value_counts()[:10]

train[train.var3 == -999999].shape
train[train == 9999999999].shape

((train == 9999999999).sum(axis=1)==1).any()


df = pd.DataFrame(train.TARGET.value_counts())
df['Percentage'] = 100*df['TARGET']/train.shape[0]
df



training = pd.read_csv("data/train.csv", index_col=0)
test = pd.read_csv("data/test.csv", index_col=0)

print(training.shape)
print(test.shape)

X = training.iloc[:,:-1]
y = training.TARGET

y.value_counts() / float(y.size)

X.apply(lambda x:x[x!=0].size).sum() / float(np.prod(training.shape))
test.apply(lambda x:x[x!=0].size).sum() / float(np.prod(test.shape))
X.dtypes.value_counts()
X.columns

name_component = pd.Series(sum([re.sub("\d+", "", s).split("_") for s in X.columns], []))
name_component.replace("", "_0", inplace=True)
name_component.value_counts()

nuniques_train = X.apply(lambda x:x.nunique())
nuniques_test = test.apply(lambda x:x.nunique())


no_variation_train = nuniques_train[nuniques_train==1].index
no_variation_test = nuniques_train[nuniques_test==1].index

print(no_variation_train.size, no_variation_test.size)

print('\nTrain[no variation in test]\n#unique cnt\n',nuniques_train[no_variation_test].value_counts())
print('\nTest[no variation in train]\n#unique cnt\n', nuniques_test[no_variation_train].value_counts())

X, test = [df.drop(no_variation_train, axis=1) for df in [X, test]]
nuniques_train, nuniques_test = [s.drop(no_variation_train) for s in [nuniques_train, nuniques_test]]

ax = nuniques_train[nuniques_train<100].hist(bins=100, figsize=(10, 7))
ax.set_xlabel("Unique Feature")
ax.set_title("Histogram of Unique Features")
plt.show()

nuniques_train[nuniques_train<100].size
nuniques_train[nuniques_train>=100].size
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X, y)

feat_imp = pd.Series(rf.feature_importances_, index=X.columns)
feat_imp.sort_values(inplace=True)
ax = feat_imp.tail(20).plot(kind='barh', figsize=(10,7), title='Feature importance')

def intersect(a, b):
    return list(set(a) & set(b))


def get_features(train, test):
    trainval = list(train.columns.values)
    testval = list(test.columns.values)
    output = intersect(trainval, testval)
    output.remove('ID')
    return output


def prepare_dataset():
    train = pd.read_csv("../input/train.csv")
    test = pd.read_csv("../input/test.csv")
    features = train.columns.values
    
    norm_f = []
    for f in features:
        u = train[f].unique()
        if len(u) != 1:
            norm_f.append(f)


    remove = []
    for i in range(len(norm_f)):
        v1 = train[norm_f[i]].values
        for j in range(i+1, len(norm_f)):
            v2 = train[norm_f[j]].values
            if np.array_equal(v1, v2):
                remove.append(norm_f[j])
    
    for r in remove:
        norm_f.remove(r)

    train = train[norm_f]
    norm_f.remove('TARGET')
    test = test[norm_f]
    features = get_features(train, test)
    return train, test, features


def find_min_max_features(df, f):
    return df[f].min(), df[f].max()


def analayze_data(train, test):
    print('Length of train: ', len(train.index))
    train_zero = train[train['TARGET'] == 0]
    print('Length of train [TARGET = 0]: ', len(train_zero.index))
    train_one = train[train['TARGET'] == 1]
    print('Length of train [TARGET = 1]: ', len(train_one.index))
    # train_one.to_csv("debug.csv", index=False)
    one_range = dict()
    for f in train.columns:
        mn0, mx0 = find_min_max_features(train_zero, f)
        mn1, mx1 = find_min_max_features(train_one, f)
        mnt = 'N/A'
        mxt = 'N/A'
        if f in test.columns:
            mnt, mxt = find_min_max_features(test, f)
        one_range[f] = (mn1, mx1)
        if mn0 != mn1 or mn1 != mnt or mx0 != mx1 or mx1 != mxt:
            print("\nFeature {}".format(f))
            print("Range target=0  ({} - {})".format(mn0, mx0))
            print("Range target=1  ({} - {})".format(mn1, mx1))
            print("Range in test   ({} - {})".format(mnt, mxt))


train, test, features = prepare_dataset()
analayze_data(train, test)

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from sklearn import cross_validation
import xgboost as xgb
from sklearn.metrics import roc_auc_score
import os
os.chdir("/mnt/Codebase/Kaggle/SantanderCustSatisf")

training = pd.read_csv("data/train.csv", index_col=0)
test = pd.read_csv("data/test.csv", index_col=0)

print(training.shape)
print(test.shape)

training = training.replace(-999999,2)
X = training.iloc[:,:-1]
y = training.TARGET


X['n0'] = (X == 0).sum(axis=1)


from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

X_normalized = normalize(X, axis=0)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_normalized)
X['PCA1'] = X_pca[:,0]
X['PCA2'] = X_pca[:,1]

from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_classif,chi2
from sklearn.preprocessing import Binarizer, scale
p = 86 
X_bin = Binarizer().fit_transform(scale(X))
selectChi2 = SelectPercentile(chi2, percentile=p).fit(X_bin, y)
selectF_classif = SelectPercentile(f_classif, percentile=p).fit(X, y)

chi2_selected = selectChi2.get_support()
chi2_selected_features = [f for i,f in enumerate(X.columns) if chi2_selected[i]]
print('Chi2 selected {} features {}.'.format(chi2_selected.sum(),
   chi2_selected_features))
f_classif_selected = selectF_classif.get_support()
f_classif_selected_features = [ f for i,f in enumerate(X.columns) if f_classif_selected[i]]
print('F_classif selected {} features {}.'.format(f_classif_selected.sum(),
   f_classif_selected_features))
selected = chi2_selected & f_classif_selected
print('Chi2 & F_classif selected {} features'.format(selected.sum()))
features = [ f for f,s in zip(X.columns, selected) if s]
print (features)

X_sel = X[features]

X_train, X_test, y_train, y_test = \
  cross_validation.train_test_split(X_sel, y, random_state=1301, stratify=y, test_size=0.4)

ratio = float(np.sum(y == 1)) / np.sum(y==0)


clf = xgb.XGBClassifier(missing=9999999999,
                max_depth = 5,
                n_estimators=1000,
                learning_rate=0.1, 
                nthread=4,
                subsample=1.0,
                colsample_bytree=0.5,
                min_child_weight = 3,
                scale_pos_weight = ratio,
                reg_alpha=0.03,
                seed=1301)
                
clf.fit(X_train, y_train, early_stopping_rounds=50, eval_metric="auc",
        eval_set=[(X_train, y_train), (X_test, y_test)])
        
print('Overall AUC:', roc_auc_score(y, clf.predict_proba(X_sel, ntree_limit=clf.best_iteration)[:,1]))

test['n0'] = (test == 0).sum(axis=1)

test_normalized = normalize(test, axis=0)
pca = PCA(n_components=2)
test_pca = pca.fit_transform(test_normalized)
test['PCA1'] = test_pca[:,0]
test['PCA2'] = test_pca[:,1]
sel_test = test[features]    
y_pred = clf.predict_proba(sel_test, ntree_limit=clf.best_iteration)

submission = pd.DataFrame({"ID":test.index, "TARGET":y_pred[:,1]})
submission.to_csv("submission.csv", index=False)

mapFeat = dict(zip(["f"+str(i) for i in range(len(features))],features))
ts = pd.Series(clf.booster().get_fscore())

ts.sort_values()[-15:].plot(kind="barh", title=("features importance"))

featp = ts.sort_values()[-15:].plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 10))
plt.title('XGBoost Feature Importance')
fig_featp = featp.get_figure()
fig_featp.savefig('feature_importance_xgb.png', bbox_inches='tight', pad_inches=1)
