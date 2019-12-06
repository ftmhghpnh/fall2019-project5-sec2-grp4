# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 10:00:02 2019

@author: tb2579
"""
import pandas as pd
import numpy as np
import pyreadr

def gbm_fn(X, y, cv):
    from sklearn.ensemble import GradientBoostingClassifier
    if(cv):
        from sklearn.model_selection import GridSearchCV
        params = {'n_estimators':[100, 250, 500, 750, 1000]}
        gbm = GradientBoostingClassifier(max_depth=1)
        clf = GridSearchCV(gbm, params, scoring='accuracy', verbose=2, n_jobs=-1)
        clf.fit(X,y)
        n_estimators = clf.best_params_['n_estimators']
        gbm = GradientBoostingClassifier(max_depth=1, n_estimators = n_estimators)
        params = {'learning_rate':[0.1, 0.05, 0.01]}
        clf = GridSearchCV(gbm, params, scoring='accuracy', verbose=2, n_jobs=-1, refit=True)
        clf.fit(X,y)
        time = clf.refit_time_
    else:
        import time
        start = time.time()
        clf = GradientBoostingClassifier(max_depth=1, learning_rate=0.05, )
        clf.fit(X,y)
        end = time.time()
        time = end-start
    
    return time, clf # model

# Baseline test
def test_clf(test, y_true, baseline_dir):
#     test = dat_test.iloc[:,:-1]
#     y_true = dat_test.iloc[:,-1]
    
    from sklearn.externals import joblib
    from sklearn.metrics import accuracy_score
    filename = baseline_dir
    loaded_model = joblib.load(filename)
    preds = loaded_model.predict(test)
    preds = pd.Series([int(p) for p in preds]) ######
    #print(preds)
    #print(y_true)
    return sum(y_true==preds)/len(preds)#accuracy_score(y_true, preds)

train_df=pandas.read_csv('C:/Users/tb2579/Documents/MA Statistics/Applied Data Analysis/Project 5/TrainFeatures.csv')
test_df=pandas.read_csv('C:/Users/tb2579/Documents/MA Statistics/Applied Data Analysis/Project 5/TestFeatures.csv')


baseline_dir = 'baseline_train_main.sav' #'baseline_train_alldata.sav' #'baseline_train_alldata.sav'#'baseline_train_main.sav' 
tm_train_baseline, baseline = gbm_fn(train_df.iloc[:,:-1], train_df.iloc[:,-1], baseline_cv)
from sklearn.externals import joblib
joblib.dump(baseline, baseline_dir) # save the model to disk

# test
start= time.time()
baseline_preds = test_clf(test_df, baseline_dir = baseline_dir) 
end = time.time()
tm_test_baseline = end-start
    
    
info_test['Baseline'] = baseline_preds

# print('training feature extraction took: {}'.format(tm_feature_train_baseline))
print('testing feature extraction took: {}'.format(tm_feature_test_baseline))
# print('model training took: {}'.format(tm_train_baseline))
print('model testing took: {}'.format(tm_test_baseline))
# print('model accuracy: {}'.format(baseline_acc))