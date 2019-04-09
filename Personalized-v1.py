#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 10:26:21 2019

@author: alireza
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import svm

training_variants_df=pd.read_csv("../input/training_variants")

training_text_df = pd.read_csv("../input/training_text",sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])

training_merge_df = training_variants_df.merge(training_text_df,left_on="ID",right_on="ID")

testing_variants_df = pd.read_csv("../input/test_variants")
                                  
                               
testing_text_df = pd.read_csv("../input/test_text", sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])

testing_merge_df = testing_variants_df.merge(testing_text_df,left_on="ID",right_on="ID")

training_merge_df= training_merge_df.iloc[0:1000]
testing_merge_df= testing_merge_df.iloc[0:1200]

train,test=train_test_split(training_merge_df, test_size=0.2)
np.random.seed(0)

X_train = train['Text'].values
X_test = test['Text'].values
y_train = train['Class'].values
y_test = test['Class'].values

text_clf = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()), ('clf', svm.LinearSVC())])
text_clf = text_clf.fit(X_train,y_train)
#
y_test_predicted = text_clf.predict(X_test)
np.mean(y_test_predicted == y_test)
#
X_test_final = testing_merge_df['Text'].values
predicted_class = text_clf.predict(X_test_final)
testing_merge_df['predicted_class'] = predicted_class
#
onehot = pd.get_dummies(testing_merge_df['predicted_class'])
testing_merge_df = testing_merge_df.join(onehot)
#
submission_df = testing_merge_df[["ID",1,2,3,4,5,6,7,8,9]]
submission_df.columns = ['ID', 'class1','class2','class3','class4','class5','class6','class7','class8','class9']
#
submission_df.to_csv('submission.csv', index=False)



