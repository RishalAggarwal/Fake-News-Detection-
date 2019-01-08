# -*- coding: utf-8 -*-

import DataPrep
import numpy as np
import pandas as pd
import pickle
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import  LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn import svm
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, f1_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import FeatureSelection
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

#punctuation counter
countV = CountVectorizer(token_pattern=r"!|\?|\"|\'|,|\.|;|:|-|\(|\)")
#tfidf vector
tfidf_ngram = TfidfVectorizer(stop_words='english',ngram_range=(1,2),use_idf=True,smooth_idf=True)
#selector based on mutual information
selector=SelectKBest(score_func=mutual_info_classif,k=20000)
#grammar syntax extractor
gram=FeatureSelection.GrammarTransformer(spacy.load('en'))
#dictionary vectorizer
dictvec=DictVectorizer()
#tfidf transformer for pcfgs
tfidftrans=TfidfTransformer()
#scaler for readability
min_max_scaler = preprocessing.MinMaxScaler()

count_punct_train=countV.fit_transform(DataPrep.train_news['Statement'].values)
tfidf_train=tfidf_ngram.fit_transform(DataPrep.train_news['Statement'].values)
tfidf_train=selector.fit_transform(tfidf_train,DataPrep.train_news['Label'].values)
gram_syntax=gram.countgrammar(DataPrep.train_news['Statement'].values)
gram_syntax= dictvec.fit_transform(gram_syntax)
gram_syntax_train= tfidftrans.fit_transform(gram_syntax)
gun_fog_train = pickle.load(open('gun_fog_train.pkl', 'rb'))
gun_fog_train=np.expand_dims(gun_fog_train,0)
flesch_train = pickle.load(open('flesch_train.pkl', 'rb'))
flesch_train=np.expand_dims(flesch_train,0)
readability_train=np.vstack((gun_fog_train,flesch_train))
readability_train=np.swapaxes(readability_train,0,1)
readability_train = min_max_scaler.fit_transform(readability_train)
train_feature_mat=np.hstack((tfidf_train.toarray(),count_punct_train.toarray(),readability_train,gram_syntax_train.toarray()))

count_punct_test=countV.transform(DataPrep.test_news['Statement'].values)
tfidf_test=tfidf_ngram.transform(DataPrep.test_news['Statement'].values)
tfidf_test=selector.transform(tfidf_test)
gram_syntax_test=gram.countgrammar(DataPrep.test_news['Statement'].values)
gram_syntax_test=dictvec.transform(gram_syntax_test)
gram_syntax_test=tfidftrans.transform(gram_syntax_test)
gun_fog_test = pickle.load(open('gun_fog_test.pkl', 'rb'))
gun_fog_test=np.expand_dims(gun_fog_test,0)
flesch_test = pickle.load(open('flesch_test.pkl', 'rb'))
flesch_test=np.expand_dims(flesch_test,0)
readability_test=np.vstack((gun_fog_test,flesch_test))
readability_test=np.swapaxes(readability_test,0,1)
readability_test = min_max_scaler.transform(readability_test)
test_feature_mat=np.hstack((tfidf_test.toarray(),count_punct_test.toarray(),readability_test,gram_syntax_test.toarray()))


count_punct_valid=countV.transform(DataPrep.valid_news['Statement'].values)
tfidf_valid=tfidf_ngram.transform(DataPrep.valid_news['Statement'].values)
tfidf_valid=selector.transform(tfidf_valid)
gram_syntax_valid=gram.countgrammar(DataPrep.valid_news['Statement'].values)
gram_syntax_valid=dictvec.transform(gram_syntax_valid)
gram_syntax_valid=tfidftrans.transform(gram_syntax_valid)
gun_fog_valid = pickle.load(open('gun_fog_valid.pkl', 'rb'))
gun_fog_valid=np.expand_dims(gun_fog_valid,0)
flesch_valid = pickle.load(open('flesch_valid.pkl', 'rb'))
flesch_valid=np.expand_dims(flesch_valid,0)
readability_valid=np.vstack((gun_fog_valid,flesch_valid))
readability_valid=np.swapaxes(readability_valid,0,1)
readability_valid = min_max_scaler.transform(readability_valid)
valid_feature_mat=np.hstack((tfidf_valid.toarray(),count_punct_valid.toarray(),readability_valid[:1284],gram_syntax_valid.toarray()))


nb=MultinomialNB()
nb.fit(train_feature_mat,DataPrep.train_news['Label'])
predicted_nb_test=nb.predict(test_feature_mat)
predicted_nb_train=nb.predict(train_feature_mat)
predicted_nb_valid=nb.predict(valid_feature_mat)

LogR=LogisticRegression(penalty="l2",C=1,solver='liblinear',max_iter=3000)
LogR.fit(train_feature_mat,DataPrep.train_news['Label'])
predicted_LogR_test=LogR.predict(test_feature_mat)
predicted_LogR_train=LogR.predict(train_feature_mat)
predicted_LogR_valid=LogR.predict(valid_feature_mat)

svm=svm.LinearSVC(max_iter=10000)
svm.fit(train_feature_mat,DataPrep.train_news['Label'])
predicted_svm_test=svm.predict(test_feature_mat)
predicted_svm_train=svm.predict(train_feature_mat)
predicted_svm_valid=svm.predict(valid_feature_mat)




print('test')
print('naive bayes')
print(classification_report(DataPrep.test_news['Label'], predicted_nb_test))
print('LogR')
print(classification_report(DataPrep.test_news['Label'], predicted_LogR_test))
print('svm')
print(classification_report(DataPrep.test_news['Label'], predicted_svm_test))

print('valid')
print('naive bayes')
print(classification_report(DataPrep.valid_news['Label'], predicted_nb_valid))
print('LogR')
print(classification_report(DataPrep.valid_news['Label'], predicted_LogR_valid))
print('svm')
print(classification_report(DataPrep.valid_news['Label'], predicted_svm_valid))


min_max_scaler_file='scaler_file.pkl'
dictvec_file='dicvec_model.pkl'
tfidftrans_file='trans_model.pkl'
selector_file='selector_model.pkl'
punct_file='punct_model.pkl'
tfidf_file='tfidf_model.pkl'
model_file = 'final_model.pkl'

#saving files for prediction
pickle.dump(min_max_scaler,open(min_max_scaler_file,'wb'))
pickle.dump(dictvec,open(dictvec_file,'wb'))
pickle.dump(tfidftrans,open(tfidftrans_file,'wb'))
pickle.dump(selector,open(selector_file,'wb'))
pickle.dump(tfidf_ngram,open(tfidf_file,'wb'))
pickle.dump(countV,open(punct_file,'wb'))
pickle.dump((LogR),open(model_file,'wb'))



