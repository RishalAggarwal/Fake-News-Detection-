# -*- coding: utf-8 -*-
import pickle
import tkinter as tk
import tkinter.simpledialog
from tkinter import *
import numpy as np
from sklearn import preprocessing
import sklearn
import FeatureSelection
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
window = tk.Tk()
window.withdraw()
var = tkinter.simpledialog.askstring("File: ","Enter the news")


#retrieving the best model and feature extractors for prediction call
tfidf=pickle.load(open('tfidf_model.pkl','rb'))
punct=pickle.load(open('punct_model.pkl','rb'))
model = pickle.load(open('final_model.pkl', 'rb'))
Scaler=pickle.load(open('scaler_file.pkl','rb'))
dictvec=pickle.load(open('dicvec_model.pkl','rb'))
tfidf_trans=pickle.load(open('trans_model.pkl','rb'))
selector=pickle.load(open('selector_model.pkl','rb'))

tf_idf=tfidf.transform([var])
tf_idf=selector.transform(tf_idf)
punct=punct.transform([var])
gram=FeatureSelection.GrammarTransformer(spacy.load('en'))
gram_syntax=gram.countgrammar([var])
gram_syntax= dictvec.transform(gram_syntax)
gram_syntax_pred= tfidf_trans.transform(gram_syntax)
readability=FeatureSelection.get_readability([var])
readability=np.swapaxes(readability,0,1)
readability_fit=Scaler.transform(readability)
feature_mat=np.hstack((tf_idf.toarray(),punct.toarray(),readability_fit,gram_syntax_pred.toarray()))

#prediction and prediction score
prediction = model.predict(feature_mat)
prob = model.predict_proba(feature_mat)


printer = 'The given statement is '+str(prediction[0])+" and the truth probability score is "+str(prob[0][1])


main = tk.Tk()
ourMessage = printer
messageVar = Message(main, text = ourMessage, width = 550)
messageVar.config(bg='lightgreen')
button = tk.Button(main, text='Stop', width=25, command=main.destroy)
messageVar.pack()
main.mainloop()
