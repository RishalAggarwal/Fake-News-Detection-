# -*- coding: utf-8 -*-

import DataPrep
import spacy
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from textstat.textstat import textstatistics, easy_word_set, legacy_round
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
import nltk
import nltk.corpus 
from nltk.tokenize import word_tokenize
import collections
import pickle



def break_sentences(text):
    nlp = spacy.load('en')
    doc = nlp(text)
    return doc.sents


# Returns Number of Words in the text
def word_count(text):
    sentences = break_sentences(text)
    words = 0
    for sentence in sentences:
        words += len([token for token in sentence])
    return words


# Returns the number of sentences in the text
def sentence_count(text):
    sentences = break_sentences(text)
    return len(list(sentences))


# Returns average sentence length
def avg_sentence_length(text):
    words = word_count(text)
    sentences = sentence_count(text)
    average_sentence_length = float(words / sentences)
    return average_sentence_length



def syllables_count(word):
    return textstatistics().syllable_count(word)


def avg_syllables_per_word(text):
    syllable = syllables_count(text)
    words = word_count(text)
    ASPW = float(syllable) / float(words)
    return legacy_round(ASPW, 1)


# Return total Difficult Words in a text
def difficult_words(text):
    words = []
    sentences = break_sentences(text)
    for sentence in sentences:
        words += [str(token) for token in sentence]

    diff_words_set = set()

    for word in words:
        syllable_count = syllables_count(word)
        if word not in easy_word_set and syllable_count >= 2:
            diff_words_set.add(word)

    return len(diff_words_set)



def poly_syllable_count(text):
    count = 0
    words = []
    sentences = break_sentences(text)
    for sentence in sentences:
        words += [token for token in sentence]

    for word in words:
        syllable_count = syllables_count(word)
        if syllable_count >= 3:
            count += 1
    return count


def flesch_reading_ease(text):
    FRE = 206.835 - float(1.015 * avg_sentence_length(text)) - \
          float(84.6 * avg_syllables_per_word(text))
    return legacy_round(FRE, 2)


def gunning_fog(text):
    per_diff_words = (difficult_words(text) / word_count(text) * 100) + 5
    grade = 0.4 * (avg_sentence_length(text) + per_diff_words)
    return grade


def smog_index(text):
   if sentence_count(text) >= 3:
        poly_syllab = poly_syllable_count(text)
        SMOG = (1.043 * (30 * (poly_syllab / sentence_count(text))) ** 0.5) \
               + 3.1291
        return legacy_round(SMOG, 1)
   else:
        return 0


def dale_chall_readability_score(text):

    words = word_count(text)
    count = word_count - difficult_words(text)
    if words > 0:


        per = float(count) / float(words) * 100


    diff_words = 100 - per

    raw_score = (0.1579 * diff_words) + \
                (0.0496 * avg_sentence_length(text))



    if diff_words > 5:
        raw_score += 3.6365

    return legacy_round(raw_score, 2)

#Returns the readability of a sentence
def get_readability(x):
    gun_fog=[]
    flesh=[]
    c=1
    for i in x:
        gun_fog.append(gunning_fog(i))
        flesh.append(flesch_reading_ease(i))
        c+=1
    return np.vstack((np.asarray(gun_fog),np.asarray(flesh)))

#Class created to extract grammar syntax
class GrammarTransformer():
    """
    Convert text to counts of syntactic structure
    """
    def __init__(self, parser):
        self.parser = spacy.load('en')

    def transform(self, X, y=None,**transform_params):
        return self.countgrammar(X)

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, deep=True):
        return {}
 #Returns the pcfg counts in form of a dictionary
    def countgrammar(self, texts):
        lookup = {}
        for i, x in enumerate(texts):
            lookup[x] = i
        grammar_counts = {}
        for doc in self.parser.pipe(texts, batch_size=1000, n_threads=4):
            counts = collections.Counter()
            for w in doc:
                counts[w.dep_] += 1
            grammar_counts[doc.text] = counts
        rv = list(range(len(texts)))
        for text, i in lookup.items():
            rv[i] = grammar_counts[text]
        for i in range(len(rv)):
            if rv[i]==i:
                rv[i]=None

        return rv




