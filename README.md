# Trump-Fake-News-Detection

The aim of this project is to classify various news headlines based on their content, syntax and readability as real or fake news. 


## Flowchart

The flowchart of the model is as follows:

![alt text](https://github.com/RishalAggarwal/Trump-Fake-News-Detection-/blob/master/Fake_news_detection/Images/flowchart%20(1).jpeg)

## Dataset

The dataset used is the LIAR dataset for this task. The data preperation and preprocessing has been borrowed from [nishitpatel01](https://github.com/nishitpatel01/Fake_News_Detection).

## Feature Extraction and Selection

### 1) TF-IDF vectors:

TF-IDF vectors were created for the documents using single word and bi-word indices and then 20,000 of these indices were selected for the classification task using Mutual information as the selection criteria.

### 2) Punctuation:

Punctuation count vectors for 11 different types of punctuations were extracted from the documents

### 3) Gram Syntax:

Probabilistic context free grammar syntax (pcfg) vectors were extracted from these documents and stored as TF-IDF vectors. The code for extracting these features is borrowed from [aldengolab](https://github.com/aldengolab/fake-news-detection).

### 4) Readability:

Flesch and Gunning fog values were extracted to use as readability parameters in the model. 
The formula for flesch reading index is:

Reading Ease score = 206.835 - (1.015 × ASL) - (84.6 × ASW)

Here,

ASL = average sentence length (number of words divided by number of sentences)

ASW = average word length in syllables (number of syllables divided by number of words)


The formula for gunning fog index is:

Grade level= 0.4 * ( (average sentence length) + (percentage of Hard Words) )

Here, 

Hard Words = words with more than two syllables.

The values were extracted and saved as pickle files as the amount of time taken to extract these values was a lot.

## Classification

Three linear classifiers (Logistic Regression, SVM and Naive Bayes) were used for the task and compared. LogR showed the best results with a Macro Average F1 score of 0.61.

## Prediction

Prediction.py file has been created that imports all the feature classes and performs feature extraction upon user input as well as classification and reports it in the form of a simple GUI.

