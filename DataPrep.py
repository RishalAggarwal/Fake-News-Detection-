# -*- coding: utf-8 -*-
import pandas as pd
import csv
import numpy as np
import seaborn as sb


test_filename = 'test.csv'
train_filename = 'train.csv'
valid_filename='valid.csv'

train_news = pd.read_csv(train_filename)
test_news = pd.read_csv(test_filename)
valid_news=pd.read_csv(valid_filename)




def data_obs():
    print("training dataset size:")
    print(train_news.shape)
    print(train_news.head(10))

    print(test_news.shape)
    print(test_news.head(10))
    
    print(valid_news.shape)
    print(valid_news.head(10))


def create_distribution(dataFile):
    
    return sb.countplot(x='Label', data=dataFile, palette='hls')
    


create_distribution(train_news)
create_distribution(test_news)
create_distribution(valid_news)


def data_qualityCheck():
    
    print("Checking data qualitites...")
    train_news.isnull().sum()
    train_news.info()
        
    print("check finished.")

    #below datasets were used to 
    test_news.isnull().sum()
    test_news.info()

    valid_news.isnull().sum()
    valid_news.info()

