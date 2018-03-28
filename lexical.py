#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 21:53:51 2018

@author: Md. Enamul Haque
@purpose: feature generation from review text
"""


import os
import sys
import numpy as np
import pandas as pd
import gensim
import csv
#from pyreadability.readability import Readability
from textstat.textstat import textstat
import re
import string
import gzip


def tokenize(document):
    characters = " '.,!#$%^&*();:\n\t\\\"?!{}[]<>"
    # terms = document.lower().split()
    return [term.strip(characters) for term in document]


def getItemNumbers(files):
    itemIndices = []
    for f in files:
        # split the file names using two level separators
        itemIndices.append(int(f.split('_',1)[1].split('.',1)[0]))
    itemIndices.sort()
    return itemIndices

def getFeatures(files):
    
    allFileData = []
    allScores = []
    allFres = []
    df = files
    rowNum = df.shape[0]
    for i in range(rowNum):
        FileData = []
        review = (df.iloc[i].reviewText).strip().split()
        summary = (df.iloc[i].summary).strip().split()
        row_data = np.append(review, summary)
        
        score = float(df.iloc[i].overall)
        fres = textstat.flesch_reading_ease(str(row_data))
        bg = bigramCount(str(row_data))


        FileData.append(bg)
        FileData.append(score)
        allFileData.append(FileData)
        allScores.append(score)
        allFres.append(fres)

    return allFileData, allScores, allFres





    
def saveFeatures(features, scores, fres, types):

    first_row = []
    with open(featureDir+str(types)+'_lexical_feature_vec.csv', 'w', newline='') as csvfile: 
        # preparing the csv feature file. First row of the csv file represents feature names and class label.
    
        for num in range(len(features[0])):    
            first_row.append('feat_'+ str(num))
        first_row.append('class_label')
        
        datawriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        datawriter.writerow(first_row)
        
        i = 0
        label = 0
        for f in features:
            # human score
            score = scores[i]
            # readability score using Flesch-Kincaid
            fkScore = fres[i]
            # check if readability score is more than 60 along with human score
            if score >= 4 and fkScore >= 80:
                label = 1
            else:
                label = 0
            i += 1
            file_data = np.append(f, label)
            datawriter.writerow(file_data)

def bigramCount(data):
    
    #f='This is a test sentence for computing bigram'
    filecontents = data
            
    # count bigrams 
    bigrams = {} 
    words_punct = filecontents.split() 
    # strip all punctuation at the beginning and end of words, and 
    # convert all words to lowercase.
    # The following is a Python list comprehension. It is a command that transforms a list,
    # here words_punct, into another list.
    words = [ w.strip(string.punctuation).lower() for w in words_punct ]
    
    # add special START, END tokens
    words = ["START"] + words + ["END"]
    
    for index, word in enumerate(words):
        if index < len(words) - 1:
            # we only look at indices up to the
            # next-to-last word, as this is
            # the last one at which a bigram starts
            w1 = words[index] 
            w2 = words[index + 1]
            # bigram is a tuple,
            # like a list, but fixed.
            # Tuples can be keys in a dictionary
            bigram = (w1, w2)
    
            if bigram in bigrams:
                bigrams[ bigram ] = bigrams[ bigram ] + 1
            else:
                bigrams[ bigram ] = 1
            # or, more simply, like this:
            # bigrams[bigram] = bigrams.get(bigram, 0) + 1
    
    # sort bigrams by their counts
    sorted_bigrams = sorted(bigrams.items(), key = lambda pair:pair[1], reverse = True)
    tcount = 0
    for bigram, count in sorted_bigrams:
        # print(bigram, ":", count)
        tcount = tcount + count
        
    return tcount

def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield eval(l)

def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')
   
if __name__=="__main__":
    global featureDir
    featureDir = './featureData/lexical/'
    global modelDir
    modelDir = './modelData/'
    dataDir = './data/'
    reviewTopics = ['automotive', 'video', 'musical', 'officeProd', 'patio', 'sports']
    fileNames = ['reviews_Automotive_5','reviews_Amazon_Instant_Video_5', 'reviews_Musical_Instruments_5','reviews_Office_Products_5', 'reviews_Patio_Lawn_and_Garden_5','reviews_Sports_and_Outdoors_5']


    for i in range(len(reviewTopics)):
        print("Processing review topic: ", reviewTopics[i])        
        df = getDF(dataDir+str(fileNames[i])+'.json.gz')
        print("Review count for ", reviewTopics[i], ' is ', df.shape[0])
        #sys.exit()
        #features, scores, fres = getFeatures(df)
        #saveFeatures(features, scores, fres, reviewTopics[i])
        #print("Processing done for type: ", reviewTopics[i])

