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
import csv
#from pyreadability.readability import Readability
from textstat.textstat import textstat
import re
import gzip
import pandas as pd

def lengthOfReview(context):
    """ counts the total number of words present in each review
    """
    lor = 0
    lor = len(context.split())
    return lor

def sentenceCount(context):
    """ counts number of sentences in each review context
        assume a sentence ends with ./!/?
    """
    sc = 0
    contextArray = re.split(r'[.!?]+', context)
    sc = len(contextArray)
    
    return sc
    
def charCount(context):
    """ total number of character in each review
    """
    cc = 0
    cc = len(context)
    return cc
    
def allCapCount(context):
    """ total number of capital letters in each review
    """
    acc = 0
    acc = len(re.findall(r'[A-Z]',context))
    return acc
    
def questionCount(context):
    
    qc = 0
    qc = len(re.findall(r'[?]', context))
    
    return qc


    
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
        lor = lengthOfReview(str(row_data))
        sc = sentenceCount(str(row_data))
        cc = charCount(str(row_data))
        acc = allCapCount(str(row_data))
        qc = questionCount(str(row_data))
        review = [element.lower() for element in review]

        FileData.append(lor)
        FileData.append(sc)
        FileData.append(cc)
        FileData.append(acc)
        FileData.append(qc)
        FileData.append(score)
        allFileData.append(FileData)
        allScores.append(score)
        allFres.append(fres)

    return allFileData, allScores, allFres



    
def saveFeatures(features, scores, fres, types):

    first_row = []
    with open(featureDir+str(types)+'_structural_feature_vec.csv', 'w', newline='') as csvfile: 
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
            if score >=4 and fkScore >= 80:
                label = 1
            else:
                label = 0
            i += 1
            file_data = np.append(f, label)
            datawriter.writerow(file_data)
            
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
    featureDir = './featureData/structural/'
    global modelDir
    modelDir = './modelData/'
    dataDir = './data/'
    reviewTopics = ['automotive', 'video', 'musical', 'officeProd', 'patio', 'sports']
    fileNames = ['reviews_Automotive_5','reviews_Amazon_Instant_Video_5', 'reviews_Musical_Instruments_5','reviews_Office_Products_5', 'reviews_Patio_Lawn_and_Garden_5','reviews_Sports_and_Outdoors_5']

    for i in range(len(reviewTopics)):
        print("Processing review topic: ", reviewTopics[i])        
        df = getDF(dataDir+str(fileNames[i])+'.json.gz')
        features, scores, fres = getFeatures(df)
        saveFeatures(features, scores, fres, reviewTopics[i])
        print("Processing done for type: ", reviewTopics[i])



    