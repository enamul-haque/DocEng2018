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
import string
#from pyreadability.readability import Readability
from textstat.textstat import textstat
import gzip



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

def getContext(files):
    
    allFileData = []
    allScores = []
    allFres = []
    allAri = []
    df = files
    rowNum = df.shape[0]
    for i in range(rowNum):
        review = (df.iloc[i].reviewText).strip().split()
        summary = (df.iloc[i].summary).strip().split()
        row_data = np.append(review, summary)
        
        score = float(df.iloc[i].overall)
        fres = textstat.flesch_reading_ease(str(row_data))
        ari = textstat.automated_readability_index(str(row_data))
        review = [element.lower() for element in review]

        allFileData.append(review)
        allScores.append(score)
        allFres.append(fres)
        allAri.append(ari)
    return allFileData, allScores, allFres, allAri


def buildModel(data, size, types):
    model = gensim.models.Word2Vec(data,size=size,window=5,min_count=1,workers=4)
    #model.wv.save_word2vec_format("term2vec_model_"+str(size))
    model.wv.save_word2vec_format(modelDir+str(types)+"_term2vec_model_"+str(size))
    return model

    
def saveEmbeddings(content, scores, fres, ari, size, types):
    model = buildModel(content, size, types)
    first_row = []
    with open(featureDir+str(types)+'_semantic_feature_vec_'+str(size)+'.csv', 'w', newline='') as csvfile: 
        # preparing the csv feature file. First row of the csv file represents feature names and class label.
    
        for num in range(size):    
            first_row.append('feat_'+ str(num))
        first_row.append('class_label')
        
        datawriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        datawriter.writerow(first_row)
        
        i = 0
        label = 0
        for eachContext in content:
            print("Working at content: ", i)
            embedding = size * [0]
            for k in range(len(eachContext)):
                embedding += model[str(eachContext[k])]
            
            try:
                avgEmbedding = embedding/len(eachContext)
                
            except:
                print(embedding)
                print(len(eachContext))
                print(avgEmbedding)
            #sys.exit()
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
            file_data = np.append(avgEmbedding, label)
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

if __name__ == "__main__":
    global featureDir
    featureDir = './featureData/semantic/'
    global modelDir
    modelDir = './modelData/'
    dataDir = './data/'
    reviewTopics = ['automotive', 'video', 'musical', 'officeProd', 'patio', 'sports']
    fileNames = ['reviews_Automotive_5','reviews_Amazon_Instant_Video_5', 'reviews_Musical_Instruments_5','reviews_Office_Products_5', 'reviews_Patio_Lawn_and_Garden_5','reviews_Sports_and_Outdoors_5']
    sizes = list(np.arange(5,105,5))

    for i in range(len(reviewTopics)):
        print("Processing review topic: ", reviewTopics[i])        
        df = getDF(dataDir+str(fileNames[i])+'.json.gz')
        content, scores, fres, ari = getContext(df)
        for size in sizes:
            saveEmbeddings(content, scores, fres, ari, size, reviewTopics[i])
        print("Processing done for type: ", reviewTopics[i])
