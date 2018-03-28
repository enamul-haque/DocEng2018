#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 13:58:02 2018

@author: Md. Enamul Haque
"""

from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, hamming_loss
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_score, recall_score
import glob
from sklearn.model_selection import cross_val_score
from sklearn import svm
import warnings
from sklearn import preprocessing
import sys
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import tree
from sklearn.metrics import classification_report
warnings.filterwarnings("ignore")


class Classification:
    
    global types
    types = ['automotive', 'video', 'musical', 'officeProd', 'patio', 'sports']
    global directory
    directory = './results/'
    
    def __init__(self, featureType, clf):
        self.fType = featureType
        self.model = clf
        self.featureDir = './featureData/'+str(featureType)+'/'
        
    def normalClassification(self):
        
        fileWriter = open(directory+'result_'+str(self.fType)+'.txt','w')
        fileWriter.write('prod_type'+ '\t' + 'accuracy' + '\t'+'acc_std'+'\t' + 'precision' + '\t'+'prec_std'+'\t'+ 'recall' + '\t'+'recall_std'+'\t' + 'f1_macro'+ '\t'+'f1_std'+'\n')
        
        for t in types:
            
            print("Working for type: ", t, "and  feature type: ", self.fType)
            print("***********************")
            data = pd.read_csv(self.featureDir+t+'_'+self.fType+'_feature_vec.csv')
            if self.fType == 'lexical':
                data=data.drop(['feat_1'],axis=1)
            if self.fType == 'structural':
                data=data.drop(['feat_5'],axis=1)
            
            class_column = data.shape[1]-1 
            sz = data.shape
            last_data_column = sz[1]-1
            #train = data.iloc[:int(sz[0] * 0.7), :]
            #test = data.iloc[int(sz[0] * 0.7):, :]
        
            #train = data.iloc[:int(sz[0] * 0.9), :]
            #test = data.iloc[int(sz[0] * 0.9):, :]
            X = data.iloc[:,0:last_data_column]
            y = data.iloc[:, class_column]
            
            acc = cross_val_score(model, X, y, cv=10, scoring='accuracy')
            #print("Accuracy: %0.2f (+/- %0.2f)" % (acc.mean(), acc.std() * 2))
            prec = cross_val_score(model, X, y, cv=10, scoring='precision')
            #print("Precision:%0.2f (+/- %0.2f)" % (prec.mean(), prec.std() * 2))
            rec = cross_val_score(model, X, y, cv=10, scoring='recall')
            #print("Recall:%0.2f (+/- %0.2f)" % (rec.mean(), rec.std() * 2))
            f1 = cross_val_score(model, X, y, cv=10, scoring='f1')
            #print("F1:%0.2f (+/- %0.2f)" % (f1.mean(), f1.std() * 2))
            fileWriter.write(str(t)+ '\t' + str(acc.mean()) + '\t'+str(acc.std())+'\t' + str(prec.mean()) + '\t'+str(prec.std())+'\t' + str(rec.mean()) +'\t'+ str(rec.std())+'\t'+ str(f1.mean())+'\t'+str(f1.std())+'\n')
        fileWriter.close()
        

    
    def semanticClassification(self, sizes):
        
        for t in types:
            print("Working for type: ", t, "and  feature type: ", self.fType)
            print("***********************")
            fileWriter = open(directory+'result_'+str(self.fType)+'_'+str(t)+'.txt','w')
            fileWriter.write('vec_length'+ '\t' + 'accuracy' + '\t' + 'precision' + '\t'+ 'recall' + '\t' + 'f1_macro'+ '\n')
        
        
            for size in sizes:
                    if self.fType == 'combined' or self.fType == 'semantic':
                        data = pd.read_csv(self.featureDir+t+'_'+self.fType+'_feature_vec_'+str(size)+'.csv')
                    
                    class_column = data.shape[1]-1 
                    #print("Class column: ", class_column)
                    
                    sz = data.shape
                    last_data_column = sz[1]-1

                    
                    X = data.iloc[:,0:last_data_column]
                    y = data.iloc[:, class_column]
                
                    train_X = train.iloc[:,0:last_data_column]
                    train_Y = train.iloc[:, class_column]
                    test_X = test.iloc[:,0:last_data_column]
                    test_Y = test.iloc[:, class_column]
                    
                    model = self.model.fit(train_X, train_Y)
                    y_hat = model.predict(test_X)
                    print ("Accuracy Rate, which is calculated by accuracy_score() is: %f" % accuracy_score(test_Y, y_hat))
                    print ("Hamming loss: ", hamming_loss(test_Y, y_hat))
                    print ("Average precision score:", precision_score(test_Y, y_hat, average='macro'))
                    print(average_precision_score(test_Y, y_hat, average='micro'))
                    fileWriter.write(str(size)+ '\t' + str(accuracy_score(test_Y, y_hat)) + '\t' + str(precision_score(test_Y, y_hat, average='macro')) + '\t' + str(recall_score(test_Y, y_hat, average='macro')) + '\t'+ str(f1_score(test_Y, y_hat, average='macro'))+'\n')
                    scores = cross_val_score(model, train_X, train_Y, cv=10, scoring='accuracy')
                    print("Accuracy:",scores.mean(), 'for size: ', size)
                    scores = cross_val_score(model, train_X, train_Y, cv=10, scoring='precision')
                    print("Precision:",scores.mean(), 'for size: ', size)
                    scores = cross_val_score(model, train_X, train_Y, cv=10, scoring='recall')
                    print("Recall:",scores.mean(), 'for size: ', size)
                    scores = cross_val_score(model, train_X, train_Y, cv=10, scoring='f1')
                    print("F1:",scores.mean(), 'for size: ', size)
            fileWriter.close()

if __name__ == "__main__":
    
    sizes = list(np.arange(5,105,5))
    model = tree.DecisionTreeClassifier()
    #model = GaussianNB()
    
#    L = Classification('structural',model)
#    L.normalClassification()
#    
#    L = Classification('lexical',model)
#    L.normalClassification()
#    
#    #sys.exit()
#    Sem = Classification('semantic', model)
#    Sem.semanticClassification(sizes)
    
    Com = Classification('combined', model)
    Com.semanticClassification(sizes)
