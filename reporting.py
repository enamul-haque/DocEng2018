#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 19:52:57 2018

@author: Md Enamul Haque
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys


if __name__ == "__main__":
    directory = './results/'
    
    # load result data
    #types = ['automotive', 'musical', 'officeProd', 'patio', 'sports', 'video']
    types = ['automotive', 'musical','sports']
    resultDir = './results/'
    lex = pd.read_csv(resultDir+'result_lexical.txt', sep='\t')
    struct = pd.read_csv(resultDir+'result_structural.txt', sep='\t')
    
    sem = pd.read_csv(resultDir+'result_semantic.txt', sep='\t')
    comb = pd.read_csv(resultDir+'result_combined.txt', sep='\t')
    
    
    
    lex_acc = [lex.accuracy[0],lex.accuracy[2],lex.accuracy[5]]
    struct_acc = [struct.accuracy[0],struct.accuracy[2],struct.accuracy[5]]
    sem_acc = [sem.accuracy[0],sem.accuracy[1],sem.accuracy[4]]
    comb_acc = [comb.accuracy[0],comb.accuracy[1],comb.accuracy[4]]

    
    plt.figure(1)
    #x = np.linspace(0, 1, 6)
    
    x=types
    plt.plot(x, lex_acc, 'r-*')
    plt.plot(x, struct_acc, 'b-o')
    plt.plot(x, sem_acc, 'g-^')
    plt.plot(x, comb_acc, 'k-x')
    plt.legend(['Lexical', 'Structural', 'Semantic', 'LS2F'])
    plt.xlabel("Review types")
    plt.ylabel("Accuracy")
    plt.savefig('accuracy.eps')
#    
#    
#    #sys.exit()
#    plt.figure(2)
#    plt.plot(x, lex.precision, 'r-*')
#    plt.plot(x, struct.precision, 'b-o')
#    plt.plot(x, sem.precision, 'g-^')
#    plt.plot(x, comb.precision, 'k-x')
#    plt.legend(['Lexical', 'Structural', 'Semantic', 'Combined'])
#    plt.xlabel("Review types")
#    plt.ylabel("Precision")
#    plt.savefig('precision.eps')
#    
#    plt.figure(3)
#    plt.plot(x, lex.recall, 'r-*')
#    plt.plot(x, struct.recall, 'b-o')
#    plt.plot(x, sem.recall, 'g-^')
#    plt.plot(x, comb.recall, 'k-x')
#    plt.legend(['Lexical', 'Structural', 'Semantic', 'Combined'])
#    plt.xlabel("Review types")
#    plt.ylabel("Recall")
#    plt.savefig('recall.eps')
#    
#    
#    plt.figure(4)
#    plt.plot(x, lex.f1_macro, 'r-*')
#    plt.plot(x, struct.f1_macro, 'b-o')
#    plt.plot(x, sem.f1_macro, 'g-^')
#    plt.plot(x, comb.f1_macro, 'k--x')
#    plt.legend(['Lexical', 'Structural', 'Semantic', 'Combined'])
#    plt.xlabel("Review types")
#    plt.ylabel("F1 macro")
#    plt.savefig('f1_macro.eps')
    


    s=15
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 macro']
    for k in range(len(types)):
        sem = pd.read_csv(resultDir+'result_semantic_'+str(types[k])+'.txt', sep='\t')
        comb = pd.read_csv(resultDir+'result_combined_'+str(types[k])+'.txt', sep='\t')
        
        plt.close('all')
        plt.subplot(2, 2, 1)
        plt.plot(sem.vec_length, sem.accuracy, 'k-x', label='sem')
        plt.plot(sem.vec_length, comb.accuracy, 'g-o', mfc='none',label='ls2f')
        plt.yticks(fontsize=s)
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
        #plt.legend(['sem', 'com'])
        #plt.title('A tale of 2 subplots')
        plt.ylabel('Accuracy')
        
        plt.subplot(2, 2, 2)
        plt.plot(sem.vec_length, sem.precision, 'k-x', label='sem')
        plt.plot(sem.vec_length, comb.precision, 'g-o', mfc='none', label='ls2f')
        plt.yticks(fontsize=s)
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
        plt.ylabel('Precision')
        
        plt.subplot(2, 2, 3)
        plt.plot(sem.vec_length, sem.recall, 'k-x')
        plt.plot(sem.vec_length, comb.recall, 'g-o', mfc='none')
        plt.yticks(fontsize=s)
        plt.xlabel('Embedding length')
        plt.ylabel('Recall')
        
        
        plt.subplot(2, 2, 4)
        plt.plot(sem.vec_length, sem.f1_macro, 'k-x')
        plt.plot(sem.vec_length, comb.f1_macro, 'g-o', mfc='none')
        plt.xlabel('Embedding length')
        plt.ylabel('F1 macro')
        plt.yticks(fontsize=s)
        plt.tight_layout()
        plt.savefig(str(types[k])+'.eps')
        
#        sys.exit()
#        plt.figure()
#        plt.plot(sem.vec_length, sem.accuracy, 'k-x')
#        plt.plot(sem.vec_length, comb.accuracy, 'b-o')
#        plt.legend(['Semantic', 'Combined'])
#        plt.xlabel("Embedding length")
#        plt.ylabel("Accuracy")
#        #plt.savefig(str(types[k])+'_accuracy'+'.eps')
#        
#        plt.figure()
#        plt.plot(sem.vec_length, sem.precision, 'k-x')
#        plt.plot(sem.vec_length, comb.precision, 'b-o')
#        plt.legend(['Semantic', 'Combined'])
#        plt.xlabel("Embedding length")
#        plt.ylabel("Precision")
#        #plt.savefig(str(types[k])+'_precision'+'.eps')
#        
#        
#        plt.figure()
#        plt.plot(sem.vec_length, sem.recall, 'k-x')
#        plt.plot(sem.vec_length, comb.recall, 'b-o')
#        plt.legend(['Semantic', 'Combined'])
#        plt.xlabel("Embedding length")
#        plt.ylabel("Recall")
#        #plt.savefig(str(types[k])+'_recall'+'.eps')
#        
#        
#        plt.figure()
#        plt.plot(sem.vec_length, sem.f1_macro, 'k-x')
#        plt.plot(sem.vec_length, comb.f1_macro, 'b-o')
#        plt.legend(['Semantic', 'Combined'])
#        plt.xlabel("Embedding length")
#        plt.ylabel("F1 macro")
#        #plt.savefig(str(types[k])+'_f1'+'.eps')
    
        
    
    
    

        
        
            