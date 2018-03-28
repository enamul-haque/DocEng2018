#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 19:52:57 2018

@author: Md Enamul Haque
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    directory = './results/'
    
    # load result data
    types = ['automotive', 'musical', 'officeProd', 'patio', 'sports', 'video']
    
    resultDir = './results/'

    fileWriter = open(directory+'result_semantic.txt','w')
    fileWriter.write('prod_type'+ '\t' + 'accuracy' + '\t'+'acc_std'+'\t' + 'precision' + '\t'+'prec_std'+'\t'+ 'recall' + '\t'+'recall_std'+'\t' + 'f1_macro'+ '\t'+'f1_std'+'\n')
        
    for k in range(len(types)):
        sem = pd.read_csv(resultDir+'result_semantic_'+str(types[k])+'.txt', sep='\t')
        #comb = pd.read_csv(resultDir+'result_combined_'+str(types[k])+'.txt', sep='\t')
        
        # create accuracy file

        avg_acc = np.mean(sem.accuracy)
        std_acc = np.std(sem.accuracy)
        
        avg_prec = np.mean(sem.precision)
        std_prec = np.std(sem.precision)
        
        avg_rec = np.mean(sem.recall)
        std_rec = np.std(sem.recall)
        
        avg_f1 = np.mean(sem.f1_macro)
        std_f1 = np.std(sem.f1_macro)
        
        fileWriter.write(str(types[k])+ '\t' + str(avg_acc) + '\t'+str(std_acc)+'\t' + str(avg_prec) + '\t'+str(std_prec)+'\t' + str(avg_rec) +'\t'+ str(std_rec)+'\t'+ str(avg_f1)+'\t'+str(std_f1)+'\n')
        print(str(types[k])+ '\t' + str(avg_acc) + '\t'+str(std_acc)+'\t' + str(avg_prec) + '\t'+str(std_prec)+'\t' + str(avg_rec) +'\t'+ str(std_rec)+'\t'+ str(avg_f1)+'\t'+str(std_f1)+'\n')

    fileWriter.close()
            
    fileWriter = open(directory+'result_combined.txt','w')
    fileWriter.write('prod_type'+ '\t' + 'accuracy' + '\t'+'acc_std'+'\t' + 'precision' + '\t'+'prec_std'+'\t'+ 'recall' + '\t'+'recall_std'+'\t' + 'f1_macro'+ '\t'+'f1_std'+'\n')
    
    for k in range(len(types)):
        
        comb = pd.read_csv(resultDir+'result_combined_'+str(types[k])+'.txt', sep='\t')
        
        # create accuracy file

        avg_acc = np.mean(comb.accuracy)
        std_acc = np.std(comb.accuracy)
        
        avg_prec = np.mean(comb.precision)
        std_prec = np.std(comb.precision)
        
        avg_rec = np.mean(comb.recall)
        std_rec = np.std(comb.recall)
        
        avg_f1 = np.mean(comb.f1_macro)
        std_f1 = np.std(comb.f1_macro)
        
        fileWriter.write(str(types[k])+ '\t' + str(avg_acc) + '\t'+str(std_acc)+'\t' + str(avg_prec) + '\t'+str(std_prec)+'\t' + str(avg_rec) +'\t'+ str(std_rec)+'\t'+ str(avg_f1)+'\t'+str(std_f1)+'\n')
    fileWriter.close()
        
        
            