# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 08:53:56 2023

@author: jianc
"""



import dataset
import numpy as np 
import pandas as pd
import gurobipy as gp
from gurobipy import*
from gurobipy import GRB

import time
from scipy import stats

#!/usr/bin/env python
# coding: utf-8
# author: Bo Tang

from collections import namedtuple
import numpy as np
from scipy import stats
import gurobipy as gp
from gurobipy import GRB
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix   
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import d2

train_ratio = 0.5

val_ratio = 0.25

test_ratio = 0.25

seed=1



N_set=[2,3,4,5]

datasett=['AER','Australia','bene1','bene2','german','give','hmeq','polish','taiwan','thomas','UK']

#样本量690~3364
dataset1=['Australia','german','thomas','AER','bene1','hmeq']

#中等样本集5910，7190
dataset2=['polish','bene2']

#大样本30000以上
dataset3=['taiwan','UK','give']



res_sk = pd.DataFrame(columns=['Data', 'alpha','cost','acc_train', 
       'acc_test', 'acc_val', 'Precision', 'Recall', 'AUC', 'F1', 'train_time','a','b','c','d'])     


for da in dataset1:
        for N in N_set:
            x, y = dataset.loadData(da)
            scales = np.max(x, axis=0)
            x=x/scales
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1-train_ratio, random_state=seed)
            x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, 
                                                            test_size=test_ratio/(test_ratio+val_ratio), random_state=seed)
            
            timel=1800
            
            cs=len(y)/sum(y)
    
            tick = time.time()
            a, b, c, d=d2.d2(x_train, y_train, cost=cs, t=timel, max_depth=2, N=N,alpha =0.001,f1=0.1)
            tock = time.time()
            
            y_train_pre=d2.predict(x_train, a,b,c,d)

            y_test_pre=d2.predict(x_test,a,b,c,d)
            
            y_val_pre=d2.predict(x_val, a,b,c,d)

            acc_train =accuracy_score(y_train, y_train_pre)

            acc_test =accuracy_score(y_test, y_test_pre)

            acc_val =accuracy_score(y_val, y_val_pre)
            
            auc_linear = roc_auc_score(y_test, y_test_pre)
            
            precision=precision_score(y_test, y_test_pre)
            
            recall=recall_score(y_test, y_test_pre)
            
            f1=f1_score(y_test, y_test_pre)
            
            def totalcost(y_test,y_test_pred):
                n10=0
                n01=0
                for i in range(len(y_test)):
                    if y_test[i]-y_test_pred[i]>=0.2:
                        n10=n10+1
                    elif y_test[i]-y_test_pred[i]<=-0.2:
                        n01=n01+1
                costt=len(y)/sum(y)
                tcost=costt* n10 +n01
                return tcost

            tcost=totalcost(y_test, y_test_pre)
            
            tock = time.time()
            train_time = tock - tick
            
            
            
            row = {'Data':da, 'N':N,'cost':tcost,'acc_train':acc_train, 
                   'acc_test':acc_test, 'acc_val':acc_val, 'Precision': precision, 'Recall':recall,
                   'AUC':auc_linear, 'F1':f1, 'train_time':train_time,'a':a,
                   'b':b, 'c':c,'d':d}
            res_sk = res_sk.append(row, ignore_index=True)
            res_sk.to_csv('cost_d21.csv', index=False)
              



for da in dataset2:
        for N in N_set:
            x, y = dataset.loadData(da)
            scales = np.max(x, axis=0)
            x=x/scales
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1-train_ratio, random_state=seed)
            x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, 
                                                            test_size=test_ratio/(test_ratio+val_ratio), random_state=seed)
            
            cs=len(y)/sum(y)
            
            timel=3000
    
            tick = time.time()
            a, b, c, d=d2.d2(x_train, y_train, cost=cs, t=timel, max_depth=2, N=N,alpha =0.001,f1=0.1)
            tock = time.time()
            
            y_train_pre=d2.predict(x_train, a,b,c,d)

            y_test_pre=d2.predict(x_test,a,b,c,d)
            
            y_val_pre=d2.predict(x_val, a,b,c,d)

            acc_train =accuracy_score(y_train, y_train_pre)

            acc_test =accuracy_score(y_test, y_test_pre)

            acc_val =accuracy_score(y_val, y_val_pre)
            
            auc_linear = roc_auc_score(y_test, y_test_pre)
            
            precision=precision_score(y_test, y_test_pre)
            
            recall=recall_score(y_test, y_test_pre)
            
            f1=f1_score(y_test, y_test_pre)
            
            def totalcost(y_test,y_test_pred):
                n10=0
                n01=0
                for i in range(len(y_test)):
                    if y_test[i]-y_test_pred[i]>=0.2:
                        n10=n10+1
                    elif y_test[i]-y_test_pred[i]<=-0.2:
                        n01=n01+1
                costt=len(y)/sum(y)
                tcost=costt* n10 +n01
                return tcost

            tcost=totalcost(y_test, y_test_pre)
            
            tock = time.time()
            train_time = tock - tick
            
            
            
            row = {'Data':da, 'N':N,'cost':tcost,'acc_train':acc_train, 
                   'acc_test':acc_test, 'acc_val':acc_val, 'Precision': precision, 'Recall':recall,
                   'AUC':auc_linear, 'F1':f1, 'train_time':train_time,'a':a,
                   'b':b, 'c':c,'d':d}
            res_sk = res_sk.append(row, ignore_index=True)
            res_sk.to_csv('cost_d22.csv', index=False)
              

for da in dataset3:
        for N in N_set:
            x, y = dataset.loadData(da)
            scales = np.max(x, axis=0)
            x=x/scales
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1-train_ratio, random_state=seed)
            x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, 
                                                            test_size=test_ratio/(test_ratio+val_ratio), random_state=seed)
            
            timel=6000
            
            cs=len(y)/sum(y)
    
            tick = time.time()
            a, b, c, d=d2.d2(x_train, y_train, cost=cs, t=timel, max_depth=2, N=N,alpha =0.001,f1=0.1)
            tock = time.time()
            
            y_train_pre=d2.predict(x_train, a,b,c,d)

            y_test_pre=d2.predict(x_test,a,b,c,d)
            
            y_val_pre=d2.predict(x_val, a,b,c,d)

            acc_train =accuracy_score(y_train, y_train_pre)

            acc_test =accuracy_score(y_test, y_test_pre)

            acc_val =accuracy_score(y_val, y_val_pre)
            
            auc_linear = roc_auc_score(y_test, y_test_pre)
            
            precision=precision_score(y_test, y_test_pre)
            
            recall=recall_score(y_test, y_test_pre)
            
            f1=f1_score(y_test, y_test_pre)
            
            def totalcost(y_test,y_test_pred):
                n10=0
                n01=0
                for i in range(len(y_test)):
                    if y_test[i]-y_test_pred[i]>=0.2:
                        n10=n10+1
                    elif y_test[i]-y_test_pred[i]<=-0.2:
                        n01=n01+1
                costt=len(y)/sum(y)
                tcost=costt* n10 +n01
                return tcost

            tcost=totalcost(y_test, y_test_pre)
            
            tock = time.time()
            train_time = tock - tick
            
            
            
            row = {'Data':da, 'N':N,'cost':tcost,'acc_train':acc_train, 
                   'acc_test':acc_test, 'acc_val':acc_val, 'Precision': precision, 'Recall':recall,
                   'AUC':auc_linear, 'F1':f1, 'train_time':train_time,'a':a,
                   'b':b, 'c':c,'d':d}
            res_sk = res_sk.append(row, ignore_index=True)
            res_sk.to_csv('cost_d23.csv', index=False)
              


