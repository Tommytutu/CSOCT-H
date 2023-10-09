# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 12:15:18 2023

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

def LBC(x, y, cost=1, t=120, max_depth=1, N=2,alpha = 0.00,f1=0.1):
    n, p=x.shape
    
    weights = {0:1, 1:cost}
    
    a_start=np.zeros(p)
    b_start=0
    yp_start=np.zeros(n)
    
    clf = DecisionTreeClassifier(max_depth=1,class_weight=weights)
    clf.fit(x,y)
    
    decision_rules = clf.tree_

    
    af= decision_rules.feature[0]
    b1= decision_rules.threshold[0]
        

    
    leaf_labels = {}

    def get_leaf_labels(node, parent_index=0):
        if node.children_left[parent_index] == -1:
            leaf_labels[parent_index] = clf.classes_[np.argmax(node.value[parent_index])]
        else:
            get_leaf_labels(node, node.children_left[parent_index])
            get_leaf_labels(node, node.children_right[parent_index])

    get_leaf_labels(clf.tree_)
    
    if leaf_labels[1]==1:
        a_start[af] =-1
        b_start =-b1
    else:
        a_start[af] =1
        b_start =b1
        
    for i in range(n):
        if (sum(a_start[j]*x[i,j] for j in range(p))>=b_start):
            yp_start[i]=1
        else:
            yp_start[i]=0
            
    


    # create a model
    m = gp.Model('m')

    # output

    # time limit
    
    timelimit=t
    m.Params.timelimit = timelimit
    m.Params.Heuristic = 0.2
    m.Params.NoRelHeurTime=10
    
    #m.params.Method=2
    m.params.DegenMoves=0
    # parallel
    #m.params.threads = 0
    #m.params.Presolve=2
    #m.params.MIPFocus=1
    #m.params.Cuts=3
    m.params.ImproveStartTime=0.8*timelimit


    # model sense
    m.modelSense = GRB.MINIMIZE

    # variables
    a = m.addVars(p,  lb=-1, ub=1, vtype=GRB.CONTINUOUS, name='a') # splitting feature
    bara = m.addVars(p, lb=0, ub=1, vtype=GRB.CONTINUOUS, name='bara') # splitting feature
    s = m.addVars(p,  vtype=GRB.BINARY, name='s') # splitting feature
    b = m.addVar(lb=-1, ub=1, vtype=GRB.CONTINUOUS, name='b') # splitting threshold
    y_pre = m.addVars(n, vtype=GRB.BINARY, name='y_pre') # leaf node assignment

    #e = m.addVar(lb=0.01, ub=0.05, vtype=GRB.CONTINUOUS, name='e') # splitting threshold
    e=0.0002



    # objective function


    e1 = gp.quicksum((y[i] == 0) * y_pre[i] for i in range(n))/n

    e2 = cost*gp.quicksum((y[i] == 1) * (1 - y_pre[i]) for i in range(n))/n

    obj = e1 + e2 + alpha * gp.quicksum(s[j] for j in range(p))
    m.setObjective(obj)
    
    TP=gp.quicksum((y[i] == 1) * y_pre[i] for i in range(n))
    TN=gp.quicksum((y[i] == 0) * (1-y_pre[i]) for i in range(n))
    FP=gp.quicksum((y[i] == 0) * (y_pre[i]) for i in range(n))
    FN=gp.quicksum((y[i] == 1) * (1-y_pre[i]) for i in range(n))
                   

   
    m.addConstr(bara.sum()<= 1)


    m.addConstrs(bara[j] >= a[j]  for j  in range(p))

    m.addConstrs(bara[j] >= -a[j]  for j  in range(p))

    m.addConstrs(s[j] >=a[j] for j  in range(p))

    m.addConstrs(-s[j] <=a[j]  for j  in range(p))

    m.addConstr(s.sum() >=1)

    #每个节点处最多两个特征
    m.addConstr(s.sum('*')<=N)


    m.addConstrs(gp.quicksum(a[j] * x[i,j] for j in range(p)) >=b - 2*(1 - y_pre[i]) for i in range(n))

            
    m.addConstrs(gp.quicksum(a[j] * x[i,j] for j in range(p)) +e<= b+ 2*y_pre[i] for i in range(n))


    #m.addConstr(gp.quicksum((y[i] == 0) * y_pre[i] for i in range(n))/ sum((y[i] == 0)  for i in range(n)) <=0.3)

    #m.addConstr(gp.quicksum((y[i] == 0) * y_pre[i] for i in range(n)) <=0.3 * gp.quicksum(((1-y_pre[i]) for i in range(n))))
    
    # F1_score
    m.addConstr(f1*(2*TP+FP+FN)<=2*TP)
    
    m.addConstr(TP>=5)
    

    #m.addConstr(gp.quicksum((y[i] == 1) * (1-y_pre[i]) for i in range(n)) <=0.9*sum((y[i] == 1)  for i in range(n)))
    
    
    s_s=np.zeros(p)
    for j in range(p):
        if abs(a_start[j])==0:
            s_s[j]=0
        else:
            s_s[j]=1

    for j in range(p):
        a[j].start=a_start[j]
        s[j].start=s_s[j]

    b.start=b_start

    for i in range(n):
        y_pre[i].start = yp_start[i]


    m.optimize()

   
    
    _a = {ind:a[ind].x for ind in a}
    _p = {ind:y_pre[ind].x for ind in y_pre}
    _b = b.x
    
    for i in range(len(_p)):
        if _p[i]<=0.1:
            _p[i]=0
        else:
            _p[i]=1

    y0=[]
    y1=[]
    for i in range(n):
        if y[i]==0:
            y0.append(i)
        
        else:
            y1.append(i)

    p0=[]
    p1=[]
    for i in range(n):
        if _p[i]<=0.001:
            p0.append(i)
        
        else:
            p1.append(i)
            
    y01=[]        
    for i in y0:
        if _p[i] >=0.1:
            y01.append(i)
            
    y10=[]        
    for i in y1:
        if _p[i] <=0.001:
            y10.append(i)
    return _a, _b, _p, p0, p1, y01, y10



def depth2(x_train, y_train, cost=1, t=120,  N=2,alpha = 0.001,f1=0.1):
    n, pv=x_train.shape
    
    #the first node
    a1, b1, _p1, p0, p1, y01, y10=LBC(x_train, y_train, cost=cost, t=t, max_depth=1, N=N,alpha = alpha, f1=f1)
    d1=1
    
    p=list(_p1.values())

    acc1=accuracy_score(p, y_train)
    
    #the second node
    if len(y10)/len(p0)>= 0.05:
        x0=x_train[p0,:]
        y0=y_train[p0]

        
        a2, b2, _p2, p02, p12, y012, y102=LBC(x0,y0, cost=cost, t=0.5*t, max_depth=1, N=N,alpha = alpha, f1=f1)


        p=list(_p2.values())
        for i in range(len(p)):
            if p[i]<=0.1:
                p[i]=0
            else:
                p[i]=1

       
        
        d2=1
    else:
        a2=[]
        b2=0
        print('the second node does not exist')
        d2=0
        
    #the third node

    if (len(y01)/len(p1)>= 0.05) and (len(y01)>=10):
        d3=1
        x1=x_train[p1,:]
        y1=y_train[p1]
        
        

        a3, b3, _p3, p03, p13, y013, y103=LBC(x1,y1, cost=cost, t=0.5*t, max_depth=1, N=N,alpha = alpha, f1=f1)

        p=list(_p3.values())
        for i in range(len(p)):
            if p[i]<=0.1:
                p[i]=0
            else:
                p[i]=1

        
       

    else:
        print('the third node does not exist')
        d3=0
        a3=[]
        b3=0
    
        
        
    
    return a1, a2, a3, b1, b2, b3, d1, d2,d3
        
    
        


def predict(a1, a2, a3, b1, b2, b3, d1, d2, d3,  x):
    n, p=x.shape
    y_pred = []
    mu=0.0002
    
    if (d2==0) and (d3==0):
        for xi in x:
            if (sum([a1[j] * xi[j] for j in range(p)]) >= b1):
                y_pred.append(1)
            
            else:
                y_pred.append(0)
                
        return y_pred
    
    elif (d2==1) and (d3==0):
        for xi in x:
            if (sum([a1[j] * xi[j] for j in range(p)]) +mu <= b1) and (sum([a2[j] * xi[j] for j in range(p)]) +mu <= b2):
                y_pred.append(0)
            else:
                y_pred.append(1)
    
    elif (d2==0) and (d3==1):
        for xi in x:
            if (sum([a1[j] * xi[j] for j in range(p)]) >= b1) and (sum([a3[j] * xi[j] for j in range(p)]) >= b3):
                y_pred.append(1)
            else:
                y_pred.append(0)
        
    elif (d2==1) and (d3==1):
        for xi in x:
            if (sum([a1[j] * xi[j] for j in range(p)]) >= b1) and (sum([a2[j] * xi[j] for j in range(p)]) >= b2):
                y_pred.append(1)
            elif (sum([a1[j] * xi[j] for j in range(p)]) +mu <= b1) and (sum([a3[j] * xi[j] for j in range(p)]) + mu <= b3):
                y_pred.append(1)
            
            else:
                y_pred.append(0)
    
    return y_pred





train_ratio = 0.5

val_ratio = 0.25

test_ratio = 0.25

seed=1



N_set=[1,2,3,4,5]

datasett=['Australia','bene1','bene2','german','give','taiwan','thomas','UK']

#
dataset1=['Australia','german','thomas','bene1']

#
dataset2=['bene2']

#
dataset3=['taiwan','UK','give']



res_sk = pd.DataFrame(columns=['Data', 'N','alph','cost','acc_train', 
       'acc_test', 'acc_val', 'Precision', 'Recall', 'AUC', 'F1', 'train_time','a','b','d'])     


dataset1=['bene2']
datasetUK=['UK']

alp=[0.001]



for da in dataset1:
        for N in N_set:
            for al in alp:
                x, y = dataset.loadData(da)
                
                cs=len(y)/sum(y)
                
                
                scales = np.max(x, axis=0)
                x=x/scales
                x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1-train_ratio, random_state=seed)
                x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, 
                                                                test_size=test_ratio/(test_ratio+val_ratio), random_state=seed)
                timel=1200
                tick = time.time()
                a1, a2, a3, b1, b2, b3, d1, d2,d3 =depth2(x_train, y_train, cost=cs, t=timel,  N=N, alpha = al,f1=0.1)
                tock = time.time()
                
                if d3>=0.1 and d2>=0.1:
                    
                    y_train_pre=predict(a1, a2, a3, b1, b2, b3, d1, d2, d3,  x_train)

                    acc_train=accuracy_score(y_train, y_train_pre)

                    y_val_pre=predict(a1, a2, a3, b1, b2, b3, d1, d2, d3,  x_val)

                    acc_val=accuracy_score(y_val, y_val_pre)

                    y_test_pre=predict(a1, a2, a3, b1, b2, b3, d1, d2, d3,  x_test)

                    acc_test = accuracy_score(y_test, y_test_pre)
                    
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
                    
                    a = {}
                    a.update(a1)
                    a.update(a2)
                    a.update(a3)

                    b={1:b1, 2:b2, 3:b3}

                    d={1:d1, 2:d2, 3:d3}
                    
                    
                        
                    
                    row = {'Data':da, 'N':N,'alph':al,'cost':tcost,'acc_train':acc_train, 
                            'acc_test':acc_test, 'acc_val':acc_val, 'Precision': precision, 'Recall':recall,
                            'AUC':auc_linear, 'F1':f1, 'train_time':train_time,'a':a,
                            'b':b,'d':d}
                    res_sk = res_sk.append(row, ignore_index=True)
                    res_sk.to_csv('ben2_3.csv', index=False)
                    
                    
                    d3=0
                    d2=0
                    y_train_pre=predict(a1, a2, a3, b1, b2, b3, d1, d2, d3,  x_train)

                    acc_train=accuracy_score(y_train, y_train_pre)

                    y_val_pre=predict(a1, a2, a3, b1, b2, b3, d1, d2, d3,  x_val)

                    acc_val=accuracy_score(y_val, y_val_pre)

                    y_test_pre=predict(a1, a2, a3, b1, b2, b3, d1, d2, d3,  x_test)

                    acc_test = accuracy_score(y_test, y_test_pre)
                    
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
                    
                    a = {}
                    a.update(a1)
                    a.update(a2)
                    a.update(a3)

                    b={1:b1, 2:b2, 3:b3}

                    d={1:d1, 2:d2, 3:d3}
                    
                    
                        
                    
                    row = {'Data':da, 'N':N,'alph':al,'cost':tcost,'acc_train':acc_train, 
                            'acc_test':acc_test, 'acc_val':acc_val, 'Precision': precision, 'Recall':recall,
                            'AUC':auc_linear, 'F1':f1, 'train_time':train_time,'a':a,
                            'b':b,'d':d}
                    res_sk = res_sk.append(row, ignore_index=True)
                    res_sk.to_csv('ben2_31.csv', index=False)
                    
                    
                    d2=1
                    d3=0
                    
                    y_train_pre=predict(a1, a2, a3, b1, b2, b3, d1, d2, d3,  x_train)

                    acc_train=accuracy_score(y_train, y_train_pre)

                    y_val_pre=predict(a1, a2, a3, b1, b2, b3, d1, d2, d3,  x_val)

                    acc_val=accuracy_score(y_val, y_val_pre)

                    y_test_pre=predict(a1, a2, a3, b1, b2, b3, d1, d2, d3,  x_test)

                    acc_test = accuracy_score(y_test, y_test_pre)
                    
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
                    
                    a = {}
                    a.update(a1)
                    a.update(a2)
                    a.update(a3)

                    b={1:b1, 2:b2, 3:b3}

                    d={1:d1, 2:d2, 3:d3}
                    
                    
                        
                    
                    row = {'Data':da, 'N':N,'alph':al,'cost':tcost,'acc_train':acc_train, 
                            'acc_test':acc_test, 'acc_val':acc_val, 'Precision': precision, 'Recall':recall,
                            'AUC':auc_linear, 'F1':f1, 'train_time':train_time,'a':a,
                            'b':b,'d':d}
                    res_sk = res_sk.append(row, ignore_index=True)
                    res_sk.to_csv('ben2_32.csv', index=False)
                    
                    d2=0
                    d3=1
                    y_train_pre=predict(a1, a2, a3, b1, b2, b3, d1, d2, d3,  x_train)

                    acc_train=accuracy_score(y_train, y_train_pre)

                    y_val_pre=predict(a1, a2, a3, b1, b2, b3, d1, d2, d3,  x_val)

                    acc_val=accuracy_score(y_val, y_val_pre)

                    y_test_pre=predict(a1, a2, a3, b1, b2, b3, d1, d2, d3,  x_test)

                    acc_test = accuracy_score(y_test, y_test_pre)
                    
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
                    
                    a = {}
                    a.update(a1)
                    a.update(a2)
                    a.update(a3)

                    b={1:b1, 2:b2, 3:b3}

                    d={1:d1, 2:d2, 3:d3}
                    
                    
                        
                    
                    row = {'Data':da, 'N':N,'alph':al,'cost':tcost,'acc_train':acc_train, 
                            'acc_test':acc_test, 'acc_val':acc_val, 'Precision': precision, 'Recall':recall,
                            'AUC':auc_linear, 'F1':f1, 'train_time':train_time,'a':a,
                            'b':b,'d':d}
                    res_sk = res_sk.append(row, ignore_index=True)
                    res_sk.to_csv('ben2_33.csv', index=False)
                
                elif d3==0 and d2>=0.1:
                    
                    y_train_pre=predict(a1, a2, a3, b1, b2, b3, d1, d2, d3,  x_train)

                    acc_train=accuracy_score(y_train, y_train_pre)

                    y_val_pre=predict(a1, a2, a3, b1, b2, b3, d1, d2, d3,  x_val)

                    acc_val=accuracy_score(y_val, y_val_pre)

                    y_test_pre=predict(a1, a2, a3, b1, b2, b3, d1, d2, d3,  x_test)

                    acc_test = accuracy_score(y_test, y_test_pre)
                    
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
                    
                    a = {}
                    a.update(a1)
                    a.update(a2)
                    a.update(a3)

                    b={1:b1, 2:b2, 3:b3}

                    d={1:d1, 2:d2, 3:d3}
                    
                    
                        
                    
                    row = {'Data':da, 'N':N,'alph':al,'cost':tcost,'acc_train':acc_train, 
                            'acc_test':acc_test, 'acc_val':acc_val, 'Precision': precision, 'Recall':recall,
                            'AUC':auc_linear, 'F1':f1, 'train_time':train_time,'a':a,
                            'b':b,'d':d}
                    res_sk = res_sk.append(row, ignore_index=True)
                    res_sk.to_csv('ben2_32.csv', index=False)
                    
                    
                    d3=0
                    d2=0
                    y_train_pre=predict(a1, a2, a3, b1, b2, b3, d1, d2, d3,  x_train)

                    acc_train=accuracy_score(y_train, y_train_pre)

                    y_val_pre=predict(a1, a2, a3, b1, b2, b3, d1, d2, d3,  x_val)

                    acc_val=accuracy_score(y_val, y_val_pre)

                    y_test_pre=predict(a1, a2, a3, b1, b2, b3, d1, d2, d3,  x_test)

                    acc_test = accuracy_score(y_test, y_test_pre)
                    
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
                    
                    a = {}
                    a.update(a1)
                    a.update(a2)
                    a.update(a3)

                    b={1:b1, 2:b2, 3:b3}

                    d={1:d1, 2:d2, 3:d3}
                    
                    
                        
                    
                    row = {'Data':da, 'N':N,'alph':al,'cost':tcost,'acc_train':acc_train, 
                            'acc_test':acc_test, 'acc_val':acc_val, 'Precision': precision, 'Recall':recall,
                            'AUC':auc_linear, 'F1':f1, 'train_time':train_time,'a':a,
                            'b':b,'d':d}
                    res_sk = res_sk.append(row, ignore_index=True)
                    res_sk.to_csv('ben2_31.csv', index=False)
                
                elif d3>=0.1 and d2==0:
                    
                    y_train_pre=predict(a1, a2, a3, b1, b2, b3, d1, d2, d3,  x_train)

                    acc_train=accuracy_score(y_train, y_train_pre)

                    y_val_pre=predict(a1, a2, a3, b1, b2, b3, d1, d2, d3,  x_val)

                    acc_val=accuracy_score(y_val, y_val_pre)

                    y_test_pre=predict(a1, a2, a3, b1, b2, b3, d1, d2, d3,  x_test)

                    acc_test = accuracy_score(y_test, y_test_pre)
                    
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
                    
                    a = {}
                    a.update(a1)
                    a.update(a2)
                    a.update(a3)

                    b={1:b1, 2:b2, 3:b3}

                    d={1:d1, 2:d2, 3:d3}
                    
                    
                        
                    
                    row = {'Data':da, 'N':N,'alph':al,'cost':tcost,'acc_train':acc_train, 
                            'acc_test':acc_test, 'acc_val':acc_val, 'Precision': precision, 'Recall':recall,
                            'AUC':auc_linear, 'F1':f1, 'train_time':train_time,'a':a,
                            'b':b,'d':d}
                    res_sk = res_sk.append(row, ignore_index=True)
                    res_sk.to_csv('ben2_33.csv', index=False)
                    
                    
                    d3=0
                    d2=0
                    y_train_pre=predict(a1, a2, a3, b1, b2, b3, d1, d2, d3,  x_train)

                    acc_train=accuracy_score(y_train, y_train_pre)

                    y_val_pre=predict(a1, a2, a3, b1, b2, b3, d1, d2, d3,  x_val)

                    acc_val=accuracy_score(y_val, y_val_pre)

                    y_test_pre=predict(a1, a2, a3, b1, b2, b3, d1, d2, d3,  x_test)

                    acc_test = accuracy_score(y_test, y_test_pre)
                    
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
                    
                    a = {}
                    a.update(a1)
                    a.update(a2)
                    a.update(a3)

                    b={1:b1, 2:b2, 3:b3}

                    d={1:d1, 2:d2, 3:d3}
                    
                    
                        
                    
                    row = {'Data':da, 'N':N,'alph':al,'cost':tcost,'acc_train':acc_train, 
                            'acc_test':acc_test, 'acc_val':acc_val, 'Precision': precision, 'Recall':recall,
                            'AUC':auc_linear, 'F1':f1, 'train_time':train_time,'a':a,
                            'b':b,'d':d}
                    res_sk = res_sk.append(row, ignore_index=True)
                    res_sk.to_csv('ben2_31.csv', index=False)
                


                

             
                
                