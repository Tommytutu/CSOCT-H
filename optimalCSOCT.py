# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 22:17:46 2023

@author: JC TU
"""

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





from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier



def d2(x, y, cost=1, t=120, max_depth=2, N=2,alpha =0.001,f1=0.1):
    n,p=x.shape
    labels = np.unique(y)

    n_index = [i+1 for i in range(2 ** (max_depth + 1) - 1)]
    b_index = n_index[:-2**max_depth] # branch nodes
    l_index = n_index[-2**max_depth:] # leaf nodes

    #
    def get_l(t):
        lls=[]
        lrs=[]
        left=(t % 2 == 0)
        right=(t % 2 == 1)
        
        if t>=2 and left:
            while (t % 2 == 0):
                 lls.append(t)
                 t=t//2
            lls.append(t)
            lls.pop(0)
        
        if t>=3 and right:
            while (t % 2 == 1) and (t>=3):
                 lrs.append(t)
                 t=t//2
            lrs.append(t)
            lrs.pop(0)
        
        if left:
            return lls
        else:
            return lrs

    #
    def getd(t,d):
        lls=[]
        lrs=[]
        left=(t % 2 == 0)
        right=(t % 2 == 1)
        
        if t>=2 and left:
            while (t % 2 == 0):
                if d[t]>=0.1:
                    lls.append(t)
                    t=t//2
            lls.pop(0)
        
        if t>=3 and right:
            while (t % 2 == 1) and (t>=3):
                if d[t]>=0.1:
                    lrs.append(t)
                    t=t//2
            lrs.pop(0)
        
        return lls, lrs
    
    weights = {0:1, 1:cost}
    
    a_start=np.zeros(p)
    b_start=0
    yp_start=np.zeros(n)
    
    clf=DecisionTreeClassifier(max_depth=1, class_weight=weights)
    clf.fit(x,y)
    # 
    decision_rules = clf.tree_

    # 
    af= decision_rules.feature[0]
    b1= decision_rules.threshold[0]
        

    # 
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
    m1 = gp.Model('m')

    # output

    # time limit
    
    
    timelimit=0.1*t
    m1.Params.timelimit = timelimit
    m1.Params.Heuristic = 0.2
    m1.Params.NoRelHeurTime=10
    #m.Params.NumericFocus =1
    
    m1.Params.IntegralityFocus=1
    #m.params.Method=2
    m1.params.DegenMoves=0
    # parallel
    #m.params.threads = 0
    #m.params.Presolve=2
    #m.params.MIPFocus=1
    #m.params.Cuts=3
    m1.params.ImproveStartTime=0.8*timelimit


    # model sense
    m1.modelSense = GRB.MINIMIZE

    # variables
    a = m1.addVars(p,  lb=-1, ub=1, vtype=GRB.CONTINUOUS, name='a') # splitting feature
    bara = m1.addVars(p, lb=0, ub=1, vtype=GRB.CONTINUOUS, name='bara') # splitting feature
    s = m1.addVars(p,  vtype=GRB.BINARY, name='s') # splitting feature
    b = m1.addVar(lb=-1, ub=1, vtype=GRB.CONTINUOUS, name='b') # splitting threshold
    y_pre = m1.addVars(n, vtype=GRB.BINARY, name='y_pre') # leaf node assignment

    #e = m.addVar(lb=0.01, ub=0.05, vtype=GRB.CONTINUOUS, name='e') # splitting threshold
    e=0.0005



    # objective function


    e1 = gp.quicksum((y[i] == 0) * y_pre[i] for i in range(n))/n

    e2 = cost*gp.quicksum((y[i] == 1) * (1 - y_pre[i]) for i in range(n))/n

    obj = e1 + e2 + alpha * gp.quicksum(s[j] for j in range(p))
    m1.setObjective(obj)
    
    TP=gp.quicksum((y[i] == 1) * y_pre[i] for i in range(n))
    TN=gp.quicksum((y[i] == 0) * (1-y_pre[i]) for i in range(n))
    FP=gp.quicksum((y[i] == 0) * (y_pre[i]) for i in range(n))
    FN=gp.quicksum((y[i] == 1) * (1-y_pre[i]) for i in range(n))
                   

    #m.addConstr(b==0)
    m1.addConstr(bara.sum()<= 1)


    m1.addConstrs(bara[j] >= a[j]  for j  in range(p))

    m1.addConstrs(bara[j] >= -a[j]  for j  in range(p))

    m1.addConstrs(s[j] >=a[j] for j  in range(p))

    m1.addConstrs(-s[j] <=a[j]  for j  in range(p))

    m1.addConstr(s.sum() >=1)

    #
    m1.addConstr(s.sum('*')<=N)


    m1.addConstrs(gp.quicksum(a[j] * x[i,j] for j in range(p)) >=b - 2*(1 - y_pre[i]) for i in range(n))

            
    m1.addConstrs(gp.quicksum(a[j] * x[i,j] for j in range(p)) +e<= b+ 2*y_pre[i] for i in range(n))


    #m.addConstr(gp.quicksum((y[i] == 0) * y_pre[i] for i in range(n))/ sum((y[i] == 0)  for i in range(n)) <=0.3)

    #m.addConstr(gp.quicksum((y[i] == 0) * y_pre[i] for i in range(n)) <=0.3 * gp.quicksum(((1-y_pre[i]) for i in range(n))))
    
    m1.addConstr(gp.quicksum(y_pre[i] for i in range(n)) >=5)
    
    m1.addConstr(gp.quicksum((1-y_pre[i]) for i in range(n)) >=5)
    
    
    # F1_score
    m1.addConstr(f1*(2*TP+FP+FN)<=2*TP)
    
    

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


    m1.optimize()

    _a = {ind:a[ind].x for ind in a}
    _b = b.x
    _y=  {ind:y_pre[ind].x for ind in y_pre}
        

    # create a model
    m = gp.Model('m')

    # output

    # time limit
    timelimit=t
    m.Params.timelimit = timelimit
    m.Params.Heuristic = 0.2
    m.Params.NoRelHeurTime=10
    #m.Params.NumericFocus =1
    m.Params.IntegralityFocus=1
    #m.params.Method=2
    m.params.DegenMoves=0
    # parallel




    # model sense
    m.modelSense = GRB.MINIMIZE

    # variables
    a = m.addVars(p, b_index, lb=-1, ub=1, vtype=GRB.CONTINUOUS, name='a') # splitting feature
    bara = m.addVars(p, b_index, lb=0, ub=1, vtype=GRB.CONTINUOUS, name='bara') # splitting feature
    s = m.addVars(p, b_index, vtype=GRB.BINARY, name='s') # splitting feature
    b = m.addVars(b_index, lb=-1, ub=1, vtype=GRB.CONTINUOUS, name='b') # splitting threshold
    c = m.addVars(labels, l_index, vtype=GRB.BINARY, name='c') # node prediction
    d = m.addVars(b_index, vtype=GRB.BINARY, name='d') # splitting option
    l = m.addVars(l_index, vtype=GRB.BINARY, name='l') # leaf node activation
    z = m.addVars(n, l_index, vtype=GRB.BINARY, name='z') # leaf node assignment
    zc= m.addVars(n,labels, lb=0, ub=1, vtype=GRB.BINARY, name='zc') # leaf node activation


    mu=0.0005;


    # objective function
    error=gp.quicksum((y[i] != k) * zc[i,k] for i in range(n) for k in labels)/n 
    obj = error + alpha * gp.quicksum(s[j,t] for j in range(p) for t in b_index)
    m.setObjective(obj)


    m.addConstr(d[1]==1)
    #m.addConstrs(l[t]==1 for t in l_index)

    m.addConstrs(bara.sum('*', t) <= d[t] for t in b_index)

    m.addConstrs(bara[j,t] >= a[j,t] for t in b_index for j  in range(p))

    m.addConstrs(bara[j,t] >= -a[j,t] for t in b_index for j  in range(p))

    m.addConstrs(s[j,t] >=a[j,t] for t in b_index for j  in range(p))

    m.addConstrs(-s[j,t] <=a[j,t] for t in b_index for j  in range(p))

    m.addConstrs(s[j,t] <=d[t] for t in b_index for j  in range(p))

    m.addConstrs(s.sum('*',t) >=d[t] for t in b_index)

    #
    m.addConstrs(s.sum('*',t)<=N for t in b_index)

    m.addConstrs(b[t]<=d[t] for t in b_index)

    m.addConstrs(b[t]>=-d[t] for t in b_index)

    # (4)
    m.addConstrs(d[t] <= d[t//2] for t in b_index if t != 1)

    m.addConstrs(c.sum('*', t) == l[t] for t in l_index)

    m.addConstrs(z.sum(i,'*') == 1 for i in range(n))
    # (7)
    m.addConstrs(z[i,t] <= l[t] for t in l_index for i in range(n))
    # (8)
    m.addConstrs(z.sum('*', t) >= 5 * l[t] for t in l_index)
    
    # (9) and (10)
    for t in l_index:
        left = (t % 2 == 0)
        ta = t // 2
        while ta != 0:
            if left:
                m.addConstrs(gp.quicksum(a[j,ta] * x[i,j] for j in range(p))
                             <=b[ta] + 2*(1 - z[i,t]) - mu* d[ta]
                             for i in range(n))
            else:
                m.addConstrs(gp.quicksum(a[j,ta] * x[i,j] for j in range(p))
                             >=b[ta]
                               - 2*(1 - z[i,t])
                             for i in range(n))
            left = (ta % 2 == 0)
            ta //= 2




    m.addConstrs(z[i,t]+c[k,t]-1<=zc[i,k] for i in range(n) for t in l_index for k in labels)

    m.addConstrs(gp.quicksum(zc[i,k]  for k in labels) ==1 for i in range(n))
        
    m.addConstrs(l[t]<=gp.quicksum(d[s] for s in get_l(t)) for t in l_index)

    m.addConstrs(max_depth*l[t]>=gp.quicksum(d[s] for s in get_l(t)) for t in l_index)
    
    
    for j in range(p):
        a[j,1].start=_a[j]
        a[j,2].start=0
        a[j,3].start=0
        
    for t in b_index:
        b[1].start=_b
        b[2].start=0
        b[3].start=0
        
    for i in range(n):
        if _y[i]==1:
            z[i,7].start = 1
            zc[i,1].start =1
        else:
            z[i,4].start = 1
            zc[i,0].start =1


    
    m.optimize()
    
   

    _a = {ind:a[ind].x for ind in a}
    _b = {ind:b[ind].x for ind in b}
    _c = {ind:c[ind].x for ind in c}
    _d = {ind:d[ind].x for ind in d}
    
    return _a, _b, _c, _d

#提取干结点t对应的叶节点




def predict(x, a, b, c, d):
    n,p=x.shape
    labels = np.array([0, 1])
    max_depth=2
    n_index = [i+1 for i in range(2 ** (max_depth + 1) - 1)]
    b_index = n_index[:-2**max_depth] # branch nodes
    l_index = n_index[-2**max_depth:] # leaf nodes
    
    def getleaf(tb):
        t=tb
        if tb % 2 == 1:
        
            t=2*t+1
        else:
          
                t=2*t
        
        return t
    
    labelmap = {}
    for t in l_index:
        for k in labels:
            if c[k,t] >= 1e-2:
                labelmap[t] = k

    y_pred = []
    for xi in x:
        t = 1
        while t not in l_index:
            right = (sum([a[j,t] * xi[j] for j in range(p)]) >= b[t])
            if d[t] >= 0.1:
                if right:
                   t = 2 * t + 1
                   
                else:
                     t = 2 * t
                 
            else:
                   t = getleaf(t)
        
        y_pred.append(labelmap[t])
    
    return y_pred
    


