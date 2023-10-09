# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 11:11:20 2023

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
import data




from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier



def LBC(x, y, cost=1, t=120, max_depth=1, N=2,alpha = 0.00,f1=0.1):
    n, p=x.shape
    
    weights = {0:1, 1:cost}
    
    a_start=np.zeros(p)
    b_start=0
    yp_start=np.zeros(n)
    
    clf = DecisionTreeClassifier(max_depth=1,class_weight=weights)
    clf.fit(x,y)
    # 获取决策树的分类规则
    decision_rules = clf.tree_

    # 获取根节点处的分类规则
    af= decision_rules.feature[0]
    b1= decision_rules.threshold[0]
        

    # 获取决策树的每个叶节点对应的标签
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
    # m.Params.NoRelHeurTime=10
    # #m.Params.NumericFocus =1
    # m.Params.FeasibilityTol=1e-5
    # m.Params.IntFeasTol=1e-5
    # m.Params.IntegralityFocus=1
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
    #m.addConstr(f1*(2*TP+FP+FN)<=2*TP)
    
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
    y_pre = {ind:y_pre[ind].x for ind in y_pre}
    _b = b.x
    
    return _a, _b, y_pre


def predict(a,b,  x, y):
    n, p=x.shape
    y_pred = []
    mu=0.0002
    
    for xi in x:
        if (sum([a[j] * xi[j] for j in range(p)]) >= b):
            y_pred.append(1)
        
        else:
            y_pred.append(0)
            
    return y_pred
    
    