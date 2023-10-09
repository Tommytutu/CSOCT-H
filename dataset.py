# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 21:48:53 2023

@author: JC TU
"""



import pandas as pd
import numpy as np



def loadData(dataname):
    """
    load training and testing data from different dataset
    """
    if dataname=='AER':
        x,y=loadAER()
        return x,y
    
    if dataname=='Australia':
        x,y=loadAustralia()
        return x,y
    
    if dataname=='bene1':
        x,y=loadbene1()
        return x,y
    
    if dataname=='bene2':
        x,y=loadbene2()
        return x,y
    
    if dataname=='german':
        x,y=loadgerman()
        return x,y
    
    if dataname=='give':
        x,y=loadGMC()
        return x,y
    
    if dataname=='hmeq':
        x,y=loadhmeq()
        return x,y
    
    if dataname=='polish':
        x,y=loadpolish()
        return x,y
    
    
    if dataname=='taiwan':
        x,y=loadtaiwan()
        return x,y
    
    
    if dataname=='thomas':
        x,y=loadthomas()
        return x,y
    
    if dataname=='UK':
        x,y=loadUK()
        return x,y
    
    
   
    raise NameError('No dataset "{}".'.format(dataname))

def loadAER():
    df=pd.read_excel(r'data\AER.xlsx')
    x=df.drop('BAD',axis=1)
    y=df['BAD']
    x = np.array(x.values)
    y = np.array(y.values)
    return x,y


def loadAustralia():
    df=pd.read_excel(r'data\Australia.xlsx')
    x=df.drop('BAD',axis=1)
    y=df['BAD']
    x = np.array(x.values)
    y = np.array(y.values)
    return x,y

def loadbene1():
    df=pd.read_excel(r'data\bene1.xlsx')
    x=df.drop('BAD',axis=1)
    y=df['BAD']
    x = np.array(x.values)
    y = np.array(y.values)
    return x,y


def loadbene2():
    df=pd.read_excel(r'data\bene2.xlsx')
    x=df.drop('BAD',axis=1)
    y=df['BAD']
    x = np.array(x.values)
    y = np.array(y.values)
    return x,y

def loadgerman():
    df=pd.read_excel(r'data\german.xlsx')
    x=df.drop('BAD',axis=1)
    y=df['BAD']
    x = np.array(x.values)
    y = np.array(y.values)
    return x,y


def loadGMC():
    df=pd.read_excel(r'data\GiveMeSomeCredit.xlsx')
    x=df.drop('BAD',axis=1)
    y=df['BAD']
    x = np.array(x.values)
    y = np.array(y.values)
    return x,y

def loadhmeq():
    df=pd.read_excel(r'data\hmeq_numerical.xlsx')
    x=df.drop('BAD',axis=1)
    y=df['BAD']
    x = np.array(x.values)
    y = np.array(y.values)
    return x,y

def loadpolish():
    df=pd.read_excel(r'data\Polish_5year.xlsx')
    x=df.drop('BAD',axis=1)
    y=df['BAD']
    x = np.array(x.values)
    y = np.array(y.values)
    return x,y


def loadtaiwan():
    df=pd.read_excel(r'data\Taiwan.xlsx')
    x=df.drop('BAD',axis=1)
    y=df['BAD']
    x = np.array(x.values)
    y = np.array(y.values)
    return x,y


def loadthomas():
    df=pd.read_excel(r'data\thomas.xlsx')
    x=df.drop('BAD',axis=1)
    y=df['BAD']
    x = np.array(x.values)
    y = np.array(y.values)
    return x,y

def loadUK():
    df=pd.read_excel(r'data\UK.xlsx')
    x=df.drop('BAD',axis=1)
    y=df['BAD']
    x = np.array(x.values)
    y = np.array(y.values)
    return x,y
