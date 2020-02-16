# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 16:09:52 2020

@author: ENVY13
"""

#import numpy as np
from math import log
import treePlotter

def calcinfoent(dataset):
    num=len(dataset)
    labelcounts={}
    for i in dataset:
        label=i[-1]
        if label not in labelcounts.keys():
            labelcounts[label]=0
        labelcounts[label]+=1
    infoent=0.0
    for key in labelcounts:
        p=float(labelcounts[key])/num
        infoent-=p*log(p,2)
    return infoent

def createDataset():
    dataset=[[1,1,'yes'],[1,1,'yes'],[1,0,'no'],[0,1,'no'],[0,1,'no']]
    labels=['no surfacing','flippers']
    return dataset,labels

def splitDataset(dataset,axis,value):
    retDataset=[]
    for featVec in dataset:
        if featVec[axis]==value:
            rf=featVec[:axis]
            rf.extend(featVec[axis+1:])
            retDataset.append(rf)
    return retDataset

def bestFeatSplit(dataset):    #选出用于划分数据集的最佳特征
    numfeat=len(dataset[0])-1
    baseEnt=calcinfoent(dataset)
    bestinfoGain=0.0
    bestfeature=-1
    for i in range(numfeat):
        featlist=[example[i] for example in dataset]
        uniqueVals=set(featlist)
        newEnt=0.0
        for value in uniqueVals:
            subDataset=splitDataset(dataset,i,value)
            p=len(subDataset)/len(dataset)
            newEnt += p*calcinfoent(subDataset)
        infoGain=baseEnt-newEnt
        if infoGain>bestinfoGain:
            bestinfoGain=infoGain
            bestfeature=i
    return bestfeature

def majorcnt(classList):    #一个叶子节点中存在多个类时，选择出现次数最多的类作为该叶子的类别
    classCount={}
    for i in classList:
        if i not in classCount.keys():
            classCount[i]=0
            classCount[i]+=1
    majorclass=max(zip(classCount.values(),classCount.keys()))
    return majorclass[1]

def createTree(dataset,labels):
    classList=[example[-1] for example in dataset]
    if classList.count(classList[0])==len(classList):
        return classList[0]
    if len(dataset[0])==1:
        return majorcnt(classList)
    bestFeat=bestFeatSplit(dataset)
    bestFeatLabel=labels[bestFeat]
    myTree={bestFeatLabel:{}}
    del(labels[bestFeat])
    featVals=[x[bestFeat] for x in dataset]
    uniqueVals=set(featVals)
    for value in uniqueVals:
        subLabels=labels[:]
        myTree[bestFeatLabel][value]=createTree(splitDataset(dataset,bestFeat,value),subLabels)
    return myTree
    
def classify(inputTree,featLabels,testVec):  #用于对新样本分类
    firstStr=list(inputTree)[0]
    featVals=inputTree[firstStr]
    featIndex=featLabels.index(firstStr)
    for key in featVals.keys():
        if testVec[featIndex]==key:
            if type(featVals[key]).__name__=='dict':
                classLabel=classify(featVals[key],featLabels,testVec)
            else:
                classLabel=featVals[key]
    return classLabel
            
def storeTree(inputTree,filename):  #用pickle存储树，需要时自行调用
    import pickle
    fw=open(filename,'wb')
    pickle.dump(inputTree,fw)
    fw.close()
    
def grabTree(filename):  #读取树，需要时自行调用
    import pickle
    fr=open(filename,'rb')
    return pickle.load(fr)

def train(filename):  #此处没有划分训练集，测试集
    fr=open(filename,'r')
    X=[x.strip().split('\t') for x in fr.readlines()]
    labels=[]  #此处要自己填入特征标签
    myTree=createTree(X,labels)
    treePlotter.createPlot(myTree)
    return myTree

labels=[]  #此处重复一次labels，否则有bug

myTree=train("filename")

            