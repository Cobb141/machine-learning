# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 09:20:36 2020

@author: ENVY13
"""

import numpy as np
#from sklearn import datasets
#from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import ShuffleSplit

def fileload(filename):
    fr = open(filename,'r')
    ar=fr.readlines()
    m = len(ar)         #get the number of lines in the file
    aline=ar[0]
    aline=aline.strip().split('\t')
    n=len(aline)-1
    returnMat = np.zeros((m,n))        #prepare matrix to return
    classLabelVector = np.zeros(m)                     #prepare labels return   
    index = 0
    for line in ar:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:n]
        classLabelVector[index]=(listFromLine[-1])
        index += 1
    fr.close()
    return returnMat,classLabelVector

def norm(X):
    minvals=X.min(0)
    maxvals=X.max(0)
    u=X.mean(0)
    X_norm=(X-u)/(maxvals-minvals)
    return X_norm

def classify0(inx,dataset,labels,k):
    dist=(np.sum((dataset-inx)**2,axis=1))**0.5
    sortdist=dist.argsort()
    klabels=[]
    for i in sortdist.flat:
        klabels.append(labels[i])
    klabels=klabels[0:k]
    count_dict = {}
    for label in set(klabels):
        count_dict[klabels.count(label)]=label
    a=list(reversed(sorted(count_dict.keys())))
    return count_dict[a[0]]

def test():
    filename=input("输入数据所在的文本文件的路径：")
    X,y=fileload(filename)
    X=norm(X)
    rs=ShuffleSplit(n_splits=3,test_size=.3,random_state=0)  #划分测试集，训练集
    rs.get_n_splits(X)
    for train_index,test_index in rs.split(X,y):    
        X_train,X_test=X[train_index],X[test_index]
        y_train,y_test=y[train_index],y[test_index]
    group,labels=X_train,y_train
    m=np.shape(X_test)[0]
    result=[]
    for i in range(m):
        result.append(classify0(X_test[i],group,labels,50))
    error=0
    print("分类错误样本：")
    for i in range(45):
        if result[i]!=y_test[i]:
            error+=1
            print(X_test[i])
    print("错误率：")
    return error/m

print(test())