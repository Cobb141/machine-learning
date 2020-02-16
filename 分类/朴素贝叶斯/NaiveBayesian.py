# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 20:52:33 2020

@author: ENVY13
"""

import numpy as np

def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                  ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                  ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                  ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                  ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                  ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    labels = [0,1,0,1,0,1]    
    return postingList,labels
                 
def createVocabList(dataSet):
    vocabSet = set([])  
    for document in dataSet:
        vocabSet = vocabSet | set(document)   #取并集
    return list(vocabSet)

def bagOfWordsVec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
        else: 
            print("the word: %s is not in my Vocabulary!" % word)
    return returnVec

def train(trainMat,labels):
    numDocs = len(trainMat)
    numWords = len(trainMat[0])
    pPositive = sum(labels)/float(numDocs)
    p0Num = np.ones(numWords)
    p1Num = np.ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numDocs):
        if labels[i] == 1:
            p1Num += trainMat[i]
            p1Denom += sum(trainMat[i])
        else:
            p0Num += trainMat[i]
            p0Denom += sum(trainMat[i])
    p1Vec = np.log(p1Num/p1Denom)
    p0Vec = np.log(p0Num/p0Denom)
    return p0Vec,p1Vec,pPositive
    
def classify(xVec,p0Vec,p1Vec,pPositive):
    p1 = sum(xVec * p1Vec) + np.log(pPositive)
    #print(p1)
    p0 = sum(xVec * p0Vec) + np.log(1.0 - pPositive)
    #print(p0)
    if p1 > p0:
        return 1
    else:
        return 0

x=[]

def test():
    dataset,labels = loadDataSet()
    myList = createVocabList(dataset)
    trainMat = []
    for i in dataset:
        trainMat.append(bagOfWordsVec(myList, i))
    p0V,p1V,pPositive = train(trainMat,labels)
    xVec = bagOfWordsVec(myList, x)
    result = classify(xVec,p0V,p1V,pPositive)
    return result

def textParse(bigStr):
    reg = ",.;:!@?%"
    for i in reg:
        bigStr = bigStr.replace(i,'')
    tokenList = bigStr.split()
    return [tok.lower() for tok in tokenList if len(tok) > 2]

def spamTest():  #垃圾邮件识别
    docList=[]; classList = []; fullText =[]
    for i in range(1,26):
        wordList = textParse(open('C:\Python专用\《机器学习实战》源代码\machinelearninginaction\Ch04\email\spam\%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('C:\Python专用\《机器学习实战》源代码\machinelearninginaction\Ch04\email\ham\%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    trainingSet = list(range(50)); testSet=[]           
    for i in range(10):
        randIndex = int(np.random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])  
    trainMat=[]
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(bagOfWordsVec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = train(np.array(trainMat),np.array(trainClasses))
    errorCount = 0
    for docIndex in testSet:        
        wordVector = bagOfWordsVec(vocabList, docList[docIndex])
        if classify(np.array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
            print("classification error",docList[docIndex])
    print('the error rate is: ',float(errorCount)/len(testSet))

spamTest()
    
    
    
    
    
    
    
    
    



