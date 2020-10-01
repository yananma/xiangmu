import numpy as np
import re 
import random


def textParse(input_string):
    listofTokens = re.split(r'\W+',input_string)
    return [tok.lower() for tok in listofTokens if len(listofTokens)>2]

def creatVocablist(doclist):
    vocabSet = set([])
    for document in doclist:
        vocabSet = vocabSet|set(document)
    return list(vocabSet)
def setOfWord2Vec(vocablist,inputSet):
    returnVec  = [0]*len(vocablist)
    for word in inputSet:
        if word in vocablist:
            returnVec[vocablist.index(word)] = 1
    return returnVec
    
def trainNB(trainMat,trainClass):
    numTrainDocs = len(trainMat)
    numWords = len(trainMat[0])
    p1 = sum(trainClass)/float(numTrainDocs)
    p0Num = np.ones((numWords)) #做了一个平滑处理
    p1Num = np.ones((numWords)) #拉普拉斯平滑
    p0Denom = 2
    p1Denom = 2 #通常情况下都是设置成类别个数
    
    for i in range(numTrainDocs):
        if trainClass[i] == 1: #垃圾邮件
            p1Num += trainMat[i]
            p1Denom += sum(trainMat[i])
        else:
            p0Num += trainMat[i]
            p0Denom += sum(trainMat[i])
    p1Vec = np.log(p1Num/p1Denom)
    p0Vec = np.log(p0Num/p0Denom)
    return p0Vec,p1Vec,p1
    
def classifyNB(wordVec,p0Vec,p1Vec,p1_class):    
    p1 = np.log(p1_class) + sum(wordVec*p1Vec)
    p0 = np.log(1.0 - p1_class) + sum(wordVec*p0Vec)
    if p0>p1:
        return 0
    else:
        return 1
    
       


def spam():
    doclist = []
    classlist = []
    for i in range(1,26):
        wordlist = textParse(open('email/spam/%d.txt'%i,'r').read())
        doclist.append(wordlist)
        classlist.append(1) #1表示垃圾邮件
        
        wordlist = textParse(open('email/ham/%d.txt'%i,'r').read())
        doclist.append(wordlist)
        classlist.append(0) #1表示垃圾邮件
        
    vocablist = creatVocablist(doclist)
    trainSet = list(range(50))
    testSet = []
    for i in range(10):
        randIndex = int(random.uniform(0,len(trainSet)))
        testSet.append(trainSet[randIndex])
        del (trainSet[randIndex])
    trainMat = []
    trainClass = []
    for docIndex in trainSet:
        trainMat.append(setOfWord2Vec(vocablist,doclist[docIndex]))
        trainClass.append(classlist[docIndex])
    p0Vec,p1Vec,p1 = trainNB(np.array(trainMat),np.array(trainClass))
    errorCount = 0
    for docIndex in testSet:
        wordVec = setOfWord2Vec(vocablist,doclist[docIndex])
        if classifyNB(np.array(wordVec),p0Vec,p1Vec,p1) != classlist[docIndex]:
            errorCount+=1
    print ('当前10个测试样本，错了：',errorCount) 

if __name__ == '__main__':
    spam()
        
    
    
    
    
    
    
    
    
    
    
        
