from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn import preprocessing
from gensim.models import word2vec
from sklearn.svm import SVC#,LinearSVC,NuSVC
from bs4 import BeautifulSoup
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec
from gensim import utils
from preProcess import *
from gensim.models import word2vec
from sklearn.svm import SVC
import tensorflow
import skflow

import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import gensim
import nltk
import os
import logging
import time

datadir = '/home/kpe/projects/deepLearning/data/'
tokenizer=nltk.data.load('/home/kpe/nltk_data/tokenizers/punkt/english.pickle')

def createDict(trainData):
    trainVarKeys=trainData[0].split(',')
    trainVarKeys[-1]=trainVarKeys[-1].strip()#strip the /n character from the last entry
    trainDict=dict()
    for i in trainVarKeys:
        trainDict[i]=list()
    keyOrder=['ID','Gene','Variation','Class']
    for i in trainData[1:]: #start from index 1, because index 0 is just labels
        splitTrainVar=i.split(',')
        splitTrainVar[-1]=splitTrainVar[-1].strip()
        for j in range(0,len(keyOrder)):
            if j==0 or j==3:
                trainDict[keyOrder[j]].append(int(splitTrainVar[j]))
            else:
                trainDict[keyOrder[j]].append(splitTrainVar[j])
    print trainDict
    return trainDict

"""processing training data 
"""
trainVarnts = open(datadir + 'training_variants_70.csv', 'rb').readlines()
trainText = open(datadir + 'training_text_70.csv', 'rb').readlines()

trainVarDict = createDict(trainVarnts)

trainSentences = []
for text in trainText:
    trainSentences += corpusTxt_to_sentences(text, tokenizer,remove_stopWords=True)

"""processing validation data 
"""

valVarnts = open(datadir + 'training_variants_30.csv', 'rb').readlines()
valText = open(datadir + 'training_text_30.csv', 'rb').readlines()

valVarDict = createDict(valVarnts)
 
valSentences = []
for text in valText:
    valSentences += corpusTxt_to_sentences(text, tokenizer,remove_stopWords=True)

"""
processing test data 
"""
testVarnts = open(datadir + 'test_variants', 'rb').readlines()
testText = open(datadir + 'test_text', 'rb').readlines()
del(testText[0])

for i in range(0, len(testText)):
    numLen = len(str(i))
    testText[i] = testText[i][2+numLen:]


testSentences = []
for text in testText:
    testSentences += corpusTxt_to_sentences(text, tokenizer,remove_stopWords=True)

testVarKeys=testVarnts[0].split(',')
testVarKeys[-1]=testVarKeys[-1].strip()#strip the /n character from the last entry
testVarDict=dict()
for i in testVarKeys:
    testVarDict[i]=list()
keyOrder=['ID','Gene','Variation']
for i in testVarnts[1:]: #start from index 1, because index 0 is just labels
    splitTestVar=i.split(',')
    splitTestVar[-1]=splitTestVar[-1].strip()
    for j in range(0,len(keyOrder)):
        if j==0 or j==3:
            testVarDict[keyOrder[j]].append(int(splitTestVar[j]))
        else:
            testVarDict[keyOrder[j]].append(splitTestVar[j])


print '###################################start word2vec###################################'
def run_word_2_vec(trainSentences, num_workers=4, num_features=300, iter=5, min_word_count=40, context=10, downsampling=1e-3,skipGramVal=0,preLoad = False, modelPath = 'test', fileName = 'dummy'):
    if preLoad == False: 
        startword2vec=time.time()
        train_model=word2vec.Word2Vec(trainSentences,workers=num_workers,size=num_features,min_count=min_word_count,sg=skipGramVal,window=context,sample=downsampling)
#if you don't plan to train the model any further, calling init_sims will make the model much more memory efficient
        train_model.init_sims(replace=True)
        finishword2vec=time.time()
        train_model.save(fileName)
        print 'total time for word2vec: '+str(finishword2vec-startword2vec)+' seconds'
    else:
        train_model = word2vec.Word2Vec.load(modelPath)
    return train_model

def textBlock_To_Vec(txt, model, num_features):
    textBlockVec = norm_vector(txt, model, num_features)
    return textBlockVec


features = 300
trainingModel = run_word_2_vec(trainSentences, fileName= 'word2vecEmbeddingsTrain70BOW.w2v')
valModel = run_word_2_vec(valSentences, fileName = 'word2vecEmbeddingsVal30BOW.w2v')
testModel = run_word_2_vec(testSentences, fileName='word2vecEmbeddingsvalALLBOW.w2v')
 
textBlockTrain = textBlock_To_Vec(trainText,trainingModel, features)
textBlockTest = textBlock_To_Vec(testText, testModel, features)
valBlockTrain = textBlock_To_Vec(valText, valModel, features)


def run_doc_to_vec(sentences, min_count, window, size, sample, negative, workers, iter, seed, preLoad=False, modelPath='test'):
    if preLoad == False:
        startdoc2vec=time.time()
        train_model_doc=doc2Vec.Doc2Vec(min_count=min_count, window=window, size=size, sample=sample, negative=negative, workers=workers,iter=iter,seed=seed)
        train_model_doc.build_vocab(sentences)
        train_model_doc.train(sentences, total_examples=train_model_doc.corpus_count, epochs=100)
        train_model_doc.init_sims(replace=True)
        finishdoc2vec=time.time()
        train_model_doc.save("docEmbeddings.d2v")
        print 'total time for doc2vec: '+str(finishdoc2vec-startdoc2vec)+' seconds'
    else:
        train_model_doc = Doc2Vec.load(modelPath)

    return train_model_doc



def train_model(modelType, textBlockVecs, trainVarDict, textBlockTest, n_classes, valBlockTrain, valVarDict):
    if modelType == 'SVM':
        startsvm=time.time()
        svm_classifier=SVC(kernel='linear', probability=True)
        svm_classifier.fit(textBlockVecs, trainVarDict['Class'])
        endsvm=time.time()
        print "Total time to train SVM was: "+str(endsvm-startsvm)+" seconds"
        probas = svm_classifier.predict_proba(textBlockVecs)
        probas_val = svm_classifier.predict_proba(valBlockTrain)
        probas_test = svm_classifier.predict_proba(textBlockTest)
        print 'Log Loss Training 70: {}'.format(log_loss(trainVarDict['Class'], probas))
        pred_indices = np.argmax(probas, axis=1)
        classes=np.array(range(1,10))
        preds = classes[pred_indices]
        print  'accurcacy Train 30: {}'.format(accuracy_score(classes[np.argmax(probas_val, axis=1)], preds)) 
        print 'log loss train 30: {}'.format(log_loss(valVarDict['Class'], probas_val))
#        print 'multi class log loss train 30: {}'.format(multiclass_log_loss(valVarDict['Class'], probas_val))
#        print 'multi class log loss train 70: {}'.format(multiclass_log_loss(trainVarDict['Class'], probas))
#        print 'log loss test: {}'.format(log_loss( 
        #Kaggle submission
#        submission_df = pd.DataFrame(probas_test, columns=['class'+str(c+1) for c in range(9)])
#        submission_df['ID'] = testVarDict['ID']
#        submission_df.to_csv('submission_svmTest.csv', index=False)


train_model('SVM', textBlockTrain,trainVarDict, textBlockTest, 9, valBlockTrain, valVarDict)
