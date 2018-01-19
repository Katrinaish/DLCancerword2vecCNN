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

def run_word_2_vec(trainSentences, num_workers=4, num_features=300, iter=5, min_word_count=40, context=10, downsampling=1e-3,skipGramVal=1,preLoad = True, modelPath = 'test', fileName = 'dummy'):
    if preLoad == False: 
        startword2vec=time.time()
        train_model=word2vec.Word2Vec(trainSentences,workers=num_workers,size=num_features,min_count=min_word_count,sg=skipGramVal,window=context,sample=downsampling)
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
