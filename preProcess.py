from nltk import word_tokenize
from nltk.corpus import stopwords
from bs4 import BeautifulSoup

import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
import re
import seaborn as sns
import gensim
import nltk
import os
import logging
import time

stop_words = stopwords.words('english')

#def corpusText_to_LabeledSentences(text, tokenizer, remove_stopWords=False):
#    text = text.strip()
#    raw_sentences = tokenizer.tokenize(text.decode('utf-8'))
#    sentences=[]
#    for raw_sent in raw_sentences:
#    
#    for index, row in data.iteritems():
#        sentences.append(LabeledSentence(utils.to_unicode(row).split(), ['Text' + '_%s' % str(index)]))
#    return sentences
"""
input: takes corpus text document and removes stop words
@returns list of words 
"""
def corpusTxt_to_wordList(text, remove_stopWords = False):
    text_fixed = BeautifulSoup(text).get_text()
    text_fixed = re.sub("[^a-zA-z]"," " ,text_fixed)
    words = text_fixed.lower().split()
    if remove_stopWords:
            words = [w for w in words if not w in stop_words]
    return(words)

"""
input: corpus text
@returns: list of sentences where sentence is list of words
"""
def corpusTxt_to_sentences(text, tokenizer, remove_stopWords=False):
    text = text.strip()
    raw_sentences = tokenizer.tokenize(text.decode('utf-8'))
    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence)>0:
            sentences.append(corpusTxt_to_wordList(raw_sentence, remove_stopWords))
    return sentences

"""
input: corpus text
@returns: average word vector representation
"""
def corpusTxtBlock_to_vector(text, model_var, num_features):
    wordVector = np.zeros((num_features,), dtype="float32")
    index2wordSet = set(model_var.wv.index2word)
    numWords = 0
    for word in text:
        if word in index2wordSet:
            numWords+=1
            wordVector = np.add(wordVector, model_var[word])

    wordVector = np.divide(wordVector, numWords)
    return wordVector

"""
calculates the average feature vector for each list of words 
@ returns: 2D numpy array 
"""
def norm_vector(text,model_var, num_features):
    normVec = np.zeros((len(text), num_features), dtype="float32")
    counter = 0
    for textBlock in text:
        if counter%1000==0:
            print "TextBlock %d of %d" % (counter, len(text))


        normVec[counter] = corpusTxtBlock_to_vector(textBlock, model_var, num_features)
        counter+=1
    return normVec
