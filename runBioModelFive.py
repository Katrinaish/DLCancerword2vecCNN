"""
https://github.com/dennybritz/cnn-text-classification-tf/blob/master/train.py
"""
from gensim.models import word2vec
import tensorflow
import skflow
from preProcess import *
import sys
import matplotlib.pyplot as plt
import logging
import time
import re
import gensim
import nltk
import logging
import time
import os
import collections
import math
import random
import datetime as dt
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf 
from embeddingsTuning import *

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(__name__)


datadir = '/home/kpe/projects/deepLearning/data/'

#test = pd.read_csv(datadir + 'stage2_test_variants.csv')
#testx = pd.read_csv(datadir + 'stage2_test_text.csv', sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])

def convertToListOfStrings(data):
    data = tf.compat.as_str(data.split())

#trainVarnts = pd.read_csv(datadir + 'training_variants')
#trainText = pd.read_csv(datadir + 'training_text', sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])
##trainTextList = convertToListOfStrings(trainText))
#logger.info("read in training data")
#
#
#testVarnts = pd.read_csv(datadir + 'test_variants')
#testText = pd.read_csv(datadir + 'test_text', sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])
#
#trainAll = pd.merge(trainText, trainVarnts, how='left', on='ID').fillna('')
#testAll = pd.merge(testText, testVarnts, how='left', on='ID').fillna('')
#
#train_arrays = np.zeros((trainAll.shape[0], 100))
#train_labels = np.zeros(trainAll.shape[0])
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
#    print trainDict
    return trainDict

"""processing training data 
"""
trainVarnts = open(datadir + 'training_variants_70.csv', 'rb').readlines()
trainText = open(datadir + 'training_text_70.csv', 'rb').readlines()

trainVarDict = createDict(trainVarnts)
logger.info("read in training data 70")


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

logger.info("read in validation data 30")


"""
processing test data 
"""
testVarnts = open(datadir + 'test_variants', 'rb').readlines()
testText = open(datadir + 'test_text', 'rb').readlines()
del(testText[0])

logger.info("read in test data all")
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


features = 300
trainingModel = run_word_2_vec(trainSentences,modelPath='/home/kpe/projects/deepLearning/python3env/word2vecEmbeddingsTrain70_actual.w2v')
testModel = run_word_2_vec(testSentences,modelPath='/home/kpe/projects/deepLearning/python3env/word2vecEmbeddingsTest_actual.w2v')

logger.info("read in word2vectors")
 
textBlockTrain = textBlock_To_Vec(trainText,trainingModel, features)
textBlockTest = textBlock_To_Vec(testText, testModel, features)


def build_vocab(data):
    dataList = tf.compat.as_str(data).split()
    return dataList

def convert_embed_to_npMatrix(model, vector_dim=300):
    embedding_matrix = np.zeros((len(model.wv.vocab), vector_dim))
    for i in range(len(model.wv.vocab)):
        embedding_vector = model.wv[model.wv.index2word[i]]
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return embedding_matrix

embedding_matrix = convert_embed_to_npMatrix(trainingModel, vector_dim=300)
logger.info("created embedding matrix")


logger.info(type(embedding_matrix))




#saved_embeddings = tf.constant(embedding_matrix)
#embedding = tf.Variable(initial_value=saved_embeddings, trainable=False)

#vocabulary = build_vocab(train_text)
#logger.info("build vocab")
#print vocabulary[1]
x_train = np.array(trainSentences)
y_train = np.array(trainVarDict['Class'])
x_dev = np.array(valSentences)
y_dev = np.array(valVarDict['Class'])
logger.info("x_train is")
logger.info(x_train)
logger.info("y_train is")
logger.info(y_train)

class TextCNN(object):
    def __init__(self, sequence_length, num_classes, vocab_size, embedding_size, filter_sizes, num_filters):

        self.input_x = tf.placeholder(tf.float64, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float64, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        with tf.device('/cpu:0'), tf.name_scope("embedding"):
#            W = tf.Variable(
#                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
#                name="W")
            W = tf.Variable(tf.constant(0.0, shape=[vocab_size, embedding_size]), trainable = False, name="W")
            self.embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, embedding_size])
            self.embedding_init = W.assign(embedding_placeholder)

#            self.embedded_chars = tf.nn.embedding_lookup(embedding_matrix, self.input_x)
#            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Max-pooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(3, pooled_outputs)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])


        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        with tf.name_scope("output"):
            W = tf.Variable(tf.truncated_normal([num_filters_total, num_classes], stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(self.scores, self.input_y)
            self.loss = tf.reduce_mean(losses)

        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")


filter_sizes = "3,4,5"
num_filters = 128
embedding_dim = 128
numClasses = 9
batch_size = 64
num_epochs = 200
dropout_keep_prob = 0.5
evaluate_every = 100
check_every = 100

cnn = TextCNN(
    sequence_length=len(x_train),
    num_classes=numClasses,
    vocab_size=len(trainingModel.wv.vocab),
    embedding_size=embedding_dim,
    filter_sizes=map(int, filter_sizes.split(",")),
    num_filters=num_filters)


global_step = tf.Variable(0, name="global_step", trainable=False)
optimizer = tf.train.AdamOptimizer(1e-3)
grads_and_vars = optimizer.compute_gradients(cnn.loss)
train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)


sess.run(tf.initialize_all_variables())


logs_path = '/home/kpe/projects/deepLearning'
def train_step(x_batch, y_batch):
    writer = tf.train.SummaryWriter(logs_path, graph=tf.get_default_graph())

    feed_dict = {
      cnn.embedding_placeholder: embedding_matrix,
      cnn.input_x: x_batch,
      cnn.input_y: y_batch,
      cnn.dropout_keep_prob: dropout_keep_prob
    }
    _, step, summaries, loss, accuracy = sess.run(
        [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
        feed_dict)
    time_str = datetime.datetime.now().isoformat()
    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
    train_summary_writer.add_summary(summaries, step)

batches = data_helpers.batch_iter(
    zip(x_train, y_train), batch_size, num_epochs)

for batch in batches:
    x_batch, y_batch = zip(*batch)
    train_step(x_batch, y_batch)
    current_step = tf.train.global_step(sess, global_step)
    if current_step % evaluate_every == 0:
        print("\nEvaluation:")
        dev_step(x_dev, y_dev, writer=dev_summary_writer)
        print("")
    if current_step % checkpoint_every == 0:
        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
        print("Saved model checkpoint to {}\n".format(path))


