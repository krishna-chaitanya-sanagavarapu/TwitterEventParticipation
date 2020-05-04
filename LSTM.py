import pprint
import collections
import io
import numpy as np
import logging
from sklearn import metrics, preprocessing
import sklearn.model_selection
from sklearn.metrics import f1_score, make_scorer
from sklearn.metrics import classification_report
from sklearn import tree
import keras
from keras.layers import Dropout
from keras.layers.embeddings import Embedding
from keras.layers import Input, Dense
from keras.layers import Dense, Dropout, LSTM, TimeDistributed, Bidirectional
from keras.models import Model
from keras.preprocessing import sequence
from sklearn.metrics import *
from keras.callbacks import Callback
from keras.preprocessing import image
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import pandas as pd

WORD_EMBEDDING_DIM = 50
WORD_GLOVE_FILE = "glove.6B.%sd.txt" % WORD_EMBEDDING_DIM
TWEET_EMBEDDING_DIM = 25
TWEET_GLOVE_FILE = "glove.twitter.27B.%sd.txt" % TWEET_EMBEDDING_DIM
EVENT_EMBEDDING_DIM = 50

mapp_l = {"certYES": 6, "probYES": 5, "INV": 4, "UNK": 3, "probNO": 2,"certNO":1}
tmp_tag = ['bld','bgd','d','ald','agd']
tag_2_desc = {'bld':'Before <day', 'bgd': 'Before >=day', 'd':'During posting tweet', 'ald':'After <day', 'agd':'After >=day'}

def encode_onehot(n, i):
    encoding = np.zeros(n)
    encoding[i] = 1
    return encoding

def prepare_tweets(tweets, word2ind, all_starts,all_ends):
    all_words = []
    all_events = []
    for i in range(0,len(tweets)):
        words = [word2ind[token] for token in tweets[i].split(' ')]
        all_words.append(words)

        evs = [1] * len(tweets[i].split(' '))
        for i in range(all_starts[i], all_ends[i]):
            evs[i] = 2
        all_events.append(evs)

    all_words = sequence.pad_sequences(all_words)
    all_events = sequence.pad_sequences(all_events)
    print(all_words.shape)
    print(all_events.shape)
    all_inputs = [all_words, all_events]

    return all_inputs

def learn(tr_tweets, te_tweets):
    all_tweets = tr_tweets['TWEET'].tolist() + te_tweets['TWEET'].tolist()
    all_starts = tr_tweets['EVENT_INI'].tolist() + te_tweets['EVENT_INI'].tolist()
    all_ends = tr_tweets['EVENT_END'].tolist() + te_tweets['EVENT_END'].tolist()

    all_words = set([token for tweet in all_tweets for token in tweet.split(' ')])
    word2ind = {word: index for index, word in enumerate(all_words, start=1)}
    ind2word = {index: word for word, index in word2ind.items()}
    print("Vocabulary size: %s" % len(word2ind))

    inputs = prepare_tweets(all_tweets, word2ind,all_starts,all_ends)
    inputs_tr = [input[:len(tr_tweets['TWEET'].tolist())] for input in inputs]
    inputs_te = [input[len(tr_tweets['TWEET'].tolist()):] for input in inputs]

    all_labels = []
    tr_labels = collections.OrderedDict()
    te_labels = collections.OrderedDict()
    for tmp in tmp_tag:
      tr_labels[tmp] = [lbl for lbl in tr_tweets[tmp].tolist()]
      te_labels[tmp] = [lbl for lbl in te_tweets[tmp].tolist()]
      for ll in tr_tweets[tmp].to_list():
        all_labels.append(ll)
    all_labels = set(all_labels)

    label2ind = {label: index for index, label in enumerate(all_labels)}
    ind2label = {index: label for label, index in label2ind.items()}
    print(pprint.pformat(label2ind))
    print("Label size: %s" % len(label2ind))

    word_seq_length = len(inputs[0][0])
    print("word_seq_length:", word_seq_length)

    print("Loading word embeddings ...")
    word2embedding = {}
    with open(WORD_GLOVE_FILE) as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            word2embedding[word] = coefs
    print("... word embeddings loaded")

    word_embedding_matrix = np.zeros((len(word2ind)+1, WORD_EMBEDDING_DIM))
    for word, i in word2ind.items():
        embedding_vector = word2embedding.get(word)
        if embedding_vector is not None:
            word_embedding_matrix[i] = embedding_vector

    word_embedding_layer = Embedding(len(word2ind)+1,
                                     WORD_EMBEDDING_DIM,
                                     input_length=word_seq_length,
                                     weights=[word_embedding_matrix],
                                     trainable=False,
                                     name='word_embeddings')

    print("Loading tweet embeddings ...")
    tweet2embedding = {}
    with open(TWEET_GLOVE_FILE) as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            tweet2embedding[word] = coefs
    print("... tweet embeddings loaded")
    tweet_embedding_matrix = np.zeros((len(word2ind) + 1, TWEET_EMBEDDING_DIM))
    for word, i in word2ind.items():
        embedding_vector = tweet2embedding.get(word)
        if embedding_vector is not None:
            tweet_embedding_matrix[i] = embedding_vector
    tweet_embedding_layer = Embedding(len(word2ind) + 1,
                                      TWEET_EMBEDDING_DIM,
                                      input_length=word_seq_length,
                                      weights=[tweet_embedding_matrix],
                                      trainable=False,
                                      name='tweet_embeddings')

    event_embedding_matrix = np.zeros((3, EVENT_EMBEDDING_DIM))
    event_embedding_matrix[0] = np.random.sample(EVENT_EMBEDDING_DIM)
    event_embedding_matrix[1] = np.random.sample(EVENT_EMBEDDING_DIM)
    event_embedding_matrix[2] = np.random.sample(EVENT_EMBEDDING_DIM)
    event_embedding_layer = Embedding(3,
                                    EVENT_EMBEDDING_DIM,
                                    input_length=word_seq_length,
                                    weights=[event_embedding_matrix],
                                    trainable=True,
                                    name='event_embeddings')

    for tmp in tr_labels.keys():
        print("*" * 80)
        print("*" * 80)
        print(tag_2_desc[tmp])

        tr_labels_tmp = np.array([encode_onehot(len(label2ind),
                                                label2ind[label]) for label in tr_labels[tmp]])
        te_labels_tmp = np.array([encode_onehot(len(label2ind),
                                                label2ind[label]) for label in te_labels[tmp]])

        print(tr_labels_tmp.shape, te_labels_tmp.shape)

        words_in = Input(shape=(word_seq_length,), name='input_words')
        x_words = word_embedding_layer(words_in)
        x_words = keras.layers.Masking(mask_value=0.)(x_words)
        x_tweet = tweet_embedding_layer(words_in)
        x_tweet = keras.layers.Masking(mask_value=0.)(x_tweet)

        events_in = Input(shape=(word_seq_length,), name='input_events')
        x_event = event_embedding_layer(events_in)
        x_event = keras.layers.Masking(mask_value=0.)(x_event)

        x = keras.layers.concatenate([x_words, x_tweet, x_event], name='concat1')
        x = Bidirectional(LSTM(50, dropout=0.3, recurrent_dropout=0.3))(x)

        predictions = Dense(len(label2ind), activation='softmax', name='output')(x)

        model = Model(inputs=[words_in, events_in], outputs=predictions)

        model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['acc','mae'])
        early_stop = keras.callbacks.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto', baseline=None, restore_best_weights=False)
        learning_rate_param = keras.callbacks.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=0, mode='auto', min_delta=0.0001)

        model.fit(inputs_tr, tr_labels_tmp,callbacks=[learning_rate_param, early_stop],epochs=100, batch_size=8, validation_split=0.2,verbose=1)

        preds = [ind2label[pred.argmax()] for pred in model.predict(inputs_te)]
        labels_gold = [ind2label[label.argmax()] for label in te_labels_tmp]
        print("Accuracy:", sklearn.metrics.accuracy_score(labels_gold, preds))
        print(classification_report(labels_gold, preds))
        op_fl = open(tmp+"_results.txt",'w')
        op_fl.write(pprint.pformat(label2ind))
        op_fl.write(classification_report(labels_gold,preds))

        print("*" * 80)
        print("*" * 80)


#LOAD THE DATA FROM TWEETS FILE
input_df = pd.read_csv('tweets.csv')
train_df = input_df.loc[input_df['split'] == 'train']
test_df = input_df.loc[input_df['split'] == 'test']
# LEARNING
learn(train_df,test_df)
