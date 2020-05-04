# Determining Whether and When People Participate in the Events They Tweet About

This code implements an LSTM model to predict the class lables for the task described in the following ACL paper:

Sanagavarapu Krishna Chaitanya, Alakananda Vempala, and Eduardo Blanco. "Determining whether and when people participate in the events they tweet about." Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers). 2017.

Link to the paper: https://www.aclweb.org/anthology/P17-2101.pdf

Abstract: This project describes an approach to determine whether people participate in the events they tweet about. Specifically, we determine whether people are participants in events with respect to the tweet timestamp. We target all events expressed by verbs in tweets, including past, present and events that may occur in the future. We present new annotations using 1,096 event mentions, and experimental results showing that the task is challenging.

The published paper reports experimental results using SVM trained with linguistically motivated features. This repository contains code that implements an LSTM model. The LSTM takes as input GloVe word embeddings, Twitter word embeddings and positional embeddings for the even in the tweet. We train five LSTMs one for each temporal tag as described in the paper.

The annotated data is in the file: tweets.csv

To reproduce the results, download the embedding files, copy them to this repository folder and execute: python LSTM.py
The following are required: python(>=3), keras, pandas, numpy and scikit-learn
The embedding files can be dwnloaded from: https://nlp.stanford.edu/projects/glove/
  
