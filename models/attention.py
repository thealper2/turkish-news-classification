import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import time
import tqdm

import numpy as np
import pandas as pd
tqdm.tqdm.pandas()

import pickle

import io
import string
import re
import nltk
import emoji

import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud
from textblob import TextBlob
from nltk.util import ngrams
from nltk import word_tokenize

from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer
from simpletransformers.classification import ClassificationModel

import fasttext

import gensim
from gensim import corpora, models, similarities
import pyLDAvis.gensim

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from mlxtend.plotting import plot_confusion_matrix
from scikitplot.metrics import plot_roc_curve

from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks, preprocessing
from tensorflow.keras.utils import to_categorical

from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from IPython.display import Markdown

def bold(string):
    display(Markdown("**" + string + "**"))

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

df = pd.read_csv("./data/processed/sample_cleaned.csv", usecols=["Haber Gövdesi Cleaned", "Sınıf"])
df = df.dropna()
df = df.reset_index(drop=True)

X = df["Haber Gövdesi Cleaned"]
y = df["Sınıf"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

tokenizer = preprocessing.text.Tokenizer(num_words=10000,
    filters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
    lower = True,
    split = " ")
tokenizer.fit_on_texts(X)

X_train_tokenizer = tokenizer.texts_to_sequences(X_train)
X_test_tokenizer = tokenizer.texts_to_sequences(X_test)

num_tokens = [len(tokens) for tokens in X_train_tokenizer + X_test_tokenizer]
num_tokens = np.array(num_tokens)
maxlen = int(np.mean(num_tokens) + (2 * np.std(num_tokens)))

X_train_tokenizer = preprocessing.sequence.pad_sequences(X_train_tokenizer, maxlen=maxlen)
X_test_tokenizer = preprocessing.sequence.pad_sequences(X_test_tokenizer, maxlen=maxlen)

input_dim = len(tokenizer.word_index) + 1

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.dense_proj = models.Sequential([
            layers.Dense(dense_dim, activation="relu"),
            layers.Dense(embed_dim)
        ])
        self.layernorm1 = layers.LayerNormalization()
        self.layernorm2 = layers.LayerNormalization()

    def call(self, inputs, mask=None):
        if mask is not None:
            mask = mask[:, tf.newaxis, :]
        attention_output = self.attention(inputs, inputs, attention_mask=mask)
        proj_input = self.layernorm1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm2(proj_input + proj_output)
    
class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, input_dim, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_embeddings = layers.Embedding(input_dim, output_dim)
        self.position_embeddings = layers.Embedding(sequence_length, output_dim)
        self.sequence_length = sequence_length
        self.input_dim = input_dim
        self.output_dim = output_dim
        
    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions
    
input_layer = layers.Input(shape=(None,), dtype="int64")
posemb_layer = PositionalEmbedding(maxlen, 10000, 256)(input_layer)
transformer_block = TransformerBlock(256, 64, 2)(posemb_layer)
gmaxpool_layer = layers.GlobalMaxPooling1D()(transformer_block)
dropout_layer = layers.Dropout(0.5)(gmaxpool_layer)
output_layer = layers.Dense(9, activation="softmax")(dropout_layer)

simplernn = models.Model(input_layer, output_layer)
simplernn.compile(optimizer="adam",
              loss="categorical_crossentropy", 
              metrics=["accuracy"])


simplernn.compile(loss='categorical_crossentropy', 
                  optimizer='adam', 
                  metrics=['accuracy'])    

simplernn.summary()

model1_train_start = time.time()
simplernn_history = simplernn.fit(X_train_tokenizer, 
                                  y_train, 
                                  epochs=10, 
                                  batch_size=128, 
                                  validation_split=0.1, 
                                  callbacks=[callbacks.EarlyStopping(monitor="val_accuracy", patience=3)])
model1_train_time = time.time() - model1_train_start
print(f"Attention Train Time = {model1_train_time:.4f}")

model1_test_start = time.time()
simplernn_pred_test = simplernn.predict(X_test_tokenizer, verbose=0)
model1_test_time = time.time() - model1_test_start
print(f"Attention Test Time = {model1_test_time:.4f}")

y_train = np.argmax(y_train, axis=1)
y_test = np.argmax(y_test, axis=1)

simplernn_pred_train = simplernn.predict(X_train_tokenizer, verbose=0)
simplernn_pred_train = np.argmax(simplernn_pred_train, axis=1)
simplernn_pred_test = np.argmax(simplernn_pred_test, axis=1)
simplernn_train_score = accuracy_score(simplernn_pred_train, y_train)
simplernn_test_score = accuracy_score(simplernn_pred_test, y_test)
print(f"Attention Train Score = {simplernn_train_score * 100:.4f}%")
print(f"Attention Test Score = {simplernn_test_score * 100:.4f}%")

simplernn_precision_score = precision_score(y_test, simplernn_pred_test, average="weighted")
simplernn_f1_score = f1_score(y_test, simplernn_pred_test, average="weighted")
simplernn_recall_score = recall_score(y_test, simplernn_pred_test, average="weighted")
simplernn_accuracy_score = accuracy_score(y_test, simplernn_pred_test)

print(f"Attention Precision Score = {simplernn_precision_score * 100:.4f}%")
print(f"Attention F1 Score = {simplernn_f1_score * 100:.4f}%")
print(f"Attention Recall Score = {simplernn_recall_score * 100:.4f}%")
print(f"Attention Accuracy Score = {simplernn_accuracy_score * 100:.4f}%")

print(classification_report(y_test, simplernn_pred_test, target_names=le.classes_))

simplernn_cm = confusion_matrix(y_test, simplernn_pred_test)
fig, ax = plot_confusion_matrix(conf_mat=simplernn_cm, show_absolute=True, show_normed=True, colorbar=True, class_names=le.classes_, figsize=(10, 10))
plt.savefig("./output/attention.png")
plt.show()