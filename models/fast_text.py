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

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=y)

train_df.reset_index(drop=True, inplace=True)
test_df.reset_index(drop=True, inplace=True)

train_df["label_format"] = 0
for i in range(len(train_df)):
    train_df.label_format[i] = "__label__" + str(train_df["Sınıf"][i]) + " " + str(train_df["Haber Gövdesi Cleaned"][i])

test_df["label_format"] = 0
for i in range(len(test_df)):
    test_df.label_format[i] = "__label__" + str(test_df["Sınıf"][i]) + " " + str(test_df["Haber Gövdesi Cleaned"][i])

train_df.label_format.to_csv("fasttext_train.txt", index=None, header=None)
test_df.label_format.to_csv("fasttext_test.txt", index=None, header=None)

model5_train_start = time.time()
fasttext_model = fasttext.train_supervised("fasttext_train.txt", epoch=50, lr=0.05, label_prefix="__label__", dim=300)
model5_train_time = time.time() - model5_train_start
print(f"FastText Train Time = {model5_train_time:.4f}")

def predict_fasttext(row):
    pred = fasttext_model.predict(row)[0][0].replace("__label__", "")
    return dict(zip(le.classes_, le.transform(le.classes_)))[pred]

model5_test_start = time.time()
fasttext_pred_test = [predict_fasttext(test) for test in X_test]
model5_test_time = time.time() - model5_test_start
print(f"FastText Test Time = {model5_test_time:.4f}")

fasttext_pred_train = [predict_fasttext(train) for train in X_train]
fasttext_train_score = accuracy_score(fasttext_pred_train, y_train)
fasttext_test_score = accuracy_score(fasttext_pred_test, y_test)
print(f"FastText Train Score = {fasttext_train_score * 100:.4f}%")
print(f"FastText Test Score = {fasttext_test_score * 100:.4f}%")

fasttext_precision_score = precision_score(y_test, fasttext_pred_test, average="weighted")
fasttext_f1_score = f1_score(y_test, fasttext_pred_test, average="weighted")
fasttext_recall_score = recall_score(y_test, fasttext_pred_test, average="weighted")
fasttext_accuracy_score = accuracy_score(y_test, fasttext_pred_test)

print(f"FastText Precision Score = {fasttext_precision_score * 100:.4f}%")
print(f"FastText F1 Score = {fasttext_f1_score * 100:.4f}%")
print(f"FastText Recall Score = {fasttext_recall_score * 100:.4f}%")
print(f"FastText Accuracy Score = {fasttext_accuracy_score * 100:.4f}%")

print(classification_report(y_test, fasttext_pred_test, target_names=le.classes_))

simplernn_cm = confusion_matrix(y_test, fasttext_pred_test)
fig, ax = plot_confusion_matrix(conf_mat=simplernn_cm, show_absolute=True, show_normed=True, colorbar=True, class_names=le.classes_, figsize=(10, 10))
plt.savefig("./output/fasttext.png")
plt.show()