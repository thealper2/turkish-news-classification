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

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
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
tfidf = TfidfVectorizer()
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

xgb = XGBClassifier(device="cuda")
xgb_tfidf_train_start = time.time()
xgb.fit(X_train_tfidf, y_train)
xgb_tfidf_train_time = time.time() - xgb_tfidf_train_start
print(f"TFIDF + XGB Train Time = {xgb_tfidf_train_time:.4f}")

xgb_tfidf_pred_train = xgb.predict(X_train_tfidf)
xgb_tfidf_test_start = time.time()
xgb_tfidf_pred_test = xgb.predict(X_test_tfidf)
xgb_tfidf_test_time = time.time() - xgb_tfidf_test_start

xgb_tfidf_train_score = accuracy_score(xgb_tfidf_pred_train, y_train)
xgb_tfidf_test_score = accuracy_score(xgb_tfidf_pred_test, y_test)
print(f"TFIDF + XGB Train Score = {xgb_tfidf_train_score * 100:.4f}%")
print(f"TFIDF + XGB Test Score = {xgb_tfidf_test_score * 100:.4f}%")
print(f"TFIDF + XGB Test Time = {xgb_tfidf_test_time:.4f}")

xgb_tfidf_precision_score = precision_score(y_test, xgb_tfidf_pred_test, average='weighted')
xgb_tfidf_f1_score = f1_score(y_test, xgb_tfidf_pred_test, average='weighted')
xgb_tfidf_recall_score = recall_score(y_test, xgb_tfidf_pred_test, average='weighted')
xgb_tfidf_accuracy_score = accuracy_score(y_test, xgb_tfidf_pred_test)

print(f"TFIDF + XGB Precision Score = {xgb_tfidf_precision_score * 100:.4f}%")
print(f"TFIDF + XGB F1 Score = {xgb_tfidf_f1_score * 100:.4f}%")
print(f"TFIDF + XGB Recall Score = {xgb_tfidf_recall_score * 100:.4f}%")
print(f"TFIDF + XGB Accuracy Score = {xgb_tfidf_accuracy_score * 100:.4f}%")

print(classification_report(y_test, xgb_tfidf_pred_test, target_names=le.classes_))

xgb_tfidf_cm = confusion_matrix(y_test, xgb_tfidf_pred_test)
fig, ax = plot_confusion_matrix(conf_mat=xgb_tfidf_cm, show_absolute=True, show_normed=True, colorbar=True, class_names=le.classes_, figsize=(10, 10))
plt.savefig("./output/xgb_tfidf.png")
plt.show()