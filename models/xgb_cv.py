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

cv = CountVectorizer()
X_train_cv = cv.fit_transform(X_train)
X_test_cv = cv.transform(X_test)

xgb = XGBClassifier(device="cuda")
xgb_cv_train_start = time.time()
xgb.fit(X_train_cv, y_train)
xgb_cv_train_time = time.time() - xgb_cv_train_start
print(f"CV + XGB Train Time = {xgb_cv_train_time:.4f}")

xgb_cv_pred_train = xgb.predict(X_train_cv)
xgb_cv_test_start = time.time()
xgb_cv_pred_test = xgb.predict(X_test_cv)
xgb_cv_test_time = time.time() - xgb_cv_test_start

xgb_cv_train_score = accuracy_score(xgb_cv_pred_train, y_train)
xgb_cv_test_score = accuracy_score(xgb_cv_pred_test, y_test)
print(f"CV + XGB Train Score = {xgb_cv_train_score * 100:.4f}%")
print(f"CV + XGB Test Score = {xgb_cv_test_score * 100:.4f}%")
print(f"CV + XGB Test Time = {xgb_cv_test_time:.4f}")

xgb_cv_precision_score = precision_score(y_test, xgb_cv_pred_test, average='weighted')
xgb_cv_f1_score = f1_score(y_test, xgb_cv_pred_test, average='weighted')
xgb_cv_recall_score = recall_score(y_test, xgb_cv_pred_test, average='weighted')
xgb_cv_accuracy_score = accuracy_score(y_test, xgb_cv_pred_test)

print(f"CV + XGB Precision Score = {xgb_cv_precision_score * 100:.4f}%")
print(f"CV + XGB F1 Score = {xgb_cv_f1_score * 100:.4f}%")
print(f"CV + XGB Recall Score = {xgb_cv_recall_score * 100:.4f}%")
print(f"CV + XGB Accuracy Score = {xgb_cv_accuracy_score * 100:.4f}%")

print(classification_report(y_test, xgb_cv_pred_test, target_names=le.classes_))

xgb_cv_cm = confusion_matrix(y_test, xgb_cv_pred_test)
fig, ax = plot_confusion_matrix(conf_mat=xgb_cv_cm, show_absolute=True, show_normed=True, colorbar=True, class_names=le.classes_, figsize=(10, 10))
plt.savefig("./output/xgb_cv.png")
plt.show()