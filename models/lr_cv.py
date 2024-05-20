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

df = pd.read_csv("../data/processed/sample_cleaned.csv", usecols=["Haber Gövdesi Cleaned", "Sınıf"])
df = df.dropna()
df = df.reset_index(drop=True)

X = df["Haber Gövdesi Cleaned"]
y = df["Sınıf"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

pickle.dump(le, open("../assets/le.pickle", "wb"))

cv = CountVectorizer()
X_train_cv = cv.fit_transform(X_train)
X_test_cv = cv.transform(X_test)

pickle.dump(cv, open("../assets/cv.pickle", "wb"))

rf = LogisticRegression()
rf_cv_train_start = time.time()
rf.fit(X_train_cv, y_train)
rf_cv_train_time = time.time() - rf_cv_train_start
print(f"CV + Logistic Regression Train Time = {rf_cv_train_time:.4f}")

pickle.dump(rf, open("../assets/logreg_model.pickle", "wb"))

rf_cv_pred_train = rf.predict(X_train_cv)
rf_cv_test_start = time.time()
rf_cv_pred_test = rf.predict(X_test_cv)
rf_cv_test_time = time.time() - rf_cv_test_start

rf_cv_train_score = accuracy_score(rf_cv_pred_train, y_train)
rf_cv_test_score = accuracy_score(rf_cv_pred_test, y_test)
print(f"CV + Logistic Regression Train Score = {rf_cv_train_score * 100:.4f}%")
print(f"CV + Logistic Regression Test Score = {rf_cv_test_score * 100:.4f}%")
print(f"CV + Logistic Regression Test Time = {rf_cv_test_time:.4f}")

rf_cv_precision_score = precision_score(y_test, rf_cv_pred_test, average='weighted')
rf_cv_f1_score = f1_score(y_test, rf_cv_pred_test, average='weighted')
rf_cv_recall_score = recall_score(y_test, rf_cv_pred_test, average='weighted')
rf_cv_accuracy_score = accuracy_score(y_test, rf_cv_pred_test)

print(f"CV + Logistic Regression Precision Score = {rf_cv_precision_score * 100:.4f}%")
print(f"CV + Logistic Regression F1 Score = {rf_cv_f1_score * 100:.4f}%")
print(f"CV + Logistic Regression Recall Score = {rf_cv_recall_score * 100:.4f}%")
print(f"CV + Logistic Regression Accuracy Score = {rf_cv_accuracy_score * 100:.4f}%")

print(classification_report(y_test, rf_cv_pred_test, target_names=le.classes_))

mnb_cv_cm = confusion_matrix(y_test, rf_cv_pred_test)
fig, ax = plot_confusion_matrix(conf_mat=mnb_cv_cm, show_absolute=True, show_normed=True, colorbar=True, class_names=le.classes_, figsize=(10, 10))
#plt.savefig("./output/lr_cv.png")
plt.show()