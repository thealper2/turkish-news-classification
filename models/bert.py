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

df["labels"] = df["Sınıf"].apply(lambda label: dict(zip(le.classes_, le.transform(le.classes_)))[label])

import gc
gc.collect()
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=y)
del df

train_df.reset_index(drop=True, inplace=True)
test_df.reset_index(drop=True, inplace=True)

bert_model = ClassificationModel(
    "bert",
    "dbmdz/bert-base-turkish-uncased",
    num_labels=9,
    use_cuda=True,
    args={
        "reprocess_input_data": True,
        "overwrite_output_dir": True,
        "num_train_epochs": 3,
        "train_batch_size": 512,
        "fp16": False,
        "output_dir": "bert_model",
        "use_multiprocessing": False,
        "use_multiprocessing_for_evaluation": False,
        "use_multiprocessed_decoding": False,
    }
)


model6_train_start = time.time()
bert_model.train_model(train_df[["Haber Gövdesi Cleaned", "labels"]], output_dir="bert_model")
model6_train_time = time.time() - model6_train_start
print(f"BERT Train Time = {model6_train_time:.4f}")

model6_test_start = time.time()
bert_result, bert_model_outputs, bert_wrong_predictions = bert_model.eval_model(test_df[["Haber Gövdesi Cleaned", "labels"]])
model6_test_time = time.time() - model6_test_start
print(f"BERT Test Time = {model6_test_time:.4f}")

bert_pred_test = bert_model_outputs.argmax(axis=1)

bert_result, bert_model_outputs, bert_wrong_predictions = bert_model.eval_model(train_df[["Haber Gövdesi Cleaned", "labels"]])
bert_pred_train = bert_model_outputs.argmax(axis=1)
bert_train_score = accuracy_score(bert_pred_train, y_train)
bert_test_score = accuracy_score(bert_pred_test, y_test)
print(f"BERT Train Score = {bert_train_score * 100:.4f}%")
print(f"BERT Test Score = {bert_test_score * 100:.4f}%")

bert_precision_score = precision_score(y_test, bert_pred_test)
bert_f1_score = f1_score(y_test, bert_pred_test)
bert_recall_score = recall_score(y_test, bert_pred_test)
bert_accuracy_score = accuracy_score(y_test, bert_pred_test)

print(f"BERT Precision Score = {bert_precision_score * 100:.4f}%")
print(f"BERT F1 Score = {bert_f1_score * 100:.4f}%")
print(f"BERT Recall Score = {bert_recall_score * 100:.4f}%")
print(f"BERt Accuracy Score = {bert_accuracy_score * 100:.4f}%")

print(classification_report(y_test, bert_pred_test, target_names=le.classes_))

bert_cm = confusion_matrix(y_test, bert_pred_test)
fig, ax = plot_confusion_matrix(conf_mat=bert_cm, show_absolute=True, show_normed=True, colorbar=True, class_names=le.classes_, figsize=(10, 10))
plt.savefig("./output/bert.png")
plt.show()