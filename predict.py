import os
import math
import datetime

from tqdm import tqdm

import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow import keras

import bert
from bert import BertModelLayer
from bert.loader import StockBertConfig, map_stock_config_to_params, load_stock_weights
from bert.tokenization.bert_tokenization import FullTokenizer

from sklearn.metrics import confusion_matrix, classification_report

RANDOM_SEED = 42
MAX_SEQ_LEN = 38

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

train = pd.read_csv("data/train.csv")
valid = pd.read_csv("data/valid.csv")
test = pd.read_csv("data/test.csv")

train = train.append(valid).reset_index(drop=True)
print(train.shape)
print(train.head())

bert_model_name="uncased_L-12_H-768_A-12"

bert_ckpt_dir = os.path.join("model/", bert_model_name)
#bert_ckpt_file = os.path.join(bert_ckpt_dir, "bert_model.ckpt")
#bert_config_file = os.path.join(bert_ckpt_dir, "bert_config.json")

class IntentDetectionData:
  DATA_COLUMN = "text"
  LABEL_COLUMN = "intent"

  def __init__(self, train, test, tokenizer: FullTokenizer, classes, max_seq_len=192):
    self.tokenizer = tokenizer
    self.max_seq_len = 0
    self.classes = classes

    ((self.train_x, self.train_y), (self.test_x, self.test_y)) = map(self._prepare, [train, test])

    print("max seq_len", self.max_seq_len)
    self.max_seq_len = min(self.max_seq_len, max_seq_len)
    self.train_x, self.test_x = map(self._pad, [self.train_x, self.test_x])

  def _prepare(self, df):
    x, y = [], []

    for _, row in tqdm(df.iterrows()):
      text, label = row[IntentDetectionData.DATA_COLUMN], row[IntentDetectionData.LABEL_COLUMN]
      tokens = self.tokenizer.tokenize(text)
      tokens = ["[CLS]"] + tokens + ["[SEP]"]
      token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
      self.max_seq_len = max(self.max_seq_len, len(token_ids))
      x.append(token_ids)
      y.append(self.classes.index(label))

    return np.array(x), np.array(y)

  def _pad(self, ids):
    x = []
    for input_ids in ids:
      input_ids = input_ids[:min(len(input_ids), self.max_seq_len - 2)]
      input_ids = input_ids + [0] * (self.max_seq_len - len(input_ids))
      x.append(np.array(input_ids))
    return np.array(x)


tokenizer = FullTokenizer(vocab_file=os.path.join(bert_ckpt_dir, "vocab.txt"))
classes = train.intent.unique().tolist()
data = IntentDetectionData(train, test, tokenizer, classes, max_seq_len=128)
print(data.train_x.shape)
print(data.train_x[0])
print(data.train_y[0])
print(data.max_seq_len)


model = tf.keras.models.load_model("saved_model/1")

sentences = [
  "Play our song now",
  "Rate this book as awful"
]

print("============tokens============")
pred_tokens = map(tokenizer.tokenize, sentences)
print(pred_tokens)
pred_tokens = map(lambda tok: ["[CLS]"] + tok + ["[SEP]"], pred_tokens)
print(pred_tokens)

print("============token ids============")
pred_token_ids = list(map(tokenizer.convert_tokens_to_ids, pred_tokens))
print(pred_token_ids)
pred_token_ids = map(lambda tids: tids+[0]*(MAX_SEQ_LEN-len(tids)), pred_token_ids)
print(pred_token_ids)
pred_token_ids = np.array(list(pred_token_ids))
print(pred_token_ids)

predictions = model.predict(pred_token_ids).argmax(axis=-1)

for text, label in zip(sentences, predictions):
  print("text:", text, "\nintent:", classes[label])
  print()
