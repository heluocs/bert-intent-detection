import os
import numpy as np

import tensorflow as tf
from tensorflow import keras

import bert
from bert.tokenization.bert_tokenization import FullTokenizer


MAX_SEQ_LEN = 38

bert_model_name="uncased_L-12_H-768_A-12"
bert_dir = os.path.join("model/", bert_model_name)

classes = ['PlayMusic', 'AddToPlaylist', 'RateBook', 'SearchScreeningEvent', 'BookRestaurant', 'GetWeather', 'SearchCreativeWork']

tokenizer = FullTokenizer(vocab_file=os.path.join(bert_dir, "vocab.txt"))

model = tf.keras.models.load_model("saved_model/1")

#sentence = "Play our song now"
sentence = "Rate this book as awful"
pred_tokens = tokenizer.tokenize(sentence)
print(pred_tokens)
pred_tokens = ["[CLS]"] + pred_tokens + ["[SEP]"]
print(pred_tokens)
pred_token_ids = list(tokenizer.convert_tokens_to_ids(pred_tokens))
print(pred_token_ids)
pred_token_ids = pred_token_ids + [0]*(MAX_SEQ_LEN-len(pred_token_ids))
print(pred_token_ids)
#pred_token_ids = np.array([pred_token_ids,])
pred_token_ids = np.array(pred_token_ids)
pred_token_ids = np.expand_dims(pred_token_ids, axis=0)
print(pred_token_ids)
predictions = model.predict(pred_token_ids).argmax(axis=-1)
print(predictions)
print(classes[predictions[0]])


"""
sentences = [
  "Play our song now",
  "Rate this book as awful"
]

pred_tokens = map(tokenizer.tokenize, sentences)
pred_tokens = map(lambda tok: ["[CLS]"] + tok + ["[SEP]"], pred_tokens)

pred_token_ids = list(map(tokenizer.convert_tokens_to_ids, pred_tokens))
pred_token_ids = map(lambda tids: tids+[0]*(MAX_SEQ_LEN-len(tids)), pred_token_ids)
pred_token_ids = np.array(list(pred_token_ids))

print(pred_token_ids)
predictions = model.predict(pred_token_ids).argmax(axis=-1)

for text, label in zip(sentences, predictions):
  print("text:", text, "\nintent:", classes[label])
  print()
"""
