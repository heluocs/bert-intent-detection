# -*- coding:UTF-8 -*-

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import bert
from bert.tokenization.bert_tokenization import FullTokenizer


class IntentDetection:
    
    def __init__(self):
        self.MAX_SEQ_LEN = 38
        self.modelDir = 'saved_model/1'
        self.vocabDir = 'config/vocab.txt'
        self.classes = ['PlayMusic', 'AddToPlaylist', 'RateBook', 'SearchScreeningEvent', 'BookRestaurant', 'GetWeather', 'SearchCreativeWork']
        self.tokenizer = FullTokenizer(vocab_file=self.vocabDir)
        print("============load model start=============")
        self.model = self.loadModel()
        print("============load model success=============")

    def loadModel(self):
        return tf.keras.models.load_model(self.modelDir) 

    def predict(self, sentence):
        pred_tokens = self.tokenizer.tokenize(sentence)
        pred_tokens = ["[CLS]"] + pred_tokens + ["[SEP]"]
        pred_token_ids = list(self.tokenizer.convert_tokens_to_ids(pred_tokens))
        pred_token_ids = pred_token_ids + [0]*(self.MAX_SEQ_LEN-len(pred_token_ids))
        #pred_token_ids = np.array([pred_token_ids,])
        pred_token_ids = np.array(pred_token_ids)
        pred_token_ids = np.expand_dims(pred_token_ids, axis=0)
        predictions = self.model.predict(pred_token_ids).argmax(axis=-1)
        return self.classes[predictions[0]]

