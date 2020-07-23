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
        print(sentence)
        #sentences = ["Play our song now","Rate this book as awful"]
        sentences = []
        sentences.append(sentence)
        pred_tokens = map(self.tokenizer.tokenize, sentences)
        pred_tokens = map(lambda tok:["[CLS]"]+tok+["[SEP]"], pred_tokens)

        pred_token_ids = list(map(self.tokenizer.convert_tokens_to_ids, pred_tokens))
        pred_token_ids = map(lambda tids: tids+[0]*(self.MAX_SEQ_LEN-len(tids)), pred_token_ids)
        pred_token_ids = np.array(list(pred_token_ids))

        predictions = self.model.predict(pred_token_ids).argmax(axis=-1)

        for text, label in zip(sentences, predictions):
            print("text:", text, "\nintent:", self.classes[label])
            return self.classes[label]
		 

