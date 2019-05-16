import os
import time

import numpy as np
import tensorflow as tf
import unidecode
from keras_preprocessing.text import Tokenizer
from model import Model

tf.enable_eager_execution()

file_path = "shakesspeare.txt"

text = unidecode.unidecode(open(file_path).read())

tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])

encoded = tokenizer.texts_to_sequences([text])[0]

vocab_size = len(tokenizer.word_index) + 1

word2idx = tokenizer.word_index
idx2word = tokenizer.index_word

checkpoint_dir = './training_checkpoints_1'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
# checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
# checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))


text_generated = ''
#
# class Model(tf.keras.Model):
#     def __init__(self, vocab_size, embedding_dim, units, batch_size):
#         super(Model, self).__init__()
#         self.units = units
#         self.batch_size = batch_size
#
#         self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
#
#         self.gru = tf.keras.layers.GRU(self.units,
#                                        return_sequences=True,
#                                        return_state=True,
#                                        recurrent_activation='sigmoid',
#                                        recurrent_initializer='glorot_uniform')
#         self.fc = tf.keras.layers.Dense(vocab_size)
#
#     def call(self, inputs, hidden):
#         inputs = self.embedding(inputs)
#
#         output, states = self.gru(inputs, initial_state=hidden)
#
#         output = tf.reshape(output, (-1, output.shape[2]))
#
#         x = self.fc(output)
#
#         return x, states

BATCH_SIZE = 512
embedding_dim = 100
units = 512
model = Model(vocab_size, embedding_dim, units, BATCH_SIZE)

optimizer = tf.train.AdamOptimizer()
checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))


start_string = "you"

input_eval = [word2idx[start_string]]
input_eval = tf.expand_dims(input_eval, 0)

hidden = [tf.zeros((1, units))]
predictions, hidden = model(input_eval, hidden)

predicted_id = tf.argmax(predictions[-1]).numpy()

text_generated += " " + idx2word[predicted_id]

print(start_string + text_generated)