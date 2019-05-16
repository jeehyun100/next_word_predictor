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


sequences = list()

for i in range(1, len(encoded)):
    sequence = encoded[i - 1:i + 1]
    sequences.append(sequence)

sequences_np = np.array(sequences)
X, Y = sequences_np[:, 0], sequences_np[:, 1]
X = np.expand_dims(X, 1)
Y = np.expand_dims(Y, 1)


BUFFER_SIZE = 100
BATCH_SIZE = 512
EPOCHS = 100
dataset = tf.data.Dataset.from_tensor_slices((X, Y)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
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

embedding_dim = 100
units = 512
model = Model(vocab_size, embedding_dim, units, BATCH_SIZE)


optimizer = tf.train.AdamOptimizer()

checkpoint_dir = './training_checkpoints_1'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)


def loss_function(labels, logits):
    return tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)



#training
for epoch in range(EPOCHS):
    start = time.time()

    hidden = model.reset_states()

    for (batch, (input, target)) in enumerate(dataset):
        with tf.GradientTape() as tape:
            predictions, hidden = model(input, hidden)

            target = tf.reshape(target, (-1,))
            loss = loss_function(target, predictions)

            grads = tape.gradient(loss, model.variables)
            optimizer.apply_gradients(zip(grads, model.variables))

            if batch % 100 == 0:
                print('Epoch {} Batch {} Loss{:.4f}'.format(epoch + 1, batch, loss))

    if (epoch + 1) % 10 == 0:
        checkpoint.save(file_prefix=checkpoint_prefix)

hidden = [tf.zeros((1, units))]

start_string = "you"
input_eval = [word2idx[start_string]]
input_eval = tf.expand_dims(input_eval, 0)
predictions, hidden = model(input_eval, hidden)
predicted_id = tf.argmax(predictions[-1]).numpy()
text_generated += " " + idx2word[predicted_id]

print(start_string + text_generated)

start_string = "if"
input_eval = [word2idx[start_string]]
input_eval = tf.expand_dims(input_eval, 0)
predictions, hidden = model(input_eval, hidden)
predicted_id = tf.argmax(predictions[-1]).numpy()
text_generated += " " + idx2word[predicted_id]
print(start_string + text_generated)

start_string = "curse"
input_eval = [word2idx[start_string]]
input_eval = tf.expand_dims(input_eval, 0)
predictions, hidden = model(input_eval, hidden)
predicted_id = tf.argmax(predictions[-1]).numpy()
text_generated += " " + idx2word[predicted_id]
print(start_string + text_generated)
