#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  8 16:06:53 2026

@author: aaronfoote
"""

#The following document contains code for building and training a machine
#learning model for text classification.

#5400 text files written by either Thomas Jefferson or Alexander Hamilton (10800 files total)
#have been split into three folders: 50% training, 20% validation, and 30% testing.

#The code below is assumes a directory structure like the following:
    #...train/
    #......jefferson/
    #......hamilton/
    #...val/
    #......jefferson/
    #......hamilton/
    #...test/
    #......jefferson/
    #......hamilton/
    
#First, tensorflow and keras are used to create a batched dataset of each text
#file and their labels

from tensorflow import keras

batch_size = 32

train_ds = keras.utils.text_dataset_from_directory("/Users/aaronfoote/Documents/Data Science/Jefferson_Hamilton_Classification/train", batch_size = batch_size)
val_ds = keras.utils.text_dataset_from_directory("/Users/aaronfoote/Documents/Data Science/Jefferson_Hamilton_Classification/val", batch_size = batch_size)
test_ds = keras.utils.text_dataset_from_directory("/Users/aaronfoote/Documents/Data Science/Jefferson_Hamilton_Classification/test", batch_size = batch_size)


#Add a preprocessing text vectorization layer to encode the most common 20,000 words as sparse numerical vectors
from tensorflow.keras.layers import TextVectorization

text_vectorization = TextVectorization(
    ngrams=2,
    max_tokens=20000,
    output_mode='multi_hot')

text_only_train_ds = train_ds.map(lambda x, y:x)
text_vectorization.adapt(text_only_train_ds)

#Build the binary unigram datasets

binary_2gram_train_ds = train_ds.map(
    lambda x, y: (text_vectorization(x), y),
    num_parallel_calls=4)

binary_2gram_val_ds = val_ds.map(
    lambda x, y: (text_vectorization(x), y),
    num_parallel_calls=4)

binary_2gram_test_ds = test_ds.map(
    lambda x, y: (text_vectorization(x), y),
    num_parallel_calls=4)


#Now the reusable model building utility function

from tensorflow.keras import layers

def get_model(max_tokens=20000, hidden_dim=16):
    inputs = keras.Input(shape=(max_tokens, ))
    x = layers.Dense(hidden_dim, activation = 'relu')(inputs)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer='rmsprop',
                  loss = 'binary_crossentropy',
                  metrics=['accuracy'])
    return model
    
    
model = get_model()
model.summary()
callbacks = [
    keras.callbacks.ModelCheckpoint('binary_2gram.keras', 
                                    save_best_only=True)]

model.fit(binary_2gram_train_ds.cache(),
          validation_data = binary_2gram_val_ds.cache(),
          epochs=10,
          callbacks=callbacks)

model=keras.models.load_model('binary_2gram.keras')
print(f'Test acc: {model.evaluate(binary_2gram_test_ds)[1]:.3f}')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    