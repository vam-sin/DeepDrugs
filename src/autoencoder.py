# MLP for MOA prediction

# Libraries
import pandas as pd
import numpy as np 
from keras.models import Model
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.models import Model
from keras import optimizers
from keras import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Input
from sklearn.utils import shuffle
import keras
from sklearn.model_selection import train_test_split

# dataset import and feature generation
X = pd.read_csv("../Processed_Data/X_prostate.csv")
X = X.iloc[:,1:]
y = pd.read_csv("../Processed_Data/y_prostate.csv")

# Autoencoder 
input_ = Input(shape = (978,))

# Encoder
x = Dense(512, activation = 'relu')(input_)
x = BatchNormalization()(x)
x = Dense(256, activation = 'relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)

# Features
low_d = Dense(128, activation = 'relu')(x)
x = BatchNormalization()(low_d)

# Decoder
x = Dropout(0.5)(x)
x = Dense(256, activation = 'relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(512, activation = 'relu')(input_)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
out = Dense(978, activation = 'relu')(input_)

model = Model(input_, out)
model.compile(optimizer = 'adam', loss = 'mse', metrics=['accuracy'])

mcp_save = keras.callbacks.callbacks.ModelCheckpoint('auto_pros.h5', save_best_only=True, monitor='accuracy', verbose=1)
reduce_lr = keras.callbacks.callbacks.ReduceLROnPlateau(monitor='accuracy', factor=0.1, patience=10, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
callbacks_list = [reduce_lr, mcp_save]

# # Training
history = model.fit(X.values, X.values, batch_size = 256, epochs = 100, shuffle = False, callbacks = callbacks_list)

# Results:
# Prostate: loss: 0.2155 - accuracy: 0.7500 - val_loss: 0.0703 - val_accuracy: 0.9722

encoder = Model(input = input_, output = low_d)
encoded_features = pd.DataFrame(encoder.predict(X.values))

encoded_features.to_csv("../Processed_Data/Prostate_Encoded_128.csv")