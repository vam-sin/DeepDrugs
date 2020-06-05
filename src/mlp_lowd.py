# MLP for MOA prediction

# Libraries
import pandas as pd
import numpy as np 
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.models import Model
from keras import optimizers
from keras import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from sklearn.utils import shuffle
import keras
from sklearn.model_selection import train_test_split

# dataset import and feature generation
X = pd.read_csv("../Processed_Data/Prostate_Encoded_128.csv")
X = X.iloc[:,1:]
y = pd.read_csv("../Processed_Data/y_prostate.csv")
y = y.iloc[:,1:]

num_classes = 9 # prostate

# MLP 
model = Sequential()
model.add(Dense(512, activation = 'relu'))
model.add(BatchNormalization())
model.add(Dense(256, activation = 'relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(128, activation = 'relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(64, activation = 'relu'))
model.add(BatchNormalization())
model.add(Dense(32, activation = 'relu'))
model.add(BatchNormalization())
model.add(Dropout(0.9))
model.add(Dense(16, activation = 'relu'))
model.add(BatchNormalization())
model.add(Dense(num_classes, activation = 'softmax')) # Final Output

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['accuracy'])

mcp_save = keras.callbacks.callbacks.ModelCheckpoint('mlp.h5', save_best_only=True, monitor='val_accuracy', verbose=1)
reduce_lr = keras.callbacks.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=10, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
callbacks_list = [reduce_lr, mcp_save]

# Training
X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.1, shuffle = False)
history = model.fit(X_train, y_train, batch_size = 256, epochs = 100, validation_data = (X_test, y_test), shuffle = False, callbacks = callbacks_list)

# Results:
# Prostate: loss: 0.8279 - accuracy: 0.7413 - val_loss: 1.1977 - val_accuracy: 0.6750 (Benchmark: 68.3 ± 0.60)
# FULL: (Benchmark: 45.9 ± 0.43)