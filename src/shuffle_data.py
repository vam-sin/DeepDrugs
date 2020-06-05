# MLP for MOA prediction

# Libraries
import pandas as pd
import numpy as np 
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from sklearn.utils import shuffle
import keras
from sklearn.model_selection import train_test_split

# dataset import and feature generation
ds = pd.read_hdf("../Processed_Data/GSE92742_fully_restricted_prostate.hdf")

# X 
X = np.asarray(ds)
X = pd.DataFrame(X)
# print(X)

# y (moa)
y_p = pd.DataFrame(ds.index)
y_p = np.asarray(y_p)
y = []
for i in y_p:
	y.append(np.asarray(i[0])[5])
# print(y)

# One Hot Encoding
encoder = LabelEncoder()
encoder.fit(y)
encoded_Y = encoder.transform(y)
y = np_utils.to_categorical(encoded_Y)
print("The classes in y are: " + str(encoder.classes_))


# Training
X, y = shuffle(X, y, random_state = 42)
X = pd.DataFrame(X)
y = pd.DataFrame(y)

X.to_csv("../Processed_Data/X_prostate.csv")
y.to_csv("../Processed_Data/y_prostate.csv")

# Results:
# Prostate: loss: 0.8279 - accuracy: 0.7413 - val_loss: 1.1977 - val_accuracy: 0.6750 (Benchmark: 68.3 ± 0.60)
# FULL: (Benchmark: 45.9 ± 0.43)