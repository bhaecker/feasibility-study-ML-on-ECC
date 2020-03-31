import sys
import math
import numpy as np
import pandas as pd
import pickle
from collections import Counter
import keras
import seaborn as sn
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense

from dataset_creation import curve_points, arrays, mult_arrays

"""
Train a neural net on curve points with factor k as target.
Display correlation matrices and training stats. 
"""

#mode is 'id' or 'actual' for the use of enumerated points or actual points
mode = 'id'

#curve parameters
a,b,p = 2,3,97

#training and model parameters
test_size = 0.2
neurons = 128
layers = 2
epochs = 100
learning_rate = 0.5

if mode == 'id':
    x_points,k_points,y_points = arrays(a,b,p)

    X = np.vstack((x_points,y_points)).transpose()
    k = k_points.transpose()

    Data = {'X': x_points,
            'Y': y_points,
            'K': k_points
            }

    df = pd.DataFrame(Data,columns=['X','Y','K'])

    corrMatrix = df.corr()
    sn.heatmap(corrMatrix, annot=True)
    plt.show()
    X_train, X_test, k_train, k_test = train_test_split(X, k, test_size=test_size, random_state=42)
    input_dim = 2

elif mode == 'actual':
    x_points, k_points, y_points = mult_arrays(a, b, p)

    df = pd.concat([x_points.drop('point_X',axis = 1), y_points.drop('point_Y',axis = 1)], axis=1, sort=False)
    df.fillna(value=-99, inplace=True)
    print(df.shape)

    k = k_points.transpose()
    num_classes = len(Counter(k).keys())

    corrMatrix = df.corr()
    sn.heatmap(corrMatrix, annot=True)
    plt.show()

    X_train, X_test, k_train, k_test = train_test_split(df, k, test_size=test_size, random_state=42)
    input_dim = df.shape[1]

# define the model
def create_model(input_dim,neurons,num_classes):
    model = Sequential()
    model.add(Dense(neurons, input_dim=input_dim, activation='relu'))
    for i in range(layers):
        model.add(Dense(neurons, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    # compile the model
    model.compile(loss=keras.losses.sparse_categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(learning_rate=learning_rate),
                  metrics=['accuracy'])
    print(model.summary())
    return(model)


num_classes = len(Counter(k).keys())

model = create_model(input_dim,neurons,num_classes)

# train the model
history = model.fit(X_train, k_train, validation_split=0.33, epochs=epochs, batch_size=20)
# evaluate the model
score = model.evaluate(X_test,k_test, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()