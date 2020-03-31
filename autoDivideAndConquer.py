import sys
import math
import numpy as np
import pandas as pd
import pickle
import keras
import random as rd
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import seaborn as sn
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split

from dataset_creation import curve_points
from curve import add,mult

"""
Divide and Conquer approach to shrink the search space.
Set up a neural net for the decision, if k is in the left or right interval.
"""

#curve parameters
a,b,p = 2,3,199

def makesamples(a,b,p,m):
    """
    get m random samples
    """
    ds = curve_points(a,b,p)
    n = ds.shape[0]
    samples=pd.DataFrame(index=range(m), columns=['x_1','x_2','y_1','y_2','k'])
    for i in range(m):
        point_id = rd.choice(range(n))
        x_1, x_2 = ds.at[point_id, 'x'], ds.at[point_id, 'y']
        samples.at[i, 'x_1'], samples.at[i, 'x_2'] = x_1, x_2
        k = rd.choice(range(p))
        if k <= n/2:
            samples.at[i, 'k'] = 0
        else:
            samples.at[i, 'k'] = 1
        y_1, y_2 = mult(k, x_1, x_2, a, p)
        samples.at[i, 'y_1'], samples.at[i, 'y_2'] = y_1, y_2
    return(samples)

#define model here
def create_model(input_dim,neurons,layers,num_classes):
    model = Sequential()
    model.add(Dense(neurons, input_dim=input_dim, activation='relu'))
    for i in range(layers):
        model.add(Dense(neurons, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    # compile the model
    model.compile(loss=keras.losses.sparse_categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),#learning_rate=learning_rate),
                  metrics=['accuracy'])
    print(model.summary())
    return(model)

#create model
neurons, layers = 512,5
model = create_model(4,neurons,layers,2)

#training data
number_samples = 10000
samples = makesamples(a, b, p, number_samples)
#split the samples into train and test set
X_train, X_test, k_train, k_test = train_test_split(samples.iloc[:,0:4], samples.iloc[:,-1], test_size=0.2, random_state=42)

epochs = 100
history = model.fit(X_train, k_train, validation_split=0.2, epochs=epochs, batch_size=20)

# evaluate the model
score = model.evaluate(X_test, k_test, verbose=0)
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


#not needed:
def train(a,b,p,number_samples,length_samples,model,epochs_per_sample):
    for i in range(number_samples):
        print(i)
        samples = makesamples(a, b, p, length_samples)
        history = model.fit(samples.iloc[:,0:4], samples.iloc[:,-1], validation_split=0.2, epochs=epochs_per_sample, batch_size=20)

    return(history)