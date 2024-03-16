import pandas as pd 
import tensorflow as tf
from sklearn.model_selection import train_test_split
import os
import time
import numpy as np

dataset = pd.read_csv("Mental Health Dataset.csv")

x = dataset.drop(columns=["Gender(0=F, 1=M)"])
y = dataset["Gender(0=F, 1=M)"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)



model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Dense(256, input_shape = x_train.shape[1:], activation = 'sigmoid'))
model.add(tf.keras.layers.Dense(256, activation = 'sigmoid'))
model.add(tf.keras.layers.Dense(1, activation = 'sigmoid'))

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#model.fit(x_train, y_train, epochs = 30)

#model.evaluate(x_test, y_test)

#time.sleep(3)
#os.system("cls")


print(model.predict(np.array([[0, 1, 1, 1, 0]])))


