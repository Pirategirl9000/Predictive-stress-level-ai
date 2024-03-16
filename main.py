import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

#get dataset
dataset = pd.read_csv('cancer.csv')

#get input and output parameters x = input values and y = output result
x = dataset.drop(columns=["diagnosis(1=m, 0=b)"]) #will use all values except the results
y = dataset["diagnosis(1=m, 0=b)"] #only stores the results of all the x inputs

#set up test set for affirming accuracy
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2) #20% of dataset will be in testing set

#setting up the actual ai using Keras, tensor's ai
#to add tensorflow you must add it through bash by doing: pip install tensorflow
model = tf.keras.models.Sequential()

#There are three layers to traditional neural networks
#1st layer is input
#2nd layer is hidden layer
#3rd layer is output

#first number is the amount of neurons in the layer
model.add(tf.keras.layers.Dense(256, input_shape = x_train.shape[1:], activation = 'sigmoid')) #sigmoid ensures we get either 0 or 1 for answer since it's a binary question
model.add(tf.keras.layers.Dense(256, activation = 'sigmoid'))
model.add(tf.keras.layers.Dense(1, activation = 'sigmoid'))

#compile everything for usage

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])

#adding new training data into there
model.fit(x_train, y_train, epochs = 1000)

#giving it data to evaluate
model.evaluate(x_test, y_test)