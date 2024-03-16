from sklearn.model_selection import train_test_split
import pandas as pd


#get dataset
dataset = pd.read_csv('cancer.csv')

#get input and output parameters x = input values and y = output result
x = dataset.drop(columns=["diagnosis(1=m, 0=b)"]) #will use all values except the results
y = dataset["diagnosis(1=m, 0=b)"] #only stores the results of all the x inputs

#train test split takes random values from the x and y data and distributes them into testing and training variables
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

print(x_train) #x_train has the x values of 20% of data table
print("END OF x_train")
print(x_test) #x_test uses values from the data table to test accuracy of model
print("END OF x_test")
print(y_train) #y values corresponding with x_train values
print("END OF y_train")
print(y_test) #y values corresponding with x_test values