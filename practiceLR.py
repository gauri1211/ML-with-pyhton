import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets,linear_model
from sklearn.metrics import mean_squared_error

iris=datasets.load_iris()

#print(iris.DESCR)

iris_X=iris.data
#slicing only taking data from whole datsets
print(iris_X)
#we get array of arrays

iris_X_train=iris_X[:-30] #starting 30 taken for training the model
iris_X_test=iris_X[-30:] #last 30 taken for testing the model

iris_Y_train=iris.target[:-30] #label for x train
iris_Y_test=iris.target[-30:]  #label for y train

model=linear_model.LinearRegression()

model.fit(iris_X_train,iris_Y_train)
iris_Y_predict=model.predict(iris_X_test)

print("mean squared erroe is:",mean_squared_error(iris_Y_test,iris_Y_predict))
print("weights",model.coef_)
print("intercept",model.intercept_)

#plt.scatter(iris_X_test,iris_Y_test)
#plt.plot(iris_X_test,iris_Y_predict)

#plt.show()
