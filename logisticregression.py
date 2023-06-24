import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets,linear_model

iris=datasets.load_iris()

#print(list(iris.keys()))
#print(iris.data)
#print(iris.target)

X=iris["data"][:,3:]   #last row of data is taken in consideration
#print(X)
Y=(iris["target"]==2).astype(np.int32)
#print(Y)

clf=linear_model.LogisticRegression()
clf.fit(X,Y)
example=clf.predict([[2.6]])
print(example)
#plotting s graph
X_new=np.linspace(0,3,1000).reshape(-1,1)
Y_prob=clf.predict_proba(X_new)
plt.plot(X_new,Y_prob[:,1],"g-",label="virginicia")
plt.show()