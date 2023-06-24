import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets,linear_model

wine=datasets.load_wine()

#print(list(wine.keys()))
#['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names']
#print(wine.DESCR)
X=wine["data"][:,12:]
#print(wine.data)
#print(X)
Y=(wine["target"]==2).astype(np.int32)
#print(Y)
clf=linear_model.LogisticRegression()
clf.fit(X,Y)
pred=clf.predict([[34.5]])
print(pred)
#plotting the graph
X_new=np.linspace(1,3,1000).reshape(-1,1)
Y_prob=clf.predict_proba(X_new)
plt.plot(X_new,Y_prob[:,1])
plt.show()