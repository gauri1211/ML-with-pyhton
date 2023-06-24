from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

wine=datasets.load_wine()

#print(wine.keys())
#dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names'])

#print(wine.DESCR)

features=wine.data
labels=wine.target

print(features[0],labels[0])

model=KNeighborsClassifier()
model.fit(features,labels)
pred=model.predict([[1,2,3,4,5,6,7,8,9,10,11,12,13]])
print(pred)