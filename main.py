from sklearn import datasets
from KNN import KNN
from NN import NN

iris = datasets.load_iris()

x = iris.data
y = iris.target

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=.5)

model = KNN()
model.fit(xtrain,ytrain)

prediction = model.predict(xtest)

from sklearn.metrics import accuracy_score
print(accuracy_score(ytest,prediction))
