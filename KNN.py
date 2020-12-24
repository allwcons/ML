import numpy as np
from scipy.spatial import distance
from collections import Counter

def euc(a,b):
  return distance.euclidean(a,b)
class KNN:
  def __init__(self,k=3):
    self.k = 3
  def fit(self,x_train,y_train):
    self.x_train = np.array(x_train)
    self.y_train = np.array(y_train)
  
  def predict(self,x_test):
    predictions = [self.closest(x) for x in x_test]
    return predictions
  
  def closest(self,x):
    distances = [euc(x,x_train) for x_train in self.x_train]
    sort_d = np.argsort(distances)[:self.k]
    k_neighbor_labels = [self.y_train[i] for i in sort_d]
    most_common = Counter(k_neighbor_labels).most_common(1)
    return most_common[0][0]

