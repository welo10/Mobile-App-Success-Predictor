import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
data = pd.read_csv('AppleStore_training.csv')
X = data.iloc[:,:-1]
Y = data.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)
