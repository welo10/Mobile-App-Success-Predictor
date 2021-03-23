import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics
import pickle
from ClassficationDataProcessing import *


DTclf = DecisionTreeClassifier(criterion="entropy", max_depth=3)
DTclf = DTclf.fit(X_train,y_train)
y_pred = DTclf.predict(X_test)
result = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:",result)

#Save Model
filename = 'DT_model.sav'
pickle.dump(DTclf, open(filename, 'wb'))



