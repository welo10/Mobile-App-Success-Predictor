import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics
import pickle
import time
from ClassficationDataProcessing import *
from sklearn.model_selection import GridSearchCV
parameters = {'max_depth':range(3,20)}

clf = GridSearchCV(DecisionTreeClassifier(criterion="entropy"), parameters, n_jobs=4)


clf.fit(X_train,y_train)
start_trainingtime=time.time()
print (clf.best_score_, clf.best_params_)
DTclf = clf.best_estimator_
DTclf.fit(X_train,y_train)
end_trainigTime=time.time()
trainingtime=end_trainigTime-start_trainingtime

start_TestTime=time.time()

y_pred = DTclf.predict(X_test)
end_TestTime=time.time()
total_TestTime=end_TestTime-start_TestTime

result = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:",result)

#Plotting 
fig = plt.figure()
ax = fig.add_subplot(111)
Paramaters = ['Classification Accuracy', 'Total Training Time', 'Total Test Time']
Values = [result,trainingtime,total_TestTime]
ax.bar(Paramaters,Values)
ax.set_ylabel('Values')
ax.set_ylim(0,4)
ax.set_title('Decision Tree')
plt.show()

#Save Model
filename = '../Models/DT_model.sav'
pickle.dump(DTclf, open(filename, 'wb'))



