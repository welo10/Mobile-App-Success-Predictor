import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics
import pickle
import time
from ClassficationDataProcessing import *

start_trainingtime=time.time()

DTclf = DecisionTreeClassifier(criterion="entropy", max_depth=3)
DTclf = DTclf.fit(X_train,y_train)
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
filename = 'DT_model.sav'
pickle.dump(DTclf, open(filename, 'wb'))



