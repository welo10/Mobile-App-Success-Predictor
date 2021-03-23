from sklearn.linear_model import LogisticRegression
from ClassficationDataProcessing import *
from sklearn.metrics import  accuracy_score
import time
import pickle
import matplotlib.pyplot as plt

start_trainingtime=time.time()
#Train Model
LR_Model=LogisticRegression(max_iter=4790)

#Model Fitting
LR_Model.fit(X_train,y_train)

end_trainigTime=time.time()
#count_training_time
trainingtime=end_trainigTime-start_trainingtime

#prediction
start_TestTime=time.time()

LR_Predict=LR_Model.predict(X_test)

end_TestTime=time.time()
#count_test_time
total_TestTime=end_TestTime-start_TestTime

#Print
LR_Accuracy=accuracy_score(y_test,LR_Predict)
print("Accuracy is : ",LR_Accuracy)
print("training time is: ",trainingtime)
print("total test time: ",total_TestTime)

#Plotting 
fig = plt.figure()
ax = fig.add_subplot(111)
Paramaters = ['Classification Accuracy', 'Total Training Time', 'Total Test Time']
Values = [LR_Accuracy,trainingtime,total_TestTime]
ax.bar(Paramaters,Values)
ax.set_ylabel('Values')
ax.set_ylim(0,4)
ax.set_title('Logisitc Regeression')
plt.show()

#Save Model 
filename = 'LogisticRegression_model.sav'
pickle.dump(LR_Model, open(filename, 'wb'))


