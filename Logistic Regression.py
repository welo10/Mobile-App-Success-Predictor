from sklearn.linear_model import LogisticRegression
from ClassficationDataProcessing import *
from sklearn.metrics import  accuracy_score
import time
import pickle


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

#Save Model 
filename = 'LogisticRegression_model.sav'
pickle.dump(LR_Model, open(filename, 'wb'))


