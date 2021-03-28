from sklearn.metrics import accuracy_score
from ClassficationDataProcessing import *
from sklearn.svm import SVC
import pickle
import time

start_trainingtime=time.time()

#Create a svm Classifier
model = SVC(C=0.4,kernel='poly',degree=3) # Polynomial Kernal


#Train the model using the training sets
model.fit(X_train, y_train)
end_trainigTime=time.time()

trainingtime=end_trainigTime-start_trainingtime

start_TestTime=time.time()
#Predict the response for test dataset
y_pred = model.predict(X_test)
end_TestTime=time.time()
total_TestTime=end_TestTime-start_TestTime
acc=accuracy_score(y_test, y_pred)
# Model Accuracy
print("Accuracy:",acc)

#Plotting 
fig = plt.figure()
ax = fig.add_subplot(111)
Paramaters = ['Classification Accuracy', 'Total Training Time', 'Total Test Time']
Values = [acc,trainingtime,total_TestTime]
ax.bar(Paramaters,Values)
ax.set_ylabel('Values')
ax.set_ylim(0,4)
ax.set_title('SVM Regeression')
plt.show()

#Save Model 
filename = '../Models/SVM_model.sav'
pickle.dump(model, open(filename, 'wb'))