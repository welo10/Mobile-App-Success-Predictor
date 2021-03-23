from sklearn.metrics import accuracy_score
from ClassficationDataProcessing import *
from sklearn.svm import SVC
import pickle

#Create a svm Classifier
model = SVC(C=0.4,kernel='poly',degree=3) # Polynomial Kernal


#Train the model using the training sets
model.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = model.predict(X_test)


# Model Accuracy
print("Accuracy:",accuracy_score(y_test, y_pred))

#Save Model 
filename = 'SVM_model.sav'
pickle.dump(model, open(filename, 'wb'))