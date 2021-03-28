import pickle
from Regression.dataProcessing import *
from Classification.ClassficationDataProcessing import *
from sklearn.metrics import accuracy_score
path = '../Models/'
MR='MultipleRegression_model.sav'
Ridge='RidgeRegression_model.sav'
PR='Poly_model.sav'
LR='LogisticRegression_model.sav'
SVM='SVM_model.sav'
DT='DT_model.sav'

loaded_model = pickle.load(open(path + DT, 'rb'))
predictions = loaded_model.predict(Classify_X)
#print("Result "+str(predictions))
Accuracy=accuracy_score(Classify_Y, predictions)
# Model Accuracy
print("Accuracy:",Accuracy)