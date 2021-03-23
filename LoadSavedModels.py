import pickle
from dataProcessing import *
from ClassficationDataProcessing import *


MR='MultipleRegression_model.sav'
Ridge='RidgeRegression_model.sav'
PR='Poly_model.sav'
LR='LogisticRegression_model.sav'
SVM='SVM_model.sav'
DT='DT_model.sav'

loaded_model = pickle.load(open(SVM, 'rb'))
result=loaded_model.predict(Classify_X)
print("Result "+str(result))