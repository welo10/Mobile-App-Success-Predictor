from dataProcessing import *
from sklearn.metrics import r2_score
import numpy as np
from sklearn import linear_model
from sklearn import metrics
from sklearn.linear_model import Ridge 
from sklearn.linear_model import Lasso 
import pickle

#Multple Linear Regression
cls = linear_model.LinearRegression()
cls.fit(X_train,y_train)
prediction = cls.predict(X_test)

print("***********Multiple Linear Regeression********")
#print
print('Co-efficient of linear regression',cls.coef_)
print('Intercept of linear regression model',cls.intercept_)
print('Mean Square Error', metrics.mean_squared_error(np.asarray(y_test), prediction))

true_App_rate=np.asarray(y_test)[0]
predicted_App_rate=prediction[0]

print('True App rate is : ' + str(true_App_rate))
print('Predicted App rate is : ' + str(predicted_App_rate))

test_set_rmse = (np.sqrt(metrics.mean_squared_error(y_test, prediction)))

test_set_r2 = r2_score(y_test, prediction)


print('True value for the first app in the test set is : ' + str(true_App_rate))
print('Predicted value for the first app in the test set is : ' + str(predicted_App_rate))
print("RMSE : "+str(test_set_rmse))
print("Coefficient of determination : "+str(test_set_r2))

#Save Model 
filename = '../Models/MultipleRegression_model.sav'
pickle.dump(cls, open(filename, 'wb'))

# Improvement using Ridge L2 Regulazation 
print("***********Ridge Regeression********")
ridgeR = Ridge(alpha = 10) 
ridgeR.fit(X_train, y_train) 
y_pred = ridgeR.predict(X_test) 
  
# calculate mean square error 
mean_squared_error_ridge = metrics.mean_squared_error(y_test, y_pred) 
print("RMSE ")
print(np.sqrt(mean_squared_error_ridge)) 
print("Score ")
print(r2_score(y_test, y_pred))
  
# get ridge coefficient and print them 
ridge_coefficient = pd.DataFrame() 
ridge_coefficient["Columns"]= X_train.columns 
ridge_coefficient['Coefficient Estimate'] = pd.Series(ridgeR.coef_) 
print(ridge_coefficient) 

fig, ax = plt.subplots(figsize =(20, 10)) 
  
color =['tab:gray', 'tab:blue', 'tab:orange',  
'tab:green', 'tab:red', 'tab:purple', 'tab:brown',  
'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan',  
'tab:orange', 'tab:green', 'tab:blue', 'tab:olive'] 
  
ax.bar(ridge_coefficient["Columns"],  
ridge_coefficient['Coefficient Estimate'],  
color = color) 
  
ax.spines['bottom'].set_position('zero') 
  
plt.style.use('ggplot') 
plt.show() 

#Save Model 
filename = '../Models/RidgeRegression_model.sav'
pickle.dump(ridgeR, open(filename, 'wb'))