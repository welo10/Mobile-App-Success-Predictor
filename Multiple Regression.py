from dataProcessing import *
from sklearn.metrics import r2_score
import numpy as np
from sklearn import linear_model
from sklearn import metrics



#Multple Linear Regression
cls = linear_model.LinearRegression()
cls.fit(X_train,y_train)
prediction= cls.predict(X_test)


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