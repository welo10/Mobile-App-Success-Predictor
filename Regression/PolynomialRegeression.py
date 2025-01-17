from dataProcessing import *
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import pickle
poly_features = PolynomialFeatures(degree=2)


#Reg_X = data[['rating_count_tot','rating_count_ver','user_rating_ver','lang.num']]
#X_train, X_test, y_train, y_test = train_test_split(Reg_X, Reg_Y, test_size = 0.3,shuffle=True)


# transforms the existing features to higher degree features.
X_train_poly = poly_features.fit_transform(X_train)

# fit the transformed features to Linear Regression
poly_model = linear_model.LinearRegression()
poly_model.fit(X_train_poly, y_train)


# predicting on training data-set
y_train_predicted = poly_model.predict(X_train_poly)

# predicting on test data-set
prediction = poly_model.predict(poly_features.fit_transform(X_test))


print('Co-efficient of linear regression',poly_model.coef_)
print('Intercept of linear regression model',poly_model.intercept_)
print('Mean Square Error', metrics.mean_squared_error(y_test, prediction))


true_app_value=np.asarray(y_test)[0]
predicted_app_value=prediction[0]

test_set_rmse = (np.sqrt(metrics.mean_squared_error(y_test, prediction)))

test_set_r2 = r2_score(y_test, prediction)



print('True value for the first app in the test set is : ' + str(true_app_value))
print('Predicted value for the first app in the test set is : ' + str(predicted_app_value))
print("RMSE : "+str(test_set_rmse))
print("Coefficient of determination (Score) : "+str(test_set_r2))

#Save Model 
filename = '../Models/Poly_model.sav'
pickle.dump(poly_model, open(filename, 'wb'))