import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import normalize,MinMaxScaler

data = pd.read_csv('AppleStore_training_classification.csv')


# extracting features
x_cols=['rating_count_tot','rating_count_ver','prime_genre','sup_devices.num','lang.num','ipadSc_urls.num']
Y = data['rate']
X = data[x_cols]

#drop null rows
data.dropna(how='any',inplace=True)


#Onehot encoding X values
dummy=pd.get_dummies(X['prime_genre'],prefix="PrimeGenre",drop_first=False)
X=pd.concat([X,dummy],axis=1)
X=X.drop(['prime_genre'],axis=1)

#Onehot encoding Y values
dummy=pd.get_dummies(Y,prefix="Rate",drop_first=False)
Y=pd.concat([Y,dummy],axis=1)
Y=Y.drop(['rate'],axis=1)

#Handle missing data
X=X.replace(np.NaN,X.mean())
Y=Y.replace(np.NaN,Y.mean())

print("******************* X *************************")
print(X)
print("******************* Y *************************")
print(Y)

#feature scaling
scaler = MinMaxScaler()
data['rating_count_tot'] = scaler.fit_transform(np.array(data['rating_count_tot']).reshape(-1,1))
data['rating_count_ver'] = scaler.fit_transform(np.array(data['rating_count_ver']).reshape(-1,1))
print("************** Scaling ***************")
print(data[['rating_count_tot','rating_count_ver']])


#splitting data
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2,shuffle=False)
