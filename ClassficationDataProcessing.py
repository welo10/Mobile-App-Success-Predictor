import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import normalize,MinMaxScaler
from sklearn.preprocessing import LabelEncoder

Classify_X=None
Classify_Y=None
 
Classify_File_Name='AppleStore_training_classification.csv'
data = pd.read_csv(Classify_File_Name)

# extracting features
x_cols=['rating_count_tot','rating_count_ver','prime_genre','sup_devices.num','lang.num','ipadSc_urls.num']
Classify_Y = data['rate']
Classify_X = data[x_cols]

#drop null rows
data.dropna(how='any',inplace=True)


#Label Encoding Y Values
lb=LabelEncoder()
data['rate']=lb.fit_transform(data['rate'])
Classify_Y=data['rate']

"""
#Onehot encoding Y values
dummy=pd.get_dummies(Y,prefix="Rate",drop_first=False)
Y=pd.concat([Y,dummy],axis=1)
Y=Y.drop(['rate'],axis=1)
"""


#Handle missing data
Classify_Y=Classify_Y.replace(np.NaN,Classify_Y.mean())



#feature scaling
scaler = MinMaxScaler()
data['rating_count_tot'] = scaler.fit_transform(np.array(data['rating_count_tot']).reshape(-1,1))
data['rating_count_ver'] = scaler.fit_transform(np.array(data['rating_count_ver']).reshape(-1,1))
#print("************** Scaling ***************")
#print(data[['rating_count_tot','rating_count_ver']])
Classify_X=data[x_cols]
#Handle missing data
Classify_X=Classify_X.replace(np.NaN,Classify_X.mean())

#Onehot encoding X values
dummy=pd.get_dummies(Classify_X['prime_genre'],prefix="Genre",drop_first=False)
Classify_X=pd.concat([Classify_X,dummy],axis=1)
Classify_X=Classify_X.drop(['prime_genre'],axis=1)

print("******************* X *************************")
#print(Classify_X)
print("******************* Y *************************")
#print(Classify_Y)

#splitting data
X_train, X_test, y_train, y_test = train_test_split(Classify_X, Classify_Y, test_size = 0.2,shuffle=False)
    
    

