import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder

data = pd.read_csv('AppleStore_training.csv')
data.dropna(how='any',inplace=True)
cols=['rating_count_tot','rating_count_ver','user_rating_ver','prime_genre','sup_devices.num','lang.num','user_rating','price']
data = data[cols] # choosing features

enc = OneHotEncoder(handle_unknown='ignore') # creating instance of one-hot-encoder

enc_df = pd.DataFrame(enc.fit_transform(data[['prime_genre']]).toarray())

data = data.join(enc_df)

#Get the correlation between the features
corr = data.corr()
top_feature = corr.index[abs(corr['user_rating']!=0)]
plt.subplots(figsize=(12, 8))
top_corr = data[top_feature].corr()
sns.heatmap(top_corr, annot=True)
#plt.show()
###########
########### splitting data
Y = data['user_rating']
data = data.drop(['user_rating'],axis=1)
X = data[['rating_count_tot','rating_count_ver','user_rating_ver','sup_devices.num','price','lang.num']]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2,shuffle=True)
