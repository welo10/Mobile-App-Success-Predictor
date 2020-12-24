import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import normalize,MinMaxScaler

data = pd.read_csv('AppleStore_training.csv')

#Get the correlation between the features
corr = data.corr()
top_feature = corr.index[abs(corr['user_rating'])>=0]
plt.subplots(figsize=(12, 8))
top_corr = data[top_feature].corr()
sns.heatmap(top_corr, annot=True)
plt.show()

# extracting features
cols=['rating_count_tot','rating_count_ver','user_rating_ver','prime_genre','sup_devices.num','lang.num','ipadSc_urls.num','user_rating']
data = data[cols]

#drop null rows
data.dropna(how='any',inplace=True)

#one hot encoding
one_hot = pd.get_dummies(data['prime_genre'])
data = data.join(one_hot)

#splitting data
Y = data['user_rating']
data = data.drop(['0'],axis=1)
data = data.drop(['user_rating'],axis=1)
data = data.drop(['prime_genre'],axis=1)
X = data
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3,shuffle=True)
