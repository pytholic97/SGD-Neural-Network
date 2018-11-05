import pandas as pd
import numpy as np
from keras.utils import np_utils
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from nn import neural_net

data = pd.read_csv('/home/prathamesh/venv/HAR_MiniProject/final_data1.csv')
#enc = preprocessing.OneHotEncoder()
#e = enc.fit_transform(data.iloc[:,-1].values.reshape(-1,1))
#enc.transform(data.iloc[:,-1].values.reshape(-1,1))
e = np_utils.to_categorical(data.iloc[:,-1].values)
#print(e.shape)
#data.iloc[4,1:7]

X = data.iloc[:,1:7].values
y = e
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42)
#print(X_train.shape, X_test.shape, X_val.shape)

model_1 = neural_net([6,9,6])
tr,v = model_1.sgd(X_train, y_train, batch_size=578, epochs=30, eta=0.1, lmbda=0.1, vldt=X_val, vldt_labels=y_val, k_fold=False)

#for i,j in zip(tr,v):
#	print(i, j)

'''
            ftr = train_samples.shape[0]//batch_size
            samples = [train_samples[i:i+batch_size].reshape(batch_size,-1) for i in range(ftr)]
            labels = [train_labels[i:i+batch_size].reshape(batch_size,-1) for i in range(ftr)]
            if train_samples.shape[0]%batch_size != 0:
                t1 = train_labels[ftr*batch_size:].reshape(train_labels.shape[0] - batch_size*ftr,-1).all()
                labels.append(t1)
                samples.append(train_samples[ftr*batch_size:].reshape(train_samples.shape[0] - batch_size*ftr,-1).all())
'''
