import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import r2_score
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
import numpy as np


#Is there a way to reduces the number of lines ?
df_X_train = pd.read_csv('Task1Data/X_train.csv') #just to see if I understand pd.read_csv == call the method/function? inside package pd ? what is v?)
df_X_train = df_X_train[df_X_train.columns[1:]] #what does the v when write pd.read_csv stands for =?
df_y_train = pd.read_csv('Task1Data/y_train.csv')
df_y_train = df_y_train[df_y_train.columns[1]]
df_X_test = pd.read_csv('Task1Data/X_test.csv')
df_X_test = df_X_test[df_X_test.columns[1:]]

print(min(len(df_X_train) - df_X_train.count()))
print(max(len(df_X_train) - df_X_train.count()))

import math

percentages = []
n = 15 #select n

for i in range(0 ,len(df_X_train['x23'])):
    count = 0
    row = df_X_train.iloc[i]
    # row [a,b,c]
    # row[0] ==> a
    # print(row[0],row[1],row[2])
    for element in row:
        if math.isnan(element):
            count += 1
    len_of_row = len(df_X_train.iloc[i])
    percentage = (count/len_of_row)*100
    percentages.append(percentage)

percentages = np.array(percentages)
print('There are {} rows for which NaN is greater than {}. They will be dropped'.format(np.sum(percentages>n),n))
df_X_train.drop(index=df_X_train[percentages > n].index, inplace=True)
df_y_train.drop(index=df_y_train[percentages > n].index, inplace=True)


df_X_train_average = df_X_train.fillna(df_X_train.mean())
df_X_test_average = df_X_test.fillna(df_X_test.mean())
print(df_X_train_average)
print(df_y_train)


#Scaling
scaler=StandardScaler()
scaler.fit(df_X_train_average)
df_X_train_average_scaled = scaler.transform(df_X_train_average)
df_X_test_average_scaled = scaler.transform(df_X_test_average)

#Model Creation
model = ElasticNet()
parameters = {'alpha':[0.25,0.5, 1], 'l1_ratio': [0.5,0.75,1], 'fit_intercept':[True,False], 'max_iter':[10000]} #why do I need to write with this format?
grid = GridSearchCV(model,parameters, scoring='r2',cv=10)
grid.fit(df_X_train_average_scaled, df_y_train)


#Results
df_result = pd.DataFrame(grid.predict(df_X_test_average_scaled))
print(grid.best_params_)

#Writing to a csv file
df_result.to_csv('Task1Data/results.csv')
