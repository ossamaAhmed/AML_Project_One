from sklearn.impute import SimpleImputer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBRegressor
from numpy import genfromtxt, delete
from numpy.random import randint
from pandas import DataFrame
import csv

randomNumber = randint(100000000)

#Model
model = XGBRegressor(n_jobs=-1, n_estimator=100, learning_rate=0.1,subsample=0.8, colsample_bytree=0.8, reg_alpha=0.2, reg_lambda=0.2, gamma=0.1, random_state=randomNumber)

#Import Data
train_x = genfromtxt('X_train.csv', delimiter=',', skip_header=1, usecols=(range(1,887)))
test_x = genfromtxt('X_test.csv', delimiter=',', skip_header=1, usecols=(range(1,887)))
train_y = genfromtxt('y_train.csv', delimiter=',', skip_header=1, usecols=1)
print('Data loaded')


#Preprocessing
#Remove NaN
train_x = SimpleImputer(strategy='median').fit_transform(X=train_x)
test_x = SimpleImputer(strategy='median').fit_transform(X=test_x)

#Remove Outliers for Training Data
numberDeletedRows = 0
mask = LocalOutlierFactor(n_neighbors=40, n_jobs=-1).fit_predict(train_x)
for i in range(train_x.shape[0]):
    if mask[i] == -1:
        train_x = delete(train_x, i-numberDeletedRows, 0)
        train_y = delete(train_y, i-numberDeletedRows, 0)
        numberDeletedRows = numberDeletedRows + 1

#Select Features
feature_selection_model = SelectFromModel(estimator=model, threshold='3*mean').fit(X=train_x, y=train_y)
train_x = feature_selection_model.transform(X=train_x)
test_x = feature_selection_model.transform(X=test_x)
print('Preprocessing completed')

#Fit Model
model_fit = model.fit(X=train_x, y=train_y)

#Predict
test_y = model_fit.predict(data=test_x)

#Write data to file
with open('submission_task1_08.csv', 'w') as csvfile:
    fieldnames = ['id','y']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for i in range(len(test_y)):
        writer.writerow({'id': float(i), 'y': test_y[i]})


