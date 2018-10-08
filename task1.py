import pandas as pd

from sklearn.metrics import r2_score
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler


#Is there a way to reduces the number of lines ?
df_X_train = pd.read_csv('Task1Data/X_train.csv') #just to see if I understand pd.read_csv == call the method/function? inside package pd ? what is v?)
df_X_train = df_X_train[df_X_train.columns[1:]] #what does the v when write pd.read_csv stands for =?
df_y_train = pd.read_csv('Task1Data/y_train.csv')
df_y_train = df_y_train[df_y_train.columns[1]]
df_X_test = pd.read_csv('Task1Data/X_test.csv')
df_X_test = df_X_test[df_X_test.columns[1:]]

print(min(len(df_X_train) - df_X_train.count()))
print(max(len(df_X_train) - df_X_train.count()))

#How can I remove from the DataFrame all the columns with more that NaN% and also getting the name of these lines?
#Is it a correct approach?
#df_tesolinto = df_X_test
#for row in df_X_train:
#    if (len(df_X_train) - df_X_train.count() > 100)
#        df_tesolinto = df_tesolinto(df.drop('column_name', 1))




#Eliminating rows that have NaN, how to do that?
#df_X_train_epurated = df_X_train[df_X_train != np.nan] #thiscommanddoesnotwork why
#df_X_train_epurated = df_X_train.dropna()
#print(df_X_train_epurated) #result Nan --> means that all the row have at least one element missing :O
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
df_result = pd.Series(grid.predict(df_X_test_average_scaled))
print(grid.best_params_)

#Writing to a csv file
df_result.to_csv('Task1Data/results.csv')
