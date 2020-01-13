import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn import neighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import preprocessing
import seaborn as sns

data= pd.read_csv("autos.csv", encoding='latin-1')   #Loading the data set
print(data.shape)
data = data.dropna() # Dropping all the rows that has a NaN in one of their columns.
data = data.drop(["name"], 1) # Dropping the name column
print(data.shape)

def outliers(column, data2): # Detecting and removing outliers from the data set
    q25, q75 = np.percentile(data2[column], 25), np.percentile(data2[column], 75) # Determining the percentiles of the passed column in the data set.
    iqr = q75 - q25 # Determining the Interquartile Range (IQR)
    cut_off = iqr * 1.5 # Defining limits on the sample values
    lower, upper = q25 - cut_off, q75 + cut_off # Making the lower and upper bounds
    data2= data2[(data2[column] > lower) & (data2[column] < upper)] # Making the new data set without the outliers.
    print(data2.shape, column)
    return data2 # Returning the new data set.

def encoder(column): # Turn all the information in the passed column to integers
    le = preprocessing.LabelEncoder()
    le.fit(data[column])
    LabelEncoder()
    new_column= le.transform(data[column]) # Transforming the passed column.
    data[column] = new_column # Replacing the passed column with the encoded data.

encoder("dateCrawled")
# encoder("name")  # Only needed when running it without preprocessing.
encoder("seller")
encoder("offerType")
encoder("abtest")
encoder("vehicleType")
encoder("gearbox")
encoder("model")
encoder("fuelType")
encoder("brand")
encoder("notRepairedDamage")
encoder("dateCreated")
encoder("lastSeen")
data = outliers("dateCrawled", data)
data=outliers("price", data)
data =outliers("vehicleType", data)
data =outliers("yearOfRegistration", data)
data =outliers("model", data)
data =outliers("powerPS", data)
data =outliers("kilometer", data)
data =outliers("monthOfRegistration", data)
data =outliers("brand", data)

plt.figure(figsize=(12,10))
cor = data.corr() # Computing the pairwise correlation of columns.
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds, xticklabels=True, yticklabels=True) # Making the heatmap of the pairwise correlation of the columns.
plt.show() #  Showing the heatmap.
cor_target = abs(cor["price"]) # Making price the output variable
print(cor_target)
total = 0
total_iterations=0
for i in range(0,19): #Adding all the correlations
    if cor_target[i] >0:
        total+= cor_target[i]
        total_iterations+=1

total = total-1 # To remove price correlation
total_iterations = total_iterations-1 # To remove price correlation
print(total_iterations, "iterations")
print(total, "total-1")
relevant_features = cor_target[cor_target>total/(total_iterations-1)]
print(relevant_features, "relevant features")
print(data.columns.values)

data = data[["price", "yearOfRegistration", "gearbox", "powerPS", "kilometer", "fuelType", "brand", "notRepairedDamage"]] # New data set with the above average variables.
print(data.columns.values)
prices = data["price"] # Getting the price column from the data set.
data = data.drop(["price"],1) # Dropping the price column from the data set
X_train, X_test, y_train, y_test = train_test_split(data, prices, test_size=0.2, random_state=0) # Making a train test split.
print (X_train.shape, y_train.shape)
print (X_test.shape, y_test.shape)

rf = RandomForestRegressor(n_estimators = 300, random_state = 42, max_depth= 20, warm_start= True) # Making a random forest regressor.
rf.fit(X_train, y_train) # Building a forest of trees from the training set
predictions = rf.predict(X_test) # Predicting regression target for X_test.
print(predictions, "predictions")
print(y_test)
scores_regr = metrics.mean_squared_error(y_test, predictions) # Computing the MSE
print(scores_regr, "MSE")

X_train, X_test, y_train, y_test = train_test_split(data, prices, test_size=0.2, random_state=0) # Making a train test split.
print (X_train.shape, y_train.shape)
print (X_test.shape, y_test.shape)

rmse_val = [] # Store RMSE values for the different k values
for K in range(30): # To check the RMSE values for k is 1 to 30
    K = K+1
    print(K)
    model = neighbors.KNeighborsRegressor(n_neighbors = K) # Making a regression based on k-nearest neighbors using K as n_neighbors.
    model.fit(X_train, y_train)  # Fitting the model
    pred=model.predict(X_test) # Predicting regression target for X_test.
    error = sqrt(mean_squared_error(y_test,pred)) # Computing the RMSE
    rmse_val.append(error) # storing the RMSE value

plt.plot(pd.DataFrame(rmse_val)) # Plotting the RMSE values against the k values
plt.xlabel("k")
plt.ylabel("RMSE")
plt.show()
print(rmse_val)
lowest = min(rmse_val) # Looking up what the lowest RMSE valye is.
lowest_for_k = rmse_val.index(lowest) # Getting the associated k value.
lowest_for_k = lowest_for_k +1 # +1 because got the value by checking the index.
print(lowest_for_k)
print(lowest)

knn = KNeighborsClassifier(n_neighbors=lowest_for_k) # Making a regression based on k-nearest neighbors using lowest_for_k as n_neighbors.
knn.fit(X_train, y_train) # Fitting the model
y_pred = knn.predict(X_test) # Predicting regression target for X_test.
print(y_pred)
print(y_test)
scores_regr = metrics.mean_squared_error(y_test, y_pred) # Computing the MSE
print(scores_regr, "MSE")
