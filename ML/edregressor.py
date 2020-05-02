import pandas as pd
import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

def regression(df, catlist, numlist, alg):
    """
    Fits regression model to data using specified algorithms and 
    explanatory variable columns. 
    """
    
    #Put desired columns into dataframe and drop nulls. 
    dfn = df.filter(items = catlist + numlist + ['ElapsedDays'])
    dfn = dfn.dropna()
    
    #Separate data into explanatory and response variables
    XCAT = dfn.filter(items = catlist).values
    XNUM = dfn.filter(items = numlist).values
    
    y = dfn.filter(items = ['ElapsedDays']).values
    
    #Encode cateogrical data and merge with numerical data
    labelencoder_X = LabelEncoder()
    for num in range(len(catlist)):
        if type(dfn.iloc[:, num][1]) != int: 
            XCAT[:, num] = labelencoder_X.fit_transform(XCAT[:, num])
            
    onehotencoder = OneHotEncoder()
    XCAT = onehotencoder.fit_transform(XCAT).toarray()
    
    X = np.concatenate((XCAT, XNUM), axis=1)
    
    #Split Data into Traning and testing sets 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

    #Train model
    regressor = alg
    regressor.fit(X_train, y_train)

    #Test model
    y_pred = regressor.predict(X_test)

    #Print loss function results
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
    print('R-Squared:', metrics.r2_score(y_test, y_pred))