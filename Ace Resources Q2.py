# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 18:26:31 2021

@author: Klun
"""

#Importing essential library
import pandas as pd
import statsmodels.api as sm

#Reading the data
ffb=pd.read_csv('palm_ffb.csv')
ffb.drop('Date',inplace=True,axis=1) #Taking away date as it is not needed
ffb.dtypes #Knowing the columns' data type

#Check for N/A values or null values
for i in ffb:
    print(ffb[i].isna().sum())
for i in ffb:
    print(ffb[i].isnull().sum())
    
"""
The summations are all 0 which indicates there are no N/A values or null values.
"""


x=ffb[[i for i in ffb.columns[:-1]]] #Extracting the independent variables
y=ffb[ffb.columns[-1]] #Extracting the dependent variable

x=sm.tools.tools.add_constant(x) #Adding this will allow the final model having a 'constant'
model= sm.regression.linear_model.OLS(y,x).fit() #Fitting the variables into a linear model.

model_summ= model.summary() 
model_summ #The regression summary

"""
From the summary table, we can see that:
1. Having a 90% confidence level, only SoilMoisture, Precipitation, Working_days and HA_Harvested are significance to the model. 
2. Average_Temp, Min_Temp and Max_Temp are not significant. 
3. SoilMoisture and HA_Harvested are negatively associated to the FFB_Yield while Percipitation and Working_days are positively associated to FFB_Yield.
4. The model has an adjusted R-Squared of 0.211.
"""

    









