# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 16:43:58 2021

@author: Klun
"""

#importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#using pandas to read the csv file
ingre=pd.read_csv('ingredient.csv')
ingre.dtypes

###############################################################################

#Q1a

#Check for any N/A values
for i in ingre:
    print(ingre[i].isna().sum())
for i in ingre:
    print(ingre[i].isnull().sum())
"""
all the summations are zero which indicates there are no N/A values or null values.
"""

#Correlation of the variables
#Heatmap graph which shows the correlations of the addictives 'a' to 'i'. Setting annot equals to True will show the values of the correlation between the addictives.
sns.heatmap(ingre.corr(),annot=True) 
"""
Addictive a and g have a high positive correlation of 0.81.
Addictive a and e have a moderate negative correlation of -0.54.
Other addictives each have very low correlation to each other.
"""

  
#Basic descriptive statistics of each column of the dataset
#using pd.set_option('display.max_columns', None) to show all columns' result
pd.set_option('display.max_columns', None)
ingre.describe()
###############################################################################

#Q1b
fig, axes = plt.subplots(3, 3, figsize=(30, 15))
sns.distplot(ingre.a,bins=20,ax=axes[0,0]).set(title='Distribution Study of Addictive a')
sns.distplot(ingre.b,bins=20,ax=axes[0,1]).set(title='Distribution Study of Addictive b')
sns.distplot(ingre.c,bins=20,ax=axes[0,2]).set(title='Distribution Study of Addictive c')
sns.distplot(ingre.d,bins=20,ax=axes[1,0]).set(title='Distribution Study of Addictive d')
sns.distplot(ingre.e,bins=20,ax=axes[1,1]).set(title='Distribution Study of Addictive e')
sns.distplot(ingre.f,bins=20,ax=axes[1,2]).set(title='Distribution Study of Addictive f')
sns.distplot(ingre.g,bins=20,ax=axes[2,0]).set(title='Distribution Study of Addictive g')
sns.distplot(ingre.h,bins=20,ax=axes[2,1]).set(title='Distribution Study of Addictive h')
sns.distplot(ingre.i,bins=20,ax=axes[2,2]).set(title='Distribution Study of Addictive i')

"""
From the plots above, we come to a conclusion:
    Distribution of a, b, d and g are positive-skewed distribution.
    Distribution of e is negative-skewed distribution.
    Distribution of c is bimodal distribution.
    Distribution of f, h and i are zero-inflated distribution.
"""


###############################################################################

#Q1c
#Clustering using KMeans
from sklearn.cluster import KMeans

#Identifying the number of clusters using Elbow Method
#WCSS stands for Within-Cluster Sum of Square. It is the squared distance between each point and the centroid in a cluster and summed together.
wcss=[]

for i in range(1,11):
    kmeans=KMeans(n_clusters=i,random_state=2021)
    kmeans.fit(ingre)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss,c='red')
plt.title('Determine number of clusters')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show() 
"""
from the WCSS plot above, we can see that the elbow occurs at n=3. Hence, we will use K-Means clustering with n=3.
"""

#Clustering using n=3
kmeans=KMeans(n_clusters=3,random_state=2021)
y_kmeans=kmeans.fit_predict(ingre)
ingre_array=ingre.values

plt.scatter(ingre_array[y_kmeans==0,0],ingre_array[y_kmeans==0,6],s=50,c='red',label='formula 1')
plt.scatter(ingre_array[y_kmeans==1,0],ingre_array[y_kmeans==1,6],s=50,c='blue',label='formula 2')
plt.scatter(ingre_array[y_kmeans==2,0],ingre_array[y_kmeans==2,6],s=50,c='green',label='formula 3')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,6],s=100,c='yellow',label='centroids')
plt.title('KMeans Clustering scatterplot with centroids')
plt.legend()
plt.show()
"""
We are using addictive a and addictive g as the two columns to show the scatterplot as they are the most correlated addictives in the dataset.
"""
###############################################################################






