#!/usr/bin/env python
# coding: utf-8

# # Python for Data Analysis
# 

# In[2]:


#Import Python Libraries
import numpy as np
#import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Pandas is a python package that deals mostly with :
# - **Series**  (1d homogeneous array)
# - **DataFrame** (2d labeled heterogeneous array) 
# - **Panel** (general 3d array)

# ### Pandas Series

# Pandas *Series* is one-dimentional labeled array containing data of the same type (integers, strings, floating point numbers, Python objects, etc. ). The axis labels are often referred to as *index*.

# In[6]:


# Example of creating Pandas series :
s1 = pd.Series( [-3,-1,1,3,5] )
print(s1)


# We did not pass any index, so by default, it assigned the indexes ranging from 0 to len(data)-1

# In[5]:


# View index values
print(s1.index)


# In[6]:


# Creating Pandas series with index:
s2 = pd.Series( np.random.randn(5), index=['a', 'b', 'c', 'd', 'e'] )
print(s2)


# In[ ]:


# View index values
print(s2.index)


# In[8]:


# Create a Series from dictionary
data = {'pi': 3.1415, 'e': 2.71828}  # dictionary
print(data)
s3 = pd.Series ( data )
print(s3)


# In[9]:


# reordering the elements
s4 = pd.Series ( data, index = ['e', 'pi', 'tau'])
print(s4)


# NAN (non a number) - is used to specify a missing value in Pandas.

# In[ ]:


s1[:2] # First 2 elements


# In[ ]:


print( s1[ [2,1,0]])  # Elements out of order


# In[ ]:


# Series can be used as ndarray:
print("Mean:" , s4.mean())


# In[ ]:


s1[s1 > 0]


# In[ ]:


# numpy functions can be used on series as usual:
s4[s4 > s4.mean()]


# #### Popular Attributes and Methods:
# 
# |  Attribute/Method | Description |
# |-----|-----|
# | dtype | data type of values in series |
# | empty | True if series is empty |
# | size | number of elements |
# | values | Returns values as ndarray |
# | head() | First n elements |
# | tail() | Last n elements |

# *Exercise* 

# In[ ]:


# Create a series of your choice and explore it
# <your code goes here >
mys = pd.Series( np.random.randn(21))
print(mys)


# In[ ]:


mys.head()


# In[ ]:


mys.empty


# ### Pandas DataFrame

# Pandas *DataFrame* is two-dimensional, size-mutable, heterogeneous tabular data structure with labeled rows and columns ( axes ). Can be thought of a dictionary-like container to store python Series objects.

# In[ ]:


d =  pd.DataFrame({ 'Name': pd.Series(['Alice','Bob','Chris']), 
                  'Age': pd.Series([ 21,25,23]) } )
print(d)


# In[4]:


d2 = pd.DataFrame(np.array([['Alice','Bob','Chris'],[ 21,25,23]]).T, columns=['Name','Age'])


# In[5]:


d2


# In[ ]:


#Add a new column:
d['height'] = pd.Series([5.2,6.0,5.6])
d


# In[ ]:


#Read csv file
df = pd.read_csv("Salaries.csv")


# In[ ]:


#Display a few first records
df.head(10)


# ---
# *Exercise* 

# In[ ]:


#Display first 10 records
# <your code goes here>


# In[ ]:


#Display first 20 records
# <your code goes here>


# In[ ]:


#Display the last 5 records
# <your code goes here>


# ---

# In[ ]:


#Identify the type of df object
type(df)


# In[ ]:


#Check the type of a column "salary"
df['salary'].dtype


# In[ ]:


#List the types of all columns
df.dtypes


# In[ ]:


#List the column names
df.columns


# In[ ]:


#List the row labels and the column names
df.axes


# In[ ]:


#Number of dimensions
df.ndim


# In[ ]:


#Total number of elements in the Data Frame
df.size


# In[ ]:


#Number of rows and columns
df.shape


# In[ ]:


#Output basic statistics for the numeric columns
df.describe()


# In[ ]:


#Calculate mean for all numeric columns
df.mean()


# ---
# *Exercise* 

# In[ ]:


#Calculate the standard deviation (std() method) for all numeric columns
# <your code goes here>
df.std()


# In[ ]:


#Calculate average of the columns in the first 50 rows
# <your code goes here>


# ---
# ### Data slicing and grouping

# In[ ]:


#Extract a column by name (method 1)
df['sex'].head()


# In[ ]:


#Extract a column name (method 2)


# ---
# *Exercise* 

# In[ ]:


#Calculate the basic statistics for the salary column (used describe() method)
# <your code goes here>


# In[ ]:


#Calculate how many values in the salary column (use count() method)
# <your code goes here>


# In[ ]:


#Calculate the average salary


# ---

# In[ ]:


#Group data using rank
df_rank = df.groupby('rank')


# In[ ]:


#Calculate mean of all numeric columns for the grouped object
df_rank.mean()


# In[ ]:


df.groupby('sex').mean()


# In[ ]:


#Calculate the mean salary for men and women. The following produce Pandas Series (single brackets around salary)
df.groupby('sex')['salary'].mean()


# In[ ]:


# If we use double brackets Pandas will produce a DataFrame
df.groupby('sex')[['salary']].mean()


# In[ ]:


# Group using 2 variables - sex and rank:
df.groupby(['rank','sex'], sort=True)[['salary']].mean()


# ---
# *Exercise* 

# In[ ]:


# Group data by the discipline and find the average salary for each group


# ---
# ### Filtering

# In[ ]:


#Select observation with the value in the salary column > 120K
df_sub = df[ df['salary'] > 120000]
df_sub.head()


# In[ ]:


df_sub.axes


# In[ ]:


#Select data for female professors
df_w = df[ df['sex'] == 'Female']
df_w.head()


# ---
# *Exercise* 

# In[ ]:


# Using filtering, find the mean value of the salary for the discipline A
df[ df['discipline'] =='A'].mean().round(2)


# In[ ]:


# Challange:
# Extract (filter) only observations with high salary ( > 100K) and find how many female and male professors in each group


# ---
# ### More on slicing the dataset

# In[ ]:


#Select column salary
df1 = df['salary']


# In[ ]:


#Check data type of the result
type(df1)


# In[ ]:


#Look at the first few elements of the output
df1.head()


# In[ ]:


#Select column salary and make the output to be a data frame
df2 = df[['salary']]


# In[ ]:


#Check the type
type(df2)


# In[ ]:


#Select a subset of rows (based on their position):
# Note 1: The location of the first row is 0
# Note 2: The last value in the range is not included
df[0:10]


# In[ ]:


#If we want to select both rows and columns we can use method .loc
df.loc[10:20,['rank', 'sex','salary']]


# In[ ]:


df_sub.head(15)


# In[ ]:


#Let's see what we get for our df_sub data frame
# Method .loc subset the data frame based on the labels:
df_sub.loc[10:20,['rank','sex','salary']]


# In[ ]:


#  Unlike method .loc, method iloc selects rows (and columns) by poistion:
df_sub.iloc[10:20, [0,3,4,5]]


# ### Sorting the Data

# In[ ]:


#Sort the data frame by yrs.service and create a new data frame
df_sorted = df.sort_values(by = 'service')
df_sorted.head()


# In[ ]:


#Sort the data frame by yrs.service and overwrite the original dataset
df.sort_values(by = 'service', ascending = False, inplace = True)
df.head()


# In[ ]:


# Restore the original order (by sorting using index)
df.sort_index(axis=0, ascending = True, inplace = True)
df.head()


# *Exercise* 

# In[ ]:


# Sort data frame by the salary (in descending order) and display the first few records of the output (head)


# ---

# In[ ]:


#Sort the data frame using 2 or more columns:
df_sorted = df.sort_values(by = ['service', 'salary'], ascending = [True,False])
df_sorted.head(10)


# ### Missing Values

# In[ ]:


# Read a dataset with missing values
flights = pd.read_csv("flights.csv")
flights.head()


# In[ ]:


# Select the rows that have at least one missing value
flights[flights.isnull().any(axis=1)].head()


# In[ ]:


# Filter all the rows where arr_delay value is missing:
flights1 = flights[ flights['arr_delay'].notnull( )]
flights1.head()


# In[ ]:


# Remove all the observations with missing values
flights2 = flights.dropna()


# In[ ]:


# Fill missing values with zeros
nomiss =flights['dep_delay'].fillna(0)
nomiss.isnull().any()


# ---
# *Exercise* 

# In[ ]:


# Count how many missing data are in dep_delay and arr_delay columns


# ---
# ### Common Aggregation Functions:
# 
# |Function|Description
# |-------|--------
# |min   | minimum
# |max   | maximum
# |count   | number of non-null observations
# |sum   | sum of values
# |mean  | arithmetic mean of values
# |median | median
# |mad | mean absolute deviation
# |mode | mode
# |prod   | product of values
# |std  | standard deviation
# |var | unbiased variance
# 
# 

# In[ ]:


# Find the number of non-missing values in each column
flights.describe()


# In[ ]:


# Find mean value for all the columns in the dataset
flights.min()


# In[ ]:


# Let's compute summary statistic per a group':
flights.groupby('carrier')['dep_delay'].mean()


# In[ ]:


# We can use agg() methods for aggregation:
flights[['dep_delay','arr_delay']].agg(['min','mean','max'])


# In[ ]:


# An example of computing different statistics for different columns
flights.agg({'dep_delay':['min','mean',max], 'carrier':['nunique']})


# ### Basic descriptive statistics

# |Function|Description
# |-------|--------
# |min   | minimum
# |max   | maximum
# |mean  | arithmetic mean of values
# |median | median
# |mad | mean absolute deviation
# |mode | mode
# |std  | standard deviation
# |var | unbiased variance
# |sem | standard error of the mean
# |skew| sample skewness
# |kurt|kurtosis
# |quantile| value at %
# 

# In[ ]:


# Convinient describe() function computes a veriety of statistics
flights.dep_delay.describe()


# In[ ]:


# find the index of the maximum or minimum value
# if there are multiple values matching idxmin() and idxmax() will return the first match
flights['dep_delay'].idxmin()  #minimum value


# In[ ]:


# Count the number of records for each different value in a vector
flights['carrier'].value_counts()


# ### Explore data using graphics

# In[ ]:


#Show graphs withint Python notebook
# get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


#Use matplotlib to draw a histogram of a salary data
plt.hist(df['salary'],bins=8)


# In[ ]:


# #Use seaborn package to draw a histogram
# sns.distplot(df['salary']);


# # In[ ]:


# # Use regular matplotlib function to display a barplot
# df.groupby(['rank'])['salary'].count().plot(kind='bar')


# # In[ ]:


# # Use seaborn package to display a barplot
# sns.set_style("whitegrid")

# ax = sns.barplot(x='rank',y ='salary', data=df, estimator=len)


# # In[ ]:


# # Split into 2 groups:
# ax = sns.barplot(x='rank',y ='salary', hue='sex', data=df, estimator=len)


# # In[ ]:


# #Violinplot
# sns.violinplot(x = "salary", data=df)


# # In[ ]:


# #Scatterplot in seaborn
# sns.jointplot(x='service', y='salary', data=df)


# # In[ ]:


# #If we are interested in linear regression plot for 2 numeric variables we can use regplot
# sns.regplot(x='service', y='salary', data=df)


# # In[ ]:


# # box plot
# sns.boxplot(x='rank',y='salary', data=df)


# # In[ ]:


# # side-by-side box plot
# sns.boxplot(x='rank',y='salary', data=df, hue='sex')


# # In[ ]:


# # swarm plot
# sns.swarmplot(x='rank',y='salary', data=df)


# # In[ ]:


# #factorplot
# sns.factorplot(x='carrier',y='dep_delay', data=flights, kind='bar')


# # In[ ]:


# # Pairplot 
# sns.pairplot(df)


# # ---
# # *Exercise*

# In[ ]:


#Using seaborn package explore the dependency of arr_delay on dep_delay (scatterplot or regplot) using flights dataset


# # ---
# # ## Basic statistical Analysis

# # ### Linear Regression

# # In[ ]:


# # Import Statsmodel functions:
# import statsmodels.formula.api as smf


# # In[ ]:


# # create a fitted model
# lm = smf.ols(formula='salary ~ service', data=df).fit()

# #print model summary
# print(lm.summary())


# # In[ ]:


# # print the coefficients
# lm.params


# # In[ ]:


# #using scikit-learn:
# from sklearn import linear_model
# est = linear_model.LinearRegression(fit_intercept = True)   # create estimator object
# est.fit(df[['service']], df[['salary']])

# #print result
# print("Coef:", est.coef_, "\nIntercept:", est.intercept_)


# # ---
# # *Exercise* 

# # In[ ]:


# # Build a linear model for arr_delay ~ dep_delay


# #print model summary


# # ---
# # ### Student T-test

# # In[ ]:


# # Using scipy package:
# from scipy import stats
# df_w = df[ df['sex'] == 'Female']['salary']
# df_m = df[ df['sex'] == 'Male']['salary']
# stats.ttest_ind(df_w, df_m)   


# # In[ ]:




