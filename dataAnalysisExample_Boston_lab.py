# -*- coding: utf-8 -*-
"""
Exploratory data analysis Lab (data scaling and transformation)
Created on Mon Mar 12 00:53:30 2018
@author: jahan

"""


import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix
from pandas import set_option
from pandas import read_csv

# load boston data set. If the dataset is not in the current directory, you will need to provide
# a full path to filename variable. Check the dataframe after loading to determine if there is any extra columns,
# if there is eliminate the extra column. The sample code for loading dataset from current directory is given.
# Boston dataset column tags
#names = ['crim','zn','indus','chas','nox','rm','age','dis','rad','tax','ptratio','black','lstat','medv']

filename = 'boston.csv'
data = read_csv(filename)

# save features as pandas dataframe put columns zero to thirteen as X and column 14 as Y
############
# to do
############
# separate features and response into two different arrays (Numpy array). 
# This is not necessary for data analysis but will be used in the later part of the course.
# For now it is just an exercise

############
# to do
############

# First perform exploratory data analysis 
# look at the first 20 rows of data

############
# to do
############

# descriptive statistics: mean, max, min, count, 25 percentile, 50 percentile, 75 percentile

############
# to do
############

# we look at the distribution of data 
# plot the histogram and compare/match with descriptive statistics results

############
# to do
############

# perform data scaling by normalizing only the X (we don't normally perform transformation on the Y/output)

############
# to do
############

# calculate the descriptive statistics after normalization

############
# to do
############

# plot histogram of X and compare with histograms before normalization.
# Does normalization improves the data for predictive modeling?

############
# to do
############

# perform data scaling by standardizing only the X (we don't normally perform transformation on the Y/output)

############
# to do
############

# calculate the descriptive statistics after standardization

############
# to do
############

# plot histogram of X and compare with histograms before standardizing.
# Does standardizing improves the data for predictive modeling?

############
# to do
############

## scatter plot of all data below

############
# to do
############


