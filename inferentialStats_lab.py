# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 21:46:11 2020

Inferential statistics Lab


Generating a dataset with Gaussian distribution

# generate gaussian data samples
from numpy.random import seed
from numpy.random import randn
from numpy import mean
from numpy import std
# seed the random number generator
seed(1)
# generate two sets of univariate observations
data1 = 5 * randn(100) + 50
data2 = 5 * randn(100) + 51
# summarize
print('data1: mean=%.3f stdv=%.3f' % (mean(data1), std(data1)))
print('data2: mean=%.3f stdv=%.3f' % (mean(data2), std(data2)))

@author: jg
"""


#####################################################################
# Generate 4 datasets with mean of 10, 15, 20 and 25 and standard deviation of 2
# each containing 100 samples. Name the datasets data1, data2, data3 and data4 
#
#
# Part 1
# The goal is to determine if the samples from data1 and data2 come from the same distribution
# or not. Specify the null and the alternative hypothesis and appropriate significance
# test. Calculate the p-value and and a significance of 0.05 and confirm that 
# the samples come from two different distributions. The confirmation of the 
# results should be based on program. Use the sample code provided in significanceTest.py
# as a starting point



# Part 2
# Perform one-way ANOVA for data1, data2, data3 and data4. State the null hypothesis
# and the alternative hypothesis and specify what you expect to prove based 
# on the simulated data (data1 to data4), i.e. which hypothesis is accepted
# confirm the results of the hypothesis test with the output of your program.


