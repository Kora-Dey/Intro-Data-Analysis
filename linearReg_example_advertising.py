import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix
from pandas import set_option
from pandas import read_csv

dataPath ='C:/Users/jg/Documents/MLData/'
filename = 'Advertising.csv'
data = read_csv(dataPath+filename)
data1 = data.drop(data.columns[0], axis=1) # when dataframe is read it generates a new index column, this will remove the extra column, check data in variable explorer
# save features as pandas dataframe for stepwise feature selection
X1 = data1.drop(data1.columns[3], axis = 1)
Y1 = data1.drop(data1.columns[0:3], axis = 1)
# separate features and response into two different arrays
array = data1.values
#X = array[:,0:3]
X = array[:,0]
y = array[:,3]
# First perform exploratory data analysis using correlation and scatter plot
# look at the first 20 rows of data
peek = data1.head(20)
print(peek)

# descriptive statistics: mean, max, min, count, 25 percentile, 50 percentile, 75 percentile
set_option('display.width', 100)
set_option('precision', 1)
description = data1.describe()
print(description)

# we look at the distribution of data and its descriptive statistics
plt.figure() # new plot
data1.hist()
plt.show()
#
#
## correlation heat map, pay attention to correlation between all predicators/features and each predictor and the output
plt.figure() # new plot
corMat = data1.corr(method='pearson')
print(corMat)
## plot correlation matrix as a heat map
sns.heatmap(corMat, square=True)
plt.yticks(rotation=0)
plt.xticks(rotation=90)
plt.title("CORELATION MATTRIX USING HEAT MAP")
plt.show()
#
## scatter plot of all data
plt.figure()
scatter_matrix(data1)
plt.show()

"""
By observing the preliminary analysis it is obvious that some of the features 
are correlated with each other (colinearity). The ad-hoc appraoch is to keep
only one of the highly correlated features. This approach is only mathematically correct
if the features are identical. Otherwise it is not accurate but good enough for 
a preliminary analysis. 

"""

# Linear model estimation
X = X.reshape(-1,1)
reg = LinearRegression().fit(X, y)
reg.score(X, y) # r^2 score

reg.coef_

reg.intercept_

reg.predict(np.array([[3]]))


