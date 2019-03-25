#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set_style('darkgrid')
import warning
warnings.filterwarnings('ignore')


file = r'C:\Users\hp\Desktop\iris.csv'
iris = pd.read_csv(file)


iris.head()

iris.info()

iris.describe()

iris.shape


iris['Species'].value_counts()

iris.drop(['Id'],axis = 1,inplace = True)

iris.head()

g = sns.PairGrid(iris,hue = 'Species')
g.map_diag(sns.distplot)
g.map_upper(plt.scatter)
g.map_lower(sns.kdeplot)

species = pd.get_dummies(iris['Species'],drop_first = True)

species.head()


iris = pd.concat([iris,species],axis = 1)


iris.head()

iris.drop(['Species'],axis = 1,inplace = True)


iris.head()


X = iris.drop(['Iris-versicolor'],axis = 1)
y = iris['Iris-versicolor']


from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

from sklearn.linear_model import LogisticRegression

logmodel =  LogisticRegression()
logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)


from sklearn.metrics import classification_report


print(classification_report(y_test,predictions))

from sklearn.metrics import confusion_matrix


confusion_matrix(y_test,predictions)

from sklearn.linear_model import LinearRegression



lm = LinearRegression()
lm
lm.fit(X_train,y_train)


lm.intercept_
lm.coef_

predictions = lm.predict(X_test)


predictions
y_test.head()

plt.scatter(y_test,predictions)


sns.distplot((y_test-predictions)) 

from sklearn import metrics


metrics.mean_absolute_error(y_test,predictions)

metrics.mean_squared_error(y_test,predictions)

np.sqrt(metrics.mean_squared_error(y_test,predictions))

