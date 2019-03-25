#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


import seaborn as sns
sns.set_style('darkgrid')


# In[4]:


import warnings
warnings.filterwarnings('ignore')


# In[58]:


file = r'C:\Users\hp\Desktop\iris.csv'
iris = pd.read_csv(file)


# In[6]:


iris.head()


# In[7]:


iris.info()


# In[8]:


iris.describe()


# In[9]:


iris.shape


# In[10]:


iris['Species'].value_counts()


# In[60]:


iris.drop(['Id'],axis = 1,inplace = True)


# In[12]:


iris.head()


# In[62]:


g = sns.PairGrid(iris,hue = 'Species')
g.map_diag(sns.distplot)
g.map_upper(plt.scatter)
g.map_lower(sns.kdeplot)


# In[21]:


species = pd.get_dummies(iris['Species'],drop_first = True)


# In[22]:


species.head()


# In[24]:


iris = pd.concat([iris,species],axis = 1)


# In[25]:


iris.head()


# In[26]:


iris.drop(['Species'],axis = 1,inplace = True)


# In[27]:


iris.head()


# In[28]:


X = iris.drop(['Iris-versicolor'],axis = 1)
y = iris['Iris-versicolor']


# In[29]:


from sklearn.model_selection import train_test_split


# In[30]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[31]:


from sklearn.linear_model import LogisticRegression


# In[32]:


logmodel =  LogisticRegression()


# In[33]:


logmodel.fit(X_train,y_train)


# In[34]:


predictions = logmodel.predict(X_test)


# In[35]:


from sklearn.metrics import classification_report


# In[36]:


print(classification_report(y_test,predictions))


# In[37]:


from sklearn.metrics import confusion_matrix


# In[38]:


confusion_matrix(y_test,predictions)


# In[42]:


from sklearn.linear_model import LinearRegression


# In[43]:


lm = LinearRegression()
lm


# In[44]:


lm.fit(X_train,y_train)


# In[45]:


lm.intercept_


# In[46]:


lm.coef_


# In[47]:


predictions = lm.predict(X_test)


# In[48]:


predictions


# In[49]:


y_test.head()


# In[50]:


plt.scatter(y_test,predictions)


# In[51]:


sns.distplot((y_test-predictions)) 


# In[52]:


from sklearn import metrics


# In[53]:


metrics.mean_absolute_error(y_test,predictions)


# In[54]:


metrics.mean_squared_error(y_test,predictions)


# In[55]:


np.sqrt(metrics.mean_squared_error(y_test,predictions))


# In[ ]:




