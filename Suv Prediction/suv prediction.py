#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import required libs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[38]:


# Import the dataset
data = pd.read_csv('data/suv_data.csv')
data.head(5)


# ## Analysing data

# In[3]:


sns.countplot(x = 'Purchased', data = data)


# In[4]:


sns.countplot(x = 'Purchased', data = data, hue = 'Gender')


# In[9]:


data['Age'].plot.hist()


# In[12]:


data['EstimatedSalary'].plot.hist(bins = 20, figsize = (10, 5))


# In[21]:


plt.figure(figsize = (20, 5))
sns.countplot(x = 'Age', data = data)


# ## Data Wrangling

# In[23]:


data.isnull().sum()


# In[26]:


sns.boxenplot(x = 'Purchased', y = 'Age', data = data)


# In[28]:


sns.boxenplot(x = 'Purchased', y = 'EstimatedSalary', data = data)


# In[29]:


data.info()


# In[39]:


Sex = pd.get_dummies(data['Gender'], drop_first = True)
Sex


# In[40]:


data['Sex'] = Sex
data.head(5)


# In[41]:


data.drop(['User ID', 'Gender'], axis = 1, inplace = True)
data.head(5)


# ## Model

# In[43]:


from sklearn.model_selection import train_test_split


# In[55]:


X = data.iloc[:, [0, 1, 3]].values
y = data.iloc[:, 2].values


# In[54]:


y


# In[56]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[57]:


from sklearn.preprocessing import StandardScaler


# In[59]:


sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


# In[60]:


from sklearn.linear_model import LogisticRegression


# In[62]:


classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)


# In[63]:


pred = classifier.predict(X_test)


# In[67]:


from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


# In[66]:


print(classification_report(y_test, pred))


# In[70]:


cm = confusion_matrix(y_test, pred)
cm


# In[71]:


sns.heatmap(cm, yticklabels = False)


# In[74]:


accuracy_score(y_test, pred)


# In[ ]:




