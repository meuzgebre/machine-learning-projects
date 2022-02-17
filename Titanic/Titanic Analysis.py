#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import required libs

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


# load the data

data = pd.read_csv('data/titanic.csv')
data.head()


# ### Analyzing data

# In[5]:


sns.countplot(x = 'Survived', data = data)


# In[6]:


sns.countplot(x = 'Survived', data = data, hue = 'Sex')


# In[7]:


sns.countplot(x = 'Survived', data = data, hue = 'Pclass')


# In[8]:


data['Age'].plot.hist()


# In[11]:


data['Fare'].plot.hist(bins = 20, figsize = (10, 5))


# In[12]:


sns.countplot(x = 'SibSp', data = data)


# In[13]:


sns.countplot(x = 'Parch', data = data)


# ## Data Wrangling

# In[14]:


data.isnull()


# In[15]:


data.isnull().sum()


# In[23]:


sns.heatmap(data.isnull(), yticklabels = False, cmap = 'viridis')


# In[24]:


sns.boxplot(x = 'Pclass', y = 'Age', data = data)


# ### Data Inputation

# In[25]:


data.drop('Cabin', axis = 1, inplace = True)


# In[26]:


data['Embarked'].dropna(inplace = True)


# In[29]:


data['Age'].fillna(np.mean(data['Age']), inplace = True)


# In[32]:


sns.heatmap(data.isnull(), yticklabels = False, cbar = True)
data.isnull().sum()


# In[33]:


sex = pd.get_dummies(data['Sex'], drop_first = True)


# In[35]:


embark = pd.get_dummies(data['Embarked'], drop_first = True)


# In[36]:


pclass = pd.get_dummies(data['Pclass'], drop_first = True)


# In[37]:


data = pd.concat([data, sex, embark, pclass], axis = 1)
data.head()


# In[38]:


data.drop(['Sex', 'Embarked', 'PassengerId', 'Name', 'Ticket', 'Pclass'], axis = 1, inplace = True)
data.head(5)


# ## Model Creation & Training

# In[45]:


X = data.drop(['Survived'], axis = 1)
y = data['Survived']


# In[43]:


from sklearn.model_selection import train_test_split


# In[46]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[47]:


from sklearn.linear_model import LogisticRegression


# In[48]:


model = LogisticRegression()


# In[49]:


model.fit(X_train, y_train)


# In[50]:


predict = model.predict(X_test)


# In[51]:


from sklearn.metrics import classification_report 


# In[64]:


print(classification_report(y_test, predict))


# In[65]:


from sklearn.metrics import confusion_matrix


# In[63]:


cm = pd.DataFrame(confusion_matrix(y_test, predict), columns=['Predicted No', 'Predicted Yes'], index=['Actual No', 'Actual Yes'])
cm


# In[69]:


sns.heatmap(confusion_matrix(y_test, predict), yticklabels = False, cbar = True)


# In[60]:


from sklearn.metrics import accuracy_score


# In[61]:


accuracy_score(y_test, predict)


# In[ ]:




