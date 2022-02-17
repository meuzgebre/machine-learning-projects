#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import Required Libs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[29]:


# Import The Dataset
data = pd.read_csv('data/winequality.csv')
data.head(5)


# In[30]:


data.info()
data.describe()


# ## Data Analysis Visualization

# In[31]:


data.hist(bins = 10, figsize = (15, 10))


# In[40]:


plt.figure(figsize = (10, 5))
plt.bar(data['quality'], data['alcohol'], color = 'blue')
plt.xlabel('Quality')
plt.ylabel('Alchol')
plt.show()


# In[37]:


# Correlation
plt.figure(figsize = (10, 5))
sns.heatmap(data.corr(), annot = True)


# In[16]:


# Finding Features
for a in range(len(data.corr().columns)):
    for b in range(a):
        if abs(data.corr().iloc[a,b]) >0.7:
            name = data.corr().columns[a]
            print(name)


# In[38]:


data.drop('total sulfur dioxide', axis = 1, inplace = True)
data.head(5)


# In[41]:


# Null Values
data.isnull().sum()


# In[20]:


sns.heatmap(data.isnull(), yticklabels = False, cmap = 'viridis')


# In[42]:


# Data Imputation
data.update(data.fillna(data.mean()))


# In[43]:


data.isnull().sum()


# In[46]:


type_white = pd.get_dummies(data['type'], drop_first = True)
data = pd.concat([data, type_white], axis = 1)
data.head(5)


# In[48]:


data.drop('type', axis = 1, inplace = True)
data.head(5)


# ### Some Visualization

# In[50]:


sns.countplot(x = 'quality', data = data)


# In[51]:


# Red to White Wine Ratio
sns.countplot(x = 'white', data = data)


# ## Train the model

# In[52]:


# Train Test Split
from sklearn.model_selection import train_test_split


# In[53]:


X = data.drop('quality', axis = 1)
y = data['quality']


# In[54]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[55]:


# Normalization
from sklearn.preprocessing import MinMaxScaler


# In[56]:


# Creating Normalization Object 
norm = MinMaxScaler()

# fit data
norm_fit = norm.fit(X_train)
new_Xtrain = norm_fit.transform(X_train)
new_Xtest = norm_fit.transform(X_test)

print(new_Xtrain)


# In[57]:


# Applay Model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


# In[58]:


#creating RandomForestClassifier constructor
rnd = RandomForestClassifier()

# fit data
fit_rnd = rnd.fit(new_Xtrain, y_train)


# In[60]:


# predicting score
rnd_score = rnd.score(new_Xtest, y_test)
print('score of model is : ' , rnd_score)


# In[68]:


from sklearn.metrics import mean_squared_error

print('calculating the error')

# calculating mean squared error
rnd_MSE = mean_squared_error(y_test, x_predict)

# calculating root mean squared error
rnd_RMSE = np.sqrt(MSE)

# display MSE
print('mean squared error is : ', rnd_MSE)

# display RMSE
print('root mean squared error is : ', rnd_RMSE)
print(classification_report(x_predict, y_test))


# In[73]:


X_predict = list(rnd.predict(X_test))

predicted_df = {'predicted_values': X_predict, 'original_values': y_test}

pd.DataFrame(predicted_df).head(20)


# In[74]:


# Save model

import pickle

file = 'wine_quality'

#save file
save = pickle.dump(rnd, open(file,'wb'))


# In[75]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test, X_predict)


# In[ ]:




