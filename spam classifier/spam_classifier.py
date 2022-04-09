#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data = pd.read_csv('data/spamham.csv')
data.head(5)


# In[4]:


data.isna().sum()


# In[9]:


from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer()

x_train_count = tfidf_vectorizer.fit_transform(data['text'])
tfidf_vectorizer.vocabulary_


# In[10]:


from sklearn.model_selection import train_test_split


# In[11]:


y = data['spam']

X_train, X_test, y_train, y_test = train_test_split(x_train_count, y, test_size=0.2)
x_train_count.shape


# In[29]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[13]:


def decision_classifier(x_train, y_train):
    clf = DecisionTreeClassifier(max_depth=10).fit(x_train, y_train)
    
    return clf


# In[16]:


def random_forest(x_train, y_train):
    clf = RandomForestClassifier().fit(x_train, y_train)
    
    return clf


# In[26]:


def naive_bayes(x_train, y_train):
    clf = GaussianNB().fit(x_train.toarray(), y_train)
    
    return clf


# In[19]:


def neighbors(x_train, y_train):
    clf = KNeighborsClassifier(n_neighbors=10).fit(x_train, y_train)
    
    return clf


# In[32]:


def logestic(x_train, y_train):
    clf = LogisticRegression().fit(x_train, y_train)
    
    return clf


# In[21]:


def build_and_train_classification(x_train, y_train, classification_fn):
    model = classification_fn(x_train, y_train)
    
    y_pred = model.predict(X_test)
    
    train_score = model.score(X_train, y_train)
    test_score = accuracy = accuracy_score(y_test, y_pred)
    
    print(f'Training Score: {train_score}')
    print(f'Test Score: {test_score}')


# In[22]:


build_and_train_classification(X_train, y_train, decision_classifier)


# In[23]:


build_and_train_classification(X_train, y_train, random_forest)


# In[33]:


build_and_train_classification(X_train, y_train, logestic)


# In[25]:


build_and_train_classification(X_train, y_train, neighbors)


# In[ ]:




