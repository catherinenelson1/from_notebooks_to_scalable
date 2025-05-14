#!/usr/bin/env python
# coding: utf-8

# ### Download dataset

# In[2]:


# dataset from https://github.com/mwaskom/seaborn-data/blob/master/penguins.csv
import requests

url = 'https://raw.githubusercontent.com/mwaskom/seaborn-data/refs/heads/master/penguins.csv'
response = requests.get(url)

with open('penguins_data.csv', 'wb') as file:
    file.write(response.content)


# ### Explore the data

# In[3]:


import pandas as pd


# In[4]:


df = pd.read_csv('penguins_data.csv')


# In[5]:


df.head()


# In[6]:


df.isnull().sum()


# In[7]:


# drop rows with missing values
df = df.dropna(subset=['species', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g'])


# In[8]:


len(df)


# In[9]:


# select only columns with relevant features
df = df[['species', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']]


# In[10]:


df.head()


# In[11]:


df['species'].value_counts()


# In[12]:


features = df[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']].to_numpy()
values = df['species'].to_numpy()


# ### Scale and encode the data

# In[13]:


from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


# In[14]:


features


# In[15]:


scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)


# In[16]:


values


# In[17]:


# one-hot encode the species
encoder = LabelEncoder()
values_encoded = encoder.fit_transform(values)


# In[18]:


values_encoded


# In[19]:


# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_scaled, values_encoded, test_size=0.2, random_state=42)


# ### Try out some models

# In[20]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix


# In[33]:


clf = LogisticRegression()
clf.fit(X_train, y_train)


# In[22]:


y_pred = clf.predict(X_test)


# In[23]:


print(classification_report(y_test, y_pred, target_names=['Adelie', 'Gentoo', 'Chinstrap']))


# In[24]:


print(confusion_matrix(y_test, y_pred))


# In[26]:


clf.predict_proba(X_test[:4])


# In[27]:


rf = RandomForestClassifier()
rf.fit(X_train, y_train)


# In[28]:


rf.score(X_test, y_test)


# In[30]:


import matplotlib.pyplot as plt
import numpy as np


# In[31]:


coefficients = clf.coef_


# ### Explore the feature importance
# 
# Plot logistic regression coefficients

# In[32]:


plt.figure(figsize=(10, 6))
for i in range(coefficients.shape[0]):
    plt.plot(np.arange(coefficients.shape[1]), coefficients[i], marker='o', label=f'Class {i}')

plt.xticks(np.arange(coefficients.shape[1]), ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g'], rotation=45)
plt.xlabel('Features')
plt.ylabel('Coefficient Value')
plt.title('Logistic Regression Coefficients')
plt.legend()
plt.grid(True)
plt.show()

