# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Download dataset

# %%
# dataset from https://github.com/mwaskom/seaborn-data/blob/master/penguins.csv
import requests

url = 'https://raw.githubusercontent.com/mwaskom/seaborn-data/refs/heads/master/penguins.csv'
response = requests.get(url)

with open('penguins_data.csv', 'wb') as file:
    file.write(response.content)

# %% [markdown]
# ## Explore and clean the data

# %%
import pandas as pd

# %%
df = pd.read_csv('penguins_data.csv')

# %%
df.head()

# %%
df.isnull().sum()

# %%
# drop rows with missing values
df = df.dropna(subset=['species', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g'])

# %%
len(df)

# %%
# select only columns with relevant features
df = df[['species', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']]

# %%
df.head()

# %%
df['species'].value_counts()

# %%
features = df[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']].to_numpy()
values = df['species'].to_numpy()

# %% [markdown]
# ## Scale and encode the data

# %%
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# %%
features

# %%
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# %%
values

# %%
# one-hot encode the species
encoder = LabelEncoder()
values_encoded = encoder.fit_transform(values)

# %%
values_encoded

# %%
# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_scaled, values_encoded, test_size=0.2, random_state=42)

# %% [markdown]
# ### Try out some models

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# %%
clf = LogisticRegression()
clf.fit(X_train, y_train)

# %%
y_pred = clf.predict(X_test)

# %%
print(classification_report(y_test, y_pred, target_names=['Adelie', 'Gentoo', 'Chinstrap']))

# %%
print(confusion_matrix(y_test, y_pred))

# %%
clf.predict_proba(X_test[:4])

# %%
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

# %%
rf.score(X_test, y_test)

# %%
import matplotlib.pyplot as plt
import numpy as np

# %%
coefficients = clf.coef_

# %% [markdown]
# ### Explore the feature importance
#
# Plot logistic regression coefficients

# %%
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

# %% [markdown]
# ### Make a prediction on new data

# %%
