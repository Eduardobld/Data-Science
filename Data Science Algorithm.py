#!/usr/bin/env python
# coding: utf-8

# # Algoritmo Data Science

# In[1]:


#Load the libraries
import pandas as pd
import numpy as np


# In[ ]:


# Load the data
df = pd.read_csv('data.csv')


# In[ ]:


# Explore the data
df.head()
df.describe()


# In[ ]:


# Clean the data
df = df.dropna()
df = df.replace('?', np.nan)


# In[ ]:


# Visualize the data
df.plot()


# In[ ]:


# Build a model
model = LinearRegression()
model.fit(df.drop('target', axis=1), df['target'])


# In[ ]:


# Evaluate the model
model.score(df.drop('target', axis=1), df['target'])


# In[ ]:


# Make predictions
predictions = model.predict(df.drop('target', axis=1))


# In[ ]:


# Save the model
model.save('model.pkl')


# In[ ]:


# Load the model
model = pickle.load('model.pkl')


# In[ ]:


# Make new predictions
new_predictions = model.predict(new_data)

