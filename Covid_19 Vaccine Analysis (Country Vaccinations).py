#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data = pd.read_csv("D:/Internship project/country_vaccinations.csv")


# In[3]:


data.head()


# In[4]:


data.tail()


# In[5]:


data.describe()


# In[6]:


pd.to_datetime(data.date)
data.country.value_counts()


# In[8]:


data.vaccines.value_counts()


# In[10]:


df = data[["vaccines", "country"]]
df.head()


# In[11]:


dict_ = {}
for i in df.vaccines.unique():
  dict_[i] = [df["country"][j] for j in df[df["vaccines"]==i].index]
vaccines = {}
for key, value in dict_.items():
  vaccines[key] = set(value)
for i, j in vaccines.items():
  print(f"{i}:>>{j}")


# In[ ]:





# In[ ]:




