#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Covid_19 vaccine analysis


# In[12]:


import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns


# In[13]:


df = pd.read_csv("D:/Internship project/country_vaccinations_by_manufacturer.csv")


# In[14]:


df.head()


# In[15]:


df["location"].nunique()


# In[16]:


df.isnull().sum()


# In[17]:


df.dtypes


# In[19]:


df['date'] = pd.to_datetime(df['date'])
data=pd.DataFrame(columns=['Country', 'Vaccine', 'Total_vaccine'])
for country in df["location"].unique():
    for vaccine in df["vaccine"].unique():
        filtered_data = df[(df['location'] == country) & (df['vaccine'] == vaccine)]
        total_count = filtered_data['total_vaccinations'].max()
        data = pd.concat([data, pd.DataFrame({'Country': [country], 'Vaccine': [vaccine], 'Total_vaccine': [total_count]})], ignore_index=True)


# In[20]:


data.head(10)


# In[21]:


data.dropna(axis=0,inplace=True)


# In[22]:


data.head(20)


# In[23]:


# Most commonly used vaccines in countries


# In[24]:


data_2=pd.DataFrame(columns=['Country', 'Vaccine'])
data["Total_vaccine"] = pd.to_numeric(data["Total_vaccine"], errors="coerce")
for country in data["Country"].unique():
    new_data = data[data["Country"] == country]
    max_vaccine = new_data.loc[new_data["Total_vaccine"].idxmax(), "Vaccine"]
    data_2 = pd.concat([data_2, pd.DataFrame({'Country': [country], 'Vaccine': [max_vaccine]})], ignore_index=True)


# In[25]:


data_2.head() 


# In[27]:


data_2["Vaccine"].value_counts().plot(kind="bar",
                                    color=["Red","Green","Blue","Black"])


# In[28]:


# Average vaccination count in countries


# In[30]:


number_of_days = (df["date"].max() -df["date"].min() ).days
dtfrm=data[data["Vaccine"]=="Pfizer/BioNTech"]
dtfrm = dtfrm.drop(dtfrm[dtfrm['Country'] == 'European Union'].index)


# In[34]:


dtfrm.head(10)


# In[35]:


dtfrm["average_vaccination_count"] = dtfrm["Total_vaccine"] / number_of_days
dtfrm["average_vaccination_count"] =dtfrm["average_vaccination_count"].astype(int)


# In[36]:


dtfrm.head(15)


# In[37]:


dtfrm.set_index("Country",inplace=True)
color=["Lightblue","Purple","Green","Orange","darkgoldenrod","tan","Gray","Blue","Pink","Lightgreen"]
dtfrm["average_vaccination_count"].sort_values(ascending=False).head(10).plot(kind="bar",color=color)


# In[38]:


# Number of countries where vaccines are used


# In[39]:


number_of_vaccines = data.groupby('Vaccine')['Country'].nunique()


# In[43]:


number_of_vaccines.sort_values(ascending=False).plot(kind="bar",color="b")


# In[5]:


# Machine learning Part


# In[ ]:





# In[ ]:




