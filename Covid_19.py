#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd


# In[3]:


import plotly.graph_objects as go
import plotly.express as px
import plotly.offline as pyo
from plotly.subplots import make_subplots
pyo.init_notebook_mode()


# In[4]:


from datetime import date , datetime , timedelta


# In[5]:


data_detailed = pd.read_csv('D:/Internship project/country_vaccinations.csv')
data_total = pd.read_csv('D:/Internship project/country_vaccinations_by_manufacturer.csv')


# In[6]:


print("* "*10+" data_detailed "+" *"*10)
print("\nShape: rows = {} , columns = {}".format(data_detailed.shape[0] , data_detailed.shape[1]))
print(data_detailed.info())
print("* "*10+" data_total "+" *"*10)
print("\nShape: rows = {} , columns = {}".format(data_total.shape[0] , data_total.shape[1]))
print(data_total.info())


# In[7]:


data_detailed.date.max() , data_total.date.max()


# In[8]:


data_detailed.tail(3)


# In[9]:


data_total.tail(3)


# In[11]:


# Which country is using what vaccine 


# In[10]:


countries = data_detailed.country.unique()

for country in countries:
    print(country,end = ":\n")
    print(data_detailed[data_detailed.country == country]['vaccines'].unique()[0] , end = "\n"+"_"*20+"\n\n")


# In[13]:


# percentage of fully vaccinated people for each country


# In[11]:


last_date = data_detailed.sort_values(by = 'date' , ascending=False)['date'].iloc[0]


# In[16]:


# its ''2022-3-29'


# In[12]:


data_detailed[(data_detailed.date == last_date)&(data_detailed.people_fully_vaccinated_per_hundred.isnull())]


# In[13]:


data_detailed[(data_detailed.date == last_date)&(data_detailed.country == 'India')]


# In[14]:


dict_vac_percentages = {}
iso_list = data_detailed.iso_code.unique()
for iso_code in iso_list:
    dict_vac_percentages[iso_code]=data_detailed[data_detailed.iso_code==iso_code]['people_fully_vaccinated_per_hundred'].max()

df_vac_percentages = pd.DataFrame()
df_vac_percentages['iso_code'] = dict_vac_percentages.keys()
df_vac_percentages['fully vaccinated percentage'] = dict_vac_percentages.values()
df_vac_percentages['country'] = countries


# In[15]:


dict_total_vac = {}
for iso_code in iso_list:
    dict_total_vac[iso_code]=data_detailed[data_detailed.iso_code==iso_code]['total_vaccinations'].max()

df_total_vac = pd.DataFrame()
df_total_vac['iso_code'] = dict_total_vac.keys()
df_total_vac['total vaccinations'] = dict_total_vac.values()
df_total_vac['country'] = countries


# In[16]:


map_total_vac = px.choropleth(data_frame = df_total_vac , locations="iso_code" , color="total vaccinations" 
                             , hover_name="country" , color_continuous_scale=px.colors.sequential.deep)
map_total_vac.update_layout(title_text='Total vaccinations in each country'
                                  , title_font={'family':'serif','size':26} , title = {'y':0.94 , 'x':0.45})
map_total_vac.show()


# In[23]:


# Fully vaccinated people percentage


# In[17]:


map_full_percentage = px.choropleth(df_vac_percentages, locations="iso_code" , color="fully vaccinated percentage"
                                    , hover_name="country" , color_continuous_scale=px.colors.sequential.YlGn)

map_full_percentage.show()


# In[25]:


# Vaccine usage in Eropean Union


# In[18]:


euro_vaccines = data_total[(data_total.location == 'European Union') &
                         (data_total.date == last_date)][['vaccine','total_vaccinations']]
euro_vaccines.sort_values(by = 'total_vaccinations' , ascending = False , inplace = True)


# In[19]:


euro_vaccines


# In[20]:


pie_euro_vac = go.Figure(data = go.Pie(values = euro_vaccines.total_vaccinations, 
                          labels = euro_vaccines.vaccine, hole = 0.55))
pie_euro_vac.update_traces(textposition='outside', textinfo='percent+label')
pie_euro_vac.update_layout(annotations=[dict(text='Vaccines used by', x=0.5, y=0.55, font_size=16, showarrow=False),
                                       dict(text='European Union', x=0.5, y=0.45, font_size=16, showarrow=False)])
pie_euro_vac.show()


# In[28]:


# Vaccine usage in perticular country - Germany


# In[21]:


data_detailed[data_detailed.country == 'Germany']['date'].max() , data_total[data_total.location == 'Germany']['date'].max()


# In[23]:


germany_vaccines=data_total[(data_total.location=='Germany')&(data_total.date=='2021-10-21')][['vaccine','total_vaccinations']]
germany_vaccines.sort_values(by = 'total_vaccinations' , ascending = False , inplace = True)
df_germany_info = data_detailed[data_detailed.country == 'Germany']


# In[24]:


fig_germany = make_subplots(rows = 4 , cols = 2
    , specs=[[{"type": "pie","rowspan": 2}, {"type": "scatter","rowspan": 2}]
           ,[None , None]
           ,[{"type": "scatter","colspan": 2,"rowspan": 2}, None]
           ,[None , None]]
                            
    , subplot_titles=[
        '', 
        'temp',
        'temp' # i will change the titles a few lines later ...
    ])

fig_germany.add_trace(go.Pie(labels = germany_vaccines.vaccine , values = germany_vaccines.total_vaccinations
                                   , hole = 0.5 , pull = [0,0.1,0.1,0.1] , title = "Vaccines" , titleposition='middle center'
                                   , titlefont = {'family':'serif' , 'size':18}
                                   , textinfo = 'percent+label' , textposition = 'inside')
                     , row = 1 , col = 1)

fig_germany.add_trace(go.Scatter(x = df_germany_info['date']
                                , y = df_germany_info['daily_vaccinations']
                                , name = "Daily vaccinations")
                     , row = 1 , col = 2)

fig_germany.add_trace(go.Scatter(x = df_germany_info['date']
                                , y = df_germany_info['people_fully_vaccinated_per_hundred']
                                , name = "Fully vaccinated people percentage"
                                 # <br> for the next line in hover
                                , hovertemplate = "<b>%{x}</b><br>" +"Fully vaccinated people = %{y:.2f} %" +"<extra></extra>")
                     , row = 3 , col = 1)


fig_germany.layout.annotations[0].update(text="Number of daily vaccinations" , x=0.75
                                         , font = {'family':'serif','size':20})

fig_germany.layout.annotations[1].update(text="Fully vaccinated people percentage" , x=0.25 
                                         , font = {'family':'serif','size':20})

fig_germany.update_yaxes(range=[0, 100], row=3, col=1)
fig_germany.update_layout(width = 950,height=600, showlegend=True)
fig_germany.update_layout(title_text='Germany abstract informations'
                                  ,title_font={'family':'serif','size':26} , title = {'x':0.25 , 'y':0.95})
fig_germany.update_layout(template = 'plotly_dark')
fig_germany.show()


# In[35]:


# Preparing Data for Machine learning


# In[36]:


# Data for predicting percentage of fully vaccinated people in Geramny


# In[25]:


data = pd.DataFrame()
data['Date'] = pd.to_datetime(df_germany_info['date'])
data['Target'] = df_germany_info['people_fully_vaccinated_per_hundred']
data.reset_index(drop = True , inplace = True)


# In[26]:


data.Date.min() , data.Date.max() , len(data)


# In[27]:


d0 = date(2020 , 12 , 27)
d1 = date(2021 , 10 , 21)
delta = d1 - d0

days = delta.days + 1
print(days)


# In[28]:


data.isnull().sum()


# In[29]:


data['Series'] = np.arange(1 , len(data)+1)


# In[30]:


data['Shift1'] = data.Target.shift(1)


# In[31]:


window_len = 10
window = data['Shift1'].rolling(window = window_len)
means = window.mean()
data['Window_mean'] = means


# In[44]:


data.dropna(inplace = True)
data.reset_index(drop = True , inplace=True)

dates = data['Date'] # we will need this

data = data[['Series' , 'Window_mean' , 'Shift1' , 'Target']]

data


# In[ ]:




