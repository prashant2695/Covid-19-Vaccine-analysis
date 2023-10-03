#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


import plotly.graph_objects as go
import plotly.express as px
import plotly.offline as pyo
from plotly.subplots import make_subplots
pyo.init_notebook_mode()


# In[3]:


from datetime import date , datetime , timedelta


# In[4]:


data_detailed = pd.read_csv('D:/Internship project/country_vaccinations.csv')
data_total = pd.read_csv('D:/Internship project/country_vaccinations_by_manufacturer.csv')


# In[5]:


print("* "*10+" data_detailed "+" *"*10)
print("\nShape: rows = {} , columns = {}".format(data_detailed.shape[0] , data_detailed.shape[1]))
print(data_detailed.info())
print("* "*10+" data_total "+" *"*10)
print("\nShape: rows = {} , columns = {}".format(data_total.shape[0] , data_total.shape[1]))
print(data_total.info())


# In[6]:


data_detailed.date.max() , data_total.date.max()


# In[7]:


data_detailed.tail(3)


# In[8]:


data_total.tail(3)


# In[9]:


# Which country is using what vaccine 


# In[10]:


countries = data_detailed.country.unique()

for country in countries:
    print(country,end = ":\n")
    print(data_detailed[data_detailed.country == country]['vaccines'].unique()[0] , end = "\n"+"_"*20+"\n\n")


# In[11]:


# percentage of fully vaccinated people for each country


# In[12]:


last_date = data_detailed.sort_values(by = 'date' , ascending=False)['date'].iloc[0]


# In[13]:


# its ''2022-3-29'


# In[14]:


data_detailed[(data_detailed.date == last_date)&(data_detailed.people_fully_vaccinated_per_hundred.isnull())]


# In[15]:


data_detailed[(data_detailed.date == last_date)&(data_detailed.country == 'India')]


# In[16]:


dict_vac_percentages = {}
iso_list = data_detailed.iso_code.unique()
for iso_code in iso_list:
    dict_vac_percentages[iso_code]=data_detailed[data_detailed.iso_code==iso_code]['people_fully_vaccinated_per_hundred'].max()

df_vac_percentages = pd.DataFrame()
df_vac_percentages['iso_code'] = dict_vac_percentages.keys()
df_vac_percentages['fully vaccinated percentage'] = dict_vac_percentages.values()
df_vac_percentages['country'] = countries


# In[17]:


dict_total_vac = {}
for iso_code in iso_list:
    dict_total_vac[iso_code]=data_detailed[data_detailed.iso_code==iso_code]['total_vaccinations'].max()

df_total_vac = pd.DataFrame()
df_total_vac['iso_code'] = dict_total_vac.keys()
df_total_vac['total vaccinations'] = dict_total_vac.values()
df_total_vac['country'] = countries


# In[18]:


map_total_vac = px.choropleth(data_frame = df_total_vac , locations="iso_code" , color="total vaccinations" 
                             , hover_name="country" , color_continuous_scale=px.colors.sequential.deep)
map_total_vac.update_layout(title_text='Total vaccinations in each country'
                                  , title_font={'family':'serif','size':26} , title = {'y':0.94 , 'x':0.45})
map_total_vac.show()


# In[19]:


# Fully vaccinated people percentage


# In[20]:


map_full_percentage = px.choropleth(df_vac_percentages, locations="iso_code" , color="fully vaccinated percentage"
                                    , hover_name="country" , color_continuous_scale=px.colors.sequential.YlGn)

map_full_percentage.show()


# In[21]:


# Vaccine usage in Eropean Union


# In[22]:


euro_vaccines = data_total[(data_total.location == 'European Union') &
                         (data_total.date == last_date)][['vaccine','total_vaccinations']]
euro_vaccines.sort_values(by = 'total_vaccinations' , ascending = False , inplace = True)


# In[23]:


euro_vaccines


# In[24]:


pie_euro_vac = go.Figure(data = go.Pie(values = euro_vaccines.total_vaccinations, 
                          labels = euro_vaccines.vaccine, hole = 0.55))
pie_euro_vac.update_traces(textposition='outside', textinfo='percent+label')
pie_euro_vac.update_layout(annotations=[dict(text='Vaccines used by', x=0.5, y=0.55, font_size=16, showarrow=False),
                                       dict(text='European Union', x=0.5, y=0.45, font_size=16, showarrow=False)])
pie_euro_vac.show()


# In[25]:


# Vaccine usage in perticular country - Germany


# In[26]:


data_detailed[data_detailed.country == 'Germany']['date'].max() , data_total[data_total.location == 'Germany']['date'].max()


# In[27]:


germany_vaccines=data_total[(data_total.location=='Germany')&(data_total.date=='2021-10-21')][['vaccine','total_vaccinations']]
germany_vaccines.sort_values(by = 'total_vaccinations' , ascending = False , inplace = True)
df_germany_info = data_detailed[data_detailed.country == 'Germany']


# In[28]:


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


# In[29]:


# Preparing Data for Machine learning


# In[68]:


dataset = pd.read_csv('D:/Internship project/country_vaccinations.csv')


# In[69]:


dataset.head(10)


# In[74]:


dataset.columns


# In[76]:


x_df = dataset[['country', 'date','daily_vaccinations_raw', 'daily_vaccinations','total_vaccinations_per_hundred', 'people_vaccinated_per_hundred','people_fully_vaccinated_per_hundred', 'daily_vaccinations_per_million','vaccines']]

y_df = dataset[['total_vaccinations']]


# In[77]:


x = x_df.values
y = y_df.values


# In[78]:


x[0:10,:]


# In[82]:


y[0:10,:]


# In[83]:


from sklearn.impute import SimpleImputer # importing the SimpleImputer class that let's us replace the missing values 
                                         # with the average of the column

imputer = SimpleImputer(missing_values= np.nan, strategy="mean")


# In[84]:


imputer.fit(X = x[:, 2:8])
x[:,2:8] = imputer.transform(x[:,2:8])
x[0:10,:]


# In[85]:


imputer.fit(X = y) 

y = imputer.transform(y) 
y[0:10,:]


# In[86]:


x_df = pd.DataFrame(x, columns = ['country', 'date','daily_vaccinations_raw', 'daily_vaccinations','total_vaccinations_per_hundred', 'people_vaccinated_per_hundred','people_fully_vaccinated_per_hundred', 'daily_vaccinations_per_million','vaccines'])
x_df


# In[87]:


x_df_nodummies = x_df
x_df = pd.get_dummies(x_df, columns=["date","country", "vaccines"], prefix=["date","country", "vaccines"]) # now, x_df has the dummy variables
x_df


# In[88]:


x = x_df.values
x


# In[89]:


y


# In[90]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)


# In[91]:


len(x_train)


# In[92]:


len(x_test)


# In[93]:


# Multiple Linear Regression


# In[94]:


from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_train, y_train)
LinearRegression()


# In[95]:


y_pred = regressor.predict(x_test)
np.set_printoptions(precision = 0)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


# In[96]:


from sklearn.metrics import r2_score
r2_score(y_test, y_pred)


# In[97]:


#  Random Forest Regression model


# In[99]:


from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(x_train, y_train)


# In[100]:


y_pred = regressor.predict(x_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


# In[107]:


from sklearn.metrics import r2_score
r2_score(y_test, y_pred)


# In[167]:


plt.figure(figsize=(100, 80))
plt.plot_date(x_df_nodummies.date, x_df_nodummies.vaccines)
plt.title("Vaccine Data Over Time", fontsize=16)
plt.xlabel("Date", fontsize=14)
plt.ylabel("Vaccines", fontsize=14)
plt.xticks(rotation='vertical', fontsize=12)
plt.savefig("vaccine_plot.png")


# In[ ]:


# Random Forest Regression model with a fit of 99.58% which is really great! Now, you can be sure that this is the model that will perform better in case you want to make a prediction of the total vaccinations based on the features selected as predictors in x.


# In[ ]:




