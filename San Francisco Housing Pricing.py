#!/usr/bin/env python
# coding: utf-8

# # 4 Project Steps: 
# 
# 1. Take big view.
# 2. Get your data ready, explore your data.
# 3. Feature engineering for ML algorithms
# 4. Pick ML model and train it -- today we use simple linear model and randomTreeRegressor

# ## 1. Take big view

# In[1]:


import numpy as np
import pandas as pd
import os
np.random.seed(42)

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')


# ##### We load open-source data from SF MLS historical database.

# In[2]:


df = pd.read_csv('Sales.csv') 
df.head()


#  Seems like we cannot see all the columns, to do so:

# In[3]:


pd.set_option('display.max_columns', None)
df.head()


# In[4]:


df.columns


# All the records are from California

# In[5]:


df['state'].unique()


# Further, all the records are from San Francisco

# In[6]:


df['city'].unique()


# ### Summary of the housing data

# In[7]:


## Each row represents one district
df.info()


# In[8]:


df.describe()


# In[9]:


df.head()


# ***We can have a look at a columns that seem to have impacts on the housing price:***
# 
# + longitude, latitute and elevation : for precise location of the house
# + full_address: also detailed location, if we use google map or other mapping system, we could map the (longtitude, latitute, elevation) location to street and numbers
# 
# + state and city: all records are from San Francisco, California
# 
# + street no, street name, street suffix: supplemental information for full address
# 
# + zip, area, district_no, district_desc: zip code, location and neighbourhood's name
# 
# + on_market_date, cdom: listing date and cumulative days on market
# 

# ### Let's have a look at the whole data and distribution:

# In[10]:


## Heatmap for median_income
df.plot(kind="scatter", x="longitude", y="latitude", alpha=0.5,
    s=df["elevation"]/60, label="elevation", figsize=(10,7),
    c="sale_price", cmap=plt.get_cmap("jet"), colorbar=True,
    sharex=False)
plt.legend()


# ## Get data ready and explore data

# We need to drop some data which is redundant and not needed in our analysis, for example, subdist_desc is a much better measure to determine the house price rather than street_no.

# In[11]:


df.columns


# In[12]:


df.drop(['full_address', 'city', 'state', 'street_no', 'street_name', 'street_suffix', 'district_no', 'district_desc'],         axis=1, inplace = True)


# In[13]:


df.head()


# Notice that, 'area' and 'subdist_no' have same value

# In[14]:


False in (df['area'] == df['subdist_no']).values


# And actually those columns refer to the regional location of the house inside SF city.

# In[15]:


df['subdist_desc'].unique()


# ![avatar](https://m2p7s3n2.rocketcdn.me/wp-content/uploads/2019/12/SanFranciscoNeighborhoods.jpg)

# #### Currently we don't need to consider attributes like on_market_date, zip, sale_date since sale_date normally wouldn't be considered as a major factor for sale price. Those features can represent the market preference and evaluation for certain houses. 

# In[14]:


df.drop(['area', 'subdist_no', 'zip', 'on_market_date', 'sale_date', 'lot_acres', 'orig_list_price'], axis=1, inplace = True)


# In[15]:


df


# For simplicity, here we only consider houses with positive square feet/acres area.

# In[16]:


df = df[df['lot_sf'] > 0]
df


# In[17]:


df.describe()


# #### Let's plot the distribution of each feature.

# In[20]:


df.hist(figsize=(20,15))
plt.show()


# We need to look at some features more granularly:

# In[18]:


df['sale_price'].hist(bins=100)


# In[19]:


df[df['sale_price'] == df['sale_price'].min()]


# In[20]:


df[df['sale_price'] == df['sale_price'].max()]


# In[22]:


nonzero_year_built = df[df['year_built'] > 0]['year_built']
nonzero_year_built


# In[23]:


nonzero_year_built.plot.box()


# Since some of the houses are missing year_built data, and after getting rid of the 0s, the data seems to still have many outliers, the missing data is best to be filled with the median. We observe the distribution from above.

# In[24]:


df['year_built']=df['year_built'].replace(0,nonzero_year_built.median())


# In[26]:


df[df['year_built'] == df['year_built'].min()]


# In[27]:


(df[df['year_built'] == df['year_built'].max()]).head()


# In[28]:


df['year_built'].hist()


# In[29]:


df['HouseAge'] = 2022 - df['year_built']
df.drop(['year_built'], axis=1, inplace= True)
df


# #### We can have a look at internal correlations between different features by scatter plotting them:

# In[31]:


from pandas.plotting import scatter_matrix

attributes = ["sale_price", "rooms", "baths",
              "beds", "lot_sf", "num_parking", "HouseAge"]
scatter_matrix(df[attributes], figsize=(20, 16));


# Obviously, we can see some positive correlation between price and lot_sf, baths, rooms, etc.

# In[32]:


df.plot(kind="scatter", x="sale_price", y="lot_sf",
             alpha=0.1, figsize=(12, 10))


# In[33]:


df.plot(kind="scatter", x="sale_price", y="baths",
             alpha=0.1, figsize=(12, 10))


# ##### We can also have a look at their correlation matrix and then sort the values, need to understand it clearly, not all of them are meaningful:

# In[34]:


corr_matrix = df.corr()
corr_matrix


# In[35]:


corr_matrix["sale_price"].sort_values(ascending=False)


# #### These values represent relationship between sale_price and other factors. For example, we can see factors like number of baths and beds plays a important role in setting the sale price while elevation and longitude barely have any impact on sale price.

# ## Feature engineering for ML algorithms

# In short terms, feature engineering is a process which translates some representation that computer or program has difficulty to understand into something easy for them to digest.
# 
# Actually the process we translate year_built to HouseAge is already a kind of feature engineering.
# 
# We noticed that (longitude, latitude, elevation) can represent location. As common sense, exact location will not matter too much in pricing the houses. What really matters is the relative location or neighbour hood.

# In[36]:


df.drop(['longitude', 'latitude',  'elevation'], axis=1, inplace=True)


# In[37]:


np.sort(df['subdist_desc'].unique())


# So in the dataset, we already have tags for different locations. By conducting research on San Francisco map, we find that each number represents a larger area and the following name is a more details area. For simplicity, we will only consider the larger region.

# In[38]:


df['subdist_desc'] = df['subdist_desc'].apply(lambda s: s.split()[0])


# In[39]:


region_label = df[['subdist_desc']]
region_label


# Since all of them are labeled under numbers, we need to use another way of encoding to eliminate the affect of numerical values.

# In[40]:


from sklearn.preprocessing import  OneHotEncoder


# In[41]:


cat_encoder = OneHotEncoder(sparse=False)
region_1hot = cat_encoder.fit_transform(region_label)
region_1hot


# In[42]:


cat_encoder.categories_


# In[43]:


df.drop(['subdist_desc'], axis=1, inplace=True)
df.drop(['cdom'],axis = 1, inplace = True)
df


# In[44]:


df.values.shape


# In[45]:


fulldata = np.c_[df.values, region_1hot]


# In[46]:


np.c_[df.values, region_1hot].shape


# In[47]:


True in np.isnan(fulldata)


# ## Pick ML model and train it -- today we use simple linear model and random tree regressor 

# ***Split training and testing data sets:***

# #### simple linear model

# In[48]:


from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(fulldata, test_size=0.2, random_state=42)


# In[49]:


train_target = train_set[:, 0]
train_target = train_target.reshape(len(train_target),-1)
train_target


# In[50]:


train_features = train_set[:, 1:]
train_features


# In[51]:


test_target = test_set[:, 0]
test_target = test_target.reshape(len(test_target),-1)
test_features = test_set[:, 1:]


# In[52]:


train_target.shape, train_features.shape


# In[66]:


from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(train_features, train_target)


# #### Random Forest Regressor

# In[58]:



from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor(n_estimators=150,random_state = 1)
rf_model.fit(train_features, train_target)


# In[59]:


regression.coef_


# In[60]:


regression.intercept_


# ***Then we need to evaluate the performance of the model on training and testing sets:***

# ##### Error using Simple Linear Model

# In[61]:


from sklearn.metrics import mean_squared_error


# In[62]:


price_predictions_on_train = regression.predict(train_features)
mse = mean_squared_error(train_target, price_predictions_on_train)
sqrtmse = np.sqrt(mse)
sqrtmse


# In[64]:


price_predictions_on_test = regression.predict(test_features)
mse = mean_squared_error(test_target, price_predictions_on_test)
sqrtmse = np.sqrt(mse)
sqrtmse


# ##### Error using Random Forest Regressor

# In[63]:


price_predictions_on_train = rf_model.predict(train_features)
mse = mean_squared_error(train_target, price_predictions_on_train)
sqrtmse = np.sqrt(mse)
sqrtmse


# In[65]:


price_predictions_on_test = rf_model.predict(test_features)
mse = mean_squared_error(test_target, price_predictions_on_test)
sqrtmse = np.sqrt(mse)
sqrtmse

