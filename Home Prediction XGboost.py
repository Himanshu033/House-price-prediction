#!/usr/bin/env python
# coding: utf-8

# In[8]:


get_ipython().system('pip install XGboost')
get_ipython().system('pip install sklearn')
get_ipython().system('pip install matplotlib')
get_ipython().system('pip install numpy')
get_ipython().system('pip install seaborn')


# In[9]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
from sklearn import preprocessing
from collections import Counter
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
pd.set_option('precision', 3)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import xgboost as xgb


# In[10]:


houses_train = pd.read_csv("H:\\train.csv", header=0, delimiter=',')
houses_train.head()
houses_train.shape


# In[11]:


houses_train.describe()


# #We start examining our data to fill missing values

# In[12]:


null_columns = houses_train.columns[houses_train.isnull().any()]
houses_train[null_columns].isna().sum()


# In[13]:


houses_test = pd.read_csv("H:\\test.csv", header=0, delimiter=',')
houses_test.head()
houses_test.shape


# In[14]:


houses_train = houses_train.drop(columns = ['PoolQC', 'Fence', 'MiscFeature', 'Alley']) #We start deleting the features with more than half of null values
houses_test = houses_test.drop(columns = ['PoolQC', 'Fence', 'MiscFeature', 'Alley'])
null_columns.drop(['PoolQC', 'Fence', 'MiscFeature', 'Alley'])


# #Now, for the rest of data, we will first do an exploration of the data to, among other things, decide which is the best way to imput the missing values

# In[15]:


# Collect the names of the Categorical and Numeric Variables seperately
num_columns = houses_train.select_dtypes(include=np.number).columns.tolist()
num_columns.remove("SalePrice") # Capturing feature names exclusively
cat_columns = houses_train.select_dtypes(exclude=np.number).columns.tolist()

# Check if the number makes sense (+1 for the target variable that was dropped)
len(num_columns) + len(cat_columns) + 1 == len(houses_train.columns)


# In[16]:


for col in cat_columns:
    print(col + ": " + str(len(houses_train[col].unique()))) # we print categorical data columns and their distinct amount of categories


# In[17]:


#We start visualizing a heat map for all numerical features
plt.figure(figsize=(32,16))
sn.heatmap(houses_train.corr(),cmap='magma_r',annot=True) #Big correlation between YearBuilt and GarageYrBuilt -> delete GarageYrBuilt (difficult to impute null values)
houses_test = houses_test.drop(columns = 'GarageYrBlt', axis = 1)
houses_train = houses_train.drop(columns = 'GarageYrBlt', axis = 1)


# In[18]:


fig,ax = plt.subplots(3,1,figsize=(15,15))
sn.lineplot(x=houses_train['OverallQual'],y=houses_train.SalePrice,ax=ax[0],color='r') #We visualize 3 features with different levels of correlation with SalePrice
sn.lineplot(x=houses_train['YearBuilt'],y=houses_train.SalePrice,ax=ax[1],color='b')
sn.lineplot(x=houses_train['MSSubClass'],y=houses_train.SalePrice,ax=ax[2],color='g')


# procede to input the values to the categorical data

# In[19]:


def fill_nulls(houses_df):
    houses_df.loc[houses_df.FireplaceQu.isna(),'FireplaceQu'] = 'None'
    houses_df.loc[houses_df.GarageType.isna(),'GarageType'] = 'None' #No garage >>
    houses_df.loc[houses_df.GarageFinish.isna(),'GarageFinish'] = 'None' #>>
    houses_df.loc[houses_df.GarageQual.isna(),'GarageQual'] = 'None' #>>
    houses_df.loc[houses_df.GarageCond.isna(),'GarageCond'] = 'None' #>>
    houses_df.loc[houses_df.BsmtExposure.isna(),'BsmtExposure'] = 'None' #No basement >>
    houses_df.loc[houses_df.BsmtFinType2.isna(),'BsmtFinType2'] = 'None' #>>
    houses_df.loc[houses_df.BsmtCond.isna(),'BsmtCond'] = 'None' #>>
    houses_df.loc[houses_df.BsmtFinType1.isna(),'BsmtFinType1'] = 'None' #>>
    houses_df.loc[houses_df.BsmtQual.isna(),'BsmtQual'] = 'None' #>>
    houses_df.loc[houses_df.MasVnrType.isna(),'MasVnrType'] = 'None' #No masonry >>
    houses_df.loc[houses_df.MasVnrArea.isna(),'MasVnrArea'] = 0 #>>
    houses_df.loc[houses_df.LotFrontage.isna(),'LotFrontage'] = 0 #No lot frontage
    houses_df = houses_df.loc[~houses_df.Electrical.isna()] #Dropping ONE row with Nan value in 'Electrical'
    houses_df.loc[houses_df.MSZoning.isna(),'MSZoning'] = 'None'
    houses_df.loc[houses_df.Utilities.isna(),'Utilities'] = 'None'
    houses_df.loc[houses_df.Exterior1st.isna(),'Exterior1st'] = 'None'
    houses_df.loc[houses_df.Exterior2nd.isna(),'Exterior2nd'] = 'None'
    houses_df.loc[houses_df.BsmtFinSF1.isna(),'BsmtFinSF1'] = 0
    houses_df.loc[houses_df.BsmtFinSF2.isna(),'BsmtFinSF2'] = 0
    houses_df.loc[houses_df.BsmtUnfSF.isna(),'BsmtUnfSF'] = 0
    houses_df.loc[houses_df.TotalBsmtSF.isna(),'TotalBsmtSF'] = 0
    houses_df.loc[houses_df.BsmtFullBath.isna(),'BsmtFullBath'] = 0
    houses_df.loc[houses_df.BsmtHalfBath.isna(),'BsmtHalfBath'] = 0
    houses_df.loc[houses_df.KitchenQual.isna(),'KitchenQual'] = 'None'
    houses_df.loc[houses_df.Functional.isna(),'Functional'] = 'None'
    houses_df.loc[houses_df.GarageCars.isna(),'GarageCars'] = 0
    houses_df.loc[houses_df.GarageArea.isna(),'GarageArea'] = 0
    houses_df.loc[houses_df.SaleType.isna(),'SaleType'] = 'None'
    return houses_df

houses_train = fill_nulls(houses_train)
houses_test = fill_nulls(houses_test)


# In[20]:


null_columns = houses_train.columns[houses_train.isnull().any()] #No more null values left
print(null_columns)


# In[21]:


houses_train['FireplaceQu'].value_counts() #There's a lot of categorical data with this punctuation method


# In[22]:


#We substitute all categorical that follow that common punctuation method with integers because in this case the categories do follow an order.
def new_punctuation(houses_df):
    houses_df['ExterQual'] = houses_df['ExterQual'].map({'Ex':5,'Fa':2,'Gd':4,'TA':3,'Po':1,'None':0}).astype('int64')
    houses_df['ExterCond'] = houses_df['ExterCond'].map({'Ex':5,'Fa':2,'Gd':4,'TA':3,'Po':1,'None':0}).astype('int64')
    houses_df['BsmtQual'] = houses_df['BsmtQual'].map({'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,'None':0}).astype('int64')
    houses_df['BsmtCond'] = houses_df['BsmtCond'].map({'Ex':5,'Fa':2,'Gd':4,'TA':3,'Po':1,'None':0}).astype('int64')
    houses_df['BsmtExposure'] = houses_df['BsmtExposure'].map({'Av':3,'Gd':4,'Mn':2,'No':1,'None':0}).astype('int64')
    houses_df['BsmtFinType1'] = houses_df['BsmtFinType1'].map({'ALQ':5, 'BLQ':4, 'GLQ':6, 'LwQ':2, 'None':0, 'Rec':3, 'Unf':1}).astype('int64')
    houses_df['BsmtFinType2'] = houses_df['BsmtFinType2'].map({'ALQ':5, 'BLQ':4, 'GLQ':6, 'LwQ':2, 'None':0, 'Rec':3, 'Unf':1}).astype('int64')
    houses_df['HeatingQC'] = houses_df['HeatingQC'].map({'Ex':5, 'Fa':2, 'Gd':4, 'Po':1, 'TA':3,'None':0}).astype('int64')
    houses_df['KitchenQual'] = houses_df['KitchenQual'].map({'Ex':5,'Fa':2,'Gd':4,'TA':3,'Po':1,'None':0}).astype('int64')
    houses_df['FireplaceQu'] = houses_df['FireplaceQu'].map({'Ex':5, 'Fa':2, 'Gd':4, 'None':0, 'Po':1, 'TA':3}).astype('int64')
    houses_df['GarageCond'] = houses_df['GarageCond'].map({'Ex':5, 'Fa':2, 'Gd':4, 'None':0, 'Po':1, 'TA':3}).astype('int64')
    houses_df['GarageQual'] = houses_df['GarageQual'].map({'Ex':5, 'Fa':2, 'Gd':4, 'None':0, 'Po':1, 'TA':3}).astype('int64')
    return houses_df

houses_train = new_punctuation(houses_train)
houses_test = new_punctuation(houses_test)


# In[23]:


cat_columns = houses_train.select_dtypes(exclude=np.number).columns.tolist() 
cat_columns_2 = houses_test.select_dtypes(exclude=np.number).columns.tolist()
print(houses_train['SaleType'].value_counts())
print(cat_columns)
print(cat_columns == cat_columns_2) #Train and test dataframes have the same categorical columns


# In[24]:


cat_columns = houses_train.select_dtypes(exclude=np.number).columns.tolist() 
enc = preprocessing.OrdinalEncoder(dtype = int)
houses_train[cat_columns] = enc.fit_transform(houses_train[cat_columns])
houses_test[cat_columns] = enc.fit_transform(houses_test[cat_columns])

print(houses_train.select_dtypes(exclude=np.number).columns.tolist())


# In[25]:


X = houses_train.drop("SalePrice", axis=1)
Y = houses_train["SalePrice"]
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.2,random_state=0)


# In[26]:


xgb_model = xgb.XGBRegressor()
cvxgb = cross_val_score(xgb_model, x_train, y_train, cv = 5)
print(cvxgb)
print(cvxgb.mean())


# In[27]:


rf = RandomForestRegressor()
cvrf = cross_val_score(rf, x_train, y_train, cv = 5)
print(cvrf)
print(cvrf.mean())


# In[ ]:




