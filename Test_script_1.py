#!/usr/bin/env python
# coding: utf-8

# ### Importing libraries

# In[1]:


import warnings
import pandas as pd
import numpy as np
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
from sklearn.impute import SimpleImputer
import argparse
warnings.filterwarnings('ignore')
from tsfresh import extract_features,select_features,feature_selection
import pickle 


# In[ ]:


parser = argparse.ArgumentParser(description='argument parsing')
parser.add_argument('--file', type=str)
args = parser.parse_args()
file = args.file
df = pd.read_csv(file)


# ### Pre-Processing

# In[3]:


df.dropna(axis=0,inplace=True,thresh = 20) #I need atleast 20 values out of 30
imp_mean = SimpleImputer(missing_values=np.nan,strategy='median')
imp_mean.fit(df)
result = imp_mean.transform(df)
result = pd.DataFrame(result)
columns = list(result.columns)
columns.pop()
columns.append('target')
result.columns = columns
y = result.target
result.drop( 'target', axis = 1, inplace = True )
d = result.stack()
d.index.rename([ 'id', 'time' ], inplace = True )
d = d.reset_index()


# ### Extracting features and selecting the top-5

# In[9]:


with warnings.catch_warnings():
        warnings.simplefilter( "ignore" )
        f = extract_features( d, column_id = "id", column_sort = "time" )
impute(f)
assert f.isnull().sum().sum() == 0


# In[16]:


columns = ['0__spkt_welch_density__coeff_2','0__fft_coefficient__coeff_1__attr_"abs"','0__partial_autocorrelation__lag_1',
          '0__autocorrelation__lag_1','0__autocorrelation__lag_2']
f = f[columns]
f.head()


# ### Loading our old model to make predictions

# In[21]:



loaded_model = pickle.load(open('finalized_model.sav', 'rb'))
result = loaded_model.predict(f)
print(result)


# In[ ]:




