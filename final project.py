# In the Name of God

import pandas as pd
import numpy as np

data = pd.read_csv('housePrice.csv')
data.head()
data.info()
data.describe()
data_cleaned = data.dropna(subset=['Address']).copy()
data_cleaned.info()
area_target_encoding = data_cleaned.groupby('Address')['Price(USD)'].apply(lambda x: np.log(x).mean())
data_cleaned.loc[:, 'Address_encoded'] = data_cleaned['Address'].map(area_target_encoding)
data_cleaned = data_cleaned.drop(columns=['Price'])
