# In the Name of God

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import zscore
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('housePrice.csv')
data.head()
data.info()
data.describe()
data_cleaned = data.dropna(subset=['Address']).copy()
data_cleaned.info()
area_target_encoding = data_cleaned.groupby('Address')['Price(USD)'].apply(lambda x: np.log(x).mean())
data_cleaned.loc[:, 'Address_encoded'] = data_cleaned['Address'].map(area_target_encoding)
data_cleaned = data_cleaned.drop(columns=['Price'])
data_cleaned = data_cleaned.drop(columns=['Address'])

data_cleaned = data_cleaned.applymap(lambda x: float(str(x).replace(',', '')) if isinstance(x, str) else x)
print(data_cleaned.head())

z_scores = np.abs(zscore(data_cleaned.select_dtypes(include=['float64', 'int64'])))

data_no_outliers = data_cleaned[(z_scores < 3).all(axis=1)].copy()
print(data_no_outliers.head())

scaler = MinMaxScaler()
data_no_outliers[['Price(USD)', 'Area']] = scaler.fit_transform(data_no_outliers[['Price(USD)', 'Area']])
print(data_no_outliers[['Price(USD)', 'Area']].head())

plt.figure(figsize=(8, 6))
sns.histplot(data_no_outliers['Price(USD)'], kde=True)
plt.title('توزیع قیمت‌ها')
plt.xlabel('قیمت')
plt.ylabel('تعداد')
plt.show()

corr_matrix = data_no_outliers.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('همبستگی ویژگی‌ها')
plt.show()

X = data_no_outliers.drop(columns=['Price(USD)'])
y = data_no_outliers['Price(USD)']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_rf = RandomForestRegressor(n_estimators=100, random_state=42)

model_rf.fit(X_train, y_train)

y_pred_rf = model_rf.predict(X_test)

rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
mae_rf = mean_absolute_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print(f'RMSE: {rmse_rf}')
print(f'MAE: {mae_rf}')
print(f'R²: {r2_rf}')

X = data_no_outliers.drop(columns=['Price(USD)'])
y = data_no_outliers['Price(USD)']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

params = {
    'objective': 'reg:squarederror',  # هدف: رگرسیون با خطای مربعی
    'eval_metric': 'rmse',  # ارزیابی با RMSE
    'max_depth': 6,  # عمق درخت
    'learning_rate': 0.1,  # نرخ یادگیری
    'n_estimators': 100  # تعداد درخت‌ها
}

model_xgb = xgb.train(params, dtrain, num_boost_round=100)

y_pred_xgb = model_xgb.predict(dtest)

rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)

print(f'RMSE: {rmse_xgb}')
print(f'MAE: {mae_xgb}')
print(f'R²: {r2_xgb}')
