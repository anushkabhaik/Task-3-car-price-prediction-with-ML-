import numpy as np
import pandas as pd

cars = pd.read_csv('car_data.csv')
cars.head()
cars.describe()
cars.isnull().sum()

cars = pd.get_dummies(cars, columns=['Fuel_Type','Selling_type','Transmission','Owner'], drop_first = True)
cars.head()

independent = ['Year','Present_Price', 'Driven_kms','Fuel_Type_Diesel','Fuel_Type_Petrol', 'Selling_type_Individual','Transmission_Manual','Owner_1','Owner_3']
X = cars[independent]
y = cars['Selling_Price']

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X.head()

import matplotlib.pyplot as plt
import seaborn as sns

corr_matrix=cars.corr()
sns.heatmap(corr_matrix,annot=True,cmap='coolwarm')
plt.show()

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X_scaled,y,test_size=0.2,random_state=50)

from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=100,random_state=50)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)

from sklearn.metrics import mean_squared_error,r2_score
mse = mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
print("mse = ",mse)
print("r2_score = ",r2)







