import pandas as pd

#from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv('houses.csv')
df = df.dropna()  #removing rows with empty entries(NaN)
x = df.drop(columns = ['price'])
y = df['price']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.1)     #using 90% of dataset for training,rest for testing

#regr = linear_model.LinearRegression() 
regressor = RandomForestRegressor(n_estimators = 120, random_state = 0)
 


regressor.fit(x_train, y_train)
predicted = regressor.predict(x_test)
#predicted
score = regressor.score(x_test,y_test)
print(score*100,'%')