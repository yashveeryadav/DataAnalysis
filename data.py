# Dynamic Multiple Regression
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm

df = pd.read_csv("/home/yash/advertising.csv")
plt.figure()
# set the independent variables
independent = ['TV', 'Radio', 'Newspaper', 'hordings', 'apps']
# set all the dependent variables here
dependent = 'Sales'
sns.pairplot(df, x_vars=independent, y_vars= dependent, size=7)
plt.savefig('/home/yash/Downloads/sns1.png')
x = df[independent]
y = df[dependent]
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=.7, random_state=100)
lm = LinearRegression()
lm.fit(x_train, y_train)
print(lm.intercept_)
coeff_df = pd.DataFrame(lm.coef_, x_test.columns, columns=['Coefficient'])
print(coeff_df)
y_pred = lm.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
r_squred = r2_score(y_test,y_pred)
print(mse)
print("Model Accurecy = "+str(r_squred))
x_train_sm = x_train
x_train_sm = sm.add_constant(x_train_sm)
lm_l = sm.OLS(y_train,x_train_sm).fit()
print(lm_l.summary())

print("Done")
