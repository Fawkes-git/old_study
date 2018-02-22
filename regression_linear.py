import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

linestyles = ['-', '--', '-.', ':']
deg_label = ["linear","quadratic","cubic","quartic"]

#print('Type name')
#name = input()

name = "backvsmp"
data = pd.read_table(os.path.join(os.getcwd(),name,'train_data.txt'))

X = data[["Photo_back"]].values / max(data[["Photo_back"]].values)
y = data["Gesture"].values 

base = 3
X_power = np.power(X,base)

X_train, X_test, y_train, y_test = train_test_split(
    X_power, y, test_size=0.2, random_state=0)

regr = LinearRegression(n_jobs=-1)
regr = regr.fit(X_train, y_train)

# fit features
X_fit = np.arange(X_train.min(), X_train.max(), 0.001)[:, np.newaxis]
y_train_pred = regr.predict(X_fit)

y_test_pred = regr.predict(X_test)
r2_train = r2_score(y_train, regr.predict(X_train))
r2_test = r2_score(y_test, regr.predict(X_test))
mse_train = mean_squared_error(y_train,regr.predict(X_train))
mse_test = mean_squared_error(y_test,regr.predict(X_test))

print("Train MSE:{:.3}".format(mse_train))
print(" Test MSE:{:.3}".format(mse_test))
print("Train R2:{:.3}".format(r2_train))
print(" Test R2:{:.3}".format(r2_test))

plt.plot(X_fit, y_train_pred, color='blue',label='linear regression, $R^2={:.3}$, y={:.3}x+{:.3}, MSE={:.4}'.format(r2_train,regr.coef_[0],regr.intercept_,mean_squared_error(y_train,regr.predict(X_train))),linewidth=2)

w_0 = regr.intercept_
w_1 = regr.coef_[0]

print("y={:.3}x{:.3}".format(w_0,w_1))

# plot results
plt.scatter(X_train, y_train, label='training points', color='black')
plt.scatter(X_test, y_test, label='test points', color='green')
plt.xlabel('{:}^Back Photo Sensor [Normalized Voltage]'.format(base))
plt.ylabel('MP angle [degree]')
plt.legend(loc='lower left')

#plt.tight_layout()
plt.savefig(os.path.join(os.getcwd(),name,'regression_front_vs_pip'), dpi=300)
#plt.show()