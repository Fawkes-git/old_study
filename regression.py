import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

print('Type name')
name = input()
data = pd.read_table(os.path.join(os.getcwd(),name,'train_data_pip.txt'))

y = data["Photo_front"].values / max(data["Photo_front"].values)
X = data[["Gesture"]].values 

regr = LinearRegression()

# create quadratic features
quadratic = PolynomialFeatures(degree=2)
cubic = PolynomialFeatures(degree=3)
quartic = PolynomialFeatures(degree=4)

X_quad = quadratic.fit_transform(X)
X_cubic = cubic.fit_transform(X)
X_quartic = quartic.fit_transform(X)

# fit features
X_fit = np.arange(X.min(), X.max(), 0.1)[:, np.newaxis]

regr = regr.fit(X, y)
y_lin_fit = regr.predict(X_fit)
linear_r2 = r2_score(y, regr.predict(X))

regr = regr.fit(X_quad, y)
y_quad_fit = regr.predict(quadratic.fit_transform(X_fit))
quadratic_r2 = r2_score(y, regr.predict(X_quad))

regr = regr.fit(X_cubic, y)
y_cubic_fit = regr.predict(cubic.fit_transform(X_fit))
cubic_r2 = r2_score(y, regr.predict(X_cubic))

regr = regr.fit(X_quartic, y)
y_quartic_fit = regr.predict(quartic.fit_transform(X_fit))
quartic_r2 = r2_score(y, regr.predict(X_quartic))

print(regr.coef_)
print(regr.intercept_)
print(mean_squared_error(y,regr.predict(X_quartic)))


# plot results
plt.scatter(X, y, label='training points', color='black')

plt.plot(X_fit, y_lin_fit, 
         label='linear (d=1), $R^2=%.3f$' % linear_r2, 
         color='blue', 
         lw=2, 
         linestyle=':')

plt.plot(X_fit, y_quad_fit, 
         label='quadratic (d=2), $R^2=%.3f$' % quadratic_r2,
         color='red', 
         lw=2,
         linestyle='-')

plt.plot(X_fit, y_cubic_fit, 
         label='cubic (d=3), $R^2=%.3f$' % cubic_r2,
         color='green', 
         lw=2, 
         linestyle='--')

plt.plot(X_fit, y_quartic_fit, 
         label='quartic (d=4), $R^2=%.3f$' % quartic_r2,
         color='purple', 
         lw=2, 
         linestyle='-.')

plt.xlabel('PIP angle [degree]')
plt.ylabel('Front Photo Sensor [Normalized Voltage]')
plt.legend(loc='lower left')

plt.tight_layout()
plt.savefig(os.path.join(os.getcwd(),name,'regression_front_vs_pip'), dpi=300)
#plt.show()