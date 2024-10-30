import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

data = pd.read_csv('gym_churn_us.csv')
data.info()
print(data.head())
# Observa el dataset: ¿contiene alguna característica ausente? Estudia los valores promedio y la desviación estándar (utiliza el método describe())
data.describe()

# Observa los valores medios de las características en dos grupos: para las personas que se fueron (cancelación) y para las que se quedaron (utiliza el método groupby())
print(data.groupby('Churn').mean())

# Traza histogramas de barras y distribuciones de características para aquellas personas que se fueron (cancelación) y para las que se quedaron.
data_b = data.groupby('Churn')
data_b.hist()
plt.clf()
# Crea una matriz de correlación y muéstrala.
cm = data.corr()
fig = sns.heatmap(cm, annot = True, square=True)
plt.show(fig)


# Paso 3. Construir un modelo para predecir la cancelación de usuarios


X = data.drop('Churn', axis = 1)
y = data['Churn']
scaler = StandardScaler()
scaler.fit(X_train)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

X_train_st = scaler.transform(X_train)
X_test_st = scaler.transform(X_test)

# Regresión logística
model_l = LogisticRegression()
model_l.fit(X_train_st, y_train)
y_pred_l = model_l.predict(X_test_st)

# Bosque aleatorio
model_r = RandomForestRegressor(n_estimators = 100)
model_r.fit(X_train_st, y_train)
y_pred_r = model_r.predict(X_test_st)

