import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score
from scipy.cluster.hierarchy import dendrogram, linkage

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


# Paso 3. Construir un modelo para predecir la cancelación de usuarios


X = data.drop('Churn', axis = 1)
y = data['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 0)

scaler = StandardScaler()
scaler.fit(X_train)

X_train_st = scaler.transform(X_train)
X_test_st = scaler.transform(X_test)

# Regresión logística
model_l = LogisticRegression()
model_l.fit(X_train_st, y_train)
y_pred_l = model_l.predict(X_test_st)
prob_l = model_l.predict_proba(X_test_st)[:, 1]

# Bosque aleatorio
model_r = RandomForestClassifier(n_estimators = 100)
model_r.fit(X_train_st, y_train)
y_pred_r = model_r.predict(X_test_st)
prob_r = model_r.predict_proba(X_test_st)[:, 1]

# función para la métricas
def print_all_metrics(y_true, y_pred, y_proba, title = 'Métricas de clasificación'):
    print(title)
    print('\tAccuracy: {:.2f}'.format(accuracy_score(y_true, y_pred)))
    print('\tPrecision: {:.2f}'.format(precision_score(y_true, y_pred)))
    print('\tRecall: {:.2f}'.format(recall_score(y_true, y_pred)))
    print('\tF1: {:.2f}'.format(f1_score(y_true, y_pred)))
    print('\tROC_AUC: {:.2f}'.format(roc_auc_score(y_true, y_proba)))

print_all_metrics(y_test, y_pred_l, prob_l , title='Métricas de regresión logística:')
print_all_metrics(y_test, y_pred_r, prob_r , title='Métricas de regresión logística:')

# Paso 4. Crear clústeres de usuarios/as

cluster_data = data.drop('Churn', axis = 1)
sc = StandardScaler()
x_sc = sc.fit_transform(cluster_data)

linked = linkage(x_sc, method= 'ward')
# Dendograma
plt.figure(figsize=(15, 10))  
dendrogram(linked, orientation='top')
plt.title('Agrupación jerárquica')
plt.show()

km = KMeans(n_clusters=5)
# Predicción de clusters
labels = km.fit_predict(x_sc)

cluster_data['cluster_km'] = labels
cluster_data.groupby(['cluster_km']).mean()

def show_clusters_on_plot(df, x_name,y_name, cluster_name):    
    plt.figure(figsize = (10,10))
    sns.scatterplot(x = df[x_name], y = df[y_name], hue = df[cluster_name], palette = 'Paired')
    plt.title('{} vs {}'.format(x_name, y_name))
    plt.show()

show_clusters_on_plot(cluster_data, 'Lifetime', 'Age', 'cluster_km')
show_clusters_on_plot(cluster_data, 'Avg_class_frequency_total', 'Age', 'cluster_km')
show_clusters_on_plot(cluster_data, 'Avg_class_frequency_current_month', 'Age', 'cluster_km')
show_clusters_on_plot(cluster_data, 'Avg_additional_charges_total', 'Month_to_end_contract', 'cluster_km')
