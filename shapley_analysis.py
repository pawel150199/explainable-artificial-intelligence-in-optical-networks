import shap
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from helpers.import_data import import_data

RANDOM_STATE = 42

dataset = "Euro28"
X, y = import_data(dataset)
X_t = X.reshape((100, 300))
y_t = y[:,0]
X_train, X_test, y_train, y_test = train_test_split(X_t, y_t, test_size=0.33, random_state=RANDOM_STATE)


regr = LinearRegression()
fitted = regr.fit(X_train, y_train)
x_df = pd.DataFrame(X_t)
column_names = [f"{'source' if i % 3 == 0 else 'destination' if i % 3 == 1 else 'bitrate'}{i // 3}" for i in range(len(x_df.columns))]
x_df.columns = column_names
shap.initjs()
ex = shap.KernelExplainer(fitted.predict,X_t)
shap_values = ex.shap_values(x_df)
shap.summary_plot(shap_values, x_df) # tutaj mozna ograniczyc liczbe wyswietlanych cech przez dodanie max_display; w pracy którą wysłałam w poprzednim mailu uzywalam max_display=10 ze wzgledu na limit miejsca