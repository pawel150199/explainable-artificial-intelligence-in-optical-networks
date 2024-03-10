import shap
import argparse
import pandas as pd
from typing import Any
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


from helpers.import_data import import_data
from helpers.loggers import configureLogger

RANDOM_STATE = 42

def plot_shapley_values(dataset: str, estimator: Any, image_name: str, random_state: int=None):
    if estimator is None:
        estimator = LinearRegression
    
    if image_name is None:
        image_name = "shapley"

    X, y = import_data(dataset)
    X_t = X.reshape((100, 300))
    y_t = y[:,0]
    X_train, _, y_train, _ = train_test_split(X_t, y_t, test_size=0.33, random_state=RANDOM_STATE)


    regr = estimator()
    fitted = regr.fit(X_train, y_train)
    x_df = pd.DataFrame(X_t)
    column_names = [f"{'source' if i % 3 == 0 else 'destination' if i % 3 == 1 else 'bitrate'}{i // 3}" for i in range(len(x_df.columns))]
    x_df.columns = column_names
    shap.initjs()
    ex = shap.KernelExplainer(fitted.predict,X_t)
    shap_values = ex.shap_values(x_df)
    shap.summary_plot(shap_values, x_df, max_display=5, show=False)
    fig = plt.gcf()
    fig.savefig(f'{image_name}.png')
    plt.close(fig)

    #shap.plots.violin(shap_values, x_df, show=False)

def get_parameters():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        "-d",
        help="Dataset path",
        type=str,
        required=True
    )

    parser.add_argument(
        "--estimator",
        "-e",
        help="Estimator path",
        type=str,
        choices=["LinearRegression", "RandomForestRegressor", "DecisionTreeRegressor", "KNeighborsRegressor", "SVR"],
    )

    parser.add_argument(
        "--image_name",
        "-i",
        help="Image name",
        type=str
    )

    return parser.parse_args()

def main():
    logger = configureLogger()
    plot_shapley_values(dataset="Euro28", estimator=LinearRegression, image_name=None, random_state=RANDOM_STATE)

if __name__ == "__main__":
    main()


