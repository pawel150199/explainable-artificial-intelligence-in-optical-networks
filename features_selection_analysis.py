import shap
import argparse
import numpy as np
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

RANDOM_STATE = 42

def plot_feature_selection(dataset: str, image_name: str, random_state: int=None):
    
    if image_name is None:
        image_name = "shapley"
    
    parameters = [
        "highestSlot",
        "avgHighestSlot",
        "sumOfSlots",
        "avgActiveTransceivers",
    ]

    X, y = import_data(dataset)
    X_t = X.reshape((100, 300))
    for i in range(len(parameters)):
        y_t = y[:,i]
        X_train, _, y_train, _ = train_test_split(X_t, y_t, test_size=0.33, random_state=RANDOM_STATE)


        tree = DecisionTreeRegressor(random_state=RANDOM_STATE)
        fitted = tree.fit(X_train, y_train)
        x_df = pd.DataFrame(X_t)
        column_names = [f"{'source' if i % 3 == 0 else 'destination' if i % 3 == 1 else 'bitrate'}{i // 3}" for i in range(len(x_df.columns))]
        x_df.columns = column_names
        indices_of_highest_values = sorted(range(len(fitted.feature_importances_)), key=lambda i: fitted.feature_importances_[i], reverse=True)[:10]
        x_df = x_df.iloc[:, indices_of_highest_values]
        feature_importance = fitted.feature_importances_[indices_of_highest_values]

        
        fig, ax = plt.subplots()
        y_pos = np.arange(len(feature_importance))
        ax.barh(y_pos, 
            feature_importance,
            color='tomato'
        )
        ax.set_yticks(y_pos)
        ax.set_yticklabels(x_df.columns)
        ax.set_xlabel('Feature Importance')
        plt.title(f"Feature importances analysis for {parameters[i]} parameter")
        plt.tight_layout()
        fig = plt.gcf()
        fig.savefig(f'images/{image_name}_{parameters[i]}.png')
        plt.close(fig)

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
        "--image_name",
        "-i",
        help="Image name",
        type=str
    )

    return parser.parse_args()

def main():
    args = get_parameters()
    plot_feature_selection(dataset=args.dataset, image_name=args.image_name, random_state=RANDOM_STATE)

if __name__ == "__main__":
    main()