import os
import time
import argparse
import warnings
import numpy as np
from tqdm import tqdm
from tabulate import tabulate
from scipy.stats import ttest_ind
from sklearn.metrics import mean_absolute_percentage_error

from sklearn.model_selection import RepeatedKFold

from models.svm import SVR
from models.tree import DecisionTreeRegressor
from models.neighbors import KNeighborsRegressor
from models.linear import LinearRegression
from models.ensemble import RandomForestRegressor
from helpers.loggers import configureLogger
from helpers.import_data import import_data


warnings.filterwarnings("ignore")

RANDOM_STATE = 1410

clfs = {
    "CART" : DecisionTreeRegressor(random_state=RANDOM_STATE),
    "SVR" : SVR(random_state=RANDOM_STATE, kernel="poly"),
    "KNN" : KNeighborsRegressor(random_state=RANDOM_STATE),
    "RF" : RandomForestRegressor(random_state=RANDOM_STATE),
    "LR" : LinearRegression(random_state=RANDOM_STATE),
    "CART-FS" : DecisionTreeRegressor(random_state=RANDOM_STATE, feature_selection=True),
    "SVR-FS" : SVR(random_state=RANDOM_STATE, kernel="poly", feature_selection=True),
    "KNN-FS" : KNeighborsRegressor(random_state=RANDOM_STATE, feature_selection=True),
    "RF-FS" : RandomForestRegressor(random_state=RANDOM_STATE, feature_selection=True),
    "LR-FS" : LinearRegression(random_state=RANDOM_STATE, feature_selection=True),
}

parameters = [
    "highestSlot",
    "avgHighestSlot",
    "sumOfSlots",
    "avgActiveTransceivers",
]

metrics = {"MAPE": mean_absolute_percentage_error}

def experiment(datasetname: str, n_splits: int, n_repeats: int, result_name: str, save: bool, measure_time: bool):

    if n_splits is None:
        n_splits = 5
    
    if n_repeats is None:
        n_repeats = 2
    
    if result_name is None:
        result_name = "experiment"
    
    if save is None:
        save = True
    
    if measure_time is None:
        measure_time = True

    X, y = import_data(datasetname)

    param_num = 4
    rskf = RepeatedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=RANDOM_STATE,
    )
    # n_splits x n_repeats x clfs x metrics
    scores = np.zeros(
        (
            param_num,
            n_splits * n_repeats,
            len(clfs),
            len(metrics),
        )
    )

    for param_id in tqdm(range(4), desc="Predicted parameter"):
        y_selected = y[:, param_id]
        for fold_id, (train, test) in enumerate(rskf.split(X, y_selected)):
            for clf_id, clf_name in enumerate(clfs):
                X_test = X[test].reshape((len(test), 300))
                X_train = X[train].reshape((len(train), 300))
                y_test, y_train = y_selected[test], y_selected[train]
                start = time.time()
                clf = clfs[clf_name]
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                end = time.time()
                for metric_id, metric_name in enumerate(metrics):
                    # PARAM X FOLD X CLASSIFIER X METRIC
                    if measure_time:
                        scores[param_id, fold_id, clf_id, metric_id] = end - start
                    else:
                        scores[param_id, fold_id, clf_id, metric_id] = metrics[metric_name](y_test, y_pred)

    if save:
        np.save(f"results/{result_name}", scores)

    return scores

def statistic_test(scores: np.array, tablename: str, alpha: float):
    std_fmt=None
    nc="---"
    db_fmt="%s"

    if tablename is None:
        tablename = "experiment"

    if alpha is None:
        alpha = 0.05
    
    mean = np.mean(scores, axis=1)
    std = np.std(scores, axis=1)

    n_clfs = len(clfs)
    table = []

    for param_idx, param_name in enumerate(parameters):
        table.append([db_fmt % param_name] + ["%.3f" % v for v in mean[param_idx, :]])

        if std_fmt:
            table.append([std_fmt % v for v in std[param_idx, :]])

        T, p = np.array(
            [[ttest_ind(scores[param_idx, :, i],
                scores[param_idx, :, j], random_state=RANDOM_STATE)
            for i in range(len(clfs))]
            for j in range(len(clfs))]
        ).swapaxes(0, 2)

        T = -T
        _ = np.where((p < alpha) * (T > 0))
        conclusions = [list(1 + _[1][_[0] == i])
                    for i in range(n_clfs)]
                        
        table.append([''] + [", ".join(["%i" % i for i in c])
                        if len(c) > 0 and len(c) < len(clfs) - 1 else ("all" if len(c) == len(clfs)-1 else nc)
                        for c in conclusions])

    headers = ["Parameter"]
    for i in clfs:
        headers.append(i)
    print(tabulate(table, headers, tablefmt="grid"))

    with open("tables/%s.txt" % (tablename), "w") as f:
        f.write(tabulate(table, headers, tablefmt="latex"))

def get_parameters():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--evaluation",
        "-e",
        help="Do experiment evaluation",
        action='store_true',
        required=True
    )

    parser.add_argument(
        "--statistic_test",
        "-s",
        help="Do statistic test",
        action='store_true',
        required=True
    )

    parser.add_argument(
        "--read_scores",
        "-r",
        type=str,
        help="Path to scores to read for statistic evaluation"
    )
    
    parser.add_argument(
        "--dataset",
        "-d",
        help="Dataset path",
        type=str,
        required=True
    )

    parser.add_argument(
        "--result_name",
        help="Name of the result of the experiment",
        type=str
    )

    parser.add_argument(
        "--n_splits",
        help="Number of splits in reapeted strafisfied cross validation",
        type=int
    )

    parser.add_argument(
        "--n_repeats",
        help="Number of repeats in reapeted strafisfied cross validation",
        type=int
    )

    parser.add_argument(
        "--tablename",
        help="Name of generated table",
        type=str
    )

    parser.add_argument(
        "--alpha",
        help="Alpha used in statistic test",
        type=float
    )

    parser.add_argument(
        "--measure_time",
        action='store_true',
        help="Measure time of execution each algoritm"
    )

    parser.add_argument(
        "--save_results",
        action='store_true',
        help="Save result of experiment evaluation"
    )

    args = parser.parse_args()

    if args.evaluation is None and args.statistic_test is not None and args.read_scores is None:
        raise argparse.ArgumentTypeError('Used --statistic_test but no --read_scores.')

    return args



def main():
    logger = configureLogger()

    args = get_parameters()

    if os.path.exists(args.dataset):
        datasetname = args.dataset
    else:
        logger.error("Dataset path does not exist! Try again.")
        exit(1)
    
    if not args.measure_time:
        measure_time = False
    else:
        measure_time = True
    
    if not args.save_results:
        save = False
    else:
        save = True

    if args.evaluation:
        logger.info("Starting experiment evaluation...")
        scores = experiment(datasetname=datasetname, n_splits=args.n_splits, n_repeats=args.n_repeats, measure_time=measure_time, result_name=args.result_name, save=save)
        logger.info("Experiment evaluation has been done.")

    if args.statistic_test:
        if "scores" not in locals() and "scores" not in globals():
            scores = np.load(args.read_scores)
        logger.info("Starting statistic evaluation...")
        statistic_test(scores, tablename=args.tablename, alpha=args.alpha)
        logger.info("Statistic evaluation has been done.")


if __name__ == "__main__":
    main()