import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR as SVRegressor


def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


EVAL_METRIC = root_mean_squared_error
RUNTIME_SECONDS = 3600  # 1h
RANDOMSTATE = 42

RGS_SEARCH_SPACE_SVR = {
    "kernel": ["linear", "poly", "rbf", "sigmoid"],
    "gamma": ["scale", "auto"],
    "C": list(np.linspace(0.0001, 5, 30)),
    "tol": list(np.linspace(0.0001, 1, 20)),
    "epsilon": list(np.linspace(0, 1, 25)),
}

RGS_SEARCH_SPACE_SGD = {
    "loss": ["squared_error", "huber", "epsilon_insensitive", "squared_epsilon_insensitive"],
    "penalty": ["l1", "l2", "elasticnet"],
    "alpha": list(np.linspace(0.0001, 2, 60)),
    "learning_rate": ["constant", "adaptive", "optimal", "invscaling"],
    "eta0": list(np.linspace(0.0001, 2, 40)),
    "random_state": [RANDOMSTATE],
}

RGS_SEARCH_SPACE_KNB = {
    "n_neighbors": list(np.arange(1, 100, 1)),
    "weights": ["uniform", "distance"],
    "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
    "metric": ["manhattan", "euclidean", "chebyshev"],
    "leaf_size": list(np.arange(3, 102, 2)),
}

RGS_SEARCH_SPACE_RF = {
    "n_estimators": list(np.arange(11, 1012, 25)),
    "criterion": ["squared_error", "friedman_mse", "poisson"],
    "max_depth": [None] + list(np.arange(2, 84, 9)),
    "min_samples_split": list(np.arange(2, 15, 2)),
    "min_samples_leaf": list(np.arange(1, 11, 3)),
    "max_features": [None, "sqrt", "log2"],
    "random_state": [RANDOMSTATE],
}

RGS_SEARCH_SPACE_MLP = {
    "hidden_layer_sizes": list(np.arange(50, 151, 50)),
    "activation": ["identity", "logistic", "relu", "tanh"],
    "solver": ["lbfgs", "adam"],
    "alpha": list(np.linspace(0, 2, 15)),
    "learning_rate_init": list(np.linspace(0.0001, 1, 15)),
    "max_iter": list(np.arange(200, 1201, 100)),
    "early_stopping": [False, True],
    "random_state": [RANDOMSTATE],
}

ALGORITHMS_CONFIG = {
    SVRegressor: RGS_SEARCH_SPACE_SVR,
    SGDRegressor: RGS_SEARCH_SPACE_SGD,
    KNeighborsRegressor: RGS_SEARCH_SPACE_KNB,
    RandomForestRegressor: RGS_SEARCH_SPACE_RF,
    MLPRegressor: RGS_SEARCH_SPACE_MLP,
}

ALGORITHMS_CONFIG_TPOT = {
    "sklearn.svm.SVR": RGS_SEARCH_SPACE_SVR,
    "sklearn.linear_model.SGDRegressor": RGS_SEARCH_SPACE_SGD,
    "sklearn.neighbors.KNeighborsRegressor": RGS_SEARCH_SPACE_KNB,
    "sklearn.ensemble.RandomForestRegressor": RGS_SEARCH_SPACE_RF,
    "sklearn.neural_network.MLPRegressor": RGS_SEARCH_SPACE_MLP,
}  # {alg.__module__ + "." + alg.__qualname__: cnfg for alg, cnfg in ALGORITHMS_CONFIG.items()}

ALGORITHMS_CONFIG_SKL_AUTO = {
    "libsvm_svr": SVRegressor,
    "sgd": SGDRegressor,
    "k_nearest_neighbors": KNeighborsRegressor,
    "random_forest": RandomForestRegressor,
    "mlp": MLPRegressor,
}
