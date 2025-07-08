from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR as SVRegressor
from configs_data import DATA_CONFIG_AUTO_MPG, DATA_CONFIG_COMMUNITIES_CRIME, DATA_CONFIG_MIAMI_HOUSING, DATA_CONFIG_BIKE_SHARING

RANDOMSTATE = 42

AUTOML_SA = "Custom"
AUTOML_SKLAUTO = "AutoSklearn"
AUTOML_TPOT = "TPOT"

############### SIMULATED ANNEALING RESULTS ###############

BEST_MODEL_SA_AUTO_MPG = MLPRegressor(
    hidden_layer_sizes=50,
    activation="tanh",
    solver="lbfgs",
    alpha=0.2857142857142857,
    learning_rate_init=0.21436428571428567,
    max_iter=800,
    early_stopping=False,
    random_state=RANDOMSTATE,
)

BEST_MODEL_SA_COMMUNITIES_CRIME = SVRegressor(
    kernel="rbf",
    gamma="auto",
    C=3.6207172413793107,
    tol=0.0001,
    epsilon=0.041666666666666664,
)

BEST_MODEL_SA_MIAMI_HOUSING = RandomForestRegressor(
    n_estimators=561,
    criterion="squared_error",
    max_depth=47,
    max_features=None,
    min_samples_split=4,
    min_samples_leaf=1,
    random_state=RANDOMSTATE,
)

BEST_MODEL_SA_BIKE_SHARING = MLPRegressor(
    hidden_layer_sizes=100,
    activation="logistic",
    solver="adam",
    alpha=0.7142857142857142,
    learning_rate_init=0.14294285714285712,
    max_iter=1200,
    early_stopping=False,
    random_state=RANDOMSTATE,
)

############### AUTO-SKLEARN RESULTS ###############

BEST_MODEL_SKLAUTO_AUTO_MPG = RandomForestRegressor(
    n_estimators=512,
    criterion="friedman_mse",
    max_depth=None,
    max_features=0.8809284139256492,
    min_samples_split=6,
    random_state=RANDOMSTATE,
)

BEST_MODEL_SKLAUTO_COMMUNITIES_CRIME = RandomForestRegressor(
    n_estimators=512,
    criterion="friedman_mse",
    max_depth=None,
    max_features=0.38093097805384224,
    min_samples_split=10,
    min_samples_leaf=10,
    random_state=RANDOMSTATE,
)

BEST_MODEL_SKLAUTO_MIAMI_HOUSING = RandomForestRegressor(
    n_estimators=512,
    criterion="absolute_error",
    max_depth=None,
    max_features=0.23006288595741897,
    min_samples_split=8,
    min_samples_leaf=16,
    bootstrap=False,
    random_state=RANDOMSTATE,
)

BEST_MODEL_SKLAUTO_BIKE_SHARING = RandomForestRegressor(
    n_estimators=512,
    criterion="absolute_error",
    max_depth=None,
    max_features=0.23006288595741897,
    min_samples_split=8,
    min_samples_leaf=16,
    bootstrap=False,
    random_state=RANDOMSTATE,
)

############### TPOT RESULTS ###############

BEST_MODEL_TPOT_AUTO_MPG = MLPRegressor(
    hidden_layer_sizes=100,
    activation="relu",
    solver="lbfgs",
    alpha=1.0,
    learning_rate_init=0.0001,
    max_iter=800,
    early_stopping=True,
    random_state=RANDOMSTATE,
)

BEST_MODEL_TPOT_COMMUNITIES_CRIME = MLPRegressor(
    hidden_layer_sizes=50,
    activation="relu",
    solver="lbfgs",
    alpha=1.1428571428571428,
    learning_rate_init=0.14294285714285712,
    max_iter=800,
    early_stopping=False,
    random_state=RANDOMSTATE,
)

BEST_MODEL_TPOT_MIAMI_HOUSING = RandomForestRegressor(
    n_estimators=761,
    criterion="friedman_mse",
    max_depth=83,
    max_features=None,
    min_samples_split=8,
    min_samples_leaf=1,
    random_state=RANDOMSTATE,
)

BEST_MODEL_TPOT_BIKE_SHARING = MLPRegressor(
    hidden_layer_sizes=100,
    activation="relu",
    solver="adam",
    alpha=0.42857142857142855,
    learning_rate_init=0.14294285714285712,
    max_iter=400,
    early_stopping=False,
    random_state=RANDOMSTATE,
)


#############################################

BEST_MODELS = {
    DATA_CONFIG_AUTO_MPG["dataset"]: {
        AUTOML_SA: BEST_MODEL_SA_AUTO_MPG,
        AUTOML_SKLAUTO: BEST_MODEL_SKLAUTO_AUTO_MPG,
        AUTOML_TPOT: BEST_MODEL_TPOT_AUTO_MPG,
    },
    DATA_CONFIG_COMMUNITIES_CRIME["dataset"]: {
        AUTOML_SA: BEST_MODEL_SA_COMMUNITIES_CRIME,
        AUTOML_SKLAUTO: BEST_MODEL_SKLAUTO_COMMUNITIES_CRIME,
        AUTOML_TPOT: BEST_MODEL_TPOT_COMMUNITIES_CRIME,
    },
    DATA_CONFIG_MIAMI_HOUSING["dataset"]: {
        AUTOML_SA: BEST_MODEL_SA_MIAMI_HOUSING,
        AUTOML_SKLAUTO: BEST_MODEL_SKLAUTO_MIAMI_HOUSING,
        AUTOML_TPOT: BEST_MODEL_TPOT_MIAMI_HOUSING,
    },
    DATA_CONFIG_BIKE_SHARING["dataset"]: {
        AUTOML_SA: BEST_MODEL_SA_BIKE_SHARING,
        AUTOML_SKLAUTO: BEST_MODEL_SKLAUTO_BIKE_SHARING,
        AUTOML_TPOT: BEST_MODEL_TPOT_BIKE_SHARING,
    },
}
