from configs_algorithms import ALGORITHMS_CONFIG_TPOT, EVAL_METRIC, RANDOMSTATE, RUNTIME_SECONDS
from df_model import DataModel
from sklearn.metrics import make_scorer
from tpot import TPOTRegressor


class TPOTWrapper:
    def __init__(
        self,
        data_config,
        algorithms_config=ALGORITHMS_CONFIG_TPOT,
        eval_metric=EVAL_METRIC,
        runtime_total=RUNTIME_SECONDS,
        random_state=RANDOMSTATE,
    ):
        self.data_model = DataModel(data_config, random_state)

        self.tpot_model = TPOTRegressor(
            template="Regressor",
            config_dict=algorithms_config,
            scoring=make_scorer(
                eval_metric,
                greater_is_better=False,
            ),
            generations=None,
            max_time_mins=runtime_total / 60,
            random_state=random_state,
            verbosity=3,
        )

    # initialize tpot automl for each algorithm
    def find_best_algorithm_poly(self):
        self.data_model.fit_model(self.tpot_model)

    # return algorithm with best performance
    def get_best_model(self):
        return self.tpot_model.fitted_pipeline_[0]

    # print summary of all algorithm run results
    def get_search_summary(self):
        return
