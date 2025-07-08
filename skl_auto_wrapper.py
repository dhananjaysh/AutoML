import numpy as np
from autosklearn.metrics import make_scorer
from autosklearn.regression import AutoSklearnRegressor
from configs_algorithms import ALGORITHMS_CONFIG_SKL_AUTO, EVAL_METRIC, RANDOMSTATE, RUNTIME_SECONDS
from df_model import DataModel


class SKLAutoWrapper:
    def __init__(
        self,
        data_config,
        algorithms_config=ALGORITHMS_CONFIG_SKL_AUTO,
        eval_metric=EVAL_METRIC,
        runtime_total=RUNTIME_SECONDS,
        random_state=RANDOMSTATE,
    ):
        self.algorithms_config = algorithms_config
        self.data_model = DataModel(data_config, random_state)

        self.sklauto_model = AutoSklearnRegressor(
            include={
                "regressor": list(algorithms_config.keys()),
                "feature_preprocessor": ["no_preprocessing"],
            },
            metric=make_scorer(
                name=EVAL_METRIC.__name__,
                score_func=eval_metric,
                greater_is_better=False,
            ),
            time_left_for_this_task=runtime_total,
            seed=random_state,
            memory_limit=6144,
        )

    # initialize tpot automl for each algorithm
    def find_best_algorithm_poly(self):
        self.data_model.fit_model(self.sklauto_model)

    # return algorithm with best performance
    def get_best_model(self):
        best_index = np.argmin(self.sklauto_model.cv_results_["mean_test_score"])
        best_dict = self.sklauto_model.cv_results_["params"][best_index]
        best_name = best_dict["regressor:__choice__"]
        best_model = self.algorithms_config[best_name]
        best_params = {k.split(":")[-1]: v for k, v in best_dict.items() if best_name in k}
        return best_model(**best_params)

    # print summary of all algorithm run results
    def get_search_summary(self):
        return self.sklauto_model.sprint_statistics()
