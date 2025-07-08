import math
import os
import random
import time

import numpy as np
import pandas as pd
from configs_algorithms import ALGORITHMS_CONFIG, EVAL_METRIC, RANDOMSTATE, RUNTIME_SECONDS
from configs_data import RESULT_PATH
from df_model import DataModel
from sklearn.model_selection import ParameterGrid

# constants - simulated annealing
RUNTIME_ESTIMATIONS = 15
ACCEPT_INITIAL = 0.98
ACCEPT_FINAL = 0.01
REPETITION_MAX = 0.1
UPDATE_FREQUENCY = 3
NEIGHBORS_WINDOW = 3
RESULT_PERSIST = False

SOLUTION_REJECT = "reject"
SOLUTION_IMPROVE = "improve"
SOLUTION_EQUAL = "equal"
SOLUTION_RANDOM = "random"


class SimulatedAnnealingPoly:
    def __init__(
        self,
        data_config,
        algorithms_config=ALGORITHMS_CONFIG,
        eval_metric=EVAL_METRIC,
        runtime_total=RUNTIME_SECONDS,
        runtime_estimations=RUNTIME_ESTIMATIONS,
        accept_initial=ACCEPT_INITIAL,
        accept_final=ACCEPT_FINAL,
        repetition_max=REPETITION_MAX,
        update_frequency=UPDATE_FREQUENCY,
        neighbors_window=NEIGHBORS_WINDOW,
        random_state=RANDOMSTATE,
        result_persist=RESULT_PERSIST,
        result_path=RESULT_PATH,
    ):
        self.data_config = data_config
        self.algorithms_config = algorithms_config
        self.eval_metric = eval_metric
        self.runtime_total = runtime_total
        self.runtime_estimations = runtime_estimations
        self.accept_initial = accept_initial
        self.accept_final = accept_final
        self.repetition_max = repetition_max
        self.update_frequency = update_frequency
        self.neighbors_window = neighbors_window
        self.random_state = random_state
        self.result_persist = result_persist
        self.result_path = result_path

        self.runtime_partial = runtime_total / len(algorithms_config)
        self.data_model = DataModel(data_config, random_state)
        self.algorithms_runs: dict[str, SimulatedAnnealingMono] = {}

    # initialize simulated annealing for each algorithm
    def find_best_algorithm_poly(self):
        for algorithm, params_search_dict in self.algorithms_config.items():
            algorithm_annealing = SimulatedAnnealingMono(self, algorithm, params_search_dict)
            algorithm_annealing.find_best_algorithm()
            self.algorithms_runs[algorithm.__name__] = algorithm_annealing

    # return algorithm with best performance
    def get_best_model(self):
        algorithms_best = {algorithm_run.model_best: algorithm_run.eval_best for algorithm_run in self.algorithms_runs.values()}
        return min(algorithms_best, key=algorithms_best.get)

    # print summary of all algorithm run results
    def get_search_summary(self):
        for algorithm_run in self.algorithms_runs.values():
            algorithm_run.logger.log_state_initial()
            algorithm_run.logger.log_state_final()


class SimulatedAnnealingMono:
    def __init__(self, sap_config: SimulatedAnnealingPoly, algorithm, params_search_dict):
        self.sap_config = sap_config
        self.algorithm = algorithm
        self.params_search_dict = params_search_dict

        # define algorithm seed
        self.seed = np.random.RandomState(self.sap_config.random_state)

        # define and hash possible search space
        self.neighbors_potential = len(ParameterGrid(params_search_dict))
        self.neighbors_visited = set()

        # define upper limit for consecutive revisits of known models
        self.repetition_cap = math.floor(self.neighbors_potential * self.sap_config.repetition_max)

        # define heuristical values for expected model evaluation
        # based on performance and runtime of algorithm repeatedly using random parameters
        start_time = time.time()
        exp_eval_trials = [self._evaluate_model(self._select_params_neighbor()) for _ in range(self.sap_config.runtime_estimations)]
        self.exp_eval_diff = np.mean(exp_eval_trials) / 100.0
        self.exp_eval_runtime = (time.time() - start_time) / self.sap_config.runtime_estimations

        # define start/end temperature such that new neighbor is selected with probability accept_initial/accept_final
        self.temp_init = abs(self.exp_eval_diff / math.log(self.sap_config.accept_initial))
        self.temp_final = abs(self.exp_eval_diff / math.log(self.sap_config.accept_final))

        # define number of neighbors evaluated such that simulated annealing finishes around runtime_partial
        # aka define temperature cooling such that final temperature is reached after desired annealing steps
        self.temp_updates = math.ceil(self.sap_config.runtime_partial / (self.exp_eval_runtime * self.sap_config.update_frequency))
        self.temp_decay = (self.temp_final / self.temp_init) ** (1.0 / self.temp_updates)

        # define logger for printing progress
        self.logger = SimulatedAnnealingMono.ProcessLogger(self)

    def find_best_algorithm(self):
        self.runtime_apartial = time.time()

        # initialize solution with random parameters
        self.params_initial = self._select_params_neighbor()
        self.eval_initial = self._evaluate_model(self.params_initial)
        self.logger.log_state_initial()

        temp_current = self.temp_init
        self.params_current = self.params_initial
        self.eval_current = self.eval_initial

        # prepare iteration result objects
        self.iterations_params = [self.params_current]
        self.iterations_eval = [self.eval_current]
        self.iterations_reason = ["initial"]

        repetition_counter = 0
        self.termination_reason = None

        while True:
            for _ in range(self.sap_config.update_frequency):
                # select random neighbor solution
                params_new = self._select_params_neighbor(self.params_current)
                eval_new = self._evaluate_model(params_new)
                self._store_neighbor_hash(params_new)

                # count consecutive revisits of known solutions
                repetition_counter = repetition_counter + 1 if self.change_known else 0

                select_reason = SOLUTION_REJECT
                if self.eval_current > eval_new:  # select solution by improved metric (smaller rmse)
                    select_reason = SOLUTION_IMPROVE
                elif self.eval_current == eval_new:  # select solution by equal metric
                    select_reason = SOLUTION_EQUAL
                else:
                    eval_diff = abs(eval_new - self.eval_current)
                    accept_current = math.exp(-eval_diff / temp_current)
                    if random.random() < accept_current:  # select solution by random threshold
                        select_reason = SOLUTION_RANDOM

                if select_reason != SOLUTION_REJECT:
                    self.params_current = params_new
                    self.eval_current = eval_new
                    self.logger.log_state_change(select_reason)

                    self.iterations_params.append(self.params_current)
                    self.iterations_eval.append(self.eval_current)
                    self.iterations_reason.append(select_reason)

            # define regular termination condition - final temperature cooling reached
            if temp_current <= self.temp_final:
                self.termination_reason = "temperature threshold reached"

            # define termination condition against infinite loops - optimum occilates
            if repetition_counter >= self.repetition_cap:
                self.termination_reason = "too many repetitions"

            # define termination condition against infinite loops - search space exhausted
            if len(self.neighbors_visited) == self.neighbors_potential:
                self.termination_reason = "all neighbors visited"

            if self.termination_reason:
                break

            # update temperature by cooling
            temp_current *= self.temp_decay

        self.runtime_apartial = time.time() - self.runtime_apartial
        self.logger.log_state_final()

        # persist all iterations of simulated annealing
        if self.sap_config.result_persist:
            self._store_result_iterations()

        self.model_best = self.algorithm(**self.params_current)
        self.eval_best = self.eval_current

    def _select_params_neighbor(self, params=None):
        if not params:  # initialize random solution
            params_new = {key: self.seed.choice(values) for key, values in self.params_search_dict.items()}
        else:  # select random neighbor
            params_new = params.copy()
            params_variable = [pr for pr in self.params_search_dict if len(self.params_search_dict[pr]) > 1]
            param_select = self.seed.choice(params_variable)
            param_choices = self.params_search_dict[param_select]

            param_old = params_new[param_select]
            index_current = param_choices.index(param_old)
            neighbors_lower = param_choices[:index_current][-self.sap_config.neighbors_window :]
            neighbors_upper = param_choices[index_current + 1 : index_current + 1 + self.sap_config.neighbors_window]

            param_new = self.seed.choice(neighbors_lower + neighbors_upper)
            params_new[param_select] = param_new
            self.change_current = (param_select, param_old, param_new)

        return params_new

    def _evaluate_model(self, params={}):
        model = self.algorithm(**params)
        return self.sap_config.data_model.evaluate_model(model, self.sap_config.eval_metric)

    def _store_neighbor_hash(self, params_new):
        neighbor_hash = hash(frozenset(params_new.items()))
        self.change_known = neighbor_hash in self.neighbors_visited
        self.neighbors_visited.add(neighbor_hash)

    def _store_result_iterations(self):
        data_name = self.sap_config.data_config["dataset"]
        algorithm_name = self.algorithm.__name__
        results = {
            "dataset": data_name,
            "algorithm_sa": algorithm_name,
            "search_space": self.neighbors_potential,
            "temp_init": self.temp_init,
            "temp_decay": self.temp_decay,
            "runtime": self.runtime_apartial,
            "termination_reason": self.termination_reason,
            "evaluations": self.iterations_eval,
            "selection_reasons": self.iterations_reason,
            **pd.DataFrame(self.iterations_params).to_dict("list"),
        }
        results_df = pd.DataFrame(results, index=pd.RangeIndex(1, len(self.iterations_eval) + 1, name="iteration"))
        out_path = os.path.join(self.sap_config.result_path, f"{data_name}_{algorithm_name}_iterations.csv")
        results_df.to_csv(out_path, sep=";")

    class ProcessLogger:
        def __init__(self, sa_instance: "SimulatedAnnealingMono"):
            self.sa_instance = sa_instance

        def log_state_initial(self):
            print(
                "[start algorithm]\t",
                f"regressor: {self.sa_instance.algorithm.__name__},",
                f"dataset: {self.sa_instance.sap_config.data_config['dataset']}",
            )
            print(
                "[initial configuration]\t",
                f"T_init: {self.sa_instance.temp_init:.4f},",
                f"decay: {self.sa_instance.temp_decay:.4f},",
                f"n_updates: {self.sa_instance.temp_updates},",
                f"T_end: {self.sa_instance.temp_final:.4f}.",
            )
            print(
                "[initial model]\t\t",
                f"rmse: {self.sa_instance.eval_initial:.4f},",
                f"params: {self.sa_instance.params_initial}",
            )

        def log_state_change(self, reason):
            param_select, param_old, param_new = self.sa_instance.change_current
            nr_format = ".4f" if np.issubdtype(type(param_old), np.floating) else ""
            print(
                f"[{'revis' if self.sa_instance.change_known else 'new'} model: {reason}]\t",
                f"rmse: {self.sa_instance.eval_current:.4f},",
                f"params: {param_select}:",
                f"{param_old:{nr_format}} -> {param_new:{nr_format}}",
            )

        def log_state_final(self):
            print(
                "[final model]\t\t",
                f"rmse: {self.sa_instance.eval_current:.4f},",
                f"params: {self.sa_instance.params_current}",
            )
            print(
                "[complete algorithm]\t",
                f"search_space: {self.sa_instance.neighbors_potential},",
                f"runtime: {self.sa_instance.runtime_apartial:.4f}s,",
                f"termination: {self.sa_instance.termination_reason}",
            )
            print("#######################################################")
