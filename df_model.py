import os

import numpy as np
import pandas as pd
from configs_data import IMPORT_PATH
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# data train/test splits constants
SMALL_BIG_THRESHOLD = 10000
SMALL_DATA_SPLIT = 0.2
BIG_DATA_SPLIT = 0.1
RANDOMSTATE = 42


class DataModel:
    def __init__(self, data_config, random_state=RANDOMSTATE):
        self.data_config = data_config
        self.random_state = random_state

        # set up data configuration
        self.data = self.import_data()
        self.data = self.set_data_types()
        self.data = self.drop_data_columns()
        self.target_name = self.data_config["target"]

        # split the data into predictor and target sets
        self.X = self.data.drop(self.target_name, axis=1)
        self.y = self.data[self.target_name]

        # split the data into train and test sets
        self.data_split = self._generate_train_val_test_split()

    # split the data into train and test sets
    def _generate_train_val_test_split(self):
        # set the train and test split size depending on the number of observations
        test_size = SMALL_DATA_SPLIT if self.data.shape[0] < SMALL_BIG_THRESHOLD else BIG_DATA_SPLIT
        validation_size = test_size / (1 - test_size)  # relative value for same size as test

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=RANDOMSTATE)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train, self.y_train, test_size=validation_size, random_state=RANDOMSTATE)

    def fit_model(self, model):
        # define basic model pipeline for encoding/imputation/caling
        model_pipeline = make_pipeline(
            self._get_categorical_encoder(),
            SimpleImputer(strategy="mean"),
            MinMaxScaler(feature_range=(0, 1)),
            model,
        )

        model_pipeline.fit(self.X_train, self.y_train)
        return model_pipeline

    def evaluate_model(self, model, eval_metric, test=False):
        model_pipeline = self.fit_model(model)

        # evaluate fitted model on validation/test data
        X_eval, y_eval = (self.X_test, self.y_test) if test else (self.X_val, self.y_val)
        y_pred = model_pipeline.predict(X_eval)
        return eval_metric(y_eval, y_pred)

    # define data encoder for categorical values
    def _get_categorical_encoder(self):
        # handle deprecated sklearn params
        one_hot_param = "sparse_output" if "sparse_output" in OneHotEncoder.__init__.__code__.co_varnames else "sparse"

        # define categorical one hot encoding
        return ColumnTransformer(
            transformers=[
                ("categorical", OneHotEncoder(**{one_hot_param: False}), self.X.select_dtypes(exclude=[np.number]).columns),
                ("numerical", "passthrough", self.X.select_dtypes(include=[np.number]).columns),
            ],
        )

    def import_data(self):
        data_path = os.path.join(IMPORT_PATH, self.data_config["dataset"] + ".csv")
        print(f"importing from: {data_path}")
        return pd.read_csv(data_path, delimiter=";")

    def set_data_types(self):
        return self.data.astype({col: "category" for col in self.data_config["cats_list"]})

    def drop_data_columns(self):
        return self.data.drop(columns=self.data_config["drop_list"])
