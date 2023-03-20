# This file contains the code required to train, test and save all the ML models.
import csv
import os
import pickle
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import sklearn.ensemble
import sklearn.model_selection
import sklearn.neural_network
import sklearn.svm
import sklearn.tree

import functions as f


def train_and_save_models(model: sklearn.base.ClassifierMixin, X_train: np.ndarray, y_train: np.ndarray, model_name: str, model_folder_path: str, overwrite: bool = False):
    """Trains and saves the model to the models folder.

    Arguments:
    model: The model to be trained and saved
    X_train: The training features
    y_train: The training labels
    model_name: The name of the model (will be saved as {model_name}.pkl)
    model_folder_path: The path to the model folder
    overwrite: Whether to overwrite the model if it already exists in the models folder (default: False).

    Returns:
    training time in seconds (-1 if model already exists and overwrite is False)
    """

    model_file = os.path.join(model_folder_path, f"{model_name}.pkl")
    # If model already exists and overwrite is False
    if os.path.exists(model_file) and not overwrite:
        print(
            f"Model {model_name} already exists. Set overwrite=True to overwrite the model.")
        return -1
    else:
        # Train model
        start_time = time.time()
        model.fit(X_train, y_train)
        pickle.dump(model, open(model_file, "wb"))
        end_time = time.time()
        return end_time - start_time


def test_model(model: sklearn.base.ClassifierMixin, X_test: np.ndarray, y_test: np.ndarray) -> tuple:
    """Tests the model on the test set.

    Arguments:
    model: The model to be tested
    X_test: The test features
    y_test: The test labels

    Returns:
    (accuracy, f1 score) tuple
    """
    y_pred = model.predict(X_test)
    accuracy = sklearn.metrics.accuracy_score(y_test, y_pred)
    f1 = sklearn.metrics.f1_score(y_test, y_pred, average="weighted")

    return (accuracy, f1)


def load_model(model_name: str, model_folder_path: str):
    """Loads the model from the models folder.

    Arguments:
    model_name: The name of the model (will be loaded from {model_name}.pkl)
    model_folder_path: The path to the model folder

    Returns:
    The model
    """
    model_file = os.path.join(model_folder_path, f"{model_name}.pkl")
    return pickle.load(open(model_file, "rb"))


def prepare_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Prepares the data for training and testing.

    Returns:
    (X_train, X_test, y_train, y_test) tuple"""
    # import data from data folder
    parent_dir = Path(__file__).parents[1]
    file_path = os.path.join(parent_dir, 'data', "2019_dataset_en.csv")
    raw_data = f.get_data(file_path)

    # profile data
    # f.profile_data(raw_data, os.path.join(parent_dir, 'reports', 'raw_data_profile.html'))
    # preprocess data
    data = f.preprocess_data(raw_data, verbose=True,
                             columns_to_drop=['C_SEV', 'P_SAFE'])
    # profile data
    # f.profile_data(data, os.path.join(parent_dir, 'reports', 'preprocessed_data_profile.html'))

    # Extract features and labels. We will use the day of the week as the label.
    X, y = f.extract_features_and_labels(data, "C_WDAY")

    # Split data into training and test sets, 80% training, 20% test
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X, y, test_size=0.2, random_state=0)

    return X_train, X_test, y_train, y_test


def train_and_test_models(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, models_folder_path: str, models: dict, test_results_file_path: str, overwrite_output=False, overwrite_saved_models=False):
    """Trains and tests all the models. Stores models in specified models folder

    Arguments:
    - X_train: The training features
    - X_test: The test features
    - y_train: The training labels
    - y_test: The test labels
    - models_folder_path: The path to store the models to
    - models: A dictionary of models to be trained and tested
    - test_results_file_path: The path to the test results file
    - overwrite_output: Whether to overwrite the output file if it already exist or to append to it (default: False).
    - overwrite_saved_models: Whether to overwrite the saved models if they already exist or not (default: False)."""

    training_times = {}  # Dictionary to store training times
    for model_name, model in models.items():
        print(f"Training model {model_name}...")
        training_time = train_and_save_models(model, X_train, y_train,
                                              model_name, models_folder_path, overwrite=overwrite_saved_models)

        if training_time != -1:
            training_times[model_name] = training_time
        print("Done!")

    # Test models

    # If it doesn't exist, create the output file csv with the header using pandas
    if not os.path.exists(test_results_file_path):
        pd.DataFrame(columns=['Model', 'Accuracy', 'F1 score', 'Training time', 'Hyperparameters']).to_csv(
            test_results_file_path, index=False)

    for model_name, model in models.items():
        print(f"Loading model {model_name}...")
        loaded_model = load_model(model_name, models_folder_path)
        print(f"Testing model {model_name}...")
        accuracy, f1 = test_model(loaded_model, X_test, y_test)
        print("Done!")
        print(
            f"Model {model_name} accuracy: {accuracy:.2f}, F1 score: {f1:.2f}")

        # Write results to csv, replace entry if it already exists
        df = pd.read_csv(test_results_file_path)

        if model_name in df.Model.values:
            if overwrite_output:
                df["Accuracy"][df.Model == model_name] = accuracy
                df["F1 score"][df.Model == model_name] = f1
                df["Training time"][df.Model ==
                                    model_name] = training_times[model_name] if model_name in training_times else "N/A"
                df["Hyperparameters"][df.Model ==
                                      model_name] = str(model.get_params())
            else:
                print(
                    f"Model {model_name} already exists in test results file. Skipping...")
        else:
            addition = pd.DataFrame(
                [[model_name, accuracy, f1, training_times[model_name] if model_name in training_times else "N/A", str(model.get_params())]], columns=['Model', 'Accuracy', 'F1 score', 'Training time', 'Hyperparameters'])
            df = pd.concat([df, addition])

        df.to_csv(test_results_file_path, index=False)


def main():

    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data()

    parent_dir = Path(__file__).parents[1]

    model_folder_path = os.path.join(parent_dir, 'models')

    test_result_file = os.path.join(parent_dir, 'reports', 'test_results.csv')

    models_to_train = {
        "decision_tree": sklearn.tree.DecisionTreeClassifier(random_state=0),
        "mlp": sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(100, 100, 100, 100, 100), max_iter=1000, random_state=0, verbose=True),
        "random_forest": sklearn.ensemble.RandomForestClassifier(n_estimators=100, random_state=0, verbose=True, n_jobs=-1),
        "svm": sklearn.svm.LinearSVC(random_state=0, verbose=True),
        "mlp_sgd": sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(100, 100, 100, 100, 100), max_iter=5000, random_state=0, verbose=True, solver="sgd"),
    }

    # Train and test models
    train_and_test_models(X_train, X_test, y_train, y_test, model_folder_path,
                          models_to_train, test_result_file, overwrite_output=False, overwrite_saved_models=False)

    # Hyperparameter search:

    # Hyperparameter seach dictionaries
    mlp_hyperparameters = {
        "hidden_layer_sizes": [(100, 100, 100), (1000, 500, 100)],
        "max_iter": [1000],
        "random_state": [0],
        "verbose": [True],
        "solver": ["adam", "sgd"],
        "learning_rate": ["constant", "adaptive"],
    }

    decision_tree_hyperparameters = {
        "criterion": ["gini", "entropy"],
        "splitter": ["best", "random"],
        "max_depth": [None, 1,  5, 10],
        "min_samples_split": [2, 5, 10],
        "max_features": [None, "sqrt", "log2"],
        "random_state": [0]
    }

    random_forest_hyperparameters = {
        "n_estimators": [10, 500],
        "criterion": ["gini", "entropy"],
        "max_depth": [None, 1, 5],
        "max_features": [None, "sqrt", "log2"],
        "n_jobs": [-1],
        "random_state": [0],
        "verbose": [True],
    }

    base_mlp = sklearn.neural_network.MLPClassifier(
        random_state=0, verbose=True)
    base_decision_tree = sklearn.tree.DecisionTreeClassifier(random_state=0)
    base_random_forest = sklearn.ensemble.RandomForestClassifier(
        random_state=0, verbose=True)

    # Grid search uses cross validation to find the best hyperparameters, so we don't need to make a separate validation set.
    models_to_train_search = {
        "gridsearch_mlp": sklearn.model_selection.GridSearchCV(base_mlp, mlp_hyperparameters, cv=3, verbose=True, n_jobs=-1),
        "gridsearch_decision_tree": sklearn.model_selection.GridSearchCV(base_decision_tree, decision_tree_hyperparameters, cv=3, verbose=True, n_jobs=-1),
        "gridsearch_random_forest": sklearn.model_selection.GridSearchCV(base_random_forest, random_forest_hyperparameters, cv=3, verbose=True, n_jobs=-1)
    }

    train_and_test_models(X_train, X_test, y_train, y_test, model_folder_path,
                          models_to_train_search, test_result_file, overwrite_output=False, overwrite_saved_models=False)


if __name__ == "__main__":
    main()
