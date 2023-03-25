# This file contains the code required to train, test and save all the ML models.
import os
from pathlib import Path

import sklearn
import sklearn.ensemble
import sklearn.model_selection
import sklearn.neural_network
import sklearn.svm
import sklearn.tree

import functions as f


def main():

    # Prepare data
    X_train, X_test, y_train, y_test = f.prepare_data(columns_to_drop=['C_SEV', 'P_SAFE'])

    parent_dir = Path(__file__).parents[1]

    model_folder_path = os.path.join(parent_dir, 'models')

    test_result_file = os.path.join(parent_dir, 'reports', 'test_results.csv')

    training_and_testing_results_file = os.path.join(parent_dir, 'reports', 'training_and_testing_comparison_results.csv')

    models_to_train = {
        "decision_tree": sklearn.tree.DecisionTreeClassifier(random_state=0),
        "mlp": sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(100, 100, 100, 100, 100), max_iter=1000, random_state=0, verbose=True),
        "random_forest": sklearn.ensemble.RandomForestClassifier(n_estimators=100, random_state=0, verbose=True, n_jobs=-1),
        "svm": sklearn.svm.LinearSVC(random_state=0, verbose=True),
        "mlp_sgd": sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(100, 100, 100, 100, 100), max_iter=5000, random_state=0, verbose=True, solver="sgd"),
    }

    # Train and test models
    f.train_and_test_models(X_train, X_test, y_train, y_test, model_folder_path,
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

    f.train_and_test_models(X_train, X_test, y_train, y_test, model_folder_path,
                          models_to_train_search, test_result_file, overwrite_output=False, overwrite_saved_models=False)
    
     # Compare training and testing accuracy and f1-score for each model
    print("Comparing training and testing accuracy and f1-score...")
    for model_name in models_to_train | models_to_train_search:
        f.compare_training_and_testing_accuracy(model_name, model_folder_path, X_train, X_test, y_train, y_test,
                                                training_and_testing_results_file, overwrite_output=True)

    # Plot the loss curve of each model
    for model_name in models_to_train | models_to_train_search:
        f.plot_loss_curve(model_name, model_folder_path, os.path.join(
            parent_dir, 'reports', 'figures'))

if __name__ == "__main__":
    main()
