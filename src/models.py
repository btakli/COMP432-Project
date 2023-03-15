# This file contains the code required to train, test and save all the ML models.
import os
from pathlib import Path
import pickle
import functions as f

import sklearn
import sklearn.model_selection
import sklearn.tree
import sklearn.neural_network
import matplotlib.pyplot as plt

def train_and_save_models(model, X_train, y_train, model_name: str, model_folder_path: str, overwrite: bool = False):
    """Trains and saves the model to the models folder.

    Arguments:
    model: The model to be trained and saved
    X_train: The training features
    y_train: The training labels
    model_name: The name of the model (will be saved as {model_name}.pkl)
    model_folder_path: The path to the model folder
    overwrite: Whether to overwrite the model if it already exists in the models folder (default: False).
    """
    # Train model
    model.fit(X_train, y_train)
    # Save model
    model_file = os.path.join(model_folder_path, f"{model_name}.pkl")
    if os.path.exists(model_file) and not overwrite:
        print(f"Model {model_name} already exists. Set overwrite=True to overwrite the model.")
    else:
        pickle.dump(model, open(model_file, "wb"))

def test_model(model, X_test, y_test) -> tuple[float,float]:
    """Tests the model on the test set.

    Arguments:
    model: The model to be tested
    X_test: The test features
    y_test: The test labels

    Returns:
    (accuracy, f1 score) tupleß
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

def main():
    # import data from data folder
    parent_dir = Path(__file__).parents[1]
    file_path = os.path.join(parent_dir, 'data', "2019_dataset_en.csv")
    raw_data = f.get_data(file_path)
    # preprocess data

    data = f.preprocess_data(raw_data, verbose=True)

    # Extract features and labels. We will use the day of the week as the label.
    X, y = f.extract_features_and_labels(data, "C_WDAY")

    # Split data into training and test sets, 80% training, 20% test
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2, random_state=0)

    # Train and save models
    model_folder_path = os.path.join(parent_dir, 'models')

    models_to_train = {"decision_tree": sklearn.tree.DecisionTreeClassifier(random_state=0), "mlp": sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(100, 100, 100, 100, 100), max_iter=1000, random_state=0)}

    for model_name, model in models_to_train.items():
        train_and_save_models(model, X_train, y_train, model_name, model_folder_path, overwrite=True)

    # Test models
    for model_name, model in models_to_train.items():
        loaded_model = load_model(model_name, model_folder_path)
        accuracy, f1 = test_model(loaded_model, X_test, y_test)
        print(f"Model {model_name} accuracy: {accuracy:.2f}, F1 score: {f1:.2f}")

if __name__ == "__main__":
    main()
