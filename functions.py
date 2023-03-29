# This file contains miscellaneous helper functions that are used in the other files.
import os
from pathlib import Path
import pickle
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import ydata_profiling as yp


def get_data(file_name: str) -> pd.DataFrame:
    """Reads the data from the file and returns it as a pandas DataFrame.

    Description of the data:
    - C_YEAR: Year of the accident
    - C_MNTH: Month of the accident
    - C_WDAY: Day of the week of the accident (1 = Monday, 7 = Sunday))
    - C_HOUR: Hour of the accident (00 = midnight to 0:59, 23:00 - 23:59)
    - C_SEV: Severity of the accident
    - C_VEHS: Number of vehicles involved in the accident
    - C_CONF: Configuration of the accident
    - C_RCFG: Road configuration
    - C_WTHR: Weather conditions
    - C_RSUR: Road surface conditions
    - C_RALN: Road alignment
    - C_TRAF: Traffic control configuration
    - V_ID: Vehicle ID
    - V_TYPE: Type of vehicle
    - V_YEAR: Year of vehicle
    - P_ID: Person ID
    - P_SEX: Sex of the person
    - P_AGE: Age of the person
    - P_PSN: Position of the person in the vehicle
    - P_ISEV: Injury severity of the person
    - P_SAFE: Safety equipment used by the person
    - P_USER: Type of user
    - C_CASE: Case number
    
    Usage Example:

    data = get_data("./src/data/accident_data.csv")
    """
    data = pd.read_csv(file_name, header=0)

    return data


def profile_data(data: pd.DataFrame, output_path: str):
    """Profiles the data and saves the report to the output path.

    Arguments: 
    data: The data to be profiled
    output_path: The path to the output folder
    
    Usage Example:
    profile_data(data, "./src/data/profile_report.html")
    """
    report = yp.ProfileReport(data)
    report.to_file(output_path)


def preprocess_data(raw_data: pd.DataFrame, verbose: bool, columns_to_drop: list[str]) -> pd.DataFrame:
    """Preprocesses the data and returns it as a pandas DataFrame.

    Arguments:
    raw_data: The data to be preprocessed
    verbose: Whether to print the data dimensions, head and size reduction after preprocessing.
    columns_to_drop: The columns to drop from the data. Excludes C_YEAR, C_CASE, P_ID, V_ID as it is automatically dropped.

    Preprocessing steps:
    - We will drop all rows with missing/unknown values in the remaining columns of interest (i.e rows that contain U, UU, X, XX for any column of interest).
    - We will drop the C_YEAR column as all data is from 2019.
    - We will drop the C_CASE column as it is not needed.
    - We will drop the P_ID and V_ID columns as they are not needed.
    - C_WTHR: Convert Q to 0 as it is still a weather condition, but different from the other values.
    - P_SEX: Convert F to 0 and M to 1.
    - C_MNTH and C_WDAY: Convert to int, now that we have dropped all rows with missing values.
    - Normalize the data (i.e. convert all values to floats)

    Usage Example:

    data = preprocess_data(data, verbose=True, columns_to_drop=['C_WTHR','P_SAFE'])

    """

    if verbose:
        print(
            f"Raw data dimensions: [{raw_data.shape[0]} x {raw_data.shape[1]}]")
        print(raw_data.head())
        print()

    # Drop C_YEAR column as all data is from 2019
    data = raw_data.drop(columns=['C_YEAR'])

    # Drop ID based columns
    data = data.drop(columns=['C_CASE', 'P_ID', 'V_ID'])
    # Drop all rows with missing/unknown values in columns of interest
    # If row value is U, UU, X, XX, N, NN for any column drop the row (N/NN means not applicable)

    # Drop columns mentioned in the columns_to_drop argument
    # Remove columns we already dropped if they're in the array
    columns_to_drop = [
        column for column in columns_to_drop if column in data.columns]
    # List invalid columns we tried to drop
    invalid_columns = [
        column for column in columns_to_drop if column not in data.columns]
    if len(invalid_columns) > 0:
        print(
            f"Warning: Tried to drop invalid columns: {invalid_columns}. Ignoring these columns.")

    data = data.drop(columns=columns_to_drop)

    data = data[~data.isin(['U', 'UU', 'UUUU', 'X', 'XX',
                           'N', 'NN', 'NNN', 'NNNN', 'Q', 'QQ']).any(axis=1)]

    # Convert Q to 0 as it is still a weather condition, but different from the other values
    data['C_WTHR'] = data['C_WTHR'].replace('Q', '0')

    # Convert F to 0 and M to 1 for P_SEX
    data['P_SEX'] = data['P_SEX'].replace('F', '0')
    data['P_SEX'] = data['P_SEX'].replace('M', '1')

    # Convert C_MNTH and C_WDAY to int
    data["C_MNTH"] = data["C_MNTH"].astype(int)
    data["C_WDAY"] = data["C_WDAY"].astype(int)

    if verbose:
        row_count_difference, column_count_difference = get_size_difference(
            raw_data, data)
        print(
            f"Data size reduction: {row_count_difference} dropped rows, and {column_count_difference} dropped columns")
        print(
            f"yields a data size reduction of {row_count_difference/raw_data.shape[0]*100:.2f}% of rows, and {column_count_difference/raw_data.shape[1]*100:.2f}% of columns")
        print(
            f"Preprocessed data dimensions: [{data.shape[0]} x {data.shape[1]}]\n")

        print("Preprocessed data head:")
        print("\tNote: P_SEX has had M and F converted to 0 and 1 respectively, and C_WTHR has had Q converted to 0")
        print(data.head())
        print()

    return data


def get_size_difference(old_data: pd.DataFrame, new_data: pd.DataFrame) -> tuple[int, int]:
    """Gets the difference in size between the old and new data sets in terms of rows and columns.
    Returns (row_count_difference, column_count_difference) tuple.
    
    Arguments:
    - old_data: The old data set
    - new_data: The new data set

    Returns:
    tuple of (row_count_difference, column_count_difference)

    Usage Example:
    row_count_difference, column_count_difference = get_size_difference(old_data, new_data)
    """
    row_count_difference = old_data.shape[0] - new_data.shape[0]
    column_count_difference = old_data.shape[1] - new_data.shape[1]
    return (row_count_difference, column_count_difference)


def extract_features_and_labels(data: pd.DataFrame, label_column: str) -> tuple[np.ndarray, np.ndarray]:
    """Extracts the labels from the data and returns them as a numpy array. Also returns the data without the labels.

    Arguments:
    data: The data to extract the labels from
    label_column: The column to extract the labels from

    Returns:
    tuple of (features, labels)
    features: The data without the labels
    labels: The labels as a numpy array
    """
    labels = data[label_column].to_numpy(dtype=np.int32)
    features = data.drop(columns=[label_column]).to_numpy(dtype=np.int32)

    # One hot encode the features (i.e convert categorical data to numerical data)
    features = sklearn.preprocessing.OneHotEncoder().fit_transform(features).toarray()

    return (features, labels)


def train_and_save_models(model: sklearn.base.ClassifierMixin, X_train: np.ndarray, y_train: np.ndarray, model_name: str, model_folder_path: str, overwrite: bool = False) -> float:
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

    Usage Example:
    training_time = train_and_save_models(model, X_train, y_train, model_name, model_folder_path, overwrite)
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

    Usage Example:
    accuracy, f1 = test_model(model, X_test, y_test)
    """
    y_pred = model.predict(X_test)
    accuracy = sklearn.metrics.accuracy_score(y_test, y_pred)
    f1 = sklearn.metrics.f1_score(y_test, y_pred, average="weighted")

    return (accuracy, f1)


def load_model(model_name: str, model_folder_path: str) -> sklearn.base.ClassifierMixin:
    """Loads the model from the models folder.

    Arguments:
    model_name: The name of the model (will be loaded from {model_name}.pkl)
    model_folder_path: The path to the model folder

    Returns:
    The model

    Usage Example:
    model = load_model(model_name, model_folder_path)
    """

    model_file = os.path.join(model_folder_path, f"{model_name}.pkl")
    return pickle.load(open(model_file, "rb"))


def prepare_data(columns_to_drop: list =[]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Prepares the data for training and testing. Creates a profile report for the raw data and preprocessed data.

    The data is first preprocesssed by:
    - Dropping the columns specified in columns_to_drop
    - Dropping the C_YEAR, C_CASE, P_ID, V_ID columns
    - Dropping rows with missing/unknown values
    
    Then a profile report is created for the raw data and preprocessed data.

    The data is split into training and testing sets with a 80/20 split, the features are one hot encoded.

    Arguments:
    columns_to_drop: The columns to drop from the data. E.g. ['C_SEV', 'P_SAFE', 'C_RSUR', 'P_USER']. 
        Excludes C_YEAR and C_CASE which are always dropped.

    Returns:
    (X_train, X_test, y_train, y_test) tuple
    
    Usage Example:
    X_train, X_test, y_train, y_test = prepare_data(columns_to_drop)"""
    # import data from data folder
    parent_dir = Path().resolve()
    file_path = os.path.join(parent_dir, 'data', "2019_dataset_en.csv")
    raw_data = get_data(file_path)

    # profile data
    profile_data(raw_data, os.path.join(parent_dir, 'reports', 'raw_data_profile.html'))
    # preprocess data
    data = preprocess_data(raw_data, verbose=True,
                           columns_to_drop=columns_to_drop)
    # profile data
    profile_data(data, os.path.join(parent_dir, 'reports', 'preprocessed_data_profile.html'))

    # Extract features and labels. We will use the day of the week as the label.
    X, y = extract_features_and_labels(data, "C_WDAY")

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
    - overwrite_saved_models: Whether to overwrite the saved models if they already exist or not (default: False).
    
    Usage Example:
    train_and_test_models(X_train, X_test, y_train, y_test, models_folder_path, models, test_results_file_path, overwrite_output, overwrite_saved_models)"""

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
                df.loc[df.Model == model_name, "Accuracy"] = accuracy
                df.loc[df.Model == model_name, "F1 score"] = f1
                df.loc[df.Model == model_name,
                       "Training time"] = training_times[model_name] if model_name in training_times else "N/A"
                df.loc[df.Model == model_name, "Hyperparameters"] = str(
                    model.get_params())
            else:
                print(
                    f"Model {model_name} already exists in test results file. Skipping...")
        else:
            addition = pd.DataFrame(
                [[model_name, accuracy, f1, training_times[model_name] if model_name in training_times else "N/A", str(model.get_params())]], columns=['Model', 'Accuracy', 'F1 score', 'Training time', 'Hyperparameters'])
            df = pd.concat([df, addition])

        df.to_csv(test_results_file_path, index=False)


def plot_loss_curve(model_name: str, models_folder: str, output_folder_path: str):
    """Plots the loss curves for the specified model.

    Arguments:
    - model_name: The name of the model to plot the loss curve for
    - models_folder: The folder containing the models
    - output_folder_path: The folder to store the loss curve plot to
    
    Usage Example:
    plot_loss_curve(model_name, models_folder, output_folder_path)
    """

    model = load_model(model_name, models_folder)

    if "gridsearch" in model_name:
        print("Model is a gridsearch object. Getting best estimator...")
        model = model.best_estimator_

    if not hasattr(model, "loss_curve_"):
        print(
            f"Model {model_name} does not have a loss curve. Skipping plotting...")
        return

    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    # If we are using gridsearch, we need to get the best estimator,
    # otherwise it tries to get the curve from the gridsearch object,
    # which doesn't have it.

    plt.plot(model.loss_curve_)
    plt.title(f"Loss curve for model {model_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(output_folder_path,
                f"{model_name}_loss_curve.png"))
    plt.clf()


def compare_training_and_testing_accuracy(model_name: str, models_folder_path: str, X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, output_file_path: str, overwrite_output: bool = False):
    """Compares the training and testing accuracy and f1-score for the specified model. Writes to a csv file.

    Arguments:
    - model_name: The name of the model to compare the accuracy and f1-score for
    - models_folder_path: The folder containing the models
    - X_train: The training features
    - X_test: The test features
    - y_train: The training labels
    - y_test: The test labels
    
    Usage Example:
    compare_training_and_testing_accuracy(model_name, models_folder_path, X_train, X_test, y_train, y_test, output_file_path, overwrite_output)"""

    model = load_model(model_name, models_folder_path)

    if "gridsearch" in model_name:
        print("Model is a gridsearch object. Getting best estimator...")
        model = model.best_estimator_

    train_accuracy, train_f1 = test_model(model, X_train, y_train)
    test_accuracy, test_f1 = test_model(model, X_test, y_test)

    print(
        f"Model {model_name} training accuracy: {train_accuracy:.2f}, training F1 score: {train_f1:.2f}")
    print(
        f"Model {model_name} testing accuracy: {test_accuracy:.2f}, testing F1 score: {test_f1:.2f}")

    if not os.path.exists(output_file_path):
        pd.DataFrame(columns=['Model', 'Training accuracy', 'Training F1 score', 'Testing accuracy', 'Testing F1 score']).to_csv(
            output_file_path, index=False)

    df = pd.read_csv(output_file_path)

    if model_name in df.Model.values:
        if overwrite_output:
            df.loc[df.Model == model_name,
                   "Training accuracy"] = train_accuracy
            df.loc[df.Model == model_name, "Training F1 score"] = train_f1
            df.loc[df.Model == model_name, "Testing accuracy"] = test_accuracy
            df.loc[df.Model == model_name, "Testing F1 score"] = test_f1
        else:
            print(
                f"Model {model_name} already exists in comparison results file. Skipping...")

    else:
        addition = pd.DataFrame(
            [[model_name, train_accuracy, train_f1, test_accuracy, test_f1]], columns=['Model', 'Training accuracy', 'Training F1 score', 'Testing accuracy', 'Testing F1 score'])
        df = pd.concat([df, addition])

    df.to_csv(output_file_path, index=False)
