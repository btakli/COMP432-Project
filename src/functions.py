# This file contains miscellaneous helper functions that are used in the other files.
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
    - C_CASE: Case number"""
    data = pd.read_csv(file_name, header=0)

    return data

def profile_data(data: pd.DataFrame, output_path: str):
    """Profiles the data and saves the report to the output path.

    Arguments: 
    data: The data to be profiled
    output_path: The path to the output folder"""
    report = yp.ProfileReport(data)
    report.to_file(output_path)

def preprocess_data(raw_data: pd.DataFrame, verbose: bool, columns_to_drop: list[str]) -> pd.DataFrame:
    """Preprocesses the data and returns it as a pandas DataFrame.

    Arguments:
    raw_data: The data to be preprocessed
    verbose: Whether to print the data dimensions, head and size reduction after preprocessing.
    columns_to_drop: The columns to drop from the data. Excludes C_YEAR, C_CASE as it is automatically dropped.

    Preprocessing steps:
    - We will drop all rows with missing/unknown values in the remaining columns of interest (i.e rows that contain U, UU, X, XX for any column of interest).
    - We will drop the C_YEAR column as all data is from 2019.
    - We will drop the C_CASE column as it is not needed.
    - C_WTHR: Convert Q to 0 as it is still a weather condition, but different from the other values.
    - P_SEX: Convert F to 0 and M to 1.
    - C_MNTH and C_WDAY: Convert to int, now that we have dropped all rows with missing values.
    - Normalize the data (i.e. convert all values to floats)

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
    columns_to_drop = [column for column in columns_to_drop if column in data.columns]
    # List invalid columns we tried to drop
    invalid_columns = [column for column in columns_to_drop if column not in data.columns]
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


def get_size_difference(old_data: pd.DataFrame, new_data: pd.DataFrame) -> tuple:
    """Returns the difference in size between the old and new data sets in terms of rows and columns by percentage."""
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
