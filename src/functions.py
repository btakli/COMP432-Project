# This file contains miscellaneous helper functions that are used in the other files.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_data(file_name: str) -> pd.DataFrame:
    """Reads the data from the file and returns it as a pandas DataFrame"""
    data = pd.read_csv(file_name, header=0)
    return data


def preprocess_data(raw_data: pd.DataFrame, verbose: bool) -> pd.DataFrame:
    """Preprocesses the data and returns it as a pandas DataFrame.

    Arguments:
    raw_data: The data to be preprocessed
    verbose: Whether to print the data dimensions, head and size reduction after preprocessing.

    Drops the C_YEAR column as all data is from 2019, as well as the following columns which are innapropriate for the analysis:
    - C_HOUR: We are only interested in the day of the week, not the time of day
    - C_SEV: We are only interested if an accident occured, not the severity of the accident
    - C_VEHS: We are only interested if an accident occured, not the number of vehicles involved
    - C_CONF: We are only interested if an accident occured, not the configuration of the accident
    - C_RCFG: Not interested in road configuration
    - C_RSUR: We will use the weather conditions instead, as they are more relevant and partially imply the road surface.
    - C_RALN: We are only interested if an accident occured, not the road alignment
    - C_TRAF: Not interested in the traffic control configuration
    - V_ID: Not interested in the vehicle ID
    - V_YEAR: Not interested in the vehicle year, it is too specific and may cause overfitting
    - P_ID: Not interested in the person ID
    - P_PSN: Not interested in the person's position in the car
    - P_ISEV: Not interested in the person's injury severity
    - P_SAFE: Not interested in the person's safety equipment
    - C_CASE: Not interested in the case number, no bearing on the outcome

    We will also drop all rows with missing/unknown values in the remaining columns of interest (i.e rows that contain U, UU, X, XX for any column of interest).

    This leaves us with the following columns:
    - C_MNTH: Month of the accident
    - C_WDAY: Day of the week of the accident
    - C_WTHR: Weather conditions of the accident. We convert Q to 0 as it is still a weather condition, but different from the other values.
    - V_TYPE: Type of vehicle involved in the accident
    - P_SEX: Sex of the person involved in the accident. Either F or M. We will convert this to 0 and 1 respectively.
    - P_AGE: Age of the person involved in the accident
    - P_USER: Type of user involved in the accident (driver, passenger, pedestrian, etc.)
    """

    if verbose:
        print(f"Raw data dimensions: [{raw_data.shape[0]} x {raw_data.shape[1]}]")
        print(raw_data.head())
        print()

    # Drop C_YEAR column as all data is from 2019
    data = raw_data.drop(columns=['C_YEAR'])
    # Drop columns that are not needed
    data = data.drop(columns=['C_HOUR', 'C_SEV', 'C_VEHS', 'C_CONF', 'C_RCFG', 'C_RSUR',
                     'C_RALN', 'C_TRAF', 'V_ID', 'V_YEAR', 'P_ID', 'P_PSN', 'P_ISEV', 'P_SAFE', 'C_CASE'])
    # Drop all rows with missing/unknown values in columns of interest
    # If row value is U, UU, X, XX, N, NN for any column drop the row (N/NN means not applicable)
    data = data[~data.isin(['U', 'UU', 'X', 'XX', 'N', 'NN']).any(axis=1)]

    # Convert Q to 0 as it is still a weather condition, but different from the other values
    data['C_WTHR'] = data['C_WTHR'].replace('Q', '0')

    # Convert F to 0 and M to 1 for P_SEX
    data['P_SEX'] = data['P_SEX'].replace('F', '0')
    data['P_SEX'] = data['P_SEX'].replace('M', '1')

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


def extract_features_and_labels(data: pd.DataFrame, label_column: str) -> tuple[pd.DataFrame, np.ndarray]:
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
    return (features, labels)
