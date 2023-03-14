# This file contains miscellaneous helper functions that are used in the other files.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_data(file_name: str) -> pd.DataFrame:
    """Reads the data from the file and returns it as a pandas DataFrame"""
    data = pd.read_csv(file_name)
    return data
