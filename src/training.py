# This file contains the code required to train and save all the ML models.
import os 
import functions as f

# Importing the data
parent_dir = os.path.dirname(os.getcwd())
file_path = os.path.join(parent_dir, 'data', "2019_dataset_en.csv")

raw_data = f.get_data(file_path)

print(raw_data.head())


