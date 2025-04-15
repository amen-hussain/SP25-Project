import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
diabetes = pd.read_csv('/workspaces/SP25-Project/data/diabetic_data.csv')

# Preview the first 10 rows 
print(diabetes.head(10))

print(diabetes.columns)

# Replace missing values marked as '?' with NaN
diabetes.replace('?', np.nan, inplace=True)

# Drop duplicate records
diabetes.drop_duplicates(inplace=True)

# Get list of column names
columns = diabetes.columns.tolist()

# List of columns to keep
columns_to_keep = ['patient_nbr', 'race', 'gender', 'age', 'weight',
       'time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications',
       'number_outpatient', 'number_emergency', 'number_inpatient','number_diagnoses', 'max_glu_serum', 'A1Cresult',
    'change', 'readmitted']

# Keep only the specified columns
diabetes = diabetes[columns_to_keep]

# Drop columns with any null values
diabetes.dropna(axis=1, inplace=True)

# Drop rows with any null (NaN) values
diabetes.dropna(axis=0, inplace=True)

# Preview the first 10 rows 
print(diabetes.head(10))