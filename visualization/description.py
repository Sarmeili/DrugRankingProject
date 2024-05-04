import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import norm


def balance_data(df, target_column='area_under_curve'):
    # Calculate the frequency of each unique value in the target column
    value_counts = df[target_column].value_counts()

    # Find the maximum frequency
    max_frequency = value_counts.max()

    # Create a DataFrame to store balanced data
    balanced_df = pd.DataFrame()

    # Iterate over unique values in the target column
    for value in df[target_column].unique():
        # Filter rows where target column equals the current value
        subset = df[df[target_column] == value]

        # Repeat rows in the subset to match the maximum frequency
        repeated_subset = subset.sample(n=max_frequency, replace=True)

        # Append the repeated subset to the balanced DataFrame
        balanced_df = pd.concat([balanced_df, repeated_subset], ignore_index=True)

    return balanced_df


response_df = pd.read_csv('../data/wrangled/ctrp.csv')
response_df = balance_data(response_df)
plt.figure()
auc = response_df['area_under_curve'].copy()
print(auc)
plt.hist(auc, bins=10)
plt.title('Histogram of Value Counts')  # Add title
plt.xlabel('Value')  # Add x-axis label
plt.ylabel('Count')  # Add y-axis label # Add grid lines
plt.show()
plt.savefig('res_freq')