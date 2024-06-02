import numpy as np
from scipy.ndimage import convolve1d
from scipy.ndimage import gaussian_filter1d
from scipy.signal.windows import triang
import pandas as pd
import matplotlib.pyplot as plt


response_df = pd.read_csv('../../data/wrangled/ctrp.csv')
plt.figure()

print(response_df[['area_under_curve', 'weight']])
plt.hist(auc, bins=10)
plt.title('Histogram of Value Counts')  # Add title
plt.xlabel('Value')  # Add x-axis label
plt.ylabel('Count')  # Add y-axis label # Add grid lines
plt.show()
plt.savefig('res_freq')