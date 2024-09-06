import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import os
import shutil

# Load the uploaded Excel files
nightlights_df = pd.read_excel('İLGECEIŞIKLARI_LN.xlsx')
gdp_df = pd.read_excel('ilgerçekgdp.xlsx')

# Convert datetime column names to string format (year)
nightlights_df.columns = nightlights_df.columns.astype(str)
gdp_df.columns = gdp_df.columns.astype(str)

# List of years available in the datasets
years = [col for col in nightlights_df.columns if col != 'Unnamed: 0']

# Create a directory to store the scatter plot images
output_dir = '/mnt/data/scatter_plots'
os.makedirs(output_dir, exist_ok=True)

# Save individual scatter plots as separate images in the directory
for year in years:
    x = nightlights_df[year].values.reshape(-1, 1)
    y = gdp_df[year].values.reshape(-1, 1)

    # Linear Regression model
    model = LinearRegression()
    model.fit(x, y)
    y_pred = model.predict(x)

    # Calculate R-squared value
    r_squared = r2_score(y, y_pred)

    # Scatter plot
    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, color='blue', label='Data Point')
    plt.plot(x, y_pred, color='red', label=f'Regression Line (R²={r_squared:.2f})')
    plt.title(f'Scatter Plot  {year[:4]}')
    plt.xlabel('NTL (ln)')
    plt.ylabel('Province GDP (ln)')
    plt.legend()
    plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import math

# Load the uploaded Excel files
nightlights_df = pd.read_excel('İLGECEIŞIKLARI_LN.xlsx')
gdp_df = pd.read_excel('ilgerçekgdp.xlsx')

# Convert datetime column names to string format (year)
nightlights_df.columns = nightlights_df.columns.astype(str)
gdp_df.columns = gdp_df.columns.astype(str)

# List of years available in the datasets
years = [col for col in nightlights_df.columns if col != 'Unnamed: 0']

# Calculate the number of rows and columns needed for the grid
n_years = len(years)
n_cols = 3  # Number of columns you want
n_rows = math.ceil(n_years / n_cols)

# Set font size for the entire plot
plt.rc('font', size=6)  # controls default text sizes
plt.rc('axes', titlesize=8)  # fontsize of the axes title
plt.rc('axes', labelsize=6)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=4)  # fontsize of the tick labels
plt.rc('ytick', labelsize=4)  # fontsize of the tick labels
plt.rc('legend', fontsize=9)  # legend fontsize
plt.rc('figure', titlesize=9)  # fontsize of the figure title

# Create a figure with specified size
fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 5 * n_rows))

# Flatten the axes array for easy iteration
axes = axes.flatten()

# Save individual scatter plots as separate images in the directory
for idx, year in enumerate(years):
    x = nightlights_df[year].values.reshape(-1, 1)
    y = gdp_df[year].values.reshape(-1, 1)

    # Linear Regression model
    model = LinearRegression()
    model.fit(x, y)
    y_pred = model.predict(x)

    # Calculate R-squared value
    r_squared = r2_score(y, y_pred)

    # Scatter plot
    axes[idx].scatter(x, y, color='blue', label='Data Point')
    axes[idx].plot(x, y_pred, color='red', label=f'Regression Line (R²={r_squared:.2f})')
    axes[idx].set_title(f'Scatter Plot {year[:4]}', fontsize = 6)
    axes[idx].set_xlabel('NTL (ln)',fontsize = 6)
    axes[idx].set_ylabel('Province GDP (ln)',fontsize = 6)
    axes[idx].legend()

# Hide any unused subplots
for i in range(len(years), len(axes)):
    fig.delaxes(axes[i])

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()
