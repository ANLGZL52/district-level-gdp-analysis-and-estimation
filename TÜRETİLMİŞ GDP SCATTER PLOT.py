import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import math

# Load the uploaded Excel files
real_gdp_df = pd.read_excel('ilgerçekgdp.xlsx')
derived_gdp_df = pd.read_excel('iltüretilmişgdp.xlsx')

# Display the first few rows of each dataframe to understand their structure
real_gdp_df.head(), derived_gdp_df.head()

# Convert datetime column names to year format
real_gdp_df.columns = [str(col).split()[0] if col != real_gdp_df.columns[0] else col for col in real_gdp_df.columns]
derived_gdp_df.columns = [str(col).split()[0] if col != derived_gdp_df.columns[0] else col for col in derived_gdp_df.columns]

# Merge the dataframes on the province names (first column in both)
merged_df = pd.merge(real_gdp_df, derived_gdp_df, left_on=real_gdp_df.columns[0], right_on=derived_gdp_df.columns[0])

# Drop the redundant column created by the merge
merged_df = merged_df.drop(columns=['YIL'])

# Get the list of years from the columns
years = real_gdp_df.columns[1:]

# Calculate the number of rows and columns needed for the grid
n_years = len(years)
n_cols = 3  # Number of columns you want
n_rows = math.ceil(n_years / n_cols)

# Set font size for the entire plot
plt.rc('font', size=10)  # controls default text sizes
plt.rc('axes', titlesize=12)  # fontsize of the axes title
plt.rc('axes', labelsize=10)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=8)  # fontsize of the tick labels
plt.rc('ytick', labelsize=8)  # fontsize of the tick labels
plt.rc('legend', fontsize=8)  # legend fontsize
plt.rc('figure', titlesize=14)  # fontsize of the figure title

# Create a figure with specified size
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))

# Flatten the axes array for easy iteration
axes = axes.flatten()

# Create scatter plots with regression lines and R-squared values
for i, year in enumerate(years):
    x = merged_df[f'{year}_x'].values.reshape(-1, 1)
    y = merged_df[f'{year}_y'].values.reshape(-1, 1)

    model = LinearRegression()
    model.fit(x, y)
    y_pred = model.predict(x)

    r_squared = r2_score(y, y_pred)

    axes[i].scatter(x, y, label='Data')
    axes[i].plot(x, y_pred, color='red', label=f'Regression Line (R^2 = {r_squared:.2f})')
    axes[i].set_xlabel('Real GDP', fontsize = 8)
    axes[i].set_ylabel('Derived GDP', fontsize = 8)
    axes[i].set_title(f'Scatter Plot {year}', fontsize = 8)
    axes[i].legend()

# Hide any unused subplots
for i in range(len(years), len(axes)):
    fig.delaxes(axes[i])

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()
