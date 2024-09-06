import pandas as pd
import  numpy as np
import matplotlib.pyplot as plt
# Load the Excel files
nightlights_data = pd.read_excel('ilgeceısıkları_LN.xlsx')
gdp_data = pd.read_excel('ilgdp_LN.xlsx')

# Set the first row as header for both datasets
nightlights_data.columns = nightlights_data.iloc[0]
gdp_data.columns = gdp_data.iloc[0]

# Remove the first row now that it's set as the header
nightlights_data = nightlights_data.iloc[1:]
gdp_data = gdp_data.iloc[1:]

# Set the first column as the index for both datasets
nightlights_data.set_index(nightlights_data.columns[0], inplace=True)
gdp_data.set_index(gdp_data.columns[0], inplace=True)

# Convert index to datetime
nightlights_data.index = pd.to_datetime(nightlights_data.index, format='%Y-%m-%d')
gdp_data.index = pd.to_datetime(gdp_data.index, format='%Y-%m-%d')

# Check the cleaned data
nightlights_data.head(), gdp_data.head()

# Function to create line plots with secondary y-axis for nightlights
def create_dual_axis_plots(data1, data2, start_index, end_index):
    plt.figure(figsize=(15 , 12))
    provinces = data1.columns[start_index:end_index]
    num_plots = len(provinces)
    plot_rows = num_plots  # Each plot in a separate row for clarity

    for i, province in enumerate(provinces, 1):
        ax1 = plt.subplot(plot_rows, 1, i)
        ax2 = ax1.twinx()  # Create a second y-axis sharing the same x-axis
        ax1.tick_params(axis='x', labelsize=6)  # Adjust the font size for x-axis labels
        ax2.tick_params(axis='x', labelsize=6)  # Ensure it is consistent for both axes
        ax1.plot(data1.index, data1[province], 'o-', color='blue', label=f'{province} ln(GSYİH)')
        ax2.plot(data2.index, data2[province], 'x-', color='green', label=f'{province} ln(NTL)')

        ax1.set_title(f'{province} ', fontsize=10)
        ax1.set_xlabel('',fontsize=8)
        ax1.set_ylabel('ln(GSYİH)', color='blue',fontsize=8)
        ax2.set_ylabel('ln(NTL)', color='green',fontsize=8)

        ax1.tick_params(axis='y', labelcolor='blue')
        ax2.tick_params(axis='y', labelcolor='green')

        # Calculate and display correlation
        correlation = np.corrcoef(data1[province].dropna().astype(float), data2[province].dropna().astype(float))[0, 1]
        ax1.annotate(f'Correlation: {correlation:.2f}', xy=(0.18, 0.95), xycoords='axes fraction', fontsize=8,
                     # smaller font size and moved to bottom right
                     horizontalalignment='right', verticalalignment='top',
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="b", lw=1))

        # Adding legends
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', fontsize=5)

    plt.tight_layout()
    plt.show()


# Generate dual-axis line plots for the first 10 provinces
create_dual_axis_plots(gdp_data, nightlights_data, 0, 10)
# Define the total number of provinces
total_provinces = len(gdp_data.columns)


# Function to generate all plots in batches of 10 provinces
def generate_all_province_plots(data1, data2):
    # Calculate how many batches are needed
    num_batches = (total_provinces + 8) // 9  # Ensure we cover all provinces

    for i in range(num_batches):
        start_index = i * 9
        end_index = min((i + 1) * 9, total_provinces)
        create_dual_axis_plots(data1, data2, start_index, end_index)


# Generate plots for all provinces in batches of 10
generate_all_province_plots(gdp_data, nightlights_data)

# Display the first few rows of each file to understand their structure
nightlights_data.head(), gdp_data.head()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the Excel files
nightlights_data = pd.read_excel('ilgeceısıkları_LN.xlsx')
gdp_data = pd.read_excel('ilgdp_LN.xlsx')

# Set the first row as header for both datasets
nightlights_data.columns = nightlights_data.iloc[0]
gdp_data.columns = gdp_data.iloc[0]

# Remove the first row now that it's set as the header
nightlights_data = nightlights_data.iloc[1:]
gdp_data = gdp_data.iloc[1:]

# Set the first column as the index for both datasets
nightlights_data.set_index(nightlights_data.columns[0], inplace=True)
gdp_data.set_index(gdp_data.columns[0], inplace=True)

# Convert index to datetime
nightlights_data.index = pd.to_datetime(nightlights_data.index, format='%Y-%m-%d')
gdp_data.index = pd.to_datetime(gdp_data.index, format='%Y-%m-%d')

# Check the cleaned data
nightlights_data.head(), gdp_data.head()

# List of selected provinces
selected_provinces = ['ISTANBUL', 'IZMIR', 'ANKARA', 'GUMUSHANE', 'ZONGULDAK', 'KARABUK', 'ANTALYA', 'SAMSUN', 'VAN']

# Function to create line plots with secondary y-axis for nightlights for selected provinces
def create_selected_province_plots(data1, data2, provinces):
    plt.figure(figsize=(15, 18))
    num_plots = len(provinces)
    plot_rows = num_plots  # Each plot in a separate row for clarity

    for i, province in enumerate(provinces, 1):
        ax1 = plt.subplot(plot_rows, 1, i)
        ax2 = ax1.twinx()  # Create a second y-axis sharing the same x-axis
        ax1.tick_params(axis='x', labelsize=6)  # Adjust the font size for x-axis labels
        ax2.tick_params(axis='x', labelsize=6)  # Ensure it is consistent for both axes
        ax1.plot(data1.index, data1[province], 'o-', color='blue', label=f'{province} ln(GSYİH)')
        ax2.plot(data2.index, data2[province], 'x-', color='green', label=f'{province} ln(NTL)')

        ax1.set_title(f'{province}', fontsize=10)
        ax1.set_xlabel('', fontsize=8)
        ax1.set_ylabel('ln(GSYİH)', color='blue', fontsize=8)
        ax2.set_ylabel('ln(NTL)', color='green', fontsize=8)

        ax1.tick_params(axis='y', labelcolor='blue')
        ax2.tick_params(axis='y', labelcolor='green')

        # Calculate and display correlation
        correlation = np.corrcoef(data1[province].dropna().astype(float), data2[province].dropna().astype(float))[0, 1]
        ax1.annotate(f'Correlation: {correlation:.2f}', xy=(0.18, 0.95), xycoords='axes fraction', fontsize=8,
                     # smaller font size and moved to bottom right
                     horizontalalignment='right', verticalalignment='top',
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="b", lw=1))

        # Adding legends
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', fontsize=5)

    plt.tight_layout()
    plt.show()

# Generate dual-axis line plots for the selected provinces
create_selected_province_plots(gdp_data, nightlights_data, selected_provinces)





