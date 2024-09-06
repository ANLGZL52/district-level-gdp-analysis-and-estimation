import pandas as pd
import matplotlib.pyplot as plt

# Load the Excel files
ilgdp_ln_df = pd.read_excel("ilgdp_LN.xlsx")
türetilmişgdp_ln_df = pd.read_excel("türetilmişgdp_LN.xlsx")

# Display the first few rows of each dataframe to understand their structure
ilgdp_ln_df.head(), türetilmişgdp_ln_df.head()

# Rename the columns based on the first row and drop the first row which is not needed anymore
ilgdp_ln_df.columns = ilgdp_ln_df.iloc[0]
türetilmişgdp_ln_df.columns = türetilmişgdp_ln_df.iloc[0]

# Drop the first row in both dataframes as it is now redundant
ilgdp_ln_df = ilgdp_ln_df.drop(0)
türetilmişgdp_ln_df = türetilmişgdp_ln_df.drop(0)

# Set the first column as the index because it contains the dates
ilgdp_ln_df = ilgdp_ln_df.set_index(ilgdp_ln_df.columns[0])
türetilmişgdp_ln_df = türetilmişgdp_ln_df.set_index(türetilmişgdp_ln_df.columns[0])

# Calculate correlation between real and weighted GDP values for each province
correlations = {}
for column in ilgdp_ln_df.columns:
    if column in türetilmişgdp_ln_df.columns:
        correlations[column] = ilgdp_ln_df[column].astype(float).corr(türetilmişgdp_ln_df[column].astype(float))

correlations_df = pd.DataFrame(list(correlations.items()), columns=['Province', 'Correlation'])
correlations_df.head()

# Format the index to show only year
ilgdp_ln_df.index = pd.to_datetime(ilgdp_ln_df.index).year
türetilmişgdp_ln_df.index = pd.to_datetime(türetilmişgdp_ln_df.index).year

# Adjust the function to place correlation text next to the legend without overlap
# Function to display correlation in a box within each plot as seen in the user's uploaded example
def plot_nine_province_group_with_boxed_corr(start, end, total_provinces):
    fig, axes = plt.subplots(nrows=9, ncols=1, figsize=(15, 12))
    fig.subplots_adjust(hspace=0.5)
    provinces = total_provinces[start:end]

    for ax, province in zip(axes, provinces):
        ax.plot(ilgdp_ln_df.index, ilgdp_ln_df[province], label=f'{province} ln(GSYİH)', marker='o', color='blue')
        ax.plot(türetilmişgdp_ln_df.index, türetilmişgdp_ln_df[province], label=f'{province} ln(NTL-GSYİH)', marker='x', color='green')
        ax.set_title(f"{province}", fontsize=8)
        ax.set_xticks(ilgdp_ln_df.index)
        ax.set_xticklabels(ilgdp_ln_df.index, rotation=0, fontsize=8)
        ax.set_ylabel('LNGSYİH', fontsize=8)
        ax.legend(loc='upper left', fontsize=6, frameon=True, framealpha=1)
        ax.grid(True)
        # Display correlation in a boxed format
        correlation_text = f'Correlation: {correlations[province]:.2f}'
        ax.text(0.165, 0.95, correlation_text, fontsize=6, color='blue', bbox=dict(facecolor='none', edgecolor='blue', boxstyle='round,pad=0.3'), transform=ax.transAxes, verticalalignment='top', horizontalalignment='right')

    plt.xlabel("", fontsize=8)
    plt.tight_layout()
    plt.show()

# Adjusting the function to plot each group of nine provinces consecutively with clear correlation display
def plot_consecutive_nine_province_groups():
    total_provinces = list(correlations.keys())
    num_provinces = len(total_provinces)
    num_groups = (num_provinces + 8) // 9  # Calculate how many groups of nine are needed

    for i in range(num_groups):
        start = i * 9
        end = min(start + 9, num_provinces)
        plot_nine_province_group_with_boxed_corr(start, end, total_provinces)

# Start plotting groups of nine provinces consecutively
plot_consecutive_nine_province_groups()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the Excel files
ilgdp_ln_df = pd.read_excel('ilgdp_LN.xlsx')
türetilmişgdp_ln_df = pd.read_excel('türetilmişgdp_LN.xlsx')

# Set the first row as header for both datasets
ilgdp_ln_df.columns = ilgdp_ln_df.iloc[0]
türetilmişgdp_ln_df.columns = türetilmişgdp_ln_df.iloc[0]

# Remove the first row now that it's set as the header
ilgdp_ln_df = ilgdp_ln_df.drop(0)
türetilmişgdp_ln_df = türetilmişgdp_ln_df.drop(0)

# Set the first column as the index for both datasets
ilgdp_ln_df = ilgdp_ln_df.set_index(ilgdp_ln_df.columns[0])
türetilmişgdp_ln_df = türetilmişgdp_ln_df.set_index(türetilmişgdp_ln_df.columns[0])

# Calculate correlation between real and weighted GDP values for each province
correlations = {}
for column in ilgdp_ln_df.columns:
    if column in türetilmişgdp_ln_df.columns:
        correlations[column] = ilgdp_ln_df[column].astype(float).corr(türetilmişgdp_ln_df[column].astype(float))

correlations_df = pd.DataFrame(list(correlations.items()), columns=['Province', 'Correlation'])
correlations_df.head()

# Format the index to show only year
ilgdp_ln_df.index = pd.to_datetime(ilgdp_ln_df.index).year
türetilmişgdp_ln_df.index = pd.to_datetime(türetilmişgdp_ln_df.index).year

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
create_selected_province_plots(ilgdp_ln_df, türetilmişgdp_ln_df, selected_provinces)
