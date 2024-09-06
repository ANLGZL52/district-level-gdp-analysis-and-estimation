import pandas as pd
import matplotlib.pyplot as plt

# Load the newly uploaded Excel files
new_data_from_districts = pd.read_excel('ilcelerdeniletoplamdeger.xlsx')
new_data_from_gee = pd.read_excel('direktgeedenyıllıktoplamlar.xlsx')

# Function to prepare data by setting headers and extracting years
def prepare_data(df):
    df.columns = df.iloc[0]  # Set the first row as column headers
    df = df.drop(index=0).reset_index(drop=True)
    df['Year'] = pd.to_datetime(df[df.columns[0]]).dt.year  # Extract year
    return df.drop(df.columns[0], axis=1)  # Drop the first column after extracting the year

# Prepare both datasets
prepared_data_from_districts = prepare_data(new_data_from_districts)
prepared_data_from_gee = prepare_data(new_data_from_gee)

# Extract the list of provinces, excluding the 'Year' column
provinces = prepared_data_from_districts.columns[1:]  # Adjusted to skip 'Year'

# Number of provinces per figure
provinces_per_figure = 9

# Create multiple figures
for i in range(0, len(provinces), provinces_per_figure):
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(15, 10), sharey=True)  # Setup for 8 subplots
    axes = axes.flatten()

    # Plot data for each province in the current set
    for j, ax in enumerate(axes):
        if i + j >= len(provinces):
            break
        province = provinces[i + j]
        ax.plot(prepared_data_from_districts['Year'], prepared_data_from_districts[province], label='illerin aylık toplamları', marker='o')
        ax.plot(prepared_data_from_gee['Year'], prepared_data_from_gee[province], label='il yıllık değerler', marker='x')
        ax.set_title(province, fontsize=6)  # Adjust the font size of the title
        ax.set_xlabel( '', fontsize=8)
        ax.set_ylabel('Toplam Gece Işıkları')
        ax.legend()

    # Adjust layout and display the figure
    plt.tight_layout()
    plt.show()
