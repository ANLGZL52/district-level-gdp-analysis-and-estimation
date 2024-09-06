import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the uploaded Excel file
file_path = 'SEGE2022.xlsx'
df = pd.read_excel(file_path)

# Renaming columns for clarity and easier access
df.columns = ['Province', 'District', 'SEGE2022', 'GDP1_2022', 'GDP2_2022']

# Calculating correlation between SEGE scores and GDP1_2022 and GDP2_2022
correlation_gdp1 = df['SEGE2022'].corr(df['GDP1_2022'])
correlation_gdp2 = df['SEGE2022'].corr(df['GDP2_2022'])

# Scatter plot with regression line for SEGE2022 vs GDP1_2022
plt.figure(figsize=(7, 6))
sns.regplot(x='GDP1_2022', y='SEGE2022', data=df, scatter_kws={'s':10,'color': 'orange'}, line_kws={'color':'red'})
plt.title(f'SEGE_2022 ve GDP_2022 Scatter Plot\nCorrelation: {correlation_gdp1:.2f}')
plt.xlabel('GDP2022')
plt.ylabel('SEGE2022')
plt.tight_layout()
plt.savefig('/mnt/data/SEGE_GDP1_Correlation_Plot.png')
plt.show()

# Plot for GDP2_2022
plt.figure(figsize=(7, 6))
sns.regplot(x='GDP2_2022', y='SEGE2022', data=df, scatter_kws={'s':10, 'color': 'blue'}, line_kws={'color':'red'})
plt.title(f'SEGE2022 ve After Normalization GDP_2022 Scatter Plot\nCorrelation: {correlation_gdp2:.2f}')
plt.xlabel('GDP2022')
plt.ylabel('SEGE2022')
plt.tight_layout()
plt.savefig('/mnt/data/SEGE_GDP2_Correlation_Plot.png')
plt.show()
import pandas as pd

# Load the uploaded Excel file
file_path = 'SEGE2022.xlsx'
df = pd.read_excel(file_path)

# Renaming columns for clarity and easier access
df.columns = ['Province', 'District', 'SEGE2022', 'GDP1_2022', 'GDP2_2022']

# Group the data by Province and calculate the correlation for each province
correlation_by_province = df.groupby('Province').apply(lambda x: x['SEGE2022'].corr(x['GDP1_2022']))
correlation_by_province_gdp2 = df.groupby('Province').apply(lambda x: x['SEGE2022'].corr(x['GDP2_2022']))

# Combine the results into a single DataFrame
correlation_table = pd.DataFrame({
    'Correlation_SEGE_GDP1': correlation_by_province,
    'Correlation_SEGE_GDP2': correlation_by_province_gdp2
})

# Reset index for better readability
correlation_table.reset_index(inplace=True)

# Save the table to a CSV file
correlation_table.to_csv('Correlation_Table_by_Province.csv', index=False)

# Display the table
correlation_table
print(correlation_table)