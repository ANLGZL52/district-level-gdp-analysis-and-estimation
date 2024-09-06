import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

# Load the shapefile
shapefile_path = "tur_polbna_adm2.shp"
gdf = gpd.read_file(shapefile_path)

# Load the Excel file
excel_path = "geoda.xlsx"
df = pd.read_excel(excel_path)

# Print the first few rows to inspect the data
print("Shapefile Data:")
print(gdf.head())
print("Excel Data:")
print(df.head())

# Align OBJECTIDs in both datasets
gdf = gdf.sort_values(by="OBJECTID")
gdf['NEW_OBJECTID'] = range(1, len(gdf) + 1)
gdf = gdf.drop(columns=["OBJECTID"]).rename(columns={'NEW_OBJECTID': "OBJECTID"})

df = df.sort_values(by="OBJECTID")
df['NEW_OBJECTID'] = range(1, len(df) + 1)
df = df.drop(columns=["OBJECTID"]).rename(columns={'NEW_OBJECTID': "OBJECTID"})

# Verify the OBJECTID range in both datasets
print("Shapefile OBJECTID range:", gdf["OBJECTID"].min(), "-", gdf["OBJECTID"].max())
print("Excel OBJECTID range:", df["OBJECTID"].min(), "-", df["OBJECTID"].max())

# Merge the datasets on the reindexed OBJECTID
merged_gdf = gdf.merge(df, how='left', left_on="OBJECTID", right_on="OBJECTID")

# Verify the merged data
print("Merged Data:")
print(merged_gdf.columns)

# Plot the merged data for GDP in 2013
fig, ax = plt.subplots(1, 1, figsize=(15, 10))

# Plot the geometry with GDP for 2013
merged_gdf.plot(column='GDPpcTL19_y', ax=ax, legend=True,
                legend_kwds={'label': "GDP PC< (2013)",
                             'orientation': "horizontal"},
                cmap='viridis')

# Add title and labels
ax.set_title("Turkey Districts GDP Total (2013)", fontsize=15)
ax.set_axis_off()

# Show the plot
plt.show()
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import imageio

# Load the shapefile
shapefile_path = "tur_polbna_adm2.shp"
gdf = gpd.read_file(shapefile_path)

# Load the Excel file
excel_path = "geoda.xlsx"
df = pd.read_excel(excel_path)

# Align OBJECTIDs in both datasets
gdf = gdf.sort_values(by="OBJECTID")
gdf['NEW_OBJECTID'] = range(1, len(gdf) + 1)
gdf = gdf.drop(columns=["OBJECTID"]).rename(columns={'NEW_OBJECTID': "OBJECTID"})

df = df.sort_values(by="OBJECTID")
df['NEW_OBJECTID'] = range(1, len(df) + 1)
df = df.drop(columns=["OBJECTID"]).rename(columns={'NEW_OBJECTID': "OBJECTID"})

# Merge the datasets on the reindexed OBJECTID
merged_gdf = gdf.merge(df, how='left', left_on="OBJECTID", right_on="OBJECTID")

# Define the GDP columns to plot
gdp_columns = ['GDPTTL13_y', 'GDPTTL14_y', 'GDPTTL15_y', 'GDPTTL16_y', 'GDPTTL17_y', 'GDPTTL18_y',
               'GDPTTL19_y', 'GDPTTL20_y', 'GDPTTL21_y', 'GDPTTL22_y', 'GDPTTL23_y']

# Create a list to store the file names of the generated images
filenames = []

# Generate and save plots for each year
for year in range(2013, 2024):
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    column_name = f'GDPTTL{year % 100:02d}_y'

    merged_gdf.plot(column=column_name, ax=ax, legend=True,
                    legend_kwds={'label': f"GDP Total ({year})",
                                 'orientation': "horizontal"},
                    cmap='viridis')

    # Add title and labels
    ax.set_title(f"Turkey Districts GDP Total ({year})", fontsize=15)
    ax.set_axis_off()

    # Save the plot as an image file
    filename = f"/mnt/data/turkey_gdp_{year}.png"
    plt.savefig(filename)
    filenames.append(filename)
    plt.close()

# Create a GIF from the saved images
gif_filename = "turkey_total4_gdp.gif"
with imageio.get_writer(gif_filename, mode='I', duration=2) as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)

gif_filename
import imageio.v2 as imageio

# List of filenames for the generated images (2013 to 2023)
filenames = [
    "turkey_gdp_2013_natural_breaks_7.png",
    "turkey_gdp_2014_natural_breaks_7.png",
    "turkey_gdp_2015_natural_breaks_7.png",
    "turkey_gdp_2016_natural_breaks_7.png",
    "turkey_gdp_2017_natural_breaks_7.png",
    "turkey_gdp_2018_natural_breaks_7.png",
    "turkey_gdp_2019_natural_breaks_7.png",
    "turkey_gdp_2020_natural_breaks_7.png",
    "turkey_gdp_2021_natural_breaks_7.png",
    "turkey_gdp_2022_natural_breaks_7.png",
    "turkey_gdp_2023_natural_breaks_7.png"
]

# Output GIF path
output_gif_path = "turkey_gdp_natural_breaks_7.gif"

# Create a GIF with each frame having a 1.5-second duration and set to loop continuously
with imageio.get_writer(output_gif_path, mode='I', duration=1500, loop=0) as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)

output_gif_path