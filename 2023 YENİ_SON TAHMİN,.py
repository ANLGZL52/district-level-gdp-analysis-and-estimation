import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the Excel files
gdp_data_2022_path = 'TÜRETİLMİŞ GDP-DÜZENLENMİŞ_LN_2022.xlsx'
population_data_full_path = 'NÜFUS_ENDOĞRU_LN.xlsx'
nightlights_data_full_path = 'TOPLAM_GECE_HAZIR_LN.xlsx'

gdp_data_2022 = pd.read_excel(gdp_data_2022_path)
population_data_full = pd.read_excel(population_data_full_path)
nightlights_data_full = pd.read_excel(nightlights_data_full_path)

# Filling missing values in population data with column mean for numeric columns only
population_data_full_filled = population_data_full.apply(lambda col: col.fillna(col.mean()) if col.dtype in ['float64', 'int64'] else col)

# Merging the datasets on 'İL' and 'İLÇE' columns
merged_data_full = pd.merge(gdp_data_2022, population_data_full_filled, on=['İL', 'İLÇE'], suffixes=('_gdp', '_pop'))
merged_data_full = pd.merge(merged_data_full, nightlights_data_full, on=['İL', 'İLÇE'], suffixes=('', '_nl'))

print("Merged Data Sample:")
print(merged_data_full.head())

# Aggregating the population and nightlights data
merged_data_full['Total_Population'] = merged_data_full[[col for col in merged_data_full.columns if '_pop' in col]].sum(axis=1)
merged_data_full['Total_Nightlights'] = merged_data_full[[col for col in merged_data_full.columns if col not in ['İL', 'İLÇE', 'Total_Population'] and not '_gdp' in col and not '_pop' in col]].sum(axis=1)

# Selecting the necessary columns for the new model
X_aggregated_full = merged_data_full[['Total_Population', 'Total_Nightlights']]
y_aggregated_full = merged_data_full['1.01.2022_gdp']

print("Aggregated Data Sample:")
print(X_aggregated_full.head())
# Splitting the data into training and testing sets
X_train_agg_full, X_test_agg_full, y_train_agg_full, y_test_agg_full = train_test_split(X_aggregated_full, y_aggregated_full, test_size=0.2, random_state=42)

# Initialize and train the Random Forest model
best_model_rf = RandomForestRegressor()
best_model_rf.fit(X_train_agg_full, y_train_agg_full)

# Predicting the GDP values for 2023
gdp_predictions_2023_full = best_model_rf.predict(X_aggregated_full)

# Adding the predictions to the original dataframe
merged_data_full['GDP_2023_Prediction'] = gdp_predictions_2023_full
output_file_path = 'Predicted_GDP3değişkenENYENİ_2023_Random.xlsx'
merged_data_full.to_excel(output_file_path, index=False)

print("GDP Predictions Sample:")
print(merged_data_full[['İL', 'İLÇE', 'GDP_2023_Prediction']].head())

# Evaluating the model performance
mse_full = mean_squared_error(y_test_agg_full, best_model_rf.predict(X_test_agg_full))
r2_full = r2_score(y_test_agg_full, best_model_rf.predict(X_test_agg_full))
cross_val_r2_full = np.mean(cross_val_score(best_model_rf, X_aggregated_full, y_aggregated_full, cv=5, scoring='r2'))

performance_scores_full = {
    'MSE': mse_full,
    'R-squared': r2_full,
    'Cross-validated R-squared': cross_val_r2_full
}

print("Model Performance Scores:")
print(performance_scores_full)
# Feature importances
feature_importances = best_model_rf.feature_importances_

# Creating a DataFrame for the feature importances
feature_importances_df = pd.DataFrame({
    'Feature': X_aggregated_full.columns,
    'Importance': feature_importances
})

print("Feature Importances:")
print(feature_importances_df)
# Splitting the data into training and testing sets
X_train_agg_full, X_test_agg_full, y_train_agg_full, y_test_agg_full = train_test_split(X_aggregated_full, y_aggregated_full, test_size=0.2, random_state=42)

# Initialize the models
models_new = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(),
    'Lasso Regression': Lasso(),
    'ElasticNet Regression': ElasticNet(),
    'Random Forest': RandomForestRegressor()
}

# Training and evaluating the models
results_new = {}
for model_name, model in models_new.items():
    model.fit(X_train_agg_full, y_train_agg_full)
    y_pred_agg_new = model.predict(X_test_agg_full)
    mse = mean_squared_error(y_test_agg_full, y_pred_agg_new)
    r2 = r2_score(y_test_agg_full, y_pred_agg_new)
    cross_val_r2 = np.mean(cross_val_score(model, X_aggregated_full, y_aggregated_full, cv=5, scoring='r2'))
    results_new[model_name] = {'MSE': mse, 'R-squared': r2, 'Cross-validated R-squared': cross_val_r2}

# Displaying the results
results_new_df = pd.DataFrame(results_new).T
print("Model Performance Comparison:")
print(results_new_df)
