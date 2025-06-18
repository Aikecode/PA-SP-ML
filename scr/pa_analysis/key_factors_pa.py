import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.inspection import permutation_importance, partial_dependence
from scipy.interpolate import splrep, splev 

from src.data_processing import load_and_split_data
from src.models import create_model_with_params

plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False

print("--- Starting PA Key Factors Analysis ---")

X_full, y_full, _ = load_and_split_data(
    outcome_type='pa',
    data_purpose='pdp' 
)

optimal_rf_params = {
    'bootstrap': True,
    'criterion': 'entropy',
    'max_depth': 10,
    'max_features': 'sqrt',
    'min_samples_leaf': 4,
    'min_samples_split': 2,
    'n_estimators': 275,
    'random_state': 42
}

X_train_pfi, X_test_pfi, y_train_pfi, y_test_pfi = train_test_split(
    X_full, y_full, test_size=0.2, stratify=y_full, random_state=42
)

print("\nFitting optimal Random Forest model for PFI...")
rf_model_pfi = create_model_with_params('RandomForest', optimal_rf_params)
rf_model_pfi.fit(X_train_pfi, y_train_pfi)

print("\nCalculating Permutation Feature Importance...")
result_pfi = permutation_importance(
    rf_model_pfi,
    X_test_pfi, 
    y_test_pfi,
    n_repeats=10, 
    random_state=42,
    scoring='roc_auc',
    n_jobs=-1     
)

sorted_idx_pfi = result_pfi.importances_mean.argsort()[::-1] 
print("\nPermutation Feature Importance (PFI) Results (Mean Importance):")
for f_name, f_importance in zip(X_test_pfi.columns[sorted_idx_pfi], result_pfi.importances_mean[sorted_idx_pfi]):
    print(f"{f_name}: {f_importance:.4f}")

print(f"\n{'='*20} Random Forest Stepwise Factor Addition {'='*20}")

feature_cols_all = [
    'Work_status', 'Recreational_lifestyle', 'Region', 'Fresh_food_outlets', 'Neighborhood_care',
    'COVID19_concern', 'BMI_category', 'Neighborhood_help', 'Family_background',
    'Suitability_for_Exercise', 'Cultural_lifestyle', 'Richness_of_facilities', 'Marital_status',
    'Air_pollution', 'Sex', 'Health_score', 'New_media_use', 'Social_security',
    'Income_level', 'Household_car', 'Illness_Status', 'Drinking',
    'Learning', 'Health_examination', 'Number_of_children', 'Safety',
    'Socioeconomic_status', 'Household_registration', 'Noise_pollution', 'Family_Friend_gathering'
]

try:
    X_train_full_stepwise, X_test_full_stepwise, y_train_stepwise, y_test_stepwise = train_test_split(
        X_full[feature_cols_all], y_full, test_size=0.2, stratify=y_full, random_state=42)
except KeyError as e:
    print(f"Error: One or more features in 'feature_cols_all' not found in the loaded data for stepwise selection: {e}")
    print("Please check your data file's column names or adjust 'feature_cols_all'.")
    exit() 

results_stepwise = []
for i in range(1, len(feature_cols_all) + 1):
    current_features = feature_cols_all[:i]
    X_train_current = X_train_full_stepwise[current_features]
    X_test_current = X_test_full_stepwise[current_features]

    rf_model_stepwise = create_model_with_params('RandomForest', optimal_rf_params)

    rf_model_stepwise.fit(X_train_current, y_train_stepwise)
    
    y_pred_stepwise = rf_model_stepwise.predict(X_test_current)
    y_prob_stepwise = rf_model_stepwise.predict_proba(X_test_current)[:, 1]

    test_acc_stepwise = accuracy_score(y_test_stepwise, y_pred_stepwise)
    test_f1_stepwise = f1_score(y_test_stepwise, y_pred_stepwise)
    test_auc_stepwise = roc_auc_score(y_test_stepwise, y_prob_stepwise)

    print(f"\nUsing first {i} features: {current_features}")
    print(f"  Test Set Evaluation:")
    print(f"    Accuracy: {test_acc_stepwise:.4f}")
    print(f"    F1 Score: {test_f1_stepwise:.4f}")
    print(f"    AUC: {test_auc_stepwise:.4f}")

    results_stepwise.append({
        'Number of Features': i,
        'Feature List': current_features,
        'Test Set Accuracy': test_acc_stepwise,
        'Test Set F1': test_f1_stepwise,
        'Test Set AUC': test_auc_stepwise
    })

results_df_stepwise = pd.DataFrame(results_stepwise)
print("\nAll results record table (stepwise feature addition):")
print(results_df_stepwise)

print(f"\n{'='*20} Partial Dependence Plots {'='*20}")

pdp_data_combined = X_full.copy()
pdp_data_combined['Physical_activity'] = y_full

categorical_vars_pdp = [
    'Work_status', 'Region', 'Fresh_food_outlets', 'Neighborhood_care',
    'COVID19_concern', 'BMI_category', 'Neighborhood_help',
    'Family_background', 'Suitability_for_Exercise'
]

feature_cols_pdp = [
    'Work_status', 'Recreational_lifestyle', 'Region', 'Fresh_food_outlets', 'Neighborhood_care',
    'COVID19_concern', 'BMI_category', 'Neighborhood_help', 'Family_background',
    'Suitability_for_Exercise'
]

X_pdp = pdp_data_combined[feature_cols_pdp]
y_pdp = pdp_data_combined['Physical_activity']

X_train_pdp, X_test_pdp, y_train_pdp, y_test_pdp = train_test_split(
    X_pdp, y_pdp, test_size=0.2, stratify=y_pdp, random_state=42)

rf_model_pdp = create_model_with_params('RandomForest', optimal_rf_params)

rf_model_pdp.fit(X_train_pdp, y_train_pdp)

print("\nDrawing Partial Dependence Plots...")

num_features = len(feature_cols_pdp) 
n_cols = 2
n_rows = (num_features + 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
axes = axes.flatten()

for idx, feature in enumerate(feature_cols_pdp):
    pdp = partial_dependence(rf_model_pdp, X_train_pdp, [feature], grid_resolution=20, kind='both')
    plot_x = pd.Series(pdp['values'][0]).rename('x')     
    plot_y = pdp['average'][0]                          
    plot_i = pdp['individual'][0]                       

    plot_df = pd.DataFrame(columns=['x', 'y'])
    for ice_line in plot_i:
        ice_series = pd.Series(ice_line).rename('y')
        df_i = pd.concat([plot_x.to_frame(), ice_series], axis=1)
        if not df_i.dropna().empty:
            plot_df = pd.concat([plot_df, df_i], axis=0, ignore_index=True) 

    sns.lineplot(data=plot_df, x="x", y="y", ax=axes[idx],
                 color=(0.6157, 0.7647, 0.9059), linewidth=1, linestyle='--', alpha=0.5, errorbar=('ci', 95), estimator='mean')

    axes[idx].plot(plot_x, plot_y, color=(0.3725, 0.5922, 0.8235), alpha=0.4, linestyle='--', label='PDP (raw)')
    
    try:
        if len(plot_x) > 3 and len(plot_y) > 3: 
            tck = splrep(plot_x, plot_y, s=30, k=2)
            xnew = np.linspace(plot_x.min(), plot_x.max(), 300)
            ynew = splev(xnew, tck, der=0)
            axes[idx].plot(xnew, ynew, color=(0.5765, 0.5804, 0.9059), linewidth=2, label='PDP (smoothed)')
        else:
            print(f"⚠️ Not enough data points for spline fitting (feature '{feature}'). Skipping smoothing.")
    except Exception as e:
        print(f"⚠️ Interpolation fitting failed (feature '{feature}'): {e}")

    axes[idx].set_xlabel(feature)
    axes[idx].set_ylabel("Physical activity") 
    axes[idx].set_title("") 
    axes[idx].legend(loc='upper left')

for j in range(idx + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()

plt.show() 

print("\n--- PA Key Factors Analysis Finished ---")