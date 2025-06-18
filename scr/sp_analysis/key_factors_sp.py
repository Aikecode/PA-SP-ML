import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns 

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.inspection import partial_dependence
from xgboost import XGBClassifier # Directly import XGBoost as it's heavily used here
from scipy.interpolate import splev, splrep

from src.data_processing import load_and_split_data
from src.models import create_model_with_params # Still useful for PFI model

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("--- Starting SP Key Factors Analysis ---")

X_full, y_full, feature_names_full = load_and_split_data(
    outcome_type='sp', 
    data_purpose='pdp' 
)

print(f"\nFull data loaded for key factors analysis: X_full shape={X_full.shape}, y_full shape={y_full.shape}")

optimal_xgb_params = {
    'n_estimators': 50,
    'learning_rate': 0.05,
    'max_depth': 6,
    'min_child_weight': 1,
    'gamma': 1,
    'subsample': 1.0,
    'colsample_bytree': 1.0,
    'random_state': 42
}

X_train_pfi, X_test_pfi, y_train_pfi, y_test_pfi = train_test_split(
    X_full, y_full, test_size=0.2, stratify=y_full, random_state=42
)

print("\nFitting optimal XGBoost model for PFI...")
xgb_model_pfi = create_model_with_params('XGBoost', optimal_xgb_params)
xgb_model_pfi.fit(X_train_pfi, y_train_pfi)

print("\nCalculating Permutation Feature Importance...")
perm_importance = permutation_importance(
    xgb_model_pfi,
    X_test_pfi, 
    y_test_pfi,
    n_repeats=10, 
    random_state=42,
    scoring='roc_auc',
    n_jobs=-1     
)

sorted_idx_pfi = perm_importance.importances_mean.argsort()[::-1] # 降序
print("\nPermutation Feature Importance (PFI) Results (Mean Importance):")
for f_name, f_importance in zip(X_test_pfi.columns[sorted_idx_pfi], perm_importance.importances_mean[sorted_idx_pfi]):
    print(f"{f_name}: {f_importance:.4f}")

feature_importances = pd.DataFrame({
    'Feature': X_test_pfi.columns,
    'Importance': perm_importance.importances_mean
}).sort_values(by='Importance', ascending=False)

print(f"\n{'='*20} XGBoost Stepwise Factor Addition {'='*20}")

positive_pfi_features = feature_importances[feature_importances['Importance'] > 0]['Feature'].tolist()
print(f"\nFeatures with positive PFI: {positive_pfi_features}")

data_pfi_simulated = X_full.copy()
data_pfi_simulated[y_full.name] = y_full 

X_all_iterative = data_pfi_simulated[positive_pfi_features]
y_iterative = data_pfi_simulated['sports_participation']

X_train_full_iterative, X_test_full_iterative, y_train_iterative, y_test_iterative = train_test_split(
    X_all_iterative, y_iterative, test_size=0.2, stratify=y_iterative, random_state=42)

results_iterative = []

print(f"\n{'='*20} Iterative Feature Selection {'='*20}")
for i in range(1, len(positive_pfi_features) + 1):
    current_features = positive_pfi_features[:i]
    X_train_current = X_train_full_iterative[current_features]
    X_test_current = X_test_full_iterative[current_features]

    xgb_model_iterative = XGBClassifier(
        n_estimators=50,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=1,
        gamma=1,
        subsample=1.0,
        colsample_bytree=1.0,
        random_state=42
    )

    xgb_model_iterative.fit(X_train_current, y_train_iterative)
    best_model_iterative = xgb_model_iterative

    y_pred_iterative = best_model_iterative.predict(X_test_current)

    y_prob_iterative = best_model_iterative.predict_proba(X_test_current)[:, 1]

    test_acc_iterative = accuracy_score(y_test_iterative, y_pred_iterative)
    test_f1_iterative = f1_score(y_test_iterative, y_pred_iterative)
    test_auc_iterative = roc_auc_score(y_test_iterative, y_prob_iterative)

    print(f"\nUsing first {i} features: {current_features}")
    print("  Test set evaluation:")
    print(f"    Accuracy: {test_acc_iterative:.4f}")
    print(f"    F1 Score: {test_f1_iterative:.4f}")
    print(f"    AUC: {test_auc_iterative:.4f}")

    results_iterative.append({
        'Number of Features': i,
        'Feature List': current_features,
        'Test set Accuracy': test_acc_iterative,
        'Test set F1': test_f1_iterative,
        'Test set AUC': test_auc_iterative
    })

results_df_iterative = pd.DataFrame(results_iterative)

print("\nAll results table (iterative feature selection):")
print(results_df_iterative)

print(f"\n{'='*20} Partial Dependence Plots {'='*20}")

data_pdp_simulated = X_full.copy()
data_pdp_simulated['Sports_participation'] = y_full 

selected_features_pdp = [
    'Learning', 'Recreational_lifestyle', 'Education_level', 'Family_Friend_gathering',
    'Suitability_for_Exercise', 'Health_examination', 'Cultural_lifestyle', 'Socioeconomic_status',
    'Richness_of_facilities', 'BMI_category', 'Income_level', 'Age_group'
]

X_pdp = data_pdp_simulated[selected_features_pdp]
y_pdp = data_pdp_simulated['Sports_participation'] 

X_train_pdp, X_test_pdp, y_train_pdp, y_test_pdp = train_test_split(
    X_pdp, y_pdp, test_size=0.2, stratify=y_pdp, random_state=42)

xgb_model_pdp = XGBClassifier(
    n_estimators=50,
    learning_rate=0.05,
    max_depth=6,
    min_child_weight=1,
    gamma=1,
    subsample=1.0,
    colsample_bytree=1.0,
    random_state=42
)

xgb_model_pdp.fit(X_train_pdp, y_train_pdp)

best_model_pdp = xgb_model_pdp
y_pred_pdp = best_model_pdp.predict(X_test_pdp)

if hasattr(best_model_pdp, "predict_proba"):
    y_prob_pdp = best_model_pdp.predict_proba(X_test_pdp)[:, 1]
else:
    y_prob_pdp = best_model_pdp.decision_function(X_test_pdp)

print("\nDrawing and saving Partial Dependence Plots...")

num_features = len(selected_features_pdp) 
n_cols = 2
n_rows = int(np.ceil(num_features / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
axes = axes.flatten()

for idx, feature in enumerate(selected_features_pdp): 
    pdp = partial_dependence(best_model_pdp, X_train_pdp, [feature], grid_resolution=20, kind="both") 
    plot_x = pd.Series(pdp['values'][0]).rename('x')   
    plot_y = pdp['average'][0]                         
    plot_i = pdp['individual'][0]                      

    plot_df = pd.DataFrame(columns=['x', 'y'])
    for a in plot_i:
        a2 = pd.Series(a)
        df_i = pd.concat([plot_x.to_frame(), a2.rename('y')], axis=1) 
        plot_df = pd.concat([plot_df, df_i], axis=0, ignore_index=True) 

    sns.lineplot(data=plot_df, x="x", y="y", ax=axes[idx],
                 color=(0.6157, 0.7647, 0.9059), linewidth=1.2, linestyle='-', alpha=0.5, errorbar=('ci', 95)) 

    axes[idx].plot(plot_x, plot_y, color=(0.3725, 0.5922, 0.8235), alpha=0.4, linestyle='--', label='PDP (raw)')

    try:
        tck = splrep(plot_x, plot_y, s=30, k=2) 
        xnew = np.linspace(plot_x.min(), plot_x.max(), 300)
        ynew = splev(xnew, tck, der=0)
        axes[idx].plot(xnew, ynew, color=(0.5765, 0.5804, 0.9059), linewidth=2, label='Smoothed PDP')
    except Exception as e:
        print(f"⚠️ Interpolation fitting failed (feature '{feature}'): {e}")

    axes[idx].set_xlabel(feature)
    axes[idx].set_ylabel("Sports participation") 
    axes[idx].set_title("") 
    axes[idx].legend(loc='upper left')

for j in range(idx + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()

plt.show()

print("\n--- SP Key Factors Analysis Finished ---") 