import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# --- Step 1: Load all years ---
years = range(2014, 2025)
all_data = []

for year in years:
    file_path = f"/Users/kevinhe/orioles-project/data/out/rolling_avg/game_level_with_rolling_avg_weather_{year}.csv"
    df = pd.read_csv(file_path)
    
    # Compute home scored first inning as binary
    df['home_scored_first'] = (df['home_first_inning_runs'] > 0).astype(int)
    
    # Assign weights: more recent years have higher weight
    df['weight'] = 1 / (2025 - year)  # e.g., 2024 has weight 1, 2014 has weight 1/11
    
    all_data.append(df)

# Concatenate all years
df_all = pd.concat(all_data, ignore_index=True)

# --- Step 2: Define features ---
# Add OBP features you prepared, ERA, travel distance, etc.
features = [
    'home_OBP_x', 'away_OBP_x', 'home_OBP_y', 'away_OBP_y',
    'home_ERA', 'vis_ERA', 'home_travel', 'vis_travel',
    'home_avg_prev', 'away_avg_prev'
]

X = df_all[features]
y = df_all['home_scored_first']
sample_weight = df_all['weight']

# --- Step 3: Handle missing values with imputer ---
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# --- Step 4: Train/test split ---
X_train, X_test, y_train, y_test, sw_train, sw_test = train_test_split(
    X_imputed, y, sample_weight, test_size=0.2, random_state=42
)

# --- Step 5: Fit logistic regression ---
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train, sample_weight=sw_train)

# --- Step 6: Evaluate ---
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_prob))

# --- Step 7: Optional: inspect coefficients ---
coef_df = pd.DataFrame({
    'feature': features,
    'coefficient': model.coef_[0]
}).sort_values(by='coefficient', ascending=False)

print("\nFeature Coefficients:\n", coef_df)
