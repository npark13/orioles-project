import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.impute import SimpleImputer
import xgboost as xgb

# --- Load all years' data ---
years = range(2014, 2025)
dfs = []
for year in years:
    file_path = f"/Users/kevinhe/orioles-project/data/out/rolling_avg/game_level_with_rolling_avg_weather_{year}.csv"
    df = pd.read_csv(file_path)
    df['year'] = year
    dfs.append(df)

data = pd.concat(dfs, ignore_index=True)

# --- Target: 1 if home team scored in first inning, else 0 ---
data['home_scored_first'] = (data['home_first_inning_runs'] > 0).astype(int)

# --- Features (no travel variables) ---
feature_cols = [
    'home_OBP_x', 'away_OBP_x', 'home_OBP_y', 'away_OBP_y',
    'home_ERA', 'vis_ERA',
    'home_avg_prev', 'away_avg_prev'
]

X = data[feature_cols]
y = data['home_scored_first']

# --- Impute missing values ---
imputer = SimpleImputer(strategy='mean')
X = pd.DataFrame(imputer.fit_transform(X), columns=feature_cols)

# --- Year-based sample weights (older years smaller weight) ---
weights = (data['year'] - data['year'].min() + 1)
weights = weights / weights.max()

# --- Train/test split ---
X_train, X_test, y_train, y_test, sw_train, sw_test = train_test_split(
    X, y, weights, test_size=0.2, random_state=42, stratify=y
)

# --- Train XGBoost classifier ---
model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),  # handle imbalance
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)

model.fit(X_train, y_train, sample_weight=sw_train)

# --- Predictions ---
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# --- Evaluation ---
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_prob))

# --- Feature importance ---
importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values(by='importance', ascending=False)

print("\nFeature Importances:")
print(importance)
