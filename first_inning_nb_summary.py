import pandas as pd
import numpy as np
import os
from statsmodels.discrete.discrete_model import NegativeBinomial
import statsmodels.api as sm

# Directory containing all yearly CSVs
data_dir = "/Users/kevinhe/orioles-project/data/out"
output_csv_yearly = "/Users/kevinhe/orioles-project/first_inning_nb_summary.csv"
output_csv_decade = "/Users/kevinhe/orioles-project/first_inning_nb_summary_by_decade.csv"

results = []

for year in range(1911, 2025):
    file_path = os.path.join(data_dir, f"{year}", f"first_inning_runs_summary_{year}.csv")
    if not os.path.exists(file_path):
        continue

    df = pd.read_csv(file_path)

    # Rename columns according to your actual CSV headers
    home_df = df[['game_id', 'hometeam', 'home_first_inning_runs']].rename(
        columns={'hometeam': 'team_id', 'home_first_inning_runs': 'first_inning_runs'}
    )
    home_df['team_type'] = 'home'

    away_df = df[['game_id', 'visteam', 'visiting_first_inning_runs']].rename(
        columns={'visteam': 'team_id', 'visiting_first_inning_runs': 'first_inning_runs'}
    )
    away_df['team_type'] = 'away'

    data = pd.concat([home_df, away_df])

    # Encode team_type (home = 1, away = 0)
    data['is_home'] = (data['team_type'] == 'home').astype(int)

    # Add constant for intercept
    X = sm.add_constant(data['is_home'])
    y = data['first_inning_runs']

    # Negative binomial regression
    model = NegativeBinomial(y, X)
    result = model.fit(disp=False)

    # Extract coefficients
    intercept = result.params['const']
    beta_home = result.params['is_home']

    # Compute expected mean (E[Y]) for away and home teams
    mu_away = np.exp(intercept)
    mu_home = np.exp(intercept + beta_home)

    # Dispersion (alpha) parameter
    alpha = result.params.iloc[-1]

    # Compute home advantage (ratio)
    home_advantage = mu_home - mu_away

    results.append({
        'year': year,
        'E[Y_away]': mu_away,
        'E[Y_home]': mu_home,
        'alpha': alpha,
        'home_advantage': home_advantage
    })

# Save yearly results
results_df = pd.DataFrame(results)
results_df.to_csv(output_csv_yearly, index=False)
print(f"Saved yearly summary with home advantage to {output_csv_yearly}")

# Aggregate by decade
results_df['decade'] = (results_df['year'] // 10) * 10  # e.g., 1911 -> 1910
decade_summary = results_df.groupby('decade').agg({
    'E[Y_away]': 'mean',
    'E[Y_home]': 'mean',
    'alpha': 'mean',
    'home_advantage': 'mean'
}).reset_index()

decade_summary.to_csv(output_csv_decade, index=False)
print(f"Saved decade summary with home advantage to {output_csv_decade}")