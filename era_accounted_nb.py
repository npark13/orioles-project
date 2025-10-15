import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import NegativeBinomial
import sys
import os

# --- Handle command-line argument ---
if len(sys.argv) >= 2:
    year = int(sys.argv[1])
else:
    year = 2024  # default fallback

file_path = f"/Users/kevinhe/orioles-project/data/out/{year}/first_inning_runs_with_era_{year}.csv"
output_file = f"/Users/kevinhe/orioles-project/data/out/first_inning_nb_results.csv"

if not os.path.exists(file_path):
    print(f"Skipping {year} — file not found: {file_path}")
    sys.exit(0)

print(f"\nLoading dataset for {year}: {file_path}")
df = pd.read_csv(file_path)

# --- Prepare dataset ---
home_df = df[['game_id', 'hometeam', 'home_first_inning_runs', 'vis_ERA']].rename(
    columns={
        'hometeam': 'team_id',
        'home_first_inning_runs': 'first_inning_runs',
        'vis_ERA': 'opp_ERA'
    }
)
home_df['is_home'] = 1

away_df = df[['game_id', 'visteam', 'visiting_first_inning_runs', 'home_ERA']].rename(
    columns={
        'visteam': 'team_id',
        'visiting_first_inning_runs': 'first_inning_runs',
        'home_ERA': 'opp_ERA'
    }
)
away_df['is_home'] = 0

data = pd.concat([home_df, away_df], ignore_index=True)
data = data.dropna(subset=['first_inning_runs', 'opp_ERA'])

if data.empty:
    print(f"No valid data for {year}, skipping.")
    sys.exit(0)

# --- Fit Negative Binomial model ---
X = data[['is_home', 'opp_ERA']]
X = sm.add_constant(X)
y = data['first_inning_runs']

model = NegativeBinomial(y, X)
result = model.fit(disp=False)

# --- Extract stats ---
intercept = result.params['const']
beta_home = result.params['is_home']
beta_opp_era = result.params['opp_opp_ERA'] if 'opp_opp_ERA' in result.params else result.params['opp_ERA']

mean_era = data['opp_ERA'].mean()
mu_home = np.exp(intercept + beta_home + beta_opp_era * mean_era)
mu_away = np.exp(intercept + beta_opp_era * mean_era)
home_advantage = mu_home - mu_away

# --- Prepare row for CSV ---
row = {
    'year': year,
    'intercept': intercept,
    'beta_home': beta_home,
    'beta_opp_era': beta_opp_era,
    'mean_opp_ERA': mean_era,
    'mu_away': mu_away,
    'mu_home': mu_home,
    'home_advantage': home_advantage
}

# --- Write or append to CSV ---
if os.path.exists(output_file):
    pd.DataFrame([row]).to_csv(output_file, mode='a', header=False, index=False)
else:
    pd.DataFrame([row]).to_csv(output_file, index=False)

print(f"Year {year} results saved to {output_file}\n")