import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import NegativeBinomial
from math import radians, sin, cos, sqrt, atan2
import os
import warnings

# --- Suppress warnings ---
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=sm.tools.sm_exceptions.HessianInversionWarning)
warnings.filterwarnings("ignore", category=sm.tools.sm_exceptions.ConvergenceWarning)

# --- Year range ---
years = range(2014, 2025)

# --- File paths ---
stadium_file = "/Users/kevinhe/orioles-project/data/stadiums.csv"
output_file = "/Users/kevinhe/orioles-project/data/out/first_inning_nb_results_with_travel.csv"

if not os.path.exists(stadium_file):
    print(f"Stadium file not found: {stadium_file}")
    exit(0)

stadiums = pd.read_csv(stadium_file)
stad_dict = stadiums.set_index("team_id")[["lat", "lon"]].to_dict(orient="index")

# --- Haversine function ---
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # km
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)
    a = sin(dphi/2)**2 + cos(phi1)*cos(phi2)*sin(dlambda/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))

# --- Loop over years ---
all_rows = []

for year in years:
    year_folder = f"/Users/kevinhe/orioles-project/data/out/{year}"
    game_file = f"{year_folder}/first_inning_runs_with_era_{year}.csv"
    if not os.path.exists(game_file):
        print(f"Skipping {year} — file not found: {game_file}")
        continue

    print(f"\nProcessing {year}")
    games = pd.read_csv(game_file)

    # --- Extract game dates ---
    games["date"] = games["game_id"].apply(lambda x: pd.to_datetime(x[3:11], format="%Y%m%d"))
    games = games.sort_values("date").reset_index(drop=True)

    # --- Track last locations ---
    games["home_last_location"] = "N/A"
    games["visiting_last_location"] = "N/A"
    last_location = {}

    for idx, row in games.iterrows():
        home = row["hometeam"]
        away = row["visteam"]

        if home in last_location:
            games.at[idx, "home_last_location"] = last_location[home]
        last_location[home] = home

        if away in last_location:
            games.at[idx, "visiting_last_location"] = last_location[away]
        last_location[away] = home

    # --- Compute travel distances ---
    home_travel, vis_travel = [], []

    for _, row in games.iterrows():
        home_last = row["home_last_location"]
        vis_last = row["visiting_last_location"]
        home_team = row["hometeam"]

        if home_last == "N/A" or vis_last == "N/A":
            home_travel.append(np.nan)
            vis_travel.append(np.nan)
            continue

        try:
            lat_home_last = float(stad_dict[home_last]["lat"])
            lon_home_last = float(stad_dict[home_last]["lon"])
            lat_home_team = float(stad_dict[home_team]["lat"])
            lon_home_team = float(stad_dict[home_team]["lon"])
            home_travel.append(haversine(lat_home_last, lon_home_last, lat_home_team, lon_home_team))

            lat_vis_last = float(stad_dict[vis_last]["lat"])
            lon_vis_last = float(stad_dict[vis_last]["lon"])
            vis_travel.append(haversine(lat_vis_last, lon_vis_last, lat_home_team, lon_home_team))
        except (KeyError, ValueError):
            home_travel.append(np.nan)
            vis_travel.append(np.nan)

    games["home_travel"] = pd.to_numeric(home_travel, errors="coerce") + 1e-6
    games["vis_travel"] = pd.to_numeric(vis_travel, errors="coerce") + 1e-6

    # --- Filter only valid travel rows ---
    data_filtered = games[
        ~((games['home_last_location'] == "N/A") | (games['visiting_last_location'] == "N/A"))
    ].copy()

    # --- Save per-game data with ERA, score, and travel ---
    per_game_output = f"{year_folder}/game_level_with_travel_{year}.csv"
    cols_to_save = [
        "game_id", "date", "hometeam", "visteam",
        "home_first_inning_runs", "visiting_first_inning_runs",
        "home_ERA", "vis_ERA",
        "home_travel", "vis_travel",
        "home_last_location", "visiting_last_location"
    ]
    data_filtered[cols_to_save].to_csv(per_game_output, index=False)
    print(f"Saved per-game CSV: {per_game_output}")

    # --- Prepare home and away rows for model ---
    home_df = data_filtered[['game_id', 'hometeam', 'home_first_inning_runs', 'vis_ERA', 'home_travel']].rename(
        columns={'hometeam': 'team_id',
                 'home_first_inning_runs': 'first_inning_runs',
                 'vis_ERA': 'opp_ERA',
                 'home_travel': 'travel'}
    )
    home_df['is_home'] = 1

    away_df = data_filtered[['game_id', 'visteam', 'visiting_first_inning_runs', 'home_ERA', 'vis_travel']].rename(
        columns={'visteam': 'team_id',
                 'visiting_first_inning_runs': 'first_inning_runs',
                 'home_ERA': 'opp_ERA',
                 'vis_travel': 'travel'}
    )
    away_df['is_home'] = 0

    data_nb = pd.concat([home_df, away_df], ignore_index=True)
    data_nb = data_nb.dropna(subset=['first_inning_runs', 'opp_ERA', 'travel'])

    # --- Standardize predictors ---
    for col in ['opp_ERA', 'travel']:
        data_nb[col] = (data_nb[col] - data_nb[col].mean()) / data_nb[col].std()

    # --- Fit Negative Binomial ---
    X = sm.add_constant(data_nb[['is_home', 'opp_ERA', 'travel']])
    y = data_nb['first_inning_runs']

    try:
        model = NegativeBinomial(y, X)
        result = model.fit(disp=False)
    except:
        # fallback to Poisson
        poisson_model = sm.GLM(y, X, family=sm.families.Poisson())
        result = poisson_model.fit(cov_type='HC3')

    intercept = result.params['const']
    beta_home = result.params['is_home']
    beta_opp_era = result.params['opp_ERA']
    beta_travel = result.params['travel']

    mean_era = data_nb['opp_ERA'].mean()
    mean_travel_home = data_nb.loc[data_nb['is_home'] == 1, 'travel'].mean()
    mean_travel_away = data_nb.loc[data_nb['is_home'] == 0, 'travel'].mean()

    mu_home = np.exp(intercept + beta_home + beta_opp_era * mean_era + beta_travel * mean_travel_home)
    mu_away = np.exp(intercept + beta_opp_era * mean_era + beta_travel * mean_travel_away)
    home_advantage = mu_home - mu_away

    row = {
        'year': year,
        'intercept': intercept,
        'beta_home': beta_home,
        'beta_opp_era': beta_opp_era,
        'beta_travel': beta_travel,
        'mean_opp_ERA': mean_era,
        'mean_travel_home': mean_travel_home,
        'mean_travel_away': mean_travel_away,
        'mu_away': mu_away,
        'mu_home': mu_home,
        'home_advantage': home_advantage
    }

    all_rows.append(row)
    print(f"Completed {year}")

# --- Save all yearly results ---
if all_rows:
    pd.DataFrame(all_rows).to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")
else:
    print("No results to save.")