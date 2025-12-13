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
output_file_travel = "/Users/kevinhe/orioles-project/data/out/first_inning_nb_results_with_travel.csv"
output_file_no_travel = "/Users/kevinhe/orioles-project/data/out/first_inning_nb_results.csv"
output_file_travel_openers = "/Users/kevinhe/orioles-project/data/out/first_inning_nb_results_with_travel_openers.csv"
output_file_no_travel_openers = "/Users/kevinhe/orioles-project/data/out/first_inning_nb_results_openers.csv"
opener_master_csv = "/Users/kevinhe/orioles-project/data/out/all_series_opener_game_ids.csv"

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

# --- Prepare storage ---
all_rows_travel = []
all_rows_no_travel = []
all_rows_travel_openers = []
all_rows_no_travel_openers = []
all_opener_game_ids = []

# --- Loop over years ---
for year in years:
    year_folder = f"/Users/kevinhe/orioles-project/data/out/{year}"
    game_file = f"{year_folder}/first_inning_runs_with_era_{year}.csv"
    if not os.path.exists(game_file):
        print(f"Skipping {year} — file not found: {game_file}")
        continue

    print(f"\nProcessing {year}")
    games = pd.read_csv(game_file)
    games["date"] = games["game_id"].apply(lambda x: pd.to_datetime(x[3:11], format="%Y%m%d"))
    games = games.sort_values("date").reset_index(drop=True)

    # --- Track last locations ---
    games["home_last_location"] = "N/A"
    games["visiting_last_location"] = "N/A"
    last_location = {}
    for idx, row in games.iterrows():
        home, away = row["hometeam"], row["visteam"]
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

    # --- Filter valid rows ---
    valid_games = games[~((games['home_last_location'] == "N/A") | (games['visiting_last_location'] == "N/A"))].copy()

    # --- Identify series openers ---
    valid_games = valid_games.sort_values(['hometeam','date']).reset_index(drop=True)
    valid_games["is_series_opener"] = True
    last_home_game = {}
    for idx, row in valid_games.iterrows():
        home, away = row['hometeam'], row['visteam']
        last_opponent = last_home_game.get(home, None)
        if last_opponent == away:
            valid_games.at[idx, 'is_series_opener'] = False
        last_home_game[home] = away

    valid_games['is_series_opener'] = valid_games['is_series_opener'].fillna(False)
    openers = valid_games[valid_games['is_series_opener']]

    # --- Save opener game_ids per year ---
    opener_csv_year = f"{year_folder}/series_opener_game_ids_{year}.csv"
    openers[['game_id']].to_csv(opener_csv_year, index=False)
    all_opener_game_ids.extend(openers['game_id'].tolist())

    # --- Save per-game CSVs ---
    cols_no_travel = ["game_id","date","hometeam","visteam","home_first_inning_runs","visiting_first_inning_runs"]
    valid_games[cols_no_travel].to_csv(f"{year_folder}/game_level_no_travel_{year}.csv", index=False)
    openers[cols_no_travel].to_csv(f"{year_folder}/game_level_no_travel_openers_{year}.csv", index=False)

    # --- Fit NB models ---
    def fit_nb(df, include_travel=False):
        home_df = df[['game_id','hometeam','home_first_inning_runs','home_travel']].rename(
            columns={'hometeam':'team_id','home_first_inning_runs':'first_inning_runs','home_travel':'travel'}
        )
        home_df['is_home'] = 1
        away_df = df[['game_id','visteam','visiting_first_inning_runs','vis_travel']].rename(
            columns={'visteam':'team_id','visiting_first_inning_runs':'first_inning_runs','vis_travel':'travel'}
        )
        away_df['is_home'] = 0
        data = pd.concat([home_df, away_df], ignore_index=True)

        if include_travel:
            data['travel'] = (data['travel'] - data['travel'].mean()) / data['travel'].std()
            X = sm.add_constant(data[['is_home','travel']])
        else:
            X = sm.add_constant(data[['is_home']])
        y = data['first_inning_runs']
        model = NegativeBinomial(y, X)
        result = model.fit(disp=False)

        intercept = result.params['const']
        beta_home = result.params['is_home']

        if include_travel:
            beta_travel = result.params['travel']
            mean_travel = data['travel'].mean()
            mu_home = np.exp(intercept + beta_home + beta_travel * mean_travel)
            mu_away = np.exp(intercept + beta_travel * mean_travel)
            return {'intercept': intercept, 'beta_home': beta_home, 'beta_travel': beta_travel,
                    'mean_travel': mean_travel, 'mu_away': mu_away, 'mu_home': mu_home,
                    'home_advantage': mu_home - mu_away}
        else:
            mu_home = np.exp(intercept + beta_home)
            mu_away = np.exp(intercept)
            return {'intercept': intercept, 'beta_home': beta_home, 'mu_away': mu_away,
                    'mu_home': mu_home, 'home_advantage': mu_home - mu_away}

    # --- All games NB ---
    all_rows_no_travel.append({'year': year, **fit_nb(valid_games, include_travel=False)})
    all_rows_travel.append({'year': year, **fit_nb(valid_games, include_travel=True)})

    # --- Openers NB ---
    all_rows_no_travel_openers.append({'year': year, **fit_nb(openers, include_travel=False)})
    all_rows_travel_openers.append({'year': year, **fit_nb(openers, include_travel=True)})

# --- Save all CSVs ---
pd.DataFrame(all_rows_no_travel).to_csv(output_file_no_travel, index=False)
pd.DataFrame(all_rows_travel).to_csv(output_file_travel, index=False)
pd.DataFrame(all_rows_no_travel_openers).to_csv(output_file_no_travel_openers, index=False)
pd.DataFrame(all_rows_travel_openers).to_csv(output_file_travel_openers, index=False)

# --- Save master opener game_ids CSV ---
pd.DataFrame({'game_id': all_opener_game_ids}).to_csv(opener_master_csv, index=False)

print("All four NB CSVs and series opener game_id CSVs saved successfully.")
