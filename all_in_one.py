import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import NegativeBinomial
from math import radians, sin, cos, sqrt, atan2
import os
import warnings
import argparse
from pathlib import Path


# --- Suppress warnings ---
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=sm.tools.sm_exceptions.HessianInversionWarning)
warnings.filterwarnings("ignore", category=sm.tools.sm_exceptions.ConvergenceWarning)

# --- Year range ---
years = range(2014, 2025)

# --- CLI / root paths ---
ap = argparse.ArgumentParser()
ap.add_argument("--out_root", default="data/out", help="Path to data/out (contains year folders)")
args = ap.parse_args()

OUT_ROOT = Path(args.out_root)
DATA_DIR = OUT_ROOT.parent              # data/
stadium_file = DATA_DIR / "stadiums.csv"

output_file_game_travel = OUT_ROOT / "game_travel_distances.csv"
output_file_travel = OUT_ROOT / "first_inning_nb_results_with_travel.csv"
output_file_no_travel = OUT_ROOT / "first_inning_nb_results.csv"
output_file_travel_openers = OUT_ROOT / "first_inning_nb_results_with_travel_openers.csv"
output_file_no_travel_openers = OUT_ROOT / "first_inning_nb_results_openers.csv"
opener_master_csv = OUT_ROOT / "all_series_opener_game_ids.csv"

# --- Check stadium file (Path-safe) ---
if not stadium_file.exists():
    raise SystemExit(f"Stadium file not found: {stadium_file.resolve()}")

stadiums = pd.read_csv(stadium_file)

# force numeric lat/lon so haversine doesn't get strings
stadiums["lat"] = pd.to_numeric(stadiums["lat"], errors="coerce")
stadiums["lon"] = pd.to_numeric(stadiums["lon"], errors="coerce")
stadiums = stadiums.dropna(subset=["team_id", "lat", "lon"])

stad_dict = stadiums.set_index("team_id")[["lat", "lon"]].to_dict(orient="index")

# --- Haversine function ---
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # km
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)
    a = sin(dphi/2)**2 + cos(phi1)*cos(phi2)*sin(dlambda/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))

# --- Storage ---
all_game_rows = []
all_rows_travel = []
all_rows_no_travel = []
all_rows_travel_openers = []
all_rows_no_travel_openers = []
all_opener_game_ids = []

TRAVEL_THRESHOLD = 0.1  # km

# --- Process each year ---
for year in years:
    year_folder = OUT_ROOT / str(year)
    game_file = year_folder / f"first_inning_runs_summary_{year}.csv"
    if not game_file.exists():
        print(f"Skipping {year} — file not found: {game_file}")
        continue

    print(f"\nProcessing {year}")
    games = pd.read_csv(game_file)
    games["date"] = games["game_id"].apply(lambda x: pd.to_datetime(x[3:11], format="%Y%m%d"))
    games = games.sort_values("date").reset_index(drop=True)

    # --- Compute travel distances ---
    last_stadium = {}
    home_travel_list = []
    vis_travel_list = []
    first_game_series_flags = []

    for _, row in games.iterrows():
        home_team = row["hometeam"]
        vis_team = row["visteam"]
        home_stad = home_team

        # Home travel
        try:
            if home_team in last_stadium:
                lat_prev = float(stad_dict[last_stadium[home_team]]["lat"])
                lon_prev = float(stad_dict[last_stadium[home_team]]["lon"])
                lat_curr = float(stad_dict[home_stad]["lat"])
                lon_curr = float(stad_dict[home_stad]["lon"])
                home_travel = haversine(lat_prev, lon_prev, lat_curr, lon_curr)
            else:
                home_travel = np.nan
        except KeyError:
            home_travel = np.nan

        # Visiting travel
        try:
            if vis_team in last_stadium:
                lat_prev = float(stad_dict[last_stadium[vis_team]]["lat"])
                lon_prev = float(stad_dict[last_stadium[vis_team]]["lon"])
                lat_curr = float(stad_dict[home_stad]["lat"])
                lon_curr = float(stad_dict[home_stad]["lon"])
                vis_travel = haversine(lat_prev, lon_prev, lat_curr, lon_curr)
            else:
                vis_travel = np.nan
        except KeyError:
            vis_travel = np.nan

        # Series opener
        first_game_of_series = False
        if (home_travel is not np.nan and home_travel > TRAVEL_THRESHOLD) or \
           (vis_travel is not np.nan and vis_travel > TRAVEL_THRESHOLD):
            first_game_of_series = True

        all_game_rows.append({
            "game_id": row["game_id"],
            "date": row["date"],
            "home_team": home_team,
            "vis_team": vis_team,
            "home_score": row["home_first_inning_runs"],
            "vis_score": row["visiting_first_inning_runs"],
            "home_travel_km": home_travel,
            "vis_travel_km": vis_travel,
            "first_game_of_series": first_game_of_series
        })

        home_travel_list.append(home_travel)
        vis_travel_list.append(vis_travel)
        first_game_series_flags.append(first_game_of_series)

        last_stadium[home_team] = home_stad
        last_stadium[vis_team] = home_stad

    games["home_travel"] = home_travel_list
    games["vis_travel"] = vis_travel_list
    games["first_game_of_series"] = first_game_series_flags

    # --- Identify series openers more robustly ---
    games = games.sort_values(['hometeam','date']).reset_index(drop=True)
    games['is_series_opener'] = True
    last_home_opponent = {}
    for idx, row in games.iterrows():
        home, away = row['hometeam'], row['visteam']
        last_opponent = last_home_opponent.get(home, None)
        if last_opponent == away:
            games.at[idx, 'is_series_opener'] = False
        last_home_opponent[home] = away

    openers = games[games['is_series_opener']]
    all_opener_game_ids.extend(openers['game_id'].tolist())

    # --- Fit Negative Binomial Models ---
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

        # --- DROP ROWS WITH MISSING OR INF ---
        if include_travel:
            data = data.dropna(subset=['travel', 'first_inning_runs'])
            data = data[np.isfinite(data['travel'])]
            data['travel'] = (data['travel'] - data['travel'].mean()) / data['travel'].std()
            X = sm.add_constant(data[['is_home','travel']])
        else:
            data = data.dropna(subset=['first_inning_runs'])
            X = sm.add_constant(data[['is_home']])

        y = data['first_inning_runs']
        model = NegativeBinomial(y, X)
        result = model.fit(disp=False)

        intercept = result.params['const']
        beta_home = result.params['is_home']

        output_dict = {
            'intercept': intercept,
            'beta_home': beta_home
        }

        if include_travel:
            beta_travel = result.params['travel']
            mean_travel = data['travel'].mean()
            mu_home = np.exp(intercept + beta_home + beta_travel * mean_travel)
            mu_away = np.exp(intercept + beta_travel * mean_travel)
            output_dict.update({
                'beta_travel': beta_travel,
                'mean_travel': mean_travel,
                'mu_home': mu_home,
                'mu_away': mu_away,
                'home_advantage': mu_home - mu_away
            })
        else:
            mu_home = np.exp(intercept + beta_home)
            mu_away = np.exp(intercept)
            output_dict.update({
                'mu_home': mu_home,
                'mu_away': mu_away,
                'home_advantage': mu_home - mu_away
            })

        # Add the betas explicitly for CSV
        output_dict['beta_intercept'] = intercept
        output_dict['beta_home_only'] = beta_home
        if include_travel:
            output_dict['beta_travel_only'] = beta_travel

        return output_dict

    # --- Append results ---
    all_rows_no_travel.append({'year': year, **fit_nb(games, include_travel=False)})
    all_rows_travel.append({'year': year, **fit_nb(games, include_travel=True)})
    all_rows_no_travel_openers.append({'year': year, **fit_nb(openers, include_travel=False)})
    all_rows_travel_openers.append({'year': year, **fit_nb(openers, include_travel=True)})

# --- Save CSVs ---
pd.DataFrame(all_game_rows).to_csv(output_file_game_travel, index=False)
pd.DataFrame(all_rows_no_travel).to_csv(output_file_no_travel, index=False)
pd.DataFrame(all_rows_travel).to_csv(output_file_travel, index=False)
pd.DataFrame(all_rows_no_travel_openers).to_csv(output_file_no_travel_openers, index=False)
pd.DataFrame(all_rows_travel_openers).to_csv(output_file_travel_openers, index=False)
pd.DataFrame({'game_id': all_opener_game_ids}).to_csv(opener_master_csv, index=False)

print("All CSVs saved successfully.")
