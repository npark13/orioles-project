# orioles-project

This project investigates the phenomenon of the first-inning home advantage in Major League Baseball in which the disproportionate share of home-field advantage occurs in the first inning of baseball games. Previous research by David W. Smith suggests higher scoring in the first inning is strongly correlated with multiple factors including number of visiting batters, men on base in the top of the first inning, and pitches in the top of the first inning. However, such research is relatively dated, and there is not a single factor that the phenomenon can be attributed to. We aim to investigate if the first-inning home advantage still exists in baseball, and explore the possible causes of such an advantage.

Team Members: Kevin He and Nancy Park

1) DATA SCRAPER
INPUTS:
- data/**
COMMAND: python scraper.py parse-events-recursive data --out data/out
OUTPUTS:
- data/out/<year>/plays.csv 
- data/out/<year>/games.csv

INPUTS:
- data/**
COMMAND: python scraper.py parse-rosters-recursive data --out data/out
OUTPUTS:
- data/out/<year>/rosters.csv

INPUTS:
- data/out/<year>/rosters.csv
- data/out/<year>/games.csv
- data/out/<year>/plays.csv
COMMAND: python scraper.py join-names --out_root data/out
updates or created "named" version of our per-year outputs under data/out

INPUTS:
- data/out/<year>/plays.csv (and/or named version)
- data/out/<year>/games.csv
COMMAND: python scraper.py analyze --all_years --out_root data/out
OUTPUTS:
- data/out/first_inning_summary.csv
- data/out/inning_summary.csv

INPUTS:
- data/out/first_inning_summary.csv
COMMAND: python scraper.py home-visit-corr data/out --out data/out
OUTPUTS:
- data/out/visitor_vs_home_first_inning.csv

INPUTS:
- data/2010sbox/
- data/2020sbox/
COMMAND: python parse_box_scores_to_results.py
OUTPUTS:
- results_by_game.csv

2) RETROSHEET GRAPHS
INPUTS:
- data/out/inning_summary.csv
- data/out/first_inning_summary.csv
- data/out/visitor_vs_home_first_inning.csv
COMMAND: python visualize_first_inning.py
OUTPUTS:
- differential_runs_per_inning.png
- first_inning_diff_by_decade.png
- home_minus_vis_plot.png
- home_versus_visiting_inning.png
- runs_per_inning.png

3) BULLPEN COOLDOWN
INPUTS:
- csv_files/first_inning_features_with_tempo_all.csv
COMMAND: python bullpen_code/first_inning_bins.py
OUTPUTS:
- csv_files/binned_first_inning_stats.csv
- all_visuals/bullpen_cooldown_visuals/binned_first_inning_stats.png

INPUTS:
- csv_files/binned_first_inning_stats.csv
COMMAND: python bullpen_code/visitor_escape_bins.py
OUTPUTS:
- csv_files/binned_first_inning_stats_with_escape.csv 
- all_visuals/bullpen_cooldown_visuals/visitor_escape_bins.png


INPUTS:
pybaseball.statcast
COMMAND: python bullpen_code/build_first_inning_features_all_years.py
OUTPUTS:
- csv_files/first_inning_features_all.csv

4) TIMEZONE CHANGE
INPUTS:
- data/out/<year>/games.csv (per-year games)
- results_by_game.csv
COMMAND: python travel_code/winning_vs_travel_2013_2024.py
OUTPUTS:
- winning_vs_tzchange_2013_2024.csv
- winning_vs_tzchange.png

5) TRAVEL
INPUTS:
- data/out/first_inning_nb_results.csv
- data/out/first_inning_nb_results_with_travel.csv
- data/out/first_inning_nb_results_openers.csv
- data/out/first_inning_nb_results_with_travel_openers.csv
COMMAND: python visualize_three.py
OUTPUTS:
- data/out/home_advantage_travel_diff.csv
- data/out/home_advantage_travel_diff_openers.csv
- home_advantage_travel_all_games.png 
- home_advantage_travel_openers.png

6) UMPIRE BIAS
INPUTS: 
Statcast pitch-level data
COMMAND: python umpire_code/statcast_scrape.py 2018-03-01 2024-11-30
OUTPUTS: 
- statcast_pitches_raw_2018_2024.parquet

INPUTS:
- statcast_pitches_raw_2018_2024.parquet
COMMAND: python umpire_code/build_strike_features.py
OUTPUTS:
- pitches_features.parquet

INPUTS:
- pitches_features.parquet
COMMAND: python umpire_code/analyze_umpire_bias.py
OUTPUTS:
- park_quadrant_fringe_summary.csv
- team_home_edge_delta.csv
- fringe_home_edge_delta.png

7) MODELS 
COMMAND: python modeling/predict_first_inning_weighted.py \
  --start 2013 --end 2024 \
  --games-root data/out \
  --per-inning data/out/rolling_avg/game_level_with_rolling_avg_weather_2014_2024_all_clean.csv \
  --target yrfi \
  --outdir .
OUTPUTS:
- modeling_dataset_weighted.csv
- model_metrics_weighted.txt
- roc_logit.png
- roc_boost.png
- calib_logit.png
- calib_boost.png
- feature_importances_boost.csv
- predictions_boost.csv

8) TEST CASES
INPUTS:
- modeling/game_ids.csv
- modeling_dataset_weighted.csv
- logit_pipeline.joblib
- boost_pipeline.joblib
COMMAND: python modeling/predict_first_inning.py \
  --game-ids modeling/game_ids.csv \
  --model-dataset modeling_dataset_weighted.csv \
  --logit-pipeline logit_pipeline.joblib \
  --boost-pipeline boost_pipeline.joblib \
  --out modeling/first_inning_preds.csv
OUTPUTS:
- modeling/first_inning_preds.csv