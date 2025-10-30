import pandas as pd

# Path template
path_template = "/Users/kevinhe/orioles-project/data/out/team_home_field_by_year/first_inning_nb_results_{}_{}.csv"

# Years to process
years = range(2018, 2025)

for year in years:
    # Load home and visitor CSVs
    home_file = path_template.format("home", year)
    vis_file = path_template.format("vis", year)
    
    df_home = pd.read_csv(home_file)
    df_vis = pd.read_csv(vis_file)
    
    # Ensure both have same order of teams
    df_home = df_home.sort_values('team').reset_index(drop=True)
    df_vis = df_vis.sort_values('team').reset_index(drop=True)
    
    # Combine home_advantage
    combined_adv = df_home['home_advantage'] + (-df_vis['home_advantage'])
    
    # Add as a new column to home dataframe
    df_home['combined_home_advantage'] = combined_adv
    
    # Output to CSV
    output_file = f"/Users/kevinhe/orioles-project/data/out/team_home_field_by_year/first_inning_nb_results_{year}.csv"
    df_home.to_csv(output_file, index=False)
    
    print(f"Processed year {year} -> {output_file}")