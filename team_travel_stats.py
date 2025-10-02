import csv
import statistics

# Input and output paths
input_file = "/Users/kevinhe/orioles-project/team_travel.csv"
output_file = "/Users/kevinhe/orioles-project/team_travel_stats_with_teams.csv"

# Read the file
with open(input_file, "r") as f:
    lines = [line.strip() for line in f if line.strip()]

year_data = {}
current_year = None

# Parse data
for line in lines:
    if line.isdigit():  # Year row
        current_year = int(line)
        year_data[current_year] = []
    else:
        try:
            team, miles = line.split(",")
            year_data[current_year].append((team, int(miles)))
        except ValueError:
            continue  # Skip malformed lines

# Write statistics to new CSV
with open(output_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Year", "Low_Team", "Low", "Median_Team", "Median", "High_Team", "High", "Average"])
    
    for year in sorted(year_data.keys()):
        data = year_data[year]
        # Sort by miles
        sorted_data = sorted(data, key=lambda x: x[1])
        
        low_team, low = sorted_data[0]
        high_team, high = sorted_data[-1]
        
        # Median calculation
        miles_only = [m for _, m in sorted_data]
        median_value = int(statistics.median(miles_only))
        # Find the team closest to median
        median_team = min(sorted_data, key=lambda x: abs(x[1] - median_value))[0]
        
        average = round(sum(miles_only) / len(miles_only), 2)
        
        writer.writerow([year, low_team, low, median_team, median_value, high_team, high, average])

print(f"Statistics with team names saved to {output_file}")