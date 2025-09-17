# orioles-project

This project investigates the phenomenon of the first-inning home advantage in Major League Baseball in which the disproportionate share of home-field advantage occurs in the first inning of baseball games. Previous research by David W. Smith suggests higher scoring in the first inning is strongly correlated with multiple factors including number of visiting batters, men on base in the top of the first inning, and pitches in the top of the first inning. However, such research is relatively dated, and there is not a single factor that the phenomenon can be attributed to. We aim to investigate if the first-inning home advantage still exists in baseball, and explore the possible causes of such an advantage.

Team Members: Kevin He and Nancy Park

python scraper.py parse-events data/2020seve
python scraper.py parse-rosters data/2020seve --teamfile data/2020seve/TEAM2024
python scraper.py join-names
python scraper.py analyze
