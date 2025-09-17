import subprocess
import os

output_file = "output.log"

with open(output_file, "w", encoding="utf-8") as f:
    for year in range(2020, 1900, -10):  # 2020, 2010, ..., 1910
        event = f"{year}seve"
        teamfile = f"data/{event}/TEAM{year}"
        f.write(f"\n===== Processing {event} ({year} decade) =====\n")

        # parse-events
        subprocess.run(
            ["python3", "scraper.py", "parse-events", f"data/{event}"],
            stdout=f, stderr=f
        )

        # parse-rosters with correct TEAM file
        if os.path.exists(teamfile):
            subprocess.run(
                ["python3", "scraper.py", "parse-rosters", f"data/{event}", "--teamfile", teamfile],
                stdout=f, stderr=f
            )
        else:
            f.write(f"⚠️ Skipping rosters for {event}, missing {teamfile}\n")

        # join-names and analyze
        subprocess.run(
            ["python3", "scraper.py", "join-names"],
            stdout=f, stderr=f
        )
        subprocess.run(
            ["python3", "scraper.py", "analyze"],
            stdout=f, stderr=f
        )

print(f"All output written to {output_file}")