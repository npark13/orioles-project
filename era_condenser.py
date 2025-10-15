import subprocess
import sys
import os

# === CONFIG ===
project_dir = "/Users/kevinhe/orioles-project"
START_YEAR = 2014
END_YEAR = 2024

# Paths to scripts
scripts = {
    "first_inning_runs": os.path.join(project_dir, "first_inning_runs.py"),
    "condense_runs": os.path.join(project_dir, "condense_runs.py"),
    "era_accounted_nb": os.path.join(project_dir, "era_accounted_nb.py"),
}

def run_script(script_path, args=None):
    """Run a Python script and stream its output live."""
    cmd = [sys.executable, script_path]
    if args:
        cmd.extend(map(str, args))

    print(f"\n=== Running {os.path.basename(script_path)} {' '.join(map(str, args or []))} ===\n")
    subprocess.run(cmd, check=True)
    print(f"\nFinished {os.path.basename(script_path)} {' '.join(map(str, args or []))}\n")

# === MAIN RUN LOOP ===
print(f"\nStarting pipeline for {START_YEAR}–{END_YEAR}\n")

for year in range(START_YEAR, END_YEAR + 1):
    print(f"\n================ YEAR {year} ================\n")

    # 1️Run first_inning_runs.py
    if os.path.exists(scripts["first_inning_runs"]):
        run_script(scripts["first_inning_runs"], [year])
    else:
        print(f"Skipping {scripts['first_inning_runs']} — file not found.")

    # 2️Run condense_runs.py
    if os.path.exists(scripts["condense_runs"]):
        run_script(scripts["condense_runs"], [year])
    else:
        print(f"Skipping {scripts['condense_runs']} — file not found.")

    # 3️Run era_accounted_nb.py
    if os.path.exists(scripts["era_accounted_nb"]):
        run_script(scripts["era_accounted_nb"], [year])
    else:
        print(f"Skipping {scripts['era_accounted_nb']} — file not found.")

print(f"\nAll scripts completed for {START_YEAR}–{END_YEAR}!\n")