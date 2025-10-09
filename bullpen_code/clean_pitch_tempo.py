import pandas as pd
from pathlib import Path

IN = Path("pitch_tempo.csv")
OUT = Path("pitch_tempo_clean.csv")

df = pd.read_csv(IN)
df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

# map columns based on the schema you showed
pid_col   = "entity_id"         if "entity_id" in df.columns else None
name_col  = "entity_name"       if "entity_name" in df.columns else None
be_col    = "median_seconds_empty"            if "median_seconds_empty" in df.columns else None
ro_col    = "median_seconds_empty.1"          if "median_seconds_empty.1" in df.columns else None

if pid_col is None or name_col is None:
    raise SystemExit(f"Could not find pitcher id/name columns in: {list(df.columns)}")

out = pd.DataFrame({
    "mlbam_pitcher_id": df[pid_col],
    "player_name": df[name_col],
})

if be_col:
    out["tempo_be_sec"] = pd.to_numeric(df[be_col], errors="coerce")
if ro_col:
    out["tempo_ro_sec"] = pd.to_numeric(df[ro_col], errors="coerce")

out = out.drop_duplicates(subset=["mlbam_pitcher_id"]).reset_index(drop=True)
out.to_csv(OUT, index=False)

print(f"[OK] wrote {OUT} with {len(out):,} rows")
print(out.head())
print("[INFO] This export has no 'season' column; tempos are aggregated across years.")
