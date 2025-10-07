import re
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
from pybaseball import statcast

# ----------------------------
# helpers
# ----------------------------
SV_RE = re.compile(r"(\d{6})_(\d{6})")  # YYMMDD_HHMMSS

def parse_sv_id_to_utc(sv_id: str, game_date_str: str) -> Optional[pd.Timestamp]:
    """
    Parse Statcast sv_id 'YYMMDD_HHMMSS' into a timezone-aware UTC-ish Timestamp.
    We anchor to the game_date if needed. Times in sv_id are ET on Savant exports historically;
    for relative differences within the *same game/inning*, the exact tz won't matter.
    """
    if not isinstance(sv_id, str):
        return None
    m = SV_RE.fullmatch(sv_id)
    if not m:
        return None
    yymmdd, hhmmss = m.groups()
    # Infer century from game_date
    try:
        gd = pd.to_datetime(game_date_str).date()
        century = gd.year // 100 * 100
        year = century + int(yymmdd[:2])
        month = int(yymmdd[2:4])
        day = int(yymmdd[4:6])
        hour = int(hhmmss[:2])
        minute = int(hhmmss[2:4])
        second = int(hhmmss[4:6])
        ts = pd.Timestamp(year, month, day, hour, minute, second, tz="US/Eastern")
        # Use UTC for portability; deltas will be identical
        return ts.tz_convert(timezone.utc)
    except Exception:
        return None

def inning_elapsed_seconds(df_inning: pd.DataFrame) -> Optional[float]:
    """Compute elapsed seconds in an inning using sv_id timestamps; fall back to None if missing."""
    times = df_inning["sv_id_time"].dropna().sort_values()
    if times.empty:
        return None
    return (times.iloc[-1] - times.iloc[0]).total_seconds()

def compute_first_inning_features(pitches: pd.DataFrame) -> pd.DataFrame:
    """
    For each game_pk:
      - top1_elapsed_seconds
      - visitor_batters_top1
      - visitor_pitches_top1
      - home_runs_bottom1
    """
    # minimal columns
    cols_needed = {
        "game_pk", "game_date", "inning", "inning_topbot",
        "ab_id", "pitch_number", "sv_id", "home_score", "away_score",
        "home_team", "away_team"
    }
    missing = cols_needed - set(pitches.columns)
    if missing:
        raise SystemExit(f"Missing required Statcast columns: {missing}")

    # sv_id -> timestamp
    pitches = pitches.copy()
    pitches["sv_id_time"] = pitches.apply(
        lambda r: parse_sv_id_to_utc(r["sv_id"], r["game_date"]), axis=1
    )

    out_rows = []
    for game_pk, g in pitches.groupby("game_pk", sort=False):
        g = g.sort_values(["inning", "inning_topbot", "ab_id", "pitch_number"])
        game_date = g["game_date"].iloc[0]
        home_team = g["home_team"].iloc[0]
        away_team = g["away_team"].iloc[0]

        # ---- Top of 1st (visitors bat) ----
        top1 = g[(g["inning"] == 1) & (g["inning_topbot"] == "Top")]
        if top1.empty:
            # skip suspended/odd games
            continue
        top1_elapsed = inning_elapsed_seconds(top1)
        visitor_bf = top1["ab_id"].nunique()
        visitor_pitches = len(top1)

        # ---- Bottom of 1st (home bats) ----
        bot1 = g[(g["inning"] == 1) & (g["inning_topbot"] == "Bottom")]
        if bot1.empty:
            # no bottom first (walk-off in top? suspended? AL extra weirdness) -> set 0
            home_runs_b1 = 0
        else:
            start_home = bot1["home_score"].iloc[0]
            end_home = bot1["home_score"].iloc[-1]
            home_runs_b1 = int(end_home - start_home)

        out_rows.append({
            "game_pk": game_pk,
            "game_date": game_date,
            "home_team": home_team,
            "away_team": away_team,
            "top1_elapsed_seconds": top1_elapsed,
            "visitor_batters_top1": int(visitor_bf),
            "visitor_pitches_top1": int(visitor_pitches),
            "home_runs_bottom1": home_runs_b1,
        })

    return pd.DataFrame(out_rows)

# ----------------------------
# main: pull a date range and compute features
# ----------------------------
if __name__ == "__main__":
    # Choose a manageable window first; widen later (Statcast queries can be big).
    START = "2023-04-01"
    END   = "2023-04-30"

    print(f"Downloading Statcast pitches {START} → {END} ...")
    sc = statcast(START, END)   # returns pitch-by-pitch DataFrame

    # Some pybaseball versions use 'inning_topbot' values 'Top'/'Bot' or 'Top'/'Bottom'
    # Normalize if needed:
    if "inning_topbot" in sc.columns:
        sc["inning_topbot"] = sc["inning_topbot"].replace({"Bot": "Bottom"})

    features = compute_first_inning_features(sc)
    print(f"[OK] Built {len(features)} game rows")

    OUT = f"statcast_first_inning_features_{START}_to_{END}.csv"
    features.to_csv(OUT, index=False)
    print(f"[OK] Wrote {OUT}")

    # quick sanity peek
    print(features.head(10))
