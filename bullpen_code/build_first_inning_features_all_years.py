# build_first_inning_features_all_years.py
# Outputs:
#   yearly_features/features_<YEAR>.csv
#   first_inning_features_all.csv
#
# For each game (multi-season):
#   • Top 1st elapsed seconds from Statcast timestamps (sv_id) if available
#   • Visitor batters faced, visitor pitches, base-state counts (empty vs on)
#   • home_starter_id (pitcher of first Top-1st pitch)
#   • visiting_starter_id (pitcher of first Bottom-1st pitch)
#   • home_runs_bottom1 (runs scored by home team in Bottom-1st)

import os
import re
import warnings
from datetime import date, timezone
from typing import Optional, Iterable, Tuple

import numpy as np
import pandas as pd
from pybaseball import statcast

# (strongly recommended) enable caching so partial downloads are reused
try:
    from pybaseball import cache
    cache.enable()
except Exception:
    pass

warnings.filterwarnings("ignore", message=".*errors='ignore' is deprecated.*")

SV_RE = re.compile(r"(\d{6})_(\d{6})")  # YYMMDD_HHMMSS


def season_window(y: int) -> Tuple[str, str]:
    """
    Return (start_dt, end_dt) strings for a MLB season.
    Past seasons: Mar 1 → Nov 30
    Current season: Mar 1 → today (but not past Nov 30).
    """
    today = date.today()
    start = date(y, 3, 1)
    end = date(y, 11, 30)
    if y == today.year:
        end = min(today, end)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def parse_sv_id_to_utc(sv_id: str, game_date_str: str) -> Optional[pd.Timestamp]:
    """Parse Statcast sv_id 'YYMMDD_HHMMSS' to tz-aware Timestamp (UTC)."""
    if not isinstance(sv_id, str):
        return None
    m = SV_RE.fullmatch(sv_id)
    if not m:
        return None
    yymmdd, hhmmss = m.groups()
    try:
        gd = pd.to_datetime(game_date_str).date()
        century = gd.year // 100 * 100
        year = century + int(yymmdd[:2])
        month = int(yymmdd[2:4])
        day = int(yymmdd[4:6])
        hour = int(hhmmss[:2]); minute = int(hhmmss[2:4]); second = int(hhmmss[4:6])
        ts = pd.Timestamp(year, month, day, hour, minute, second, tz="US/Eastern")
        return ts.tz_convert(timezone.utc)
    except Exception:
        return None


def inning_elapsed_seconds(df_inning: pd.DataFrame) -> Optional[float]:
    """Elapsed seconds in a half-inning using sv_id-derived timestamps."""
    times = df_inning["sv_id_time"].dropna().sort_values()
    if times.empty:
        return None
    return float((times.iloc[-1] - times.iloc[0]).total_seconds())


def batters_faced(df_half: pd.DataFrame) -> int:
    """Robust BF count with fallbacks."""
    if "ab_number" in df_half.columns:
        return int(pd.Series(df_half["ab_number"]).nunique())
    if "pitch_number" in df_half.columns:
        return int((pd.to_numeric(df_half["pitch_number"], errors="coerce") == 1).sum())
    return int((df_half["batter"].shift() != df_half["batter"]).sum())


def normalize_columns(sc: pd.DataFrame) -> pd.DataFrame:
    """Standardize common columns and add helpers (bases_empty, sv_id_time)."""
    sc = sc.copy()

    # Normalize Top/Bot labels
    if "inning_topbot" in sc.columns:
        sc["inning_topbot"] = sc["inning_topbot"].astype(str).str.title()
        sc["inning_topbot"] = sc["inning_topbot"].replace({"Bot": "Bottom"})
    else:
        sc["inning_topbot"] = pd.NA

    # Ensure base-state flags exist; Savant uses NaN when base is empty
    for col in ["on_1b", "on_2b", "on_3b"]:
        if col not in sc.columns:
            sc[col] = pd.NA

    sc["bases_empty"] = sc["on_1b"].isna() & sc["on_2b"].isna() & sc["on_3b"].isna()

    # Timestamp from sv_id
    if "sv_id" in sc.columns and "game_date" in sc.columns:
        sc["sv_id_time"] = sc.apply(
            lambda r: parse_sv_id_to_utc(r["sv_id"], r["game_date"]), axis=1
        )
    else:
        sc["sv_id_time"] = pd.NaT

    return sc


def build_features_from_raw(sc: pd.DataFrame, season: int) -> pd.DataFrame:
    """Extract per-game first-inning features for a season."""
    required = {
        "game_pk", "game_date", "inning", "inning_topbot", "pitch_number",
        "home_score", "away_score", "home_team", "away_team",
        "pitcher", "batter", "bases_empty", "sv_id_time",
    }
    missing = required - set(sc.columns)
    if missing:
        raise SystemExit(f"[{season}] Missing columns: {missing}")

    rows = []
    for game_pk, g in sc.groupby("game_pk", sort=False):
        # Use true clock order when available
        if g["sv_id_time"].notna().any():
            g = g.sort_values("sv_id_time")
        else:
            g = g.sort_values(["inning", "inning_topbot", "pitch_number"])

        game_date = g["game_date"].iloc[0]
        home_team = g["home_team"].iloc[0]
        away_team = g["away_team"].iloc[0]

        # ---- Top of 1st (visitors bat)
        top1 = g[(g["inning"] == 1) & (g["inning_topbot"] == "Top")]
        if top1.empty:
            # weird/suspended game; skip
            continue

        top1_elapsed = inning_elapsed_seconds(top1)  # may be None
        visitor_bf = batters_faced(top1)
        visitor_pitches = int(len(top1))
        n_empty = int(top1["bases_empty"].sum())
        n_on = int(visitor_pitches - n_empty)

        # home starter is the pitcher on the first pitch of the game
        top1_sorted = top1.sort_values(["sv_id_time", "pitch_number"])
        home_starter_id = int(pd.to_numeric(top1_sorted["pitcher"].iloc[0], errors="coerce"))

        # ---- Bottom of 1st (home bats)
        bot1 = g[(g["inning"] == 1) & (g["inning_topbot"] == "Bottom")]
        if bot1.empty:
            visiting_starter_id = None
            home_runs_b1 = 0
        else:
            # make sure first pitch is first row
            if bot1["sv_id_time"].notna().any():
                bot1 = bot1.sort_values("sv_id_time")
            else:
                bot1 = bot1.sort_values(["inning", "inning_topbot", "pitch_number"])
            visiting_starter_id = int(pd.to_numeric(bot1["pitcher"].iloc[0], errors="coerce"))

            # runs scored by home in bottom 1st
            start_home = pd.to_numeric(bot1["home_score"].iloc[0], errors="coerce")
            end_home   = pd.to_numeric(bot1["home_score"].iloc[-1], errors="coerce")
            delta = (end_home - start_home) if pd.notna(start_home) and pd.notna(end_home) else 0
            home_runs_b1 = max(0, int(delta))

        rows.append({
            "season": int(season),
            "game_pk": int(game_pk),
            "game_date": game_date,
            "home_team": home_team,
            "away_team": away_team,
            "top1_elapsed_seconds": top1_elapsed,      # may be NaN
            "visitor_batters_top1": int(visitor_bf),
            "visitor_pitches_top1": int(visitor_pitches),
            "n_empty_top1": n_empty,
            "n_on_top1": n_on,
            "home_starter_id": home_starter_id,        # needed for tempo merges
            "visiting_starter_id": visiting_starter_id,
            "home_runs_bottom1": int(home_runs_b1),
        })

    return pd.DataFrame(rows)


def run(years: Iterable[int]) -> pd.DataFrame:
    os.makedirs("yearly_features", exist_ok=True)
    all_parts = []

    for y in years:
        start, end = season_window(y)
        print(f"[{y}] Downloading Statcast {start} → {end} ...")
        try:
            sc = statcast(start_dt=start, end_dt=end)
        except Exception as e:
            print(f"[{y}] ERROR downloading: {e}")
            continue

        print(f"[{y}] Raw rows: {len(sc)}")
        sc = normalize_columns(sc)

        feats = build_features_from_raw(sc, y)
        print(f"[{y}] Games extracted: {len(feats)}")

        out_y = f"yearly_features/features_{y}.csv"
        feats.to_csv(out_y, index=False)
        print(f"[{y}] Wrote {out_y}")

        all_parts.append(feats)

    if not all_parts:
        raise SystemExit("No seasons processed — nothing to write.")

    full = pd.concat(all_parts, ignore_index=True)
    full.to_csv("first_inning_features_all.csv", index=False)
    print(f"[ALL] Combined games: {len(full)}")
    print("[ALL] Wrote first_inning_features_all.csv")
    return full


if __name__ == "__main__":
    # Build 2015 → current year (inclusive)
    YEARS = list(range(2015, date.today().year + 1))
    _ = run(YEARS)
