#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
import pandas as pd


# ---------------- helpers ----------------

def american_odds_profit(odds: float, stake: float) -> float:
    odds = float(odds)
    if odds > 0:
        return stake * (odds / 100.0)
    else:
        return stake * (100.0 / abs(odds))


def opposite_odds(odds: float) -> float:
    odds = float(odds)
    if odds > 0:
        return -odds
    else:
        return abs(odds)


def nrfi_pick(prob: float, thr: float = 0.5) -> int:
    # 1 = NRFI, 0 = YRFI
    return 1 if float(prob) < thr else 0


def yes_no(x: int) -> str:
    return "yes" if int(x) == 1 else "no"


# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser(description="Evaluate NRFI/YRFI bets and payouts.")
    ap.add_argument("--game-ids", required=True)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--out-picks", default="nrfi_picks_with_result.csv")
    ap.add_argument("--out-payouts", default="nrfi_payouts.csv")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--stake", type=float, default=10.0)
    args = ap.parse_args()

    try:
        # ---------- load ----------
        gids = pd.read_csv(args.game_ids)
        preds = pd.read_csv(args.preds)
        res = pd.read_csv(args.results)
        res.columns = [c.lower() for c in res.columns]

        # ---------- merge ----------
        df = gids.merge(preds, on="game_id", how="left")

        # ---------- model picks ----------
        df["logreg_pick"] = df["logit_prob_first_inning_score"].apply(
            lambda p: nrfi_pick(p, args.thr)
        )
        df["xg_boost_pick"] = df["boost_prob_first_inning_score"].apply(
            lambda p: nrfi_pick(p, args.thr)
        )

        # ---------- true result ----------
        res_small = res[[
            "game_id",
            "home_first_inning_runs",
            "visiting_first_inning_runs"
        ]].copy()

        res_small["result"] = (
            (res_small["home_first_inning_runs"] == 0) &
            (res_small["visiting_first_inning_runs"] == 0)
        ).astype(int)

        df = df.merge(res_small[["game_id", "result"]], on="game_id", how="left")

        # ---------- picks CSV ----------
        picks_out = df[[
            "game_id", "logreg_pick", "xg_boost_pick", "result"
        ]].copy()

        picks_out["logreg_pick"] = picks_out["logreg_pick"].apply(yes_no)
        picks_out["xg_boost_pick"] = picks_out["xg_boost_pick"].apply(yes_no)
        picks_out["result"] = picks_out["result"].apply(yes_no)

        picks_out.to_csv(args.out_picks, index=False)

        # ---------- payouts ----------
        pay = df[[
            "game_id", "nrfi_odds", "mgm_pred",
            "logreg_pick", "xg_boost_pick", "result"
        ]].copy()

        pay["bet"] = args.stake

        pay["logreg_pick"] = pay["logreg_pick"].apply(yes_no)
        pay["xg_boost_pick"] = pay["xg_boost_pick"].apply(yes_no)
        pay["mgm_pick"] = pay["mgm_pred"].apply(yes_no)
        pay["result"] = pay["result"].apply(yes_no)

        def payout(pick: str, result: str, odds: float) -> float:
            if pick == "yes":   # NRFI
                win = (result == "yes")
                use_odds = odds
            else:               # YRFI
                win = (result == "no")
                use_odds = opposite_odds(odds)

            if win:
                return american_odds_profit(use_odds, args.stake)
            return -args.stake

        pay["logreg_payout"] = pay.apply(
            lambda r: payout(r["logreg_pick"], r["result"], r["nrfi_odds"]), axis=1
        )
        pay["xg_boost_payout"] = pay.apply(
            lambda r: payout(r["xg_boost_pick"], r["result"], r["nrfi_odds"]), axis=1
        )
        pay["mgm_payout"] = pay.apply(
            lambda r: payout(r["mgm_pick"], r["result"], r["nrfi_odds"]), axis=1
        )

        # cap payouts to 2 decimal places
        pay["logreg_payout"] = pay["logreg_payout"].round(2)
        pay["xg_boost_payout"] = pay["xg_boost_payout"].round(2)
        pay["mgm_payout"] = pay["mgm_payout"].round(2)


        final = pay[[
            "game_id", "nrfi_odds", "bet",
            "logreg_pick", "logreg_payout",
            "xg_boost_pick", "xg_boost_payout",
            "mgm_pick", "mgm_payout",
            "result"
        ]]

        final.to_csv(args.out_payouts, index=False)

        # ---------- accuracy ----------
        eval_df = df.dropna(subset=["result"]).copy()
        logreg_acc = (eval_df["logreg_pick"].astype(int) == eval_df["result"].astype(int)).mean()
        xgb_acc = (eval_df["xg_boost_pick"].astype(int) == eval_df["result"].astype(int)).mean()
        mgm_acc = (eval_df["mgm_pred"].astype(int) == eval_df["result"].astype(int)).mean()

        print(f"\nAccuracy (NRFI=yes, YRFI=no):")
        print(f"  LogReg:  {logreg_acc*100:.2f}%")
        print(f"  XGBoost: {xgb_acc*100:.2f}%")
        print(f"  MGM:     {mgm_acc*100:.2f}%")


        # ---------- totals ----------
        print("\nTotal profit ($10 flat bets):")
        print(f"  LogReg:  ${final['logreg_payout'].sum():.2f}")
        print(f"  XGBoost: ${final['xg_boost_payout'].sum():.2f}")
        print(f"  MGM:     ${final['mgm_payout'].sum():.2f}")

        print(f"\n[OK] wrote {args.out_picks}")
        print(f"[OK] wrote {args.out_payouts}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
