from __future__ import annotations
import argparse
from pathlib import Path

import parse_events
import parse_ros
import join_names
import analyze_home_ad

def main():
    parser = argparse.ArgumentParser(description="Retrosheet Scraper & Analyzer")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ev = subparsers.add_parser("parse-events", help="Parse EVN/EVA in a single folder")
    ev.add_argument("folder")
    ev.add_argument("--out", default="data/out")

    ros = subparsers.add_parser("parse-rosters", help="Parse ROS + TEAM in a single folder")
    ros.add_argument("folder")
    ros.add_argument("--teamfile", required=True)
    ros.add_argument("--out", default="data/out/rosters.csv")

    evr = subparsers.add_parser("parse-events-recursive", help="Walk all subfolders for EV?")
    evr.add_argument("root")
    evr.add_argument("--out", default="data/out")

    rosrec = subparsers.add_parser("parse-rosters-recursive", help="Walk all subfolders for ROS/TEAM")
    rosrec.add_argument("root")
    rosrec.add_argument("--out", default="data/out")

    jn = subparsers.add_parser("join-names", help="Join names onto per-year outputs")
    jn.add_argument("--out_root", default="data/out")

    an = subparsers.add_parser("analyze", help="Analyze 1st-inning home advantage")
    group = an.add_mutually_exclusive_group(required=True)
    group.add_argument("--plays_csv", help="Single plays.csv")
    group.add_argument("--all_years", action="store_true")
    an.add_argument("--out_root", default="data/out")

    args = parser.parse_args()

    if args.command == "parse-events":
        parse_events.main(args.folder, args.out)

    elif args.command == "parse-rosters":
        parse_ros.main(args.folder, args.teamfile, args.out)

    elif args.command == "parse-events-recursive":
        parse_events.parse_events_recursive(args.root, args.out)

    elif args.command == "parse-rosters-recursive":
        parse_ros.parse_rosters_recursive(args.root, args.out)

    elif args.command == "join-names":
        join_names.main(args.out_root)

    elif args.command == "analyze":
        if args.all_years:
            analyze_home_ad.summarize_all_innings(Path(args.out_root))
        else:
            analyze_home_ad.summarize_year(Path(args.plays_csv))

if __name__ == "__main__":
    main()
