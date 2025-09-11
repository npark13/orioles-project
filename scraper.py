import argparse
import parse_events
import parse_ros
import join_names
import analyze_home_ad

def main():
    parser = argparse.ArgumentParser(description="Retrosheet Scraper & Analyzer")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ev_parser = subparsers.add_parser("parse-events")
    ev_parser.add_argument("folder")
    ev_parser.add_argument("--out", default="out")

    ros_parser = subparsers.add_parser("parse-rosters")
    ros_parser.add_argument("folder")
    ros_parser.add_argument("--teamfile", required=True)
    ros_parser.add_argument("--out", default="out/rosters.csv")

    join_parser = subparsers.add_parser("join-names")
    join_parser.add_argument("--events_out", default="out")
    join_parser.add_argument("--rosters_csv", default="out/rosters.csv")

    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("--plays_csv", default="out/plays.csv")

    args = parser.parse_args()

    if args.command == "parse-events":
        parse_events.main(args.folder, args.out)
    elif args.command == "parse-rosters":
        parse_ros.main(args.folder, args.teamfile, args.out)
    elif args.command == "join-names":
        join_names.add_names(args.events_out, args.rosters_csv)
    elif args.command == "analyze":
        analyze_home_ad.analyze(args.plays_csv)

if __name__ == "__main__":
    main()
