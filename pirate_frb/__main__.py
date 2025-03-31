import sys
import argparse


parser = argparse.ArgumentParser(description="pirate_frb command-line driver (use --help for more info)")
subparsers = parser.add_subparsers(dest="command", required=True)

# Coming soon!
# subparsers.add_parser("test", help="Run unit tests!")

subparsers.add_parser("show_hardware", help="Show hardware information")

# Parse arguments
args = parser.parse_args()

if args.command == "show_hardware":
    from .Hardware import Hardware
    h = Hardware()
    h.show()
else:
    print(f"Command '{args.command}' not recognized", file=sys.stderr)
    sys.exit(2)


####################################################################################################
