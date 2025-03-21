import sys
import argparse


parser = argparse.ArgumentParser(description="Command-line tool with 'test' and 'show_hardware' commands.")
subparsers = parser.add_subparsers(dest="command", required=True)

# Coming soon!
# subparsers.add_parser("test", help="Run unit tests!")

subparsers.add_parser("show_hardware", help="Show hardware information")

# Parse arguments
args = parser.parse_args()

# Call the corresponding function
if args.command == "show_hardware":
    from .Hardware import Hardware
    h = Hardware()
    h.show()
else:
    print(f"Command '{args.command}' not recognized", file=sys.stderr)
    sys.exit(2)
