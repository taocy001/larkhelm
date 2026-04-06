#!/usr/bin/env python3
"""larkhelm CLI entry point"""
import argparse
import sys


def cli():
    from larkhelm import __version__

    parser = argparse.ArgumentParser(
        prog="larkhelm",
        description="Feishu (Lark) AI assistant platform powered by Claude CLI & Gemini CLI",
    )
    parser.add_argument(
        "--version", "-V",
        action="version",
        version=f"larkhelm {__version__}",
    )
    sub = parser.add_subparsers(dest="command")
    start_parser = sub.add_parser("start", help="start the service (runs in the foreground)")
    start_parser.add_argument("--config", metavar="PATH",
                              help="path to config file (default: auto-detect)")
    start_parser.add_argument("--data-dir", metavar="DIR",
                              help="path to data directory (default: auto-detect)")
    # Keep the version sub-command for backward compatibility (prefer the --version flag)
    sub.add_parser("version", help="print version (prefer --version)")

    args = parser.parse_args()

    if args.command == "version":
        print(f"larkhelm {__version__}")
    elif args.command == "start":
        from larkhelm.bridge import main
        main(config_path=args.config, data_dir=args.data_dir)
    else:
        # No subcommand given: show help and require the user to make an explicit choice
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    cli()
