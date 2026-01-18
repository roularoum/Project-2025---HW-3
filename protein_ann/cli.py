from __future__ import annotations  # epitrepei type hints me forward refs

import argparse  # gia command line arguments CLI


def add_common_io_args(parser: argparse.ArgumentParser) -> None:  # prosthetei koino argument gia logging
    parser.add_argument(  # orizei to flag -v/--verbose
        "-v",  # syntomo flag
        "--verbose",  # plires flag
        action="store_true",  # an mpei sto CLI, ginetai True alliws False
        help="Verbose logging", 
    )


def add_seed_arg(parser: argparse.ArgumentParser) -> None:  # prosthetei argument gia random seed
    parser.add_argument(  # orizei to seed
        "--seed",  # onoma argument
        type=int,  # prepei na einai int
        default=1,  # default timi
        help="Random seed (default: 1)", 
    )
