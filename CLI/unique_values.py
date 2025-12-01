import argparse
import pandas as pd

from Newspaper_Tension.utils import load_GDELT_tsv

"""
Command Line Interface for understanding GDELT CSV file.
"""

if __name__=="__main__":
    """
    Start the CLI and mine unique values from different columns.
    """
    parser = argparse.ArgumentParser(description="CLI for mining unique column values.")
    parser.add_argument("-f", "--files", nargs="+", help="List of csv files to look through")
    parser.add_argument("-c", "--column-names", nargs="+", help="Columns to list unique values for")
    args = parser.parse_args()

    unique_values = { c : set() for c in args.column_names }
    for file in args.files:
        df = load_GDELT_tsv(file)
        for c in args.column_names:
            unique_values[c] = unique_values[c].union(set(df[c].dropna().unique()))

    for c in args.column_names:
        print(f"===== {c} =====")
        values = ", ".join(sorted(list(unique_values[c])))
        print(values)
            
