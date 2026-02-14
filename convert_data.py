"""
Convert Databento .csv.zst to yearly CSV chunks for the backtest engine.

Run locally:  python convert_data.py
Requires:     pip install pandas zstandard

This reads the large .csv.zst file, converts timestamps, and splits
into yearly CSV files small enough for regular Git (no LFS needed).
"""

import pandas as pd
import zstandard as zstd
import io
import os
import sys


def main():
    src = "data/glbx-mdp3-20100606-20260212.ohlcv-1m.csv.zst"

    if not os.path.exists(src):
        print("ERROR: File not found: {}".format(src))
        print("Make sure you run this from the Trading repo root directory.")
        sys.exit(1)

    print("Reading {} (this may take a minute)...".format(src))

    # Manually decompress zstd then read CSV (works with all pandas versions)
    with open(src, "rb") as f:
        dctx = zstd.ZstdDecompressor()
        decompressed = dctx.decompress(f.read(), max_output_size=2 * 1024 * 1024 * 1024)
    df = pd.read_csv(io.BytesIO(decompressed))
    del decompressed  # free memory
    df.columns = [c.strip().lower() for c in df.columns]
    print("  Loaded {:,} rows".format(len(df)))

    # Convert nanosecond timestamps to datetime
    if "ts_event" in df.columns:
        df["ts_event"] = pd.to_datetime(df["ts_event"], unit="ns", utc=True)
        df["ts_event"] = df["ts_event"].dt.tz_convert("US/Eastern").dt.tz_localize(None)
        df = df.set_index("ts_event")
    elif "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime")

    df.index.name = "datetime"

    # Drop metadata columns
    drop_cols = ["rtype", "publisher_id", "instrument_id", "ts_recv",
                 "flags", "sequence", "symbol"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # Keep only OHLCV columns
    keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    df = df[keep]
    df = df.sort_index()

    print("  Date range: {} to {}".format(df.index[0], df.index[-1]))
    print("  Columns: {}".format(list(df.columns)))

    # Split by year and save
    os.makedirs("data/es_1min", exist_ok=True)
    years = df.index.year.unique()

    for year in sorted(years):
        chunk = df[df.index.year == year]
        if len(chunk) == 0:
            continue
        outpath = "data/es_1min/ES_{}.csv".format(year)
        chunk.to_csv(outpath)
        size_mb = os.path.getsize(outpath) / 1e6
        print("  Saved {}: {:,} bars ({:.1f} MB)".format(outpath, len(chunk), size_mb))

    # Also save a combined file for quick loading
    combined = "data/es_1min_all.csv"
    print("\nSaving combined file: {} ...".format(combined))
    df.to_csv(combined)
    size_mb = os.path.getsize(combined) / 1e6
    print("  Total: {:,} bars ({:.1f} MB)".format(len(df), size_mb))

    print("\nDone! Now run:")
    print("  git add data/es_1min/ data/es_1min_all.csv")
    print('  git commit -m "Add converted ES 1-min CSV data"')
    print("  git push")


if __name__ == "__main__":
    main()
