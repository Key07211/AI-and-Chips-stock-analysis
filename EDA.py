import pandas as pd
import os
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
SUMMARY_FILE = DATA_DIR / "_summary.csv"

summary = pd.read_csv(SUMMARY_FILE)
PRICE_COLS = ["Open", "High", "Low", "Close"]
ALL_COLS = PRICE_COLS + ["Volume"]


def load(file_name: str) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / file_name, parse_dates=["Date"])
    df.set_index("Date", inplace=True)
    df = df[ALL_COLS]
    return df


# ──────────────────────────────────────────────
# 1. Basic Info Overview
# ──────────────────────────────────────────────
def check_basic_info():
    print("=" * 70)
    print("1. Basic Info Overview")
    print("=" * 70)

    rows = []
    for _, meta in summary.iterrows():
        df = load(meta["File"])
        rows.append({
            "Ticker": meta["Ticker"],
            "Name": meta["Name"],
            "Sector": meta["Sector"],
            "Rows": len(df),
            "Columns": df.shape[1],
            "Start": df.index.min().strftime("%Y-%m-%d"),
            "End": df.index.max().strftime("%Y-%m-%d"),
            "Dtypes": ", ".join(f"{c}:{df[c].dtype}" for c in df.columns),
        })
    info_df = pd.DataFrame(rows)
    print(info_df.to_string(index=False))
    print()
    return info_df


# ──────────────────────────────────────────────
# 2. Missing Values Check
# ──────────────────────────────────────────────
def check_missing():
    print("=" * 70)
    print("2. Missing Values Check")
    print("=" * 70)

    rows = []
    for _, meta in summary.iterrows():
        df = load(meta["File"])
        missing = df.isnull().sum()
        missing_pct = (df.isnull().mean() * 100).round(2)
        row = {"Ticker": meta["Ticker"], "Total_Rows": len(df)}
        for col in ALL_COLS:
            row[f"{col}_missing"] = missing[col]
            row[f"{col}_missing%"] = missing_pct[col]
        rows.append(row)

    miss_df = pd.DataFrame(rows)
    has_missing = miss_df[[c for c in miss_df.columns if "missing%" in c]].max(axis=1) > 0
    if has_missing.any():
        print("[!] The following stocks have missing values:")
        print(miss_df[has_missing].to_string(index=False))
    else:
        print("[OK] No missing values found in any stock")
    print()
    return miss_df


# ──────────────────────────────────────────────
# 3. Duplicate Rows Check
# ──────────────────────────────────────────────
def check_duplicates():
    print("=" * 70)
    print("3. Duplicate Rows Check")
    print("=" * 70)

    rows = []
    for _, meta in summary.iterrows():
        df = load(meta["File"])
        dup_index = df.index.duplicated().sum()
        dup_rows = df.duplicated().sum()
        rows.append({
            "Ticker": meta["Ticker"],
            "Duplicate_Dates": dup_index,
            "Duplicate_Rows": dup_rows,
        })

    dup_df = pd.DataFrame(rows)
    has_dup = (dup_df["Duplicate_Dates"] > 0) | (dup_df["Duplicate_Rows"] > 0)
    if has_dup.any():
        print("[!] The following stocks have duplicates:")
        print(dup_df[has_dup].to_string(index=False))
    else:
        print("[OK] No duplicate dates or rows found")
    print()
    return dup_df


# ──────────────────────────────────────────────
# 4. Data Types Check
# ──────────────────────────────────────────────
def check_dtypes():
    print("=" * 70)
    print("4. Data Types Check")
    print("=" * 70)

    issues = []
    for _, meta in summary.iterrows():
        df = load(meta["File"])
        for col in PRICE_COLS:
            if not pd.api.types.is_float_dtype(df[col]):
                issues.append({"Ticker": meta["Ticker"], "Column": col,
                               "Actual_Dtype": str(df[col].dtype), "Expected": "float64"})
        if not pd.api.types.is_integer_dtype(df["Volume"]) and not pd.api.types.is_float_dtype(df["Volume"]):
            issues.append({"Ticker": meta["Ticker"], "Column": "Volume",
                           "Actual_Dtype": str(df["Volume"].dtype), "Expected": "int64/float64"})

    if issues:
        print("[!] Data type issues found:")
        print(pd.DataFrame(issues).to_string(index=False))
    else:
        print("[OK] All column dtypes are correct (price=float64, volume=int64/float64)")
    print()
    return issues


# ──────────────────────────────────────────────
# 5. Descriptive Statistics
# ──────────────────────────────────────────────
def check_descriptive_stats():
    print("=" * 70)
    print("5. Descriptive Statistics (Close Price)")
    print("=" * 70)

    rows = []
    for _, meta in summary.iterrows():
        df = load(meta["File"])
        desc = df["Close"].describe()
        rows.append({
            "Ticker": meta["Ticker"],
            "Name": meta["Name"],
            "Sector": meta["Sector"],
            "Min": round(desc["min"], 2),
            "Max": round(desc["max"], 2),
            "Mean": round(desc["mean"], 2),
            "Std": round(desc["std"], 2),
            "Median": round(desc["50%"], 2),
        })
    stat_df = pd.DataFrame(rows)
    for sector in stat_df["Sector"].unique():
        sub = stat_df[stat_df["Sector"] == sector]
        print(f"\n--- {sector} ---")
        print(sub.to_string(index=False))
    print()
    return stat_df


# ──────────────────────────────────────────────
# 6. Outlier Check (Price & Volume)
# ──────────────────────────────────────────────
def check_outliers():
    print("=" * 70)
    print("6. Outlier Check")
    print("=" * 70)

    issues = []
    for _, meta in summary.iterrows():
        df = load(meta["File"])
        ticker = meta["Ticker"]

        for col in PRICE_COLS:
            neg_count = (df[col] <= 0).sum()
            if neg_count > 0:
                issues.append({"Ticker": ticker, "Check": f"{col}<=0", "Count": neg_count})

        neg_vol = (df["Volume"] < 0).sum()
        if neg_vol > 0:
            issues.append({"Ticker": ticker, "Check": "Volume<0", "Count": neg_vol})

        bad_hl = (df["High"] < df["Low"]).sum()
        if bad_hl > 0:
            issues.append({"Ticker": ticker, "Check": "High<Low", "Count": bad_hl})

        out_of_range = ((df["Close"] < df["Low"]) | (df["Close"] > df["High"])).sum()
        if out_of_range > 0:
            issues.append({"Ticker": ticker, "Check": "Close outside [Low,High]", "Count": out_of_range})

        out_open = ((df["Open"] < df["Low"]) | (df["Open"] > df["High"])).sum()
        if out_open > 0:
            issues.append({"Ticker": ticker, "Check": "Open outside [Low,High]", "Count": out_open})

        daily_ret = df["Close"].pct_change().abs()
        extreme = (daily_ret > 0.5).sum()
        if extreme > 0:
            extreme_dates = df.index[daily_ret > 0.5].strftime("%Y-%m-%d").tolist()
            issues.append({"Ticker": ticker, "Check": "Daily return >50%",
                           "Count": extreme, "Dates": ", ".join(extreme_dates)})

    if issues:
        print("[!] Outliers detected:")
        print(pd.DataFrame(issues).to_string(index=False))
    else:
        print("[OK] No price/volume outliers found")
    print()
    return issues


# ──────────────────────────────────────────────
# 7. Trading Day Continuity Check
# ──────────────────────────────────────────────
def check_trading_day_continuity():
    print("=" * 70)
    print("7. Trading Day Continuity Check")
    print("=" * 70)

    ref_file = summary.loc[summary["Ticker"] == "^GSPC", "File"].values[0]
    ref_df = load(ref_file)
    ref_dates = set(ref_df.index)

    rows = []
    for _, meta in summary.iterrows():
        if meta["Sector"] == "Index":
            continue
        df = load(meta["File"])
        stock_dates = set(df.index)

        missing_dates = sorted(ref_dates - stock_dates)
        extra_dates = sorted(stock_dates - ref_dates)
        rows.append({
            "Ticker": meta["Ticker"],
            "Total_Days": len(df),
            "Missing_vs_SP500": len(missing_dates),
            "Extra_vs_SP500": len(extra_dates),
            "Missing_Dates": ", ".join(d.strftime("%Y-%m-%d") for d in missing_dates[:5]) if missing_dates else "",
        })

    cont_df = pd.DataFrame(rows)
    has_gap = (cont_df["Missing_vs_SP500"] > 0) | (cont_df["Extra_vs_SP500"] > 0)
    if has_gap.any():
        print("[!] The following stocks have trading day gaps vs S&P500:")
        print(cont_df[has_gap].to_string(index=False))
    else:
        print("[OK] All stocks align with S&P500 trading calendar")
    print()
    return cont_df


# ──────────────────────────────────────────────
# 8. Time Order Check
# ──────────────────────────────────────────────
def check_time_order():
    print("=" * 70)
    print("8. Time Series Order Check")
    print("=" * 70)

    issues = []
    for _, meta in summary.iterrows():
        df = load(meta["File"])
        if not df.index.is_monotonic_increasing:
            issues.append(meta["Ticker"])

    if issues:
        print(f"[!] The following stocks are not sorted in ascending order: {', '.join(issues)}")
    else:
        print("[OK] All stocks are sorted in ascending date order")
    print()
    return issues


# ──────────────────────────────────────────────
# 9. Zero Volume Check
# ──────────────────────────────────────────────
def check_zero_volume():
    print("=" * 70)
    print("9. Zero Volume Check")
    print("=" * 70)

    rows = []
    for _, meta in summary.iterrows():
        df = load(meta["File"])
        zero_vol = (df["Volume"] == 0).sum()
        if zero_vol > 0:
            pct = round(zero_vol / len(df) * 100, 1)
            rows.append({
                "Ticker": meta["Ticker"],
                "Name": meta["Name"],
                "Sector": meta["Sector"],
                "Zero_Volume_Days": zero_vol,
                "Pct": f"{pct}%",
            })

    if rows:
        zero_df = pd.DataFrame(rows)
        print("[!] The following stocks have zero-volume trading days:")
        print(zero_df.to_string(index=False))
        print("    (Zero volume is normal for index data)")
    else:
        print("[OK] All stocks have non-zero volume")
    print()
    return rows


# ──────────────────────────────────────────────
# 10. Summary Report
# ──────────────────────────────────────────────
def generate_report():
    print("\n")
    print("#" * 70)
    print("#  EDA Data Quality Report")
    print(f"#  Data directory: {DATA_DIR}")
    print(f"#  Total stocks/indices: {len(summary)}")
    print("#" * 70)
    print()

    check_basic_info()
    check_missing()
    check_duplicates()
    check_dtypes()
    check_descriptive_stats()
    check_outliers()
    check_trading_day_continuity()
    check_time_order()
    check_zero_volume()

    print("=" * 70)
    print("EDA Data Quality Check Complete")
    print("=" * 70)


if __name__ == "__main__":
    generate_report()
