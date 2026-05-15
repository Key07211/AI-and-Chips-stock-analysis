import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).parent / "data"
CHART_DIR = Path(__file__).parent / "charts"
CHART_DIR.mkdir(exist_ok=True)

summary = pd.read_csv(DATA_DIR / "_summary.csv")
stocks_only = summary[summary["Sector"] != "Index"]

SECTOR_COLORS = {
    "AI": "#2196F3",
    "Chip": "#FF9800",
    "NewEnergy": "#4CAF50",
}


def load_close(file_name: str) -> pd.Series:
    df = pd.read_csv(DATA_DIR / file_name, parse_dates=["Date"], index_col="Date")
    return df["Close"]


# ══════════════════════════════════════════════
# Part A: Sector correlation analysis
# ══════════════════════════════════════════════
def sector_correlation_analysis():
    print("=" * 70)
    print("SECTOR CORRELATION ANALYSIS")
    print("=" * 70)

    sector_returns = {}
    for sector in ["AI", "Chip", "NewEnergy"]:
        tickers = stocks_only[stocks_only["Sector"] == sector]
        daily_rets = []
        for _, meta in tickers.iterrows():
            close = load_close(meta["File"])
            daily_rets.append(close.pct_change().dropna().rename(meta["Ticker"]))
        sector_df = pd.concat(daily_rets, axis=1)
        sector_returns[sector] = sector_df.mean(axis=1)

    sector_df = pd.DataFrame(sector_returns)

    # --- Pairwise correlation ---
    corr = sector_df.corr()
    print("\n>>> Sector daily return correlation matrix <<<")
    print(corr.round(4).to_string())

    # --- Rolling 30-day correlation ---
    rolling_corr_ai_chip = sector_df["AI"].rolling(30).corr(sector_df["Chip"])
    rolling_corr_ai_ne = sector_df["AI"].rolling(30).corr(sector_df["NewEnergy"])
    rolling_corr_chip_ne = sector_df["Chip"].rolling(30).corr(sector_df["NewEnergy"])

    print(f"\n>>> 30-day rolling correlation summary <<<")
    print(f"  AI vs Chip      — mean: {rolling_corr_ai_chip.mean():.4f}, min: {rolling_corr_ai_chip.min():.4f}, max: {rolling_corr_ai_chip.max():.4f}")
    print(f"  AI vs NewEnergy — mean: {rolling_corr_ai_ne.mean():.4f}, min: {rolling_corr_ai_ne.min():.4f}, max: {rolling_corr_ai_ne.max():.4f}")
    print(f"  Chip vs NewEnergy — mean: {rolling_corr_chip_ne.mean():.4f}, min: {rolling_corr_chip_ne.min():.4f}, max: {rolling_corr_chip_ne.max():.4f}")

    # --- Individual stock cross-sector correlation ---
    print("\n>>> Cross-sector stock-level correlation (top 10 pairs) <<<")
    all_returns = {}
    for _, meta in stocks_only.iterrows():
        close = load_close(meta["File"])
        all_returns[meta["Ticker"]] = close.pct_change().dropna()
    all_ret_df = pd.DataFrame(all_returns)
    full_corr = all_ret_df.corr()

    ai_tickers = stocks_only[stocks_only["Sector"] == "AI"]["Ticker"].tolist()
    chip_tickers = stocks_only[stocks_only["Sector"] == "Chip"]["Ticker"].tolist()
    ne_tickers = stocks_only[stocks_only["Sector"] == "NewEnergy"]["Ticker"].tolist()

    ai_chip_pairs = []
    for a in ai_tickers:
        for c in chip_tickers:
            ai_chip_pairs.append({"Pair": f"{a} vs {c}", "Sectors": "AI-Chip", "Corr": round(full_corr.loc[a, c], 4)})
    ai_ne_pairs = []
    for a in ai_tickers:
        for n in ne_tickers:
            ai_ne_pairs.append({"Pair": f"{a} vs {n}", "Sectors": "AI-NewEnergy", "Corr": round(full_corr.loc[a, n], 4)})
    chip_ne_pairs = []
    for c in chip_tickers:
        for n in ne_tickers:
            chip_ne_pairs.append({"Pair": f"{c} vs {n}", "Sectors": "Chip-NewEnergy", "Corr": round(full_corr.loc[c, n], 4)})

    all_pairs = ai_chip_pairs + ai_ne_pairs + chip_ne_pairs
    pairs_df = pd.DataFrame(all_pairs).sort_values("Corr", ascending=False)

    print("\n  Top 10 highest correlation pairs:")
    print(pairs_df.head(10).to_string(index=False))
    print("\n  Bottom 10 lowest correlation pairs:")
    print(pairs_df.tail(10).to_string(index=False))

    # --- Average cross-sector correlation ---
    avg_ai_chip = pd.DataFrame(ai_chip_pairs)["Corr"].mean()
    avg_ai_ne = pd.DataFrame(ai_ne_pairs)["Corr"].mean()
    avg_chip_ne = pd.DataFrame(chip_ne_pairs)["Corr"].mean()

    print(f"\n>>> Average stock-level cross-sector correlation <<<")
    print(f"  AI vs Chip:       {avg_ai_chip:.4f}")
    print(f"  AI vs NewEnergy:  {avg_ai_ne:.4f}")
    print(f"  Chip vs NewEnergy:{avg_chip_ne:.4f}")

    threshold = 0.3
    include_ne = avg_ai_ne >= threshold or avg_chip_ne >= threshold
    print(f"\n>>> Decision (threshold = {threshold}) <<<")
    if include_ne:
        print(f"  NewEnergy has meaningful correlation with AI/Chip. KEEPING NewEnergy in analysis.")
    else:
        print(f"  NewEnergy has WEAK correlation with AI/Chip. DROPPING NewEnergy from subsequent analysis.")

    # --- Plot rolling correlation ---
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(rolling_corr_ai_chip.index, rolling_corr_ai_chip.values,
            color="#2196F3", linewidth=1.5, label="AI vs Chip")
    ax.plot(rolling_corr_ai_ne.index, rolling_corr_ai_ne.values,
            color="#4CAF50", linewidth=1.5, label="AI vs NewEnergy")
    ax.plot(rolling_corr_chip_ne.index, rolling_corr_chip_ne.values,
            color="#FF9800", linewidth=1.5, label="Chip vs NewEnergy")
    ax.axhline(y=threshold, color="red", linestyle="--", linewidth=1, alpha=0.7, label=f"Threshold ({threshold})")
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("30-Day Rolling Correlation", fontsize=11)
    ax.set_title("Sector Rolling Correlation (30-Day Window)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "sector_rolling_correlation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  [OK] sector_rolling_correlation.png")

    return include_ne


# ══════════════════════════════════════════════
# Step 4: Annualized Volatility & Return vs Risk
# ══════════════════════════════════════════════
def annualized_volatility(include_ne: bool):
    print("\n" + "=" * 70)
    print("STEP 4: ANNUALIZED VOLATILITY")
    print("=" * 70)

    sectors = ["AI", "Chip"]
    if include_ne:
        sectors.append("NewEnergy")
    analysis_stocks = stocks_only[stocks_only["Sector"].isin(sectors)]

    rows = []
    for _, meta in analysis_stocks.iterrows():
        close = load_close(meta["File"])
        daily_ret = close.pct_change().dropna()

        annual_return = (close.iloc[-1] / close.iloc[0] - 1) * 100
        annual_vol = daily_ret.std() * np.sqrt(252) * 100
        max_dd = ((close / close.cummax()) - 1).min() * 100
        sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(252) if daily_ret.std() > 0 else 0

        rows.append({
            "Ticker": meta["Ticker"],
            "Name": meta["Name"],
            "Sector": meta["Sector"],
            "Annual_Return%": round(annual_return, 2),
            "Annual_Vol%": round(annual_vol, 2),
            "Max_Drawdown%": round(max_dd, 2),
            "Sharpe": round(sharpe, 2),
        })

    vol_df = pd.DataFrame(rows)

    print("\n>>> Annualized volatility table (sorted by Sharpe) <<<")
    sorted_df = vol_df.sort_values("Sharpe", ascending=False).reset_index(drop=True)
    sorted_df.index = sorted_df.index + 1
    sorted_df.index.name = "Rank"
    print(sorted_df.to_string())

    # --- Best risk-adjusted (high return, low vol) ---
    print("\n>>> Best risk-adjusted stocks (Sharpe > 1.5) <<<")
    best = vol_df[vol_df["Sharpe"] > 1.5].sort_values("Sharpe", ascending=False)
    if best.empty:
        best = vol_df.nlargest(5, "Sharpe")
        print("  (No stock with Sharpe > 1.5, showing top 5)")
    print(best[["Ticker", "Name", "Sector", "Annual_Return%", "Annual_Vol%", "Sharpe"]].to_string(index=False))

    # --- Return vs Risk scatter ---
    fig, ax = plt.subplots(figsize=(14, 9))

    active_colors = {s: SECTOR_COLORS[s] for s in sectors}
    for sector in sectors:
        sub = vol_df[vol_df["Sector"] == sector]
        ax.scatter(sub["Annual_Vol%"], sub["Annual_Return%"],
                   c=active_colors[sector], s=100, alpha=0.85,
                   edgecolors="white", linewidth=0.8, label=sector, zorder=3)
        for _, row in sub.iterrows():
            ax.annotate(row["Ticker"].replace("^", ""),
                        (row["Annual_Vol%"], row["Annual_Return%"]),
                        fontsize=8, fontweight="bold",
                        ha="left", va="bottom",
                        xytext=(5, 5), textcoords="offset points")

    # --- Sharpe reference lines ---
    vol_range = np.linspace(0, vol_df["Annual_Vol%"].max() * 1.1, 100)
    for s_val, alpha in [(1.0, 0.3), (2.0, 0.2), (3.0, 0.15)]:
        ax.plot(vol_range, s_val * vol_range, "--", color="gray", alpha=alpha, linewidth=1)
        ax.text(vol_range[-1], s_val * vol_range[-1], f"Sharpe={s_val}",
                fontsize=7, color="gray", alpha=0.6)

    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_xlabel("Annualized Volatility (%)", fontsize=12)
    ax.set_ylabel("1-Year Return (%)", fontsize=12)
    ax.set_title("Return vs Risk — Annualized Volatility Scatter", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "return_vs_risk_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  [OK] return_vs_risk_scatter.png")

    vol_df.to_csv(CHART_DIR / "annualized_volatility.csv", index=False)
    print(f"  [OK] annualized_volatility.csv")

    return vol_df


# ══════════════════════════════════════════════
# Step 5: 30-Day Rolling Volatility
# ══════════════════════════════════════════════
def rolling_volatility(include_ne: bool):
    print("\n" + "=" * 70)
    print("STEP 5: 30-DAY ROLLING VOLATILITY")
    print("=" * 70)

    sectors = ["AI", "Chip"]
    if include_ne:
        sectors.append("NewEnergy")
    analysis_stocks = stocks_only[stocks_only["Sector"].isin(sectors)]

    # --- Compute rolling vol for all stocks ---
    rolling_data = {}
    for _, meta in analysis_stocks.iterrows():
        close = load_close(meta["File"])
        daily_ret = close.pct_change().dropna()
        rolling_vol = daily_ret.rolling(30).std() * np.sqrt(252) * 100
        rolling_data[meta["Ticker"]] = rolling_vol

    rolling_df = pd.DataFrame(rolling_data).dropna()

    # --- Per-sector subplot ---
    fig, axes = plt.subplots(len(sectors), 1, figsize=(16, 5 * len(sectors)), sharex=True)
    if len(sectors) == 1:
        axes = [axes]

    for ax, sector in zip(axes, sectors):
        tickers = analysis_stocks[analysis_stocks["Sector"] == sector]["Ticker"].tolist()
        for t in tickers:
            safe = t.replace("^", "")
            ax.plot(rolling_df.index, rolling_df[t], linewidth=1.2, alpha=0.8, label=safe)
        ax.set_ylabel("Annualized Vol (%)", fontsize=10)
        ax.set_title(f"{sector} — 30-Day Rolling Volatility", fontsize=12)
        ax.legend(fontsize=7, loc="upper left", ncol=3)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Date", fontsize=11)
    fig.suptitle("30-Day Rolling Volatility by Sector", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "rolling_volatility_by_sector.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] rolling_volatility_by_sector.png")

    # --- Detect volatility spikes ---
    print("\n>>> Volatility spike detection <<<")
    print("  (Days where 30-day rolling vol > 2x median for that stock)\n")

    spike_records = []
    for _, meta in analysis_stocks.iterrows():
        ticker = meta["Ticker"]
        name = meta["Name"]
        sector = meta["Sector"]

        vol_series = rolling_df[ticker]
        median_vol = vol_series.median()
        spike_threshold = median_vol * 2

        spike_mask = vol_series > spike_threshold
        if spike_mask.any():
            spike_dates = vol_series[spike_mask]

            groups = []
            current_group = []
            prev_date = None
            for date in spike_dates.index:
                if prev_date is not None and (date - prev_date).days > 5:
                    groups.append(current_group)
                    current_group = []
                current_group.append((date, spike_dates[date]))
                prev_date = date
            if current_group:
                groups.append(current_group)

            for group in groups:
                peak_date, peak_vol = max(group, key=lambda x: x[1])
                start_date = group[0][0]
                end_date = group[-1][0]
                duration = (end_date - start_date).days + 1

                close = load_close(meta["File"])
                period_close = close.loc[start_date:end_date]
                if len(period_close) >= 2:
                    period_return = (period_close.iloc[-1] / period_close.iloc[0] - 1) * 100
                else:
                    period_return = 0.0

                spike_records.append({
                    "Ticker": ticker,
                    "Name": name,
                    "Sector": sector,
                    "Spike_Start": start_date.strftime("%Y-%m-%d"),
                    "Spike_End": end_date.strftime("%Y-%m-%d"),
                    "Duration_Days": duration,
                    "Peak_Date": peak_date.strftime("%Y-%m-%d"),
                    "Peak_Vol%": round(peak_vol, 1),
                    "Median_Vol%": round(median_vol, 1),
                    "Period_Return%": round(period_return, 2),
                })

    spike_df = pd.DataFrame(spike_records)

    if not spike_df.empty:
        spike_df = spike_df.sort_values("Peak_Vol%", ascending=False).reset_index(drop=True)
        spike_df.index = spike_df.index + 1
        spike_df.index.name = "Rank"

        print("  Top 20 volatility spike events:")
        print(spike_df.head(20).to_string())

        spike_df.to_csv(CHART_DIR / "volatility_spikes.csv", index=False)
        print(f"\n  [OK] volatility_spikes.csv ({len(spike_df)} events total)")
    else:
        print("  No spikes detected.")

    # --- Plot top spike stocks ---
    if not spike_df.empty:
        top_spike_tickers = spike_df.head(10)["Ticker"].unique()
        fig, ax = plt.subplots(figsize=(16, 8))
        for t in top_spike_tickers:
            safe = t.replace("^", "")
            ax.plot(rolling_df.index, rolling_df[t], linewidth=1.5, alpha=0.85, label=safe)

        for _, row in spike_df.head(10).iterrows():
            peak = pd.Timestamp(row["Peak_Date"])
            if peak in rolling_df.index:
                ax.annotate(f"{row['Ticker']}\n{row['Peak_Date']}\n{row['Peak_Vol%']}%",
                            xy=(peak, row["Peak_Vol%"]),
                            fontsize=6, ha="center", va="bottom",
                            xytext=(0, 10), textcoords="offset points",
                            arrowprops=dict(arrowstyle="->", color="red", lw=0.8),
                            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="red", lw=0.5))

        ax.set_xlabel("Date", fontsize=11)
        ax.set_ylabel("Annualized Volatility (%)", fontsize=11)
        ax.set_title("Top Volatility Spike Stocks — 30-Day Rolling", fontsize=13)
        ax.legend(fontsize=9, loc="upper left", ncol=2)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(CHART_DIR / "volatility_spike_highlight.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  [OK] volatility_spike_highlight.png")

    return spike_df


if __name__ == "__main__":
    include_ne = sector_correlation_analysis()
    annualized_volatility(include_ne)
    rolling_volatility(include_ne)

    print("\n" + "=" * 70)
    print("Volatility analysis complete.")
    print("=" * 70)
