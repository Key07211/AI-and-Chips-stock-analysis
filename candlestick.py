import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).parent / "data"
CHART_DIR = Path(__file__).parent / "charts"
CHART_DIR.mkdir(exist_ok=True)

summary = pd.read_csv(DATA_DIR / "_summary.csv")

SECTOR_COLORS = {
    "AI": "#2196F3",
    "Chip": "#FF9800",
    "NewEnergy": "#4CAF50",
    "Index": "#9C27B0",
}


def load(file_name: str) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / file_name, parse_dates=["Date"], index_col="Date")
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    return df


# ──────────────────────────────────────────────
# 1. Line chart for each stock
# ──────────────────────────────────────────────
def generate_line_charts():
    print("=" * 70)
    print("Generating line charts...")
    print("=" * 70)

    for _, meta in summary.iterrows():
        ticker = meta["Ticker"]
        name = meta["Name"]
        sector = meta["Sector"]
        safe = ticker.replace("^", "")
        df = load(meta["File"])

        ret = (df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100
        color = SECTOR_COLORS.get(sector, "#999")

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7),
                                        gridspec_kw={"height_ratios": [3, 1]},
                                        sharex=True)

        ax1.plot(df.index, df["Close"], color=color, linewidth=1.5)
        ax1.fill_between(df.index, df["Close"], alpha=0.1, color=color)
        ax1.set_ylabel("Close Price ($)", fontsize=11)
        ax1.set_title(f"{ticker} ({name})  |  {sector}  |  Return: {ret:+.1f}%",
                      fontsize=13)
        ax1.grid(True, alpha=0.3)

        vol_colors = ["#4CAF50" if df["Close"].iloc[i] >= df["Open"].iloc[i]
                       else "#F44336" for i in range(len(df))]
        ax2.bar(df.index, df["Volume"], color=vol_colors, alpha=0.6, width=1)
        ax2.set_ylabel("Volume", fontsize=11)
        ax2.set_xlabel("Date", fontsize=11)
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        save_path = CHART_DIR / f"{safe}_line.png"
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  [OK] {safe}_line.png")

    print(f"\nAll line charts saved to {CHART_DIR}\n")


# ──────────────────────────────────────────────
# 2. Performance ranking
# ──────────────────────────────────────────────
def performance_ranking():
    print("=" * 70)
    print("Performance Ranking (1-Year)")
    print("=" * 70)

    rows = []
    for _, meta in summary.iterrows():
        df = load(meta["File"])
        close = df["Close"]

        annual_return = (close.iloc[-1] / close.iloc[0] - 1) * 100
        daily_returns = close.pct_change().dropna()
        volatility = daily_returns.std() * 100
        annualized_vol = volatility * (252 ** 0.5)
        max_drawdown = ((close / close.cummax()) - 1).min() * 100
        sharpe = (daily_returns.mean() / daily_returns.std()) * (252 ** 0.5) if daily_returns.std() > 0 else 0

        rows.append({
            "Ticker": meta["Ticker"],
            "Name": meta["Name"],
            "Sector": meta["Sector"],
            "Start_Price": round(close.iloc[0], 2),
            "End_Price": round(close.iloc[-1], 2),
            "Return%": round(annual_return, 2),
            "Daily_Vol%": round(volatility, 2),
            "Annual_Vol%": round(annualized_vol, 2),
            "Max_Drawdown%": round(max_drawdown, 2),
            "Sharpe": round(sharpe, 2),
        })

    perf = pd.DataFrame(rows)

    print("\n>>> TOP 10 GAINERS <<<")
    top_gain = perf.nlargest(10, "Return%")
    print(top_gain[["Ticker", "Name", "Sector", "Start_Price", "End_Price", "Return%"]].to_string(index=False))

    print("\n>>> BOTTOM 5 PERFORMERS <<<")
    bottom = perf.nsmallest(5, "Return%")
    print(bottom[["Ticker", "Name", "Sector", "Start_Price", "End_Price", "Return%"]].to_string(index=False))

    print("\n>>> TOP 10 MOST STABLE (lowest annualized volatility, stocks only) <<<")
    stocks_only = perf[perf["Sector"] != "Index"]
    stable = stocks_only.nsmallest(10, "Annual_Vol%")
    print(stable[["Ticker", "Name", "Sector", "Annual_Vol%", "Max_Drawdown%", "Sharpe", "Return%"]].to_string(index=False))

    print("\n>>> TOP 10 MOST VOLATILE <<<")
    volatile = stocks_only.nlargest(10, "Annual_Vol%")
    print(volatile[["Ticker", "Name", "Sector", "Annual_Vol%", "Max_Drawdown%", "Return%"]].to_string(index=False))

    print("\n>>> SECTOR AVERAGE PERFORMANCE <<<")
    sector_avg = perf.groupby("Sector").agg(
        Avg_Return=("Return%", "mean"),
        Avg_AnnualVol=("Annual_Vol%", "mean"),
        Avg_Sharpe=("Sharpe", "mean"),
        Avg_MaxDD=("Max_Drawdown%", "mean"),
    ).round(2)
    print(sector_avg.to_string())

    perf.to_csv(CHART_DIR / "performance_summary.csv", index=False)
    print(f"\nFull ranking saved to {CHART_DIR / 'performance_summary.csv'}")
    return perf


# ──────────────────────────────────────────────
# 3. Return vs volatility scatter
# ──────────────────────────────────────────────
def plot_return_vs_volatility(perf: pd.DataFrame):
    print("\n" + "=" * 70)
    print("Generating return vs volatility scatter plot...")
    print("=" * 70)

    stocks = perf[perf["Sector"] != "Index"]

    fig, ax = plt.subplots(figsize=(14, 9))

    for sector in stocks["Sector"].unique():
        sub = stocks[stocks["Sector"] == sector]
        ax.scatter(sub["Annual_Vol%"], sub["Return%"],
                   c=SECTOR_COLORS.get(sector, "#999"),
                   s=80, alpha=0.8, edgecolors="white", linewidth=0.5,
                   label=sector)
        for _, row in sub.iterrows():
            ax.annotate(row["Ticker"].replace("^", ""),
                        (row["Annual_Vol%"], row["Return%"]),
                        fontsize=7, ha="left", va="bottom",
                        xytext=(4, 4), textcoords="offset points")

    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Annualized Volatility (%)", fontsize=12)
    ax.set_ylabel("1-Year Return (%)", fontsize=12)
    ax.set_title("Return vs Volatility — All Stocks (1 Year)", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    save_path = CHART_DIR / "return_vs_volatility.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {save_path}")


# ──────────────────────────────────────────────
# 4. Sector cumulative return comparison
# ──────────────────────────────────────────────
def plot_sector_cumulative_return():
    print("\n" + "=" * 70)
    print("Generating sector cumulative return chart...")
    print("=" * 70)

    fig, ax = plt.subplots(figsize=(14, 7))

    for sector in ["AI", "Chip", "NewEnergy"]:
        sector_tickers = summary[summary["Sector"] == sector]
        cum_returns = []
        for _, meta in sector_tickers.iterrows():
            df = load(meta["File"])
            cr = (df["Close"] / df["Close"].iloc[0] - 1) * 100
            cum_returns.append(cr)
        avg_cum = pd.concat(cum_returns, axis=1).mean(axis=1)
        ax.plot(avg_cum.index, avg_cum.values,
                color=SECTOR_COLORS[sector], linewidth=2, label=sector)

    sp500 = load("GSPC.csv")
    sp_cum = (sp500["Close"] / sp500["Close"].iloc[0] - 1) * 100
    ax.plot(sp_cum.index, sp_cum.values, color="#9C27B0",
            linewidth=2, linestyle="--", label="S&P500 (benchmark)")

    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Cumulative Return (%)", fontsize=12)
    ax.set_title("Sector Average Cumulative Return vs S&P500", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    save_path = CHART_DIR / "sector_cumulative_return.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {save_path}")


# ──────────────────────────────────────────────
# 5. All stocks on one chart (normalized)
# ──────────────────────────────────────────────
def plot_all_stocks_combined():
    print("\n" + "=" * 70)
    print("Generating all-stocks combined line chart...")
    print("=" * 70)

    fig, axes = plt.subplots(2, 2, figsize=(22, 14))
    sector_list = ["AI", "Chip", "NewEnergy", "Index"]
    sector_titles = {
        "AI": "AI Tech",
        "Chip": "Chip",
        "NewEnergy": "New Energy",
        "Index": "Index",
    }

    for ax, sector in zip(axes.flat, sector_list):
        sector_tickers = summary[summary["Sector"] == sector]
        for _, meta in sector_tickers.iterrows():
            df = load(meta["File"])
            norm = (df["Close"] / df["Close"].iloc[0] - 1) * 100
            safe = meta["Ticker"].replace("^", "")
            ax.plot(norm.index, norm.values, linewidth=1.2, alpha=0.85, label=safe)

        ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_title(f"{sector_titles[sector]} — Cumulative Return (%)", fontsize=13)
        ax.set_xlabel("Date", fontsize=10)
        ax.set_ylabel("Return (%)", fontsize=10)
        ax.legend(fontsize=7, loc="upper left", ncol=2)
        ax.grid(True, alpha=0.3)

    fig.suptitle("All Stocks — 1-Year Cumulative Return by Sector", fontsize=16, y=1.01)
    fig.tight_layout()
    save_path = CHART_DIR / "all_stocks_combined.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {save_path}")


if __name__ == "__main__":
    generate_line_charts()
    perf = performance_ranking()
    plot_return_vs_volatility(perf)
    plot_sector_cumulative_return()
    plot_all_stocks_combined()

    print("\n" + "=" * 70)
    print("All done.")
    print("=" * 70)
