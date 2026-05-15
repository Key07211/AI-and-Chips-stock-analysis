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
    return df[["Open", "High", "Low", "Close", "Volume"]]


# ──────────────────────────────────────────────
# 1. Compute daily, monthly, yearly returns
# ──────────────────────────────────────────────
def compute_returns():
    print("=" * 70)
    print("Computing daily / monthly / yearly returns...")
    print("=" * 70)

    all_daily = []
    all_monthly = []
    all_yearly = []

    for _, meta in summary.iterrows():
        ticker = meta["Ticker"]
        name = meta["Name"]
        sector = meta["Sector"]
        df = load(meta["File"])
        close = df["Close"]

        # --- Daily returns ---
        daily_ret = close.pct_change().dropna()
        daily_stats = {
            "Ticker": ticker,
            "Name": name,
            "Sector": sector,
            "Mean%": round(daily_ret.mean() * 100, 4),
            "Median%": round(daily_ret.median() * 100, 4),
            "Std%": round(daily_ret.std() * 100, 4),
            "Min%": round(daily_ret.min() * 100, 2),
            "Max%": round(daily_ret.max() * 100, 2),
            "Positive_Days": (daily_ret > 0).sum(),
            "Negative_Days": (daily_ret < 0).sum(),
            "Win_Rate%": round((daily_ret > 0).sum() / len(daily_ret) * 100, 1),
        }
        all_daily.append(daily_stats)

        # --- Monthly returns ---
        monthly_close = close.resample("ME").last().dropna()
        monthly_ret = monthly_close.pct_change().dropna()
        for date, ret in monthly_ret.items():
            all_monthly.append({
                "Ticker": ticker,
                "Name": name,
                "Sector": sector,
                "Month": date.strftime("%Y-%m"),
                "Return%": round(ret * 100, 2),
            })

        # --- Yearly (full period) return ---
        yearly_ret = (close.iloc[-1] / close.iloc[0] - 1) * 100
        all_yearly.append({
            "Ticker": ticker,
            "Name": name,
            "Sector": sector,
            "Start_Price": round(close.iloc[0], 2),
            "End_Price": round(close.iloc[-1], 2),
            "Yearly_Return%": round(yearly_ret, 2),
        })

    daily_df = pd.DataFrame(all_daily)
    monthly_df = pd.DataFrame(all_monthly)
    yearly_df = pd.DataFrame(all_yearly)

    return daily_df, monthly_df, yearly_df


# ──────────────────────────────────────────────
# 2. Print ranking tables
# ──────────────────────────────────────────────
def print_rankings(daily_df, monthly_df, yearly_df):

    # --- Daily return summary ---
    print("\n" + "=" * 70)
    print("DAILY RETURN SUMMARY (all stocks)")
    print("=" * 70)
    print(daily_df.to_string(index=False))

    # --- Daily return ranking by mean ---
    print("\n" + "=" * 70)
    print("DAILY RETURN RANKING (by mean daily return)")
    print("=" * 70)
    ranked = daily_df.sort_values("Mean%", ascending=False).reset_index(drop=True)
    ranked.index = ranked.index + 1
    ranked.index.name = "Rank"
    print(ranked[["Ticker", "Name", "Sector", "Mean%", "Std%", "Win_Rate%"]].to_string())

    # --- Monthly return pivot ---
    print("\n" + "=" * 70)
    print("MONTHLY RETURN TABLE (%) — each stock by month")
    print("=" * 70)
    monthly_pivot = monthly_df.pivot_table(
        index=["Ticker", "Name", "Sector"],
        columns="Month",
        values="Return%",
    )
    monthly_pivot["Total%"] = monthly_pivot.sum(axis=1).round(2)
    monthly_pivot = monthly_pivot.sort_values("Total%", ascending=False)
    print(monthly_pivot.to_string())

    # --- Monthly return ranking (average monthly return) ---
    print("\n" + "=" * 70)
    print("MONTHLY RETURN RANKING (by average monthly return)")
    print("=" * 70)
    monthly_avg = monthly_df.groupby(["Ticker", "Name", "Sector"])["Return%"].agg(
        Avg_Monthly="mean",
        Best_Month="max",
        Worst_Month="min",
        Positive_Months=lambda x: (x > 0).sum(),
        Total_Months="count",
    ).round(2).reset_index()
    monthly_avg = monthly_avg.sort_values("Avg_Monthly", ascending=False).reset_index(drop=True)
    monthly_avg.index = monthly_avg.index + 1
    monthly_avg.index.name = "Rank"
    print(monthly_avg.to_string())

    # --- Yearly return ranking ---
    print("\n" + "=" * 70)
    print("YEARLY RETURN RANKING")
    print("=" * 70)
    yearly_ranked = yearly_df.sort_values("Yearly_Return%", ascending=False).reset_index(drop=True)
    yearly_ranked.index = yearly_ranked.index + 1
    yearly_ranked.index.name = "Rank"
    print(yearly_ranked.to_string())

    return monthly_pivot, monthly_avg, yearly_ranked


# ──────────────────────────────────────────────
# 3. Save all tables to CSV
# ──────────────────────────────────────────────
def save_tables(daily_df, monthly_df, monthly_pivot, yearly_df):
    daily_path = CHART_DIR / "daily_return_summary.csv"
    monthly_detail_path = CHART_DIR / "monthly_return_detail.csv"
    monthly_pivot_path = CHART_DIR / "monthly_return_pivot.csv"
    yearly_path = CHART_DIR / "yearly_return_ranking.csv"

    daily_df.sort_values("Mean%", ascending=False).to_csv(daily_path, index=False)
    monthly_df.to_csv(monthly_detail_path, index=False)
    monthly_pivot.to_csv(monthly_pivot_path)
    yearly_df.sort_values("Yearly_Return%", ascending=False).to_csv(yearly_path, index=False)

    print("\n" + "=" * 70)
    print("CSV files saved:")
    for p in [daily_path, monthly_detail_path, monthly_pivot_path, yearly_path]:
        print(f"  {p}")
    print("=" * 70)


# ──────────────────────────────────────────────
# 4. Bar chart — yearly return ranking
# ──────────────────────────────────────────────
def plot_yearly_ranking(yearly_df):
    print("\n" + "=" * 70)
    print("Generating yearly return ranking bar chart...")
    print("=" * 70)

    ranked = yearly_df.sort_values("Yearly_Return%", ascending=True)
    colors = [SECTOR_COLORS.get(s, "#999") for s in ranked["Sector"]]
    labels = [t.replace("^", "") for t in ranked["Ticker"]]

    fig, ax = plt.subplots(figsize=(14, 16))
    bars = ax.barh(range(len(ranked)), ranked["Yearly_Return%"], color=colors, edgecolor="white", height=0.7)

    for i, (val, label) in enumerate(zip(ranked["Yearly_Return%"], labels)):
        offset = 5 if val >= 0 else -5
        ha = "left" if val >= 0 else "right"
        ax.text(val + offset, i, f"{val:+.1f}%", va="center", ha=ha, fontsize=8)

    ax.set_yticks(range(len(ranked)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.set_xlabel("1-Year Return (%)", fontsize=12)
    ax.set_title("Yearly Return Ranking — All Stocks", fontsize=14)
    ax.grid(True, axis="x", alpha=0.3)

    from matplotlib.patches import Patch
    legend_items = [Patch(facecolor=c, label=s) for s, c in SECTOR_COLORS.items()]
    ax.legend(handles=legend_items, fontsize=10, loc="lower right")

    fig.tight_layout()
    save_path = CHART_DIR / "yearly_return_ranking.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {save_path}")


# ──────────────────────────────────────────────
# 5. Heatmap — monthly returns for all stocks
# ──────────────────────────────────────────────
def plot_monthly_heatmap(monthly_pivot):
    print("\n" + "=" * 70)
    print("Generating monthly return heatmap...")
    print("=" * 70)

    data = monthly_pivot.drop(columns=["Total%"], errors="ignore")
    labels = [f"{t} ({n})" for t, n, _ in data.index]

    fig, ax = plt.subplots(figsize=(18, 16))
    im = ax.imshow(data.values, aspect="auto", cmap="RdYlGn", vmin=-30, vmax=30)

    ax.set_xticks(range(len(data.columns)))
    ax.set_xticklabels(data.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data.values[i, j]
            if pd.notna(val):
                ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=6,
                        color="white" if abs(val) > 20 else "black")

    cbar = fig.colorbar(im, ax=ax, shrink=0.6, label="Monthly Return (%)")
    ax.set_title("Monthly Return Heatmap — All Stocks", fontsize=14)

    fig.tight_layout()
    save_path = CHART_DIR / "monthly_return_heatmap.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {save_path}")


if __name__ == "__main__":
    daily_df, monthly_df, yearly_df = compute_returns()
    monthly_pivot, monthly_avg, yearly_ranked = print_rankings(daily_df, monthly_df, yearly_df)
    save_tables(daily_df, monthly_df, monthly_pivot, yearly_df)
    plot_yearly_ranking(yearly_df)
    plot_monthly_heatmap(monthly_pivot)

    print("\n" + "=" * 70)
    print("Step 3 complete.")
    print("=" * 70)
