import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).parent / "data"
CHART_BASE = Path(__file__).parent / "charts"

summary = pd.read_csv(DATA_DIR / "_summary.csv")
stocks_only = summary[summary["Sector"] != "Index"]

SECTOR_COLORS = {
    "AI": "#2196F3",
    "Chip": "#FF9800",
    "NewEnergy": "#4CAF50",
}

RF_ANNUAL = 0.05


def load_close(file_name: str) -> pd.Series:
    df = pd.read_csv(DATA_DIR / file_name, parse_dates=["Date"], index_col="Date")
    return df["Close"]


def build_returns() -> pd.DataFrame:
    ret = {}
    for _, meta in stocks_only.iterrows():
        close = load_close(meta["File"])
        ret[meta["Ticker"]] = close.pct_change().dropna()
    return pd.DataFrame(ret)


def build_close() -> pd.DataFrame:
    close = {}
    for _, meta in stocks_only.iterrows():
        close[meta["Ticker"]] = load_close(meta["File"])
    return pd.DataFrame(close)


# ══════════════════════════════════════════════
# Step 10: Drawdown Analysis
# ══════════════════════════════════════════════
def step10_drawdown():
    print("=" * 70)
    print("STEP 10: DRAWDOWN ANALYSIS")
    print("=" * 70)

    out_dir = CHART_BASE / "Part5_Risk" / "Step10_Drawdown"
    out_dir.mkdir(parents=True, exist_ok=True)

    close_df = build_close()

    drawdown_df = pd.DataFrame()
    records = []

    for _, meta in stocks_only.iterrows():
        ticker = meta["Ticker"]
        close = close_df[ticker].dropna()

        cummax = close.cummax()
        drawdown = (close - cummax) / cummax
        drawdown_df[ticker] = drawdown

        max_dd = drawdown.min()
        trough_date = drawdown.idxmin()
        peak_date = close[:trough_date].idxmax()

        recovery = close[trough_date:]
        recovered = recovery[recovery >= close[peak_date]]
        recovery_date = recovered.index[0] if len(recovered) > 0 else None
        recovery_days = (recovery_date - trough_date).days if recovery_date else None

        records.append({
            "Ticker": ticker,
            "Name": meta["Name"],
            "Sector": meta["Sector"],
            "Max_Drawdown%": round(max_dd * 100, 2),
            "Peak_Date": peak_date.strftime("%Y-%m-%d"),
            "Trough_Date": trough_date.strftime("%Y-%m-%d"),
            "Drawdown_Days": (trough_date - peak_date).days,
            "Recovery_Date": recovery_date.strftime("%Y-%m-%d") if recovery_date else "Not recovered",
            "Recovery_Days": recovery_days if recovery_days else "N/A",
        })

    dd_df = pd.DataFrame(records).sort_values("Max_Drawdown%").reset_index(drop=True)
    dd_df.index = dd_df.index + 1
    dd_df.index.name = "Rank"

    print("\n>>> MAX DRAWDOWN RANKING (worst to best) <<<")
    print(dd_df.to_string())

    dd_raw = pd.DataFrame(records)
    sector_dd = dd_raw.groupby("Sector")["Max_Drawdown%"].agg(["mean", "min", "max"]).round(2)
    sector_dd.columns = ["Avg_MaxDD%", "Worst_MaxDD%", "Best_MaxDD%"]
    print("\n>>> SECTOR DRAWDOWN SUMMARY <<<")
    print(sector_dd.to_string())

    # --- 10a. Underwater chart: top 10 worst drawdown stocks ---
    worst_10 = dd_raw.nsmallest(10, "Max_Drawdown%")["Ticker"].tolist()

    fig, ax = plt.subplots(figsize=(16, 8))
    for ticker in worst_10:
        sector = stocks_only[stocks_only["Ticker"] == ticker]["Sector"].values[0]
        color = SECTOR_COLORS.get(sector, "#999")
        ax.fill_between(drawdown_df.index, drawdown_df[ticker] * 100, 0,
                        alpha=0.12, color=color)
        ax.plot(drawdown_df.index, drawdown_df[ticker] * 100,
                linewidth=1.2, label=ticker, color=color, alpha=0.8)

    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Drawdown (%)", fontsize=11)
    ax.set_title("Underwater Chart — Top 10 Deepest Drawdowns", fontsize=14)
    ax.legend(fontsize=9, loc="lower left", ncol=2)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="black", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(out_dir / "underwater_top10.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  [OK] underwater_top10.png")

    # --- 10b. Sector-level underwater chart ---
    fig, axes = plt.subplots(3, 1, figsize=(16, 14), sharex=True)
    for ax, sector in zip(axes, ["AI", "Chip", "NewEnergy"]):
        tickers = stocks_only[stocks_only["Sector"] == sector]["Ticker"].tolist()
        for t in tickers:
            ax.plot(drawdown_df.index, drawdown_df[t] * 100,
                    linewidth=1, alpha=0.6, label=t)
        sector_avg = drawdown_df[tickers].mean(axis=1) * 100
        ax.plot(drawdown_df.index, sector_avg, linewidth=2.5, color="black",
                linestyle="--", label=f"{sector} avg", alpha=0.9)
        ax.set_ylabel("Drawdown (%)", fontsize=10)
        ax.set_title(f"{sector} Sector — Drawdown", fontsize=12)
        ax.legend(fontsize=7, loc="lower left", ncol=4)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color="black", linewidth=0.5)

    axes[-1].set_xlabel("Date", fontsize=11)
    fig.suptitle("Drawdown by Sector", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / "drawdown_by_sector.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] drawdown_by_sector.png")

    # --- 10c. Drawdown duration vs depth scatter ---
    fig, ax = plt.subplots(figsize=(14, 9))
    for sector in ["AI", "Chip", "NewEnergy"]:
        sub = dd_raw[dd_raw["Sector"] == sector]
        ax.scatter(sub["Drawdown_Days"], sub["Max_Drawdown%"].abs(),
                   c=SECTOR_COLORS[sector], s=100, alpha=0.85,
                   edgecolors="white", linewidth=0.8, label=sector, zorder=3)
        for _, row in sub.iterrows():
            ax.annotate(row["Ticker"],
                        (row["Drawdown_Days"], abs(row["Max_Drawdown%"])),
                        fontsize=8, fontweight="bold",
                        ha="left", va="bottom",
                        xytext=(5, 5), textcoords="offset points")

    ax.set_xlabel("Drawdown Duration (days)", fontsize=12)
    ax.set_ylabel("Max Drawdown Depth (%)", fontsize=12)
    ax.set_title("Drawdown Depth vs Duration", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "drawdown_depth_vs_duration.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] drawdown_depth_vs_duration.png")

    dd_raw.to_csv(out_dir / "drawdown_analysis.csv", index=False)
    print(f"  [OK] drawdown_analysis.csv")

    return dd_raw


# ══════════════════════════════════════════════
# Step 11: Risk Metrics — VaR, Sharpe, Sortino, Calmar
# ══════════════════════════════════════════════
def step11_risk_metrics():
    print("\n" + "=" * 70)
    print("STEP 11: RISK METRICS — VaR, SHARPE, SORTINO, CALMAR")
    print("=" * 70)

    out_dir = CHART_BASE / "Part5_Risk" / "Step11_RiskMetrics"
    out_dir.mkdir(parents=True, exist_ok=True)

    ret_df = build_returns()
    close_df = build_close()

    sp500_close = load_close("GSPC.csv")
    sp500_ret = sp500_close.pct_change().dropna()
    rf_daily = RF_ANNUAL / 252

    records = []
    for _, meta in stocks_only.iterrows():
        ticker = meta["Ticker"]
        ret = ret_df[ticker].dropna()
        close = close_df[ticker].dropna()

        ann_ret = ret.mean() * 252
        ann_vol = ret.std() * np.sqrt(252)

        cummax = close.cummax()
        max_dd = ((close - cummax) / cummax).min()

        var_95_hist = ret.quantile(0.05)
        var_99_hist = ret.quantile(0.01)
        var_95_param = ret.mean() - 1.645 * ret.std()
        var_99_param = ret.mean() - 2.326 * ret.std()

        excess = ret - rf_daily
        sharpe = excess.mean() / ret.std() * np.sqrt(252) if ret.std() > 0 else 0

        downside = ret[ret < rf_daily]
        downside_std = downside.std() * np.sqrt(252)
        sortino = (ann_ret - RF_ANNUAL) / downside_std if downside_std > 0 else 0

        calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0

        aligned = pd.concat([ret, sp500_ret], axis=1, join="inner")
        aligned.columns = ["stock", "market"]
        cov = aligned.cov().iloc[0, 1]
        var_market = aligned["market"].var()
        beta = cov / var_market if var_market > 0 else 0

        treynor = (ann_ret - RF_ANNUAL) / beta if beta != 0 else 0

        records.append({
            "Ticker": ticker,
            "Name": meta["Name"],
            "Sector": meta["Sector"],
            "Ann_Return%": round(ann_ret * 100, 2),
            "Ann_Vol%": round(ann_vol * 100, 2),
            "Max_DD%": round(max_dd * 100, 2),
            "VaR_95_Hist%": round(var_95_hist * 100, 2),
            "VaR_99_Hist%": round(var_99_hist * 100, 2),
            "VaR_95_Param%": round(var_95_param * 100, 2),
            "VaR_99_Param%": round(var_99_param * 100, 2),
            "Sharpe": round(sharpe, 2),
            "Sortino": round(sortino, 2),
            "Calmar": round(calmar, 2),
            "Beta": round(beta, 2),
            "Treynor": round(treynor, 2),
        })

    risk_df = pd.DataFrame(records)

    print("\n>>> RISK METRICS (sorted by Sharpe) <<<")
    sorted_risk = risk_df.sort_values("Sharpe", ascending=False).reset_index(drop=True)
    sorted_risk.index = sorted_risk.index + 1
    sorted_risk.index.name = "Rank"
    print(sorted_risk[["Ticker", "Name", "Sector", "Ann_Return%", "Ann_Vol%",
                        "Sharpe", "Sortino", "Calmar", "Beta"]].to_string())

    print("\n>>> VaR COMPARISON <<<")
    var_sorted = risk_df.sort_values("VaR_95_Hist%").reset_index(drop=True)
    var_sorted.index = var_sorted.index + 1
    var_sorted.index.name = "Rank"
    print(var_sorted[["Ticker", "Sector", "VaR_95_Hist%", "VaR_99_Hist%",
                       "VaR_95_Param%", "VaR_99_Param%"]].to_string())

    sector_risk = risk_df.groupby("Sector").agg({
        "Ann_Return%": "mean",
        "Ann_Vol%": "mean",
        "Sharpe": "mean",
        "Sortino": "mean",
        "Max_DD%": "mean",
        "VaR_95_Hist%": "mean",
        "Beta": "mean",
    }).round(2)
    print("\n>>> SECTOR RISK SUMMARY <<<")
    print(sector_risk.to_string())

    # --- 11a. Risk-return quadrant chart ---
    fig, ax = plt.subplots(figsize=(14, 9))

    for sector in ["AI", "Chip", "NewEnergy"]:
        sub = risk_df[risk_df["Sector"] == sector]
        sizes = sub["Sharpe"].clip(lower=0) * 60 + 30
        ax.scatter(sub["Ann_Vol%"], sub["Ann_Return%"],
                   c=SECTOR_COLORS[sector], s=sizes, alpha=0.85,
                   edgecolors="white", linewidth=0.8, label=sector, zorder=3)
        for _, row in sub.iterrows():
            ax.annotate(row["Ticker"],
                        (row["Ann_Vol%"], row["Ann_Return%"]),
                        fontsize=8, fontweight="bold",
                        ha="left", va="bottom",
                        xytext=(5, 5), textcoords="offset points")

    med_vol = risk_df["Ann_Vol%"].median()
    med_ret = risk_df["Ann_Return%"].median()
    ax.axvline(x=med_vol, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.axhline(y=med_ret, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    ax.set_xlabel("Annualized Volatility (%)", fontsize=12)
    ax.set_ylabel("Annualized Return (%)", fontsize=12)
    ax.set_title("Risk-Return Quadrant (bubble size = Sharpe ratio)", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "risk_return_quadrant.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  [OK] risk_return_quadrant.png")

    # --- 11b. VaR comparison bar chart ---
    fig, ax = plt.subplots(figsize=(16, 9))
    var_plot = risk_df.sort_values("VaR_95_Hist%")
    colors = [SECTOR_COLORS[s] for s in var_plot["Sector"]]
    y_pos = range(len(var_plot))

    ax.barh(y_pos, var_plot["VaR_95_Hist%"].abs(), color=colors, alpha=0.7,
            edgecolor="white", height=0.6, label="95% VaR")
    ax.barh(y_pos, var_plot["VaR_99_Hist%"].abs(), color=colors, alpha=0.35,
            edgecolor="white", height=0.6, label="99% VaR")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(var_plot["Ticker"], fontsize=9)
    ax.set_xlabel("Daily VaR (% loss)", fontsize=12)
    ax.set_title("Value at Risk — 95% and 99% Historical VaR", fontsize=14)
    ax.grid(True, axis="x", alpha=0.3)

    from matplotlib.patches import Patch
    legend_items = [Patch(facecolor=c, label=s) for s, c in SECTOR_COLORS.items()]
    legend_items += [Patch(facecolor="gray", alpha=0.7, label="95% VaR"),
                     Patch(facecolor="gray", alpha=0.35, label="99% VaR")]
    ax.legend(handles=legend_items, fontsize=10, loc="lower right")

    fig.tight_layout()
    fig.savefig(out_dir / "var_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] var_comparison.png")

    # --- 11c. Sharpe / Sortino / Calmar comparison ---
    fig, axes = plt.subplots(1, 3, figsize=(20, 8))

    for ax, metric, title in zip(axes, ["Sharpe", "Sortino", "Calmar"],
                                  ["Sharpe Ratio", "Sortino Ratio", "Calmar Ratio"]):
        plot_data = risk_df.sort_values(metric, ascending=True)
        colors = [SECTOR_COLORS[s] for s in plot_data["Sector"]]
        y_pos = range(len(plot_data))
        ax.barh(y_pos, plot_data[metric], color=colors, edgecolor="white",
                height=0.7, alpha=0.85)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(plot_data["Ticker"], fontsize=8)
        ax.set_title(title, fontsize=13)
        ax.axvline(x=0, color="black", linewidth=0.8)
        ax.grid(True, axis="x", alpha=0.3)

    fig.suptitle("Risk-Adjusted Return Metrics Comparison", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "risk_ratios_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] risk_ratios_comparison.png")

    risk_df.to_csv(out_dir / "risk_metrics.csv", index=False)
    print(f"  [OK] risk_metrics.csv")

    return risk_df


if __name__ == "__main__":
    step10_drawdown()
    step11_risk_metrics()

    print("\n" + "=" * 70)
    print("Risk analysis complete.")
    print("=" * 70)
