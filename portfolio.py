import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
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
    return pd.DataFrame(ret).dropna()


def build_close() -> pd.DataFrame:
    close = {}
    for _, meta in stocks_only.iterrows():
        close[meta["Ticker"]] = load_close(meta["File"])
    return pd.DataFrame(close).dropna()


# ══════════════════════════════════════════════
# Step 12: Efficient Frontier & Portfolio Optimization
# ══════════════════════════════════════════════
def step12_efficient_frontier():
    print("=" * 70)
    print("STEP 12: EFFICIENT FRONTIER & PORTFOLIO OPTIMIZATION")
    print("=" * 70)

    out_dir = CHART_BASE / "Part6_Portfolio" / "Step12_EfficientFrontier"
    out_dir.mkdir(parents=True, exist_ok=True)

    ret_df = build_returns()
    tickers = ret_df.columns.tolist()
    n = len(tickers)

    mean_ret = ret_df.mean() * 252
    cov_matrix = ret_df.cov() * 252

    print(f"\n>>> Universe: {n} stocks <<<")
    print(f"  Expected return range: {mean_ret.min()*100:.1f}% to {mean_ret.max()*100:.1f}%")
    vols = np.sqrt(np.diag(cov_matrix)) * 100
    print(f"  Annualized vol range:  {vols.min():.1f}% to {vols.max():.1f}%")

    def portfolio_stats(weights):
        port_ret = np.dot(weights, mean_ret)
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe = (port_ret - RF_ANNUAL) / port_vol
        return port_ret, port_vol, sharpe

    def neg_sharpe(weights):
        return -portfolio_stats(weights)[2]

    def portfolio_vol(weights):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    bounds = tuple((0, 0.15) for _ in range(n))
    init_weights = np.array([1 / n] * n)

    # --- Min variance portfolio ---
    min_var = minimize(portfolio_vol, init_weights, method="SLSQP",
                       bounds=bounds, constraints=constraints)
    min_var_ret, min_var_vol, min_var_sharpe = portfolio_stats(min_var.x)

    print(f"\n>>> MINIMUM VARIANCE PORTFOLIO <<<")
    print(f"  Return: {min_var_ret*100:.2f}%  |  Vol: {min_var_vol*100:.2f}%  |  Sharpe: {min_var_sharpe:.2f}")
    mv_weights = pd.Series(min_var.x, index=tickers)
    mv_top = mv_weights[mv_weights > 0.01].sort_values(ascending=False)
    print(f"  Top holdings ({len(mv_top)} stocks with >1% weight):")
    for t, w in mv_top.items():
        sector = stocks_only[stocks_only["Ticker"] == t]["Sector"].values[0]
        print(f"    {t:6s} ({sector:10s}): {w*100:5.1f}%")

    # --- Max Sharpe portfolio ---
    max_sharpe = minimize(neg_sharpe, init_weights, method="SLSQP",
                          bounds=bounds, constraints=constraints)
    ms_ret, ms_vol, ms_sharpe = portfolio_stats(max_sharpe.x)

    print(f"\n>>> MAXIMUM SHARPE PORTFOLIO <<<")
    print(f"  Return: {ms_ret*100:.2f}%  |  Vol: {ms_vol*100:.2f}%  |  Sharpe: {ms_sharpe:.2f}")
    ms_weights = pd.Series(max_sharpe.x, index=tickers)
    ms_top = ms_weights[ms_weights > 0.01].sort_values(ascending=False)
    print(f"  Top holdings ({len(ms_top)} stocks with >1% weight):")
    for t, w in ms_top.items():
        sector = stocks_only[stocks_only["Ticker"] == t]["Sector"].values[0]
        print(f"    {t:6s} ({sector:10s}): {w*100:5.1f}%")

    # --- Equal weight portfolio ---
    eq_weights = np.array([1 / n] * n)
    eq_ret, eq_vol, eq_sharpe = portfolio_stats(eq_weights)

    print(f"\n>>> EQUAL WEIGHT PORTFOLIO (benchmark) <<<")
    print(f"  Return: {eq_ret*100:.2f}%  |  Vol: {eq_vol*100:.2f}%  |  Sharpe: {eq_sharpe:.2f}")

    # --- Generate efficient frontier ---
    target_returns = np.linspace(mean_ret.min(), mean_ret.max(), 100)
    frontier_vols = []
    frontier_rets = []

    for target in target_returns:
        cons = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {"type": "eq", "fun": lambda w, t=target: np.dot(w, mean_ret) - t},
        ]
        result = minimize(portfolio_vol, init_weights, method="SLSQP",
                          bounds=bounds, constraints=cons)
        if result.success:
            frontier_vols.append(portfolio_vol(result.x) * 100)
            frontier_rets.append(target * 100)

    # --- 12a. Plot efficient frontier ---
    fig, ax = plt.subplots(figsize=(14, 9))

    for sector in ["AI", "Chip", "NewEnergy"]:
        sector_tickers = stocks_only[stocks_only["Sector"] == sector]["Ticker"].tolist()
        sector_tickers = [t for t in sector_tickers if t in tickers]
        s_vols = [np.sqrt(cov_matrix.loc[t, t]) * 100 for t in sector_tickers]
        s_rets = [mean_ret[t] * 100 for t in sector_tickers]
        ax.scatter(s_vols, s_rets, c=SECTOR_COLORS[sector], s=40, alpha=0.5,
                   edgecolors="white", linewidth=0.5, label=sector, zorder=2)
        for t, v, r in zip(sector_tickers, s_vols, s_rets):
            ax.annotate(t, (v, r), fontsize=6, alpha=0.6, ha="left", va="bottom",
                        xytext=(3, 3), textcoords="offset points")

    ax.plot(frontier_vols, frontier_rets, color="#E91E63", linewidth=2.5,
            label="Efficient Frontier", zorder=4)

    ax.scatter(min_var_vol * 100, min_var_ret * 100, c="blue", s=200,
               marker="*", zorder=5, label=f"Min Variance (Sharpe={min_var_sharpe:.2f})")
    ax.scatter(ms_vol * 100, ms_ret * 100, c="red", s=200,
               marker="*", zorder=5, label=f"Max Sharpe (Sharpe={ms_sharpe:.2f})")
    ax.scatter(eq_vol * 100, eq_ret * 100, c="green", s=200,
               marker="D", zorder=5, label=f"Equal Weight (Sharpe={eq_sharpe:.2f})")

    ax.set_xlabel("Annualized Volatility (%)", fontsize=12)
    ax.set_ylabel("Annualized Return (%)", fontsize=12)
    ax.set_title("Efficient Frontier with Individual Stocks", fontsize=14)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "efficient_frontier.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  [OK] efficient_frontier.png")

    # --- 12b. Optimal weights bar chart ---
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    for ax, weights, title in zip(axes,
                                   [ms_weights, mv_weights],
                                   ["Max Sharpe Portfolio", "Min Variance Portfolio"]):
        top = weights[weights > 0.005].sort_values(ascending=True)
        colors = [SECTOR_COLORS.get(
            stocks_only[stocks_only["Ticker"] == t]["Sector"].values[0], "#999")
            for t in top.index]
        y_pos = range(len(top))
        ax.barh(y_pos, top.values * 100, color=colors, edgecolor="white",
                height=0.7, alpha=0.85)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top.index, fontsize=9)
        ax.set_xlabel("Weight (%)", fontsize=11)
        ax.set_title(title, fontsize=13)
        ax.grid(True, axis="x", alpha=0.3)
        for i, v in enumerate(top.values):
            ax.text(v * 100 + 0.3, i, f"{v*100:.1f}%", va="center", fontsize=8)

    fig.suptitle("Optimal Portfolio Weights", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "optimal_weights.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] optimal_weights.png")

    # --- 12c. Sector allocation comparison ---
    alloc_records = []
    for label, w in [("Max Sharpe", ms_weights), ("Min Variance", mv_weights),
                     ("Equal Weight", pd.Series(eq_weights, index=tickers))]:
        for sector in ["AI", "Chip", "NewEnergy"]:
            sector_tickers = stocks_only[stocks_only["Sector"] == sector]["Ticker"].tolist()
            sector_tickers = [t for t in sector_tickers if t in tickers]
            sector_weight = w[sector_tickers].sum()
            alloc_records.append({
                "Portfolio": label,
                "Sector": sector,
                "Weight%": round(sector_weight * 100, 1),
            })

    alloc_df = pd.DataFrame(alloc_records)
    pivot = alloc_df.pivot(index="Portfolio", columns="Sector", values="Weight%")
    print("\n>>> SECTOR ALLOCATION <<<")
    print(pivot.to_string())

    fig, ax = plt.subplots(figsize=(10, 6))
    pivot.plot(kind="bar", ax=ax, color=[SECTOR_COLORS[s] for s in pivot.columns],
               edgecolor="white", width=0.7)
    ax.set_xlabel("")
    ax.set_ylabel("Weight (%)", fontsize=12)
    ax.set_title("Sector Allocation by Portfolio Strategy", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    fig.tight_layout()
    fig.savefig(out_dir / "sector_allocation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] sector_allocation.png")

    weight_df = pd.DataFrame({
        "Ticker": tickers,
        "MaxSharpe_Weight%": (ms_weights * 100).round(2).values,
        "MinVar_Weight%": (mv_weights * 100).round(2).values,
        "EqualWeight%": round(100 / n, 2),
    })
    weight_df = weight_df.merge(stocks_only[["Ticker", "Name", "Sector"]], on="Ticker")
    weight_df = weight_df[["Ticker", "Name", "Sector",
                            "MaxSharpe_Weight%", "MinVar_Weight%", "EqualWeight%"]]
    weight_df.to_csv(out_dir / "optimal_weights.csv", index=False)
    print(f"  [OK] optimal_weights.csv")

    return ms_weights, mv_weights, eq_weights, tickers


# ══════════════════════════════════════════════
# Step 13: Portfolio Backtest
# ══════════════════════════════════════════════
def step13_backtest(ms_weights, mv_weights, eq_weights, tickers):
    print("\n" + "=" * 70)
    print("STEP 13: PORTFOLIO BACKTEST")
    print("=" * 70)

    out_dir = CHART_BASE / "Part6_Portfolio" / "Step13_Backtest"
    out_dir.mkdir(parents=True, exist_ok=True)

    ret_df = build_returns()[tickers]

    sp500_close = load_close("GSPC.csv")
    sp500_ret = sp500_close.pct_change().dropna()
    sp500_ret = sp500_ret.reindex(ret_df.index).fillna(0)

    n = len(tickers)
    rf_daily = RF_ANNUAL / 252

    # --- Build portfolio daily returns ---
    portfolios = {}

    portfolios["Equal Weight"] = ret_df.mean(axis=1)

    w_ms = pd.Series(
        ms_weights if isinstance(ms_weights, np.ndarray) else ms_weights.values,
        index=tickers)
    portfolios["Max Sharpe"] = (ret_df * w_ms).sum(axis=1)

    w_mv = pd.Series(
        mv_weights if isinstance(mv_weights, np.ndarray) else mv_weights.values,
        index=tickers)
    portfolios["Min Variance"] = (ret_df * w_mv).sum(axis=1)

    sector_weights = {}
    sectors = stocks_only["Sector"].unique()
    for sector in sectors:
        sector_tickers = stocks_only[stocks_only["Sector"] == sector]["Ticker"].tolist()
        sector_tickers = [t for t in sector_tickers if t in tickers]
        weight_per_stock = (1 / len(sectors)) / len(sector_tickers) if sector_tickers else 0
        for t in sector_tickers:
            sector_weights[t] = weight_per_stock
    w_sector = pd.Series(sector_weights).reindex(tickers).fillna(0)
    portfolios["Sector Equal"] = (ret_df * w_sector).sum(axis=1)

    portfolios["S&P 500"] = sp500_ret

    # --- Cumulative returns ---
    cum_ret = pd.DataFrame()
    for name, ret_series in portfolios.items():
        cum_ret[name] = (1 + ret_series).cumprod()

    print("\n>>> CUMULATIVE RETURN (final) <<<")
    for col in cum_ret.columns:
        total_ret = (cum_ret[col].iloc[-1] - 1) * 100
        print(f"  {col:15s}: {total_ret:+.2f}%")

    # --- Performance summary ---
    perf_records = []
    for name, ret_series in portfolios.items():
        total_ret = cum_ret[name].iloc[-1] - 1
        ann_ret = ret_series.mean() * 252
        ann_vol = ret_series.std() * np.sqrt(252)
        sharpe = (ret_series.mean() - rf_daily) / ret_series.std() * np.sqrt(252) if ret_series.std() > 0 else 0

        downside = ret_series[ret_series < rf_daily]
        sortino = (ann_ret - RF_ANNUAL) / (downside.std() * np.sqrt(252)) if downside.std() > 0 else 0

        cum = (1 + ret_series).cumprod()
        max_dd = ((cum - cum.cummax()) / cum.cummax()).min()
        calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0

        win_rate = (ret_series > 0).mean()

        perf_records.append({
            "Portfolio": name,
            "Total_Return%": round(total_ret * 100, 2),
            "Ann_Return%": round(ann_ret * 100, 2),
            "Ann_Vol%": round(ann_vol * 100, 2),
            "Sharpe": round(sharpe, 2),
            "Sortino": round(sortino, 2),
            "Max_DD%": round(max_dd * 100, 2),
            "Calmar": round(calmar, 2),
            "Win_Rate%": round(win_rate * 100, 1),
        })

    perf_df = pd.DataFrame(perf_records)
    print("\n>>> PORTFOLIO PERFORMANCE COMPARISON <<<")
    print(perf_df.to_string(index=False))

    # --- Chart colors ---
    PORT_COLORS = {
        "Equal Weight": "#2196F3",
        "Max Sharpe": "#E91E63",
        "Min Variance": "#FF9800",
        "Sector Equal": "#4CAF50",
        "S&P 500": "#9E9E9E",
    }

    # --- 13a. Cumulative return chart ---
    fig, ax = plt.subplots(figsize=(16, 8))
    for col in cum_ret.columns:
        style = "--" if col == "S&P 500" else "-"
        width = 1.5 if col == "S&P 500" else 2.0
        ax.plot(cum_ret.index, (cum_ret[col] - 1) * 100,
                linewidth=width, label=col, color=PORT_COLORS.get(col, "#999"),
                linestyle=style)

    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Cumulative Return (%)", fontsize=11)
    ax.set_title("Portfolio Backtest — Cumulative Returns", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "cumulative_returns.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  [OK] cumulative_returns.png")

    # --- 13b. Rolling 30-day Sharpe ---
    fig, ax = plt.subplots(figsize=(16, 7))
    for name, ret_series in portfolios.items():
        rolling_sharpe = ((ret_series.rolling(30).mean() - rf_daily)
                          / ret_series.rolling(30).std() * np.sqrt(252))
        style = "--" if name == "S&P 500" else "-"
        ax.plot(rolling_sharpe.index, rolling_sharpe.values,
                linewidth=1.5, label=name, color=PORT_COLORS.get(name, "#999"),
                linestyle=style, alpha=0.8)

    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Rolling 30-Day Sharpe Ratio", fontsize=11)
    ax.set_title("Rolling Sharpe Ratio Comparison", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "rolling_sharpe.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] rolling_sharpe.png")

    # --- 13c. Drawdown comparison ---
    fig, ax = plt.subplots(figsize=(16, 7))
    for name, ret_series in portfolios.items():
        cum = (1 + ret_series).cumprod()
        dd = ((cum - cum.cummax()) / cum.cummax()) * 100
        style = "--" if name == "S&P 500" else "-"
        ax.plot(dd.index, dd.values, linewidth=1.5, label=name,
                color=PORT_COLORS.get(name, "#999"), linestyle=style, alpha=0.8)

    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Drawdown (%)", fontsize=11)
    ax.set_title("Portfolio Drawdown Comparison", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "drawdown_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] drawdown_comparison.png")

    # --- 13d. Monthly returns heatmap for Max Sharpe portfolio ---
    ms_monthly = (portfolios["Max Sharpe"]
                  .resample("ME")
                  .apply(lambda x: (1 + x).prod() - 1) * 100)
    ms_monthly_df = pd.DataFrame({
        "Year": ms_monthly.index.year,
        "Month": ms_monthly.index.month,
        "Return%": ms_monthly.values,
    })
    pivot = ms_monthly_df.pivot(index="Year", columns="Month", values="Return%")

    fig, ax = plt.subplots(figsize=(14, 4))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="RdYlGn", center=0,
                linewidths=0.5, ax=ax, annot_kws={"size": 9},
                cbar_kws={"label": "Monthly Return (%)", "shrink": 0.8})
    ax.set_title("Max Sharpe Portfolio — Monthly Returns (%)", fontsize=13)
    ax.set_xlabel("Month", fontsize=11)
    ax.set_ylabel("Year", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "monthly_returns_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] monthly_returns_heatmap.png")

    perf_df.to_csv(out_dir / "portfolio_performance.csv", index=False)
    print(f"  [OK] portfolio_performance.csv")

    return perf_df


if __name__ == "__main__":
    ms_weights, mv_weights, eq_weights, tickers = step12_efficient_frontier()
    step13_backtest(ms_weights, mv_weights, eq_weights, tickers)

    print("\n" + "=" * 70)
    print("Portfolio analysis complete.")
    print("=" * 70)
