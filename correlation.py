import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).parent / "data"
CHART_DIR = Path(__file__).parent / "charts"
CHART_DIR.mkdir(exist_ok=True)

summary = pd.read_csv(DATA_DIR / "_summary.csv")

AI_CHIP = summary[summary["Sector"].isin(["AI", "Chip"])]
SP500_FILE = "GSPC.csv"


def load_close(file_name: str) -> pd.Series:
    df = pd.read_csv(DATA_DIR / file_name, parse_dates=["Date"], index_col="Date")
    return df["Close"]


def build_returns_df(stock_df: pd.DataFrame) -> pd.DataFrame:
    returns = {}
    for _, meta in stock_df.iterrows():
        close = load_close(meta["File"])
        returns[meta["Ticker"]] = close.pct_change().dropna()
    return pd.DataFrame(returns)


# ══════════════════════════════════════════════
# Step 6: AI & Chip Correlation Analysis
# ══════════════════════════════════════════════
def step6_correlation():
    print("=" * 70)
    print("STEP 6: AI & CHIP CORRELATION ANALYSIS")
    print("=" * 70)

    ret_df = build_returns_df(AI_CHIP)
    corr = ret_df.corr()

    # --- 6a. Full heatmap ---
    fig, ax = plt.subplots(figsize=(16, 13))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn", center=0,
                mask=mask, square=True, linewidths=0.5,
                vmin=-0.2, vmax=0.7,
                annot_kws={"size": 8}, ax=ax,
                cbar_kws={"label": "Correlation", "shrink": 0.8})
    ax.set_title("AI & Chip Stocks — Daily Return Correlation Matrix", fontsize=14)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "correlation_heatmap_ai_chip.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] correlation_heatmap_ai_chip.png")

    # --- 6b. Top correlated pairs ---
    pairs = []
    tickers = corr.columns.tolist()
    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            t1, t2 = tickers[i], tickers[j]
            s1 = AI_CHIP[AI_CHIP["Ticker"] == t1]["Sector"].values[0]
            s2 = AI_CHIP[AI_CHIP["Ticker"] == t2]["Sector"].values[0]
            pairs.append({
                "Stock_1": t1,
                "Stock_2": t2,
                "Sector_1": s1,
                "Sector_2": s2,
                "Pair_Type": f"{s1}-{s2}" if s1 != s2 else s1,
                "Correlation": round(corr.loc[t1, t2], 4),
            })
    pairs_df = pd.DataFrame(pairs).sort_values("Correlation", ascending=False).reset_index(drop=True)
    pairs_df.index = pairs_df.index + 1
    pairs_df.index.name = "Rank"

    print("\n>>> TOP 15 MOST CORRELATED PAIRS <<<")
    print(pairs_df.head(15).to_string())

    print("\n>>> BOTTOM 10 LEAST CORRELATED PAIRS <<<")
    print(pairs_df.tail(10).to_string())

    # --- 6c. Strongest linkage clusters ---
    print("\n>>> STRONGEST LINKAGE CLUSTERS (corr > 0.45) <<<")
    strong = pairs_df[pairs_df["Correlation"] > 0.45]
    if not strong.empty:
        from collections import defaultdict
        adj = defaultdict(list)
        for _, row in strong.iterrows():
            adj[row["Stock_1"]].append((row["Stock_2"], row["Correlation"]))
            adj[row["Stock_2"]].append((row["Stock_1"], row["Correlation"]))
        print(f"  {len(strong)} pairs with correlation > 0.45:")
        for node in sorted(adj.keys()):
            partners = sorted(adj[node], key=lambda x: -x[1])
            partner_str = ", ".join(f"{p}({c:.2f})" for p, c in partners)
            print(f"    {node}: {partner_str}")

    # --- 6d. NVDA vs AMD deep dive ---
    print("\n" + "=" * 70)
    print("NVDA vs AMD — Competitor Correlation Deep Dive")
    print("=" * 70)

    nvda_ret = ret_df["NVDA"]
    amd_ret = ret_df["AMD"]

    overall_corr = nvda_ret.corr(amd_ret)
    print(f"\n  Overall correlation: {overall_corr:.4f}")

    rolling_30 = nvda_ret.rolling(30).corr(amd_ret)
    rolling_60 = nvda_ret.rolling(60).corr(amd_ret)
    print(f"  30-day rolling corr — mean: {rolling_30.mean():.4f}, min: {rolling_30.min():.4f}, max: {rolling_30.max():.4f}")
    print(f"  60-day rolling corr — mean: {rolling_60.mean():.4f}, min: {rolling_60.min():.4f}, max: {rolling_60.max():.4f}")

    nvda_close = load_close("NVDA.csv")
    amd_close = load_close("AMD.csv")
    nvda_norm = (nvda_close / nvda_close.iloc[0]) * 100
    amd_norm = (amd_close / amd_close.iloc[0]) * 100

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={"height_ratios": [2, 1]}, sharex=True)

    ax1.plot(nvda_norm.index, nvda_norm.values, color="#76B900", linewidth=2, label="NVDA (NVIDIA)")
    ax1.plot(amd_norm.index, amd_norm.values, color="#ED1C24", linewidth=2, label="AMD")
    ax1.set_ylabel("Normalized Price (base=100)", fontsize=11)
    ax1.set_title("NVDA vs AMD — Price Movement & Rolling Correlation", fontsize=13)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    ax2.plot(rolling_30.index, rolling_30.values, color="#2196F3", linewidth=1.5, alpha=0.6, label="30-day rolling")
    ax2.plot(rolling_60.index, rolling_60.values, color="#FF9800", linewidth=2, label="60-day rolling")
    ax2.axhline(y=overall_corr, color="red", linestyle="--", linewidth=1, label=f"Overall ({overall_corr:.2f})")
    ax2.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
    ax2.set_ylabel("Correlation", fontsize=11)
    ax2.legend(fontsize=10)
    ax2.set_xlabel("Date", fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.3, 1.0)

    fig.tight_layout()
    fig.savefig(CHART_DIR / "nvda_vs_amd.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] nvda_vs_amd.png")

    # --- Same-day move agreement ---
    same_dir = ((nvda_ret > 0) & (amd_ret > 0)) | ((nvda_ret < 0) & (amd_ret < 0))
    agreement_pct = same_dir.mean() * 100
    print(f"  Same-direction days: {agreement_pct:.1f}%")

    # --- Scatter plot ---
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.scatter(nvda_ret * 100, amd_ret * 100, alpha=0.4, s=20, c="#2196F3", edgecolors="white", linewidth=0.3)
    z = np.polyfit(nvda_ret, amd_ret, 1)
    p = np.poly1d(z)
    x_line = np.linspace(nvda_ret.min(), nvda_ret.max(), 100)
    ax.plot(x_line * 100, p(x_line) * 100, color="red", linewidth=2,
            label=f"Regression: AMD = {z[0]:.2f} * NVDA + {z[1]*100:.3f}%")
    ax.set_xlabel("NVDA Daily Return (%)", fontsize=11)
    ax.set_ylabel("AMD Daily Return (%)", fontsize=11)
    ax.set_title(f"NVDA vs AMD — Daily Return Scatter (corr={overall_corr:.2f})", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(CHART_DIR / "nvda_vs_amd_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] nvda_vs_amd_scatter.png")

    pairs_df.to_csv(CHART_DIR / "correlation_pairs.csv", index=False)
    print(f"  [OK] correlation_pairs.csv")

    return corr, pairs_df


# ══════════════════════════════════════════════
# Step 7: Beta Analysis vs S&P500
# ══════════════════════════════════════════════
def step7_beta():
    print("\n" + "=" * 70)
    print("STEP 7: BETA ANALYSIS vs S&P500")
    print("=" * 70)

    sp500_close = load_close(SP500_FILE)
    sp500_ret = sp500_close.pct_change().dropna()

    rows = []
    for _, meta in AI_CHIP.iterrows():
        close = load_close(meta["File"])
        stock_ret = close.pct_change().dropna()

        aligned = pd.concat([stock_ret, sp500_ret], axis=1, join="inner")
        aligned.columns = ["Stock", "SP500"]

        corr = aligned["Stock"].corr(aligned["SP500"])
        cov = aligned["Stock"].cov(aligned["SP500"])
        var_sp = aligned["SP500"].var()
        beta = cov / var_sp
        alpha = (aligned["Stock"].mean() - beta * aligned["SP500"].mean()) * 252 * 100

        annual_return = (close.iloc[-1] / close.iloc[0] - 1) * 100

        rows.append({
            "Ticker": meta["Ticker"],
            "Name": meta["Name"],
            "Sector": meta["Sector"],
            "Beta": round(beta, 2),
            "Corr_SP500": round(corr, 4),
            "Alpha%": round(alpha, 2),
            "Annual_Return%": round(annual_return, 2),
        })

    beta_df = pd.DataFrame(rows)

    # --- Sorted by beta ---
    print("\n>>> FULL BETA TABLE (sorted by beta, descending) <<<")
    sorted_beta = beta_df.sort_values("Beta", ascending=False).reset_index(drop=True)
    sorted_beta.index = sorted_beta.index + 1
    sorted_beta.index.name = "Rank"
    print(sorted_beta.to_string())

    # --- Highest beta ---
    print("\n>>> HIGHEST BETA (moves most with market) <<<")
    top_beta = beta_df.nlargest(5, "Beta")
    print(top_beta[["Ticker", "Name", "Sector", "Beta", "Corr_SP500", "Annual_Return%"]].to_string(index=False))

    # --- Lowest beta ---
    print("\n>>> LOWEST BETA (most independent, best diversifier) <<<")
    low_beta = beta_df.nsmallest(5, "Beta")
    print(low_beta[["Ticker", "Name", "Sector", "Beta", "Corr_SP500", "Annual_Return%"]].to_string(index=False))

    # --- Highest correlation with SP500 ---
    print("\n>>> HIGHEST CORRELATION WITH S&P500 <<<")
    top_corr = beta_df.nlargest(5, "Corr_SP500")
    print(top_corr[["Ticker", "Name", "Sector", "Beta", "Corr_SP500"]].to_string(index=False))

    # --- Beta bar chart ---
    plot_df = beta_df.sort_values("Beta", ascending=True)
    colors = [{"AI": "#2196F3", "Chip": "#FF9800"}[s] for s in plot_df["Sector"]]

    fig, ax = plt.subplots(figsize=(12, 10))
    bars = ax.barh(range(len(plot_df)), plot_df["Beta"], color=colors, edgecolor="white", height=0.7)
    ax.axvline(x=1.0, color="red", linestyle="--", linewidth=1.5, label="Beta = 1.0 (market)")

    for i, (val, ticker) in enumerate(zip(plot_df["Beta"], plot_df["Ticker"])):
        offset = 0.03 if val >= 0 else -0.03
        ha = "left" if val >= 0 else "right"
        ax.text(val + offset, i, f"{val:.2f}", va="center", ha=ha, fontsize=9, fontweight="bold")

    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(plot_df["Ticker"], fontsize=10)
    ax.set_xlabel("Beta", fontsize=12)
    ax.set_title("Beta vs S&P500 — AI & Chip Stocks", fontsize=14)
    ax.grid(True, axis="x", alpha=0.3)

    from matplotlib.patches import Patch
    legend_items = [Patch(facecolor="#2196F3", label="AI"),
                    Patch(facecolor="#FF9800", label="Chip")]
    ax.legend(handles=legend_items + [plt.Line2D([0], [0], color="red", linestyle="--", label="Market (Beta=1)")],
              fontsize=10, loc="lower right")

    fig.tight_layout()
    fig.savefig(CHART_DIR / "beta_ranking.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  [OK] beta_ranking.png")

    # --- Beta vs Return scatter ---
    fig, ax = plt.subplots(figsize=(14, 9))
    for sector in ["AI", "Chip"]:
        sub = beta_df[beta_df["Sector"] == sector]
        color = "#2196F3" if sector == "AI" else "#FF9800"
        ax.scatter(sub["Beta"], sub["Annual_Return%"], c=color, s=100, alpha=0.85,
                   edgecolors="white", linewidth=0.8, label=sector, zorder=3)
        for _, row in sub.iterrows():
            ax.annotate(row["Ticker"], (row["Beta"], row["Annual_Return%"]),
                        fontsize=9, fontweight="bold", ha="left", va="bottom",
                        xytext=(5, 5), textcoords="offset points")

    ax.axvline(x=1.0, color="red", linestyle="--", linewidth=1, alpha=0.7, label="Market Beta")
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
    ax.set_xlabel("Beta (vs S&P500)", fontsize=12)
    ax.set_ylabel("1-Year Return (%)", fontsize=12)
    ax.set_title("Beta vs Return — AI & Chip Stocks", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "beta_vs_return.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] beta_vs_return.png")

    # --- Correlation heatmap with SP500 ---
    ret_df = build_returns_df(AI_CHIP)
    ret_df["SP500"] = sp500_ret
    corr_with_sp = ret_df.corr()

    fig, ax = plt.subplots(figsize=(17, 14))
    mask = np.triu(np.ones_like(corr_with_sp, dtype=bool), k=1)
    sns.heatmap(corr_with_sp, annot=True, fmt=".2f", cmap="RdYlGn", center=0,
                mask=mask, square=True, linewidths=0.5,
                vmin=-0.2, vmax=0.7,
                annot_kws={"size": 8}, ax=ax,
                cbar_kws={"label": "Correlation", "shrink": 0.8})
    ax.set_title("AI & Chip Stocks + S&P500 — Correlation Matrix", fontsize=14)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "correlation_heatmap_with_sp500.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] correlation_heatmap_with_sp500.png")

    beta_df.to_csv(CHART_DIR / "beta_analysis.csv", index=False)
    print(f"  [OK] beta_analysis.csv")

    return beta_df


if __name__ == "__main__":
    corr, pairs_df = step6_correlation()
    beta_df = step7_beta()

    print("\n" + "=" * 70)
    print("Correlation & Beta analysis complete.")
    print("=" * 70)
