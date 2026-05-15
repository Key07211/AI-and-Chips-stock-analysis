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
TICKERS = AI_CHIP["Ticker"].tolist()
MAX_LAG = 5

SECTOR_COLORS = {"AI": "#2196F3", "Chip": "#FF9800"}


def load_close(file_name: str) -> pd.Series:
    df = pd.read_csv(DATA_DIR / file_name, parse_dates=["Date"], index_col="Date")
    return df["Close"]


def build_returns() -> pd.DataFrame:
    ret = {}
    for _, meta in AI_CHIP.iterrows():
        close = load_close(meta["File"])
        ret[meta["Ticker"]] = close.pct_change().dropna()
    return pd.DataFrame(ret)


# ══════════════════════════════════════════════
# Step 8: Lagged Correlation Analysis
# ══════════════════════════════════════════════
def step8_lagged_correlation():
    print("=" * 70)
    print("STEP 8: LAGGED CORRELATION ANALYSIS")
    print("=" * 70)

    ret = build_returns()

    # --- 8a. Compute lagged corr for all pairs at all lags ---
    records = []
    for leader in TICKERS:
        for follower in TICKERS:
            if leader == follower:
                continue
            for lag in range(1, MAX_LAG + 1):
                shifted = ret[follower].shift(-lag)
                corr = ret[leader].corr(shifted.dropna())
                records.append({
                    "Leader": leader,
                    "Follower": follower,
                    "Lag": lag,
                    "Lagged_Corr": round(corr, 4),
                })

    lag_df = pd.DataFrame(records)

    # --- 8b. Find best lag for each pair ---
    best_lag = lag_df.loc[lag_df.groupby(["Leader", "Follower"])["Lagged_Corr"].idxmax()]
    best_lag = best_lag.sort_values("Lagged_Corr", ascending=False).reset_index(drop=True)
    best_lag.index = best_lag.index + 1
    best_lag.index.name = "Rank"

    print("\n>>> TOP 20 STRONGEST LEAD-LAG PAIRS <<<")
    print("    (Leader_t predicts Follower_t+lag)")
    print(best_lag.head(20).to_string())

    # --- 8c. NVDA as leader ---
    print("\n>>> NVDA AS LEADER — who follows NVDA? <<<")
    nvda_leads = best_lag[best_lag["Leader"] == "NVDA"].sort_values("Lagged_Corr", ascending=False)
    print(nvda_leads.head(10).to_string(index=False))

    # --- 8d. NVDA -> AMD deep dive ---
    print("\n>>> NVDA -> AMD DETAILED LAG ANALYSIS <<<")
    nvda_amd = lag_df[(lag_df["Leader"] == "NVDA") & (lag_df["Follower"] == "AMD")]
    print(nvda_amd.to_string(index=False))

    amd_nvda = lag_df[(lag_df["Leader"] == "AMD") & (lag_df["Follower"] == "NVDA")]
    print("\n>>> AMD -> NVDA DETAILED LAG ANALYSIS <<<")
    print(amd_nvda.to_string(index=False))

    # --- 8e. Plot NVDA->AMD lag structure ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    lags_nvda_amd = nvda_amd["Lag"].values
    corrs_nvda_amd = nvda_amd["Lagged_Corr"].values
    lags_amd_nvda = amd_nvda["Lag"].values
    corrs_amd_nvda = amd_nvda["Lagged_Corr"].values

    axes[0].bar(lags_nvda_amd, corrs_nvda_amd, color="#76B900", edgecolor="white", width=0.6)
    axes[0].set_title("NVDA leads AMD", fontsize=13)
    axes[0].set_xlabel("Lag (days)", fontsize=11)
    axes[0].set_ylabel("Lagged Correlation", fontsize=11)
    axes[0].set_xticks(range(1, MAX_LAG + 1))
    axes[0].grid(True, axis="y", alpha=0.3)
    for i, v in enumerate(corrs_nvda_amd):
        axes[0].text(lags_nvda_amd[i], v + 0.002, f"{v:.4f}", ha="center", fontsize=9)

    axes[1].bar(lags_amd_nvda, corrs_amd_nvda, color="#ED1C24", edgecolor="white", width=0.6)
    axes[1].set_title("AMD leads NVDA", fontsize=13)
    axes[1].set_xlabel("Lag (days)", fontsize=11)
    axes[1].set_ylabel("Lagged Correlation", fontsize=11)
    axes[1].set_xticks(range(1, MAX_LAG + 1))
    axes[1].grid(True, axis="y", alpha=0.3)
    for i, v in enumerate(corrs_amd_nvda):
        axes[1].text(lags_amd_nvda[i], v + 0.002, f"{v:.4f}", ha="center", fontsize=9)

    y_min = min(corrs_nvda_amd.min(), corrs_amd_nvda.min()) - 0.02
    y_max = max(corrs_nvda_amd.max(), corrs_amd_nvda.max()) + 0.03
    axes[0].set_ylim(y_min, y_max)
    axes[1].set_ylim(y_min, y_max)

    fig.suptitle("NVDA vs AMD — Lead-Lag Correlation Structure", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "nvda_amd_lag.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  [OK] nvda_amd_lag.png")

    # --- 8f. Lag-1 heatmap for all pairs ---
    lag1 = lag_df[lag_df["Lag"] == 1].pivot(index="Leader", columns="Follower", values="Lagged_Corr")
    lag1 = lag1.reindex(index=TICKERS, columns=TICKERS)

    fig, ax = plt.subplots(figsize=(16, 13))
    sns.heatmap(lag1, annot=True, fmt=".2f", cmap="RdYlGn", center=0,
                square=True, linewidths=0.5, vmin=-0.15, vmax=0.15,
                annot_kws={"size": 7}, ax=ax,
                cbar_kws={"label": "Lag-1 Correlation", "shrink": 0.8})
    ax.set_title("Lag-1 Correlation Matrix (rows lead, columns follow next day)", fontsize=13)
    ax.set_xlabel("Follower (t+1)", fontsize=11)
    ax.set_ylabel("Leader (t)", fontsize=11)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "lag1_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] lag1_heatmap.png")

    lag_df.to_csv(CHART_DIR / "lagged_correlations.csv", index=False)
    print(f"  [OK] lagged_correlations.csv")

    return lag_df, best_lag


# ══════════════════════════════════════════════
# Step 9: Leadership Score — who drives the market?
# ══════════════════════════════════════════════
def step9_leadership(lag_df: pd.DataFrame):
    print("\n" + "=" * 70)
    print("STEP 9: LEADERSHIP ANALYSIS — WHO DRIVES THE MARKET?")
    print("=" * 70)

    lag1 = lag_df[lag_df["Lag"] == 1]

    # --- 9a. Leadership score = avg abs lag-1 corr as leader ---
    leader_scores = []
    for ticker in TICKERS:
        as_leader = lag1[lag1["Leader"] == ticker]
        as_follower = lag1[lag1["Follower"] == ticker]

        avg_lead = as_leader["Lagged_Corr"].abs().mean()
        avg_follow = as_follower["Lagged_Corr"].abs().mean()
        max_lead_row = as_leader.loc[as_leader["Lagged_Corr"].abs().idxmax()]
        max_follow_row = as_follower.loc[as_follower["Lagged_Corr"].abs().idxmax()]

        sector = AI_CHIP[AI_CHIP["Ticker"] == ticker]["Sector"].values[0]
        name = AI_CHIP[AI_CHIP["Ticker"] == ticker]["Name"].values[0]

        leader_scores.append({
            "Ticker": ticker,
            "Name": name,
            "Sector": sector,
            "Avg_Lead_Corr": round(avg_lead, 4),
            "Avg_Follow_Corr": round(avg_follow, 4),
            "Lead_Ratio": round(avg_lead / avg_follow, 2) if avg_follow > 0 else 0,
            "Strongest_Leads": max_lead_row["Follower"],
            "Lead_Corr": round(max_lead_row["Lagged_Corr"], 4),
            "Strongest_Follows": max_follow_row["Leader"],
            "Follow_Corr": round(max_follow_row["Lagged_Corr"], 4),
        })

    score_df = pd.DataFrame(leader_scores)

    print("\n>>> LEADERSHIP RANKING (sorted by Avg Lead Correlation) <<<")
    ranked = score_df.sort_values("Avg_Lead_Corr", ascending=False).reset_index(drop=True)
    ranked.index = ranked.index + 1
    ranked.index.name = "Rank"
    print(ranked.to_string())

    print("\n>>> PURE LEADERS (Lead_Ratio > 1.2 = leads more than follows) <<<")
    leaders = score_df[score_df["Lead_Ratio"] > 1.2].sort_values("Lead_Ratio", ascending=False)
    print(leaders[["Ticker", "Name", "Sector", "Avg_Lead_Corr", "Lead_Ratio", "Strongest_Leads", "Lead_Corr"]].to_string(index=False))

    print("\n>>> PURE FOLLOWERS (Lead_Ratio < 0.8 = follows more than leads) <<<")
    followers = score_df[score_df["Lead_Ratio"] < 0.8].sort_values("Lead_Ratio")
    print(followers[["Ticker", "Name", "Sector", "Avg_Follow_Corr", "Lead_Ratio", "Strongest_Follows", "Follow_Corr"]].to_string(index=False))

    # --- 9b. Leadership bar chart ---
    plot_df = score_df.sort_values("Avg_Lead_Corr", ascending=True)
    colors = [SECTOR_COLORS[s] for s in plot_df["Sector"]]

    fig, ax = plt.subplots(figsize=(14, 10))
    y_pos = range(len(plot_df))
    ax.barh(y_pos, plot_df["Avg_Lead_Corr"], color=colors, edgecolor="white",
            height=0.7, alpha=0.85, label="As Leader")
    ax.barh(y_pos, -plot_df["Avg_Follow_Corr"], color=colors, edgecolor="white",
            height=0.7, alpha=0.4, label="As Follower (inverted)")

    for i, (lead, follow, ticker) in enumerate(zip(
            plot_df["Avg_Lead_Corr"], plot_df["Avg_Follow_Corr"], plot_df["Ticker"])):
        ax.text(lead + 0.002, i, f"{lead:.3f}", va="center", fontsize=8)
        ax.text(-follow - 0.002, i, f"{follow:.3f}", va="center", ha="right", fontsize=8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df["Ticker"], fontsize=10)
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.set_xlabel("Avg |Lag-1 Correlation|", fontsize=11)
    ax.set_title("Leadership Score — Lead (right) vs Follow (left)", fontsize=14)
    ax.grid(True, axis="x", alpha=0.3)

    from matplotlib.patches import Patch
    legend_items = [Patch(facecolor="#2196F3", label="AI"),
                    Patch(facecolor="#FF9800", label="Chip")]
    ax.legend(handles=legend_items, fontsize=10, loc="lower right")

    fig.tight_layout()
    fig.savefig(CHART_DIR / "leadership_score.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  [OK] leadership_score.png")

    # --- 9c. Network diagram: who leads whom ---
    lag1_data = lag_df[lag_df["Lag"] == 1]
    strong_links = lag1_data[lag1_data["Lagged_Corr"].abs() > 0.08].copy()
    strong_links = strong_links.sort_values("Lagged_Corr", ascending=False)

    print("\n>>> SIGNIFICANT LEAD-LAG LINKS (|lag-1 corr| > 0.08) <<<")
    print(strong_links[["Leader", "Follower", "Lagged_Corr"]].to_string(index=False))

    # --- 9d. Influence flow chart ---
    fig, ax = plt.subplots(figsize=(14, 10))

    top_links = strong_links.head(30)
    leaders_unique = top_links["Leader"].unique()
    followers_unique = top_links["Follower"].unique()
    all_nodes = list(dict.fromkeys(list(leaders_unique) + list(followers_unique)))

    node_y = {node: i for i, node in enumerate(all_nodes)}
    node_sector = {}
    for t in all_nodes:
        row = AI_CHIP[AI_CHIP["Ticker"] == t]
        node_sector[t] = row["Sector"].values[0] if len(row) > 0 else "AI"

    for node, y in node_y.items():
        color = SECTOR_COLORS.get(node_sector[node], "#999")
        ax.scatter(0.2, y, s=200, c=color, edgecolors="black", linewidth=1, zorder=5)
        ax.text(0.15, y, node, ha="right", va="center", fontsize=10, fontweight="bold")
        ax.scatter(0.8, y, s=200, c=color, edgecolors="black", linewidth=1, zorder=5)
        ax.text(0.85, y, node, ha="left", va="center", fontsize=10, fontweight="bold")

    for _, row in top_links.iterrows():
        leader_y = node_y.get(row["Leader"])
        follower_y = node_y.get(row["Follower"])
        if leader_y is not None and follower_y is not None:
            corr_val = row["Lagged_Corr"]
            color = "#4CAF50" if corr_val > 0 else "#F44336"
            alpha = min(abs(corr_val) * 5, 0.9)
            width = abs(corr_val) * 15
            ax.annotate("", xy=(0.78, follower_y), xytext=(0.22, leader_y),
                         arrowprops=dict(arrowstyle="->", color=color, alpha=alpha, lw=width))

    ax.text(0.2, len(all_nodes) + 0.3, "LEADER (t)", ha="center", fontsize=12, fontweight="bold")
    ax.text(0.8, len(all_nodes) + 0.3, "FOLLOWER (t+1)", ha="center", fontsize=12, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(-1, len(all_nodes) + 1)
    ax.axis("off")
    ax.set_title("Lead-Lag Influence Flow (top 30 links, |corr| > 0.08)", fontsize=14)

    fig.tight_layout()
    fig.savefig(CHART_DIR / "influence_flow.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] influence_flow.png")

    score_df.to_csv(CHART_DIR / "leadership_scores.csv", index=False)
    print(f"  [OK] leadership_scores.csv")

    return score_df


if __name__ == "__main__":
    lag_df, best_lag = step8_lagged_correlation()
    score_df = step9_leadership(lag_df)

    print("\n" + "=" * 70)
    print("Lead-lag analysis complete.")
    print("=" * 70)
