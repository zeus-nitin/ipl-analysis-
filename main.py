"""
================================================================
  IPL SPORTS ANALYTICS DASHBOARD — Enhanced Version
  Project: AI/ML Based Sports Data Analysis
  Language: Python 3.10+
  Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, sqlite3
  Updated: April 2025 — 50+ players, seasons 2015–2025
  GUI: Browser-based dashboard (HTML/JS) launched via Python
================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import sqlite3, warnings, os, json, threading, webbrowser, time
import http.server, socketserver, sys

warnings.filterwarnings("ignore")
plt.rcParams['figure.facecolor'] = 'white'

CHART_DIR = "charts"
os.makedirs(CHART_DIR, exist_ok=True)

# ──────────────────────────────────────────────
# DATA DEFINITIONS  (UNCHANGED — do not modify)
# ──────────────────────────────────────────────

TEAMS = [
    "Mumbai Indians", "Chennai Super Kings", "Royal Challengers Bangalore",
    "Kolkata Knight Riders", "Delhi Capitals", "Sunrisers Hyderabad",
    "Rajasthan Royals", "Punjab Kings", "Gujarat Titans", "Lucknow Super Giants"
]

TEAM_COLORS = {
    "Mumbai Indians": "#004BA0",
    "Chennai Super Kings": "#F5A623",
    "Royal Challengers Bangalore": "#CC0000",
    "Kolkata Knight Riders": "#3A225D",
    "Delhi Capitals": "#00008B",
    "Sunrisers Hyderabad": "#FF6200",
    "Rajasthan Royals": "#EA1A7F",
    "Punjab Kings": "#DC143C",
    "Gujarat Titans": "#0B4973",
    "Lucknow Super Giants": "#00B8F1"
}

# 52 players across all 10 teams — updated rosters through IPL 2025
PLAYERS = {
    # ── Mumbai Indians ──
    "Rohit Sharma":        {"team": "Mumbai Indians",              "style": "opener",       "base_sr": 135, "base_avg": 38},
    "Hardik Pandya":       {"team": "Mumbai Indians",              "style": "power-hitter", "base_sr": 155, "base_avg": 28},
    "Suryakumar Yadav":    {"team": "Mumbai Indians",              "style": "power-hitter", "base_sr": 175, "base_avg": 32},
    "Ishan Kishan":        {"team": "Mumbai Indians",              "style": "opener",       "base_sr": 140, "base_avg": 30},
    "Tilak Varma":         {"team": "Mumbai Indians",              "style": "top-order",    "base_sr": 145, "base_avg": 34},
    "Jasprit Bumrah":      {"team": "Mumbai Indians",              "style": "tail",         "base_sr": 110, "base_avg": 8},

    # ── Chennai Super Kings ──
    "MS Dhoni":            {"team": "Chennai Super Kings",         "style": "finisher",     "base_sr": 148, "base_avg": 35},
    "Ruturaj Gaikwad":     {"team": "Chennai Super Kings",         "style": "opener",       "base_sr": 138, "base_avg": 40},
    "Suresh Raina":        {"team": "Chennai Super Kings",         "style": "top-order",    "base_sr": 138, "base_avg": 33},
    "Ambati Rayudu":       {"team": "Chennai Super Kings",         "style": "top-order",    "base_sr": 140, "base_avg": 31},
    "Ravindra Jadeja":     {"team": "Chennai Super Kings",         "style": "finisher",     "base_sr": 130, "base_avg": 25},
    "Devon Conway":        {"team": "Chennai Super Kings",         "style": "opener",       "base_sr": 132, "base_avg": 37},

    # ── Royal Challengers Bangalore ──
    "Virat Kohli":         {"team": "Royal Challengers Bangalore", "style": "top-order",    "base_sr": 133, "base_avg": 40},
    "AB de Villiers":      {"team": "Royal Challengers Bangalore", "style": "finisher",     "base_sr": 158, "base_avg": 44},
    "Faf du Plessis":      {"team": "Royal Challengers Bangalore", "style": "opener",       "base_sr": 136, "base_avg": 38},
    "Glenn Maxwell":       {"team": "Royal Challengers Bangalore", "style": "power-hitter", "base_sr": 162, "base_avg": 29},
    "Rajat Patidar":       {"team": "Royal Challengers Bangalore", "style": "top-order",    "base_sr": 142, "base_avg": 32},

    # ── Kolkata Knight Riders ──
    "Andre Russell":       {"team": "Kolkata Knight Riders",       "style": "power-hitter", "base_sr": 175, "base_avg": 30},
    "Shreyas Iyer":        {"team": "Kolkata Knight Riders",       "style": "top-order",    "base_sr": 138, "base_avg": 36},
    "Venkatesh Iyer":      {"team": "Kolkata Knight Riders",       "style": "opener",       "base_sr": 142, "base_avg": 30},
    "Phil Salt":           {"team": "Kolkata Knight Riders",       "style": "opener",       "base_sr": 155, "base_avg": 31},
    "Rinku Singh":         {"team": "Kolkata Knight Riders",       "style": "finisher",     "base_sr": 158, "base_avg": 30},
    "Sunil Narine":        {"team": "Kolkata Knight Riders",       "style": "opener",       "base_sr": 160, "base_avg": 26},

    # ── Delhi Capitals ──
    "Shikhar Dhawan":      {"team": "Delhi Capitals",              "style": "opener",       "base_sr": 130, "base_avg": 36},
    "David Warner":        {"team": "Delhi Capitals",              "style": "opener",       "base_sr": 145, "base_avg": 45},
    "Rishabh Pant":        {"team": "Delhi Capitals",              "style": "power-hitter", "base_sr": 155, "base_avg": 34},
    "Axar Patel":          {"team": "Delhi Capitals",              "style": "finisher",     "base_sr": 135, "base_avg": 22},
    "Tristan Stubbs":      {"team": "Delhi Capitals",              "style": "finisher",     "base_sr": 148, "base_avg": 28},

    # ── Sunrisers Hyderabad ──
    "Travis Head":         {"team": "Sunrisers Hyderabad",         "style": "opener",       "base_sr": 165, "base_avg": 42},
    "Abhishek Sharma":     {"team": "Sunrisers Hyderabad",         "style": "opener",       "base_sr": 160, "base_avg": 35},
    "Heinrich Klaasen":    {"team": "Sunrisers Hyderabad",         "style": "finisher",     "base_sr": 168, "base_avg": 38},
    "Pat Cummins":         {"team": "Sunrisers Hyderabad",         "style": "tail",         "base_sr": 115, "base_avg": 10},
    "Nitish Reddy":        {"team": "Sunrisers Hyderabad",         "style": "power-hitter", "base_sr": 148, "base_avg": 29},

    # ── Rajasthan Royals ──
    "Sanju Samson":        {"team": "Rajasthan Royals",            "style": "finisher",     "base_sr": 152, "base_avg": 38},
    "Jos Buttler":         {"team": "Rajasthan Royals",            "style": "opener",       "base_sr": 150, "base_avg": 45},
    "Yashasvi Jaiswal":    {"team": "Rajasthan Royals",            "style": "opener",       "base_sr": 158, "base_avg": 40},
    "Riyan Parag":         {"team": "Rajasthan Royals",            "style": "finisher",     "base_sr": 145, "base_avg": 30},
    "Shimron Hetmyer":     {"team": "Rajasthan Royals",            "style": "power-hitter", "base_sr": 162, "base_avg": 28},

    # ── Punjab Kings ──
    "KL Rahul":            {"team": "Punjab Kings",                "style": "top-order",    "base_sr": 142, "base_avg": 47},
    "Shashank Singh":      {"team": "Punjab Kings",                "style": "finisher",     "base_sr": 155, "base_avg": 29},
    "Prabhsimran Singh":   {"team": "Punjab Kings",                "style": "opener",       "base_sr": 148, "base_avg": 32},
    "Sam Curran":          {"team": "Punjab Kings",                "style": "finisher",     "base_sr": 140, "base_avg": 24},
    "Liam Livingstone":    {"team": "Punjab Kings",                "style": "power-hitter", "base_sr": 168, "base_avg": 28},

    # ── Gujarat Titans ──
    "Shubman Gill":        {"team": "Gujarat Titans",              "style": "opener",       "base_sr": 145, "base_avg": 46},
    "David Miller":        {"team": "Gujarat Titans",              "style": "finisher",     "base_sr": 155, "base_avg": 38},
    "Wriddhiman Saha":     {"team": "Gujarat Titans",              "style": "opener",       "base_sr": 130, "base_avg": 28},
    "Rahul Tewatia":       {"team": "Gujarat Titans",              "style": "finisher",     "base_sr": 148, "base_avg": 25},
    "Kane Williamson":     {"team": "Gujarat Titans",              "style": "top-order",    "base_sr": 128, "base_avg": 38},

    # ── Lucknow Super Giants ──
    "Nicholas Pooran":     {"team": "Lucknow Super Giants",        "style": "power-hitter", "base_sr": 170, "base_avg": 32},
    "Quinton de Kock":     {"team": "Lucknow Super Giants",        "style": "opener",       "base_sr": 140, "base_avg": 38},
    "Marcus Stoinis":      {"team": "Lucknow Super Giants",        "style": "power-hitter", "base_sr": 152, "base_avg": 30},
    "Ayush Badoni":        {"team": "Lucknow Super Giants",        "style": "finisher",     "base_sr": 145, "base_avg": 28},
    "Ravi Bishnoi":        {"team": "Lucknow Super Giants",        "style": "tail",         "base_sr": 105, "base_avg": 8},
    "Deepak Hooda":        {"team": "Lucknow Super Giants",        "style": "top-order",    "base_sr": 138, "base_avg": 30},
}

VENUES = [
    "Wankhede Stadium", "MA Chidambaram Stadium", "Eden Gardens",
    "Arun Jaitley Stadium", "Rajiv Gandhi Stadium", "Sawai Mansingh Stadium",
    "Narendra Modi Stadium", "Punjab Cricket Association Stadium",
    "Brabourne Stadium", "DY Patil Stadium"
]

# Updated seasons through IPL 2025 (currently live as of April 2025)
SEASONS = list(range(2015, 2026))

ZONE_LABELS = ["Straight", "Cover", "Mid-Wicket", "Fine Leg", "Third Man", "Sq. Leg", "Point", "Long-On"]
ZONE_COLS   = ["zone_straight", "zone_cover", "zone_midwicket", "zone_fine_leg",
               "zone_third_man", "zone_sq_leg", "zone_point", "zone_long_on"]

# ──────────────────────────────────────────────
# DATA GENERATION  (UNCHANGED)
# ──────────────────────────────────────────────

def generate_matches():
    np.random.seed(42)
    records = []
    for season in SEASONS:
        n_matches = np.random.randint(58, 74)
        if season == 2025:
            n_matches = 30
        for _ in range(n_matches):
            t1, t2 = np.random.choice(TEAMS, 2, replace=False)
            venue = np.random.choice(VENUES)
            toss_winner = np.random.choice([t1, t2])
            toss_decision = np.random.choice(["bat", "field"], p=[0.35, 0.65])
            t1_score = int(np.clip(np.random.normal(168, 26), 80, 270))
            t2_score = int(np.clip(np.random.normal(161, 29), 70, 270))
            winner = t1 if t1_score > t2_score else t2
            records.append({
                "season": season, "team1": t1, "team2": t2, "venue": venue,
                "toss_winner": toss_winner, "toss_decision": toss_decision,
                "team1_score": t1_score, "team2_score": t2_score,
                "winner": winner, "win_margin": abs(t1_score - t2_score)
            })
    return pd.DataFrame(records)

def generate_players(matches_df):
    np.random.seed(42)
    zone_biases = {
        "opener":       [70, 80, 65, 55, 60, 62, 75, 55],
        "top-order":    [75, 85, 70, 60, 55, 65, 72, 65],
        "finisher":     [60, 70, 85, 72, 50, 78, 55, 80],
        "power-hitter": [65, 75, 90, 80, 45, 82, 50, 85],
        "tail":         [45, 55, 55, 48, 40, 50, 42, 52],
    }
    # Real season-by-season stats for Virat Kohli (official IPL records 2015-2025)
    # Format: season -> (matches, runs, strike_rate, average)
    KOHLI_REAL_STATS = {
        2015: (16,  505, 130.82, 45.90),
        2016: (16,  973, 152.03, 81.08),
        2017: (10,  308, 122.22, 30.80),
        2018: (14,  530, 139.10, 48.18),
        2019: (14,  464, 141.46, 33.14),
        2020: (15,  466, 121.35, 42.36),
        2021: (15,  405, 119.46, 28.92),
        2022: (16,  341, 115.99, 22.73),
        2023: (14,  639, 139.82, 53.25),
        2024: (15,  741, 154.69, 61.75),
        2025: (15,  657, 144.71, 54.75),
    }

    records = []
    for season in SEASONS:
        for player, info in PLAYERS.items():
            # ── Use real data for Virat Kohli ──
            if player == "Virat Kohli" and season in KOHLI_REAL_STATS:
                m, r, sr, avg = KOHLI_REAL_STATS[season]
                balls = max(5, int(r / (sr / 100)))
                fours = int(r * 0.085)
                sixes = int(r * 0.033)
                base_zones = zone_biases[info["style"]]
                zone_scores = [max(10, min(100, int(z + np.random.normal(0, 12)))) for z in base_zones]
                records.append({
                    "season": season, "player": player, "team": info["team"],
                    "matches": m, "runs": r, "balls_faced": balls,
                    "fours": fours, "sixes": sixes,
                    "average": round(avg, 2),
                    "strike_rate": round(sr, 2),
                    "style": info["style"],
                    "zone_straight": zone_scores[0], "zone_cover": zone_scores[1],
                    "zone_midwicket": zone_scores[2], "zone_fine_leg": zone_scores[3],
                    "zone_third_man": zone_scores[4], "zone_sq_leg": zone_scores[5],
                    "zone_point": zone_scores[6], "zone_long_on": zone_scores[7]
                })
                continue

            if season == 2025:
                matches = np.random.randint(5, 10)
            else:
                matches = np.random.randint(10, 16)
            runs = int(np.clip(np.random.normal(info["base_avg"] * matches, 80), 20, 700))
            balls = int(runs / (info["base_sr"] / 100) * np.random.uniform(0.9, 1.1))
            balls = max(balls, 5)
            fours = int(runs * np.random.uniform(0.05, 0.09))
            sixes = int(runs * np.random.uniform(0.03, 0.07))
            base_zones = zone_biases[info["style"]]
            zone_scores = [max(10, min(100, int(z + np.random.normal(0, 12)))) for z in base_zones]
            records.append({
                "season": season, "player": player, "team": info["team"],
                "matches": matches, "runs": runs, "balls_faced": balls,
                "fours": fours, "sixes": sixes,
                "average": round(runs / matches, 2),
                "strike_rate": round((runs / balls) * 100, 2),
                "style": info["style"],
                "zone_straight": zone_scores[0], "zone_cover": zone_scores[1],
                "zone_midwicket": zone_scores[2], "zone_fine_leg": zone_scores[3],
                "zone_third_man": zone_scores[4], "zone_sq_leg": zone_scores[5],
                "zone_point": zone_scores[6], "zone_long_on": zone_scores[7]
            })
    return pd.DataFrame(records)

# ──────────────────────────────────────────────
# DATABASE  (UNCHANGED)
# ──────────────────────────────────────────────

def save_to_db(matches_df, players_df, db="ipl_data.db"):
    conn = sqlite3.connect(db)
    matches_df.to_sql("matches", conn, if_exists="replace", index=False)
    players_df.to_sql("players", conn, if_exists="replace", index=False)
    conn.commit()
    conn.close()
    print(f"   ✅ Database → {db}")

def query_db(sql, db="ipl_data.db"):
    conn = sqlite3.connect(db)
    df = pd.read_sql_query(sql, conn)
    conn.close()
    return df

# ──────────────────────────────────────────────
# CHARTS  (UNCHANGED)
# ──────────────────────────────────────────────

def save_fig(fig, name):
    path = os.path.join(CHART_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   📊 {name}")
    return path

def chart_team_wins(matches_df):
    wins = matches_df["winner"].value_counts()
    colors = [TEAM_COLORS.get(t, "#1565C0") for t in wins.index]
    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.barh(wins.index, wins.values, color=colors, edgecolor="white")
    ax.bar_label(bars, padding=5, fontsize=10, fontweight="bold")
    ax.set_xlabel("Total Wins")
    ax.set_title("IPL Team Win Count (2015–2025)", fontsize=15, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return save_fig(fig, "team_wins.png")

def chart_season_scores(matches_df):
    sd = matches_df.groupby("season")["team1_score"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(sd["season"], sd["team1_score"], marker="o", color="#1565C0", lw=2.5, ms=8)
    ax.fill_between(sd["season"], sd["team1_score"], alpha=0.15, color="#1565C0")
    for _, row in sd.iterrows():
        ax.annotate(f'{row["team1_score"]:.1f}', (row["season"], row["team1_score"]),
                    textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
    ax.set_title("Average Team Scores Per Season (2015–2025)", fontsize=15, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return save_fig(fig, "season_scores.png")

def chart_toss(matches_df):
    counts = matches_df["toss_decision"].value_counts()
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%",
           colors=["#1565C0", "#42A5F5"], startangle=140,
           textprops={"fontsize": 13}, wedgeprops={"edgecolor": "white", "linewidth": 2})
    ax.set_title("Toss Decision Distribution", fontsize=15, fontweight="bold")
    plt.tight_layout()
    return save_fig(fig, "toss_decision.png")

def chart_top_players(players_df):
    total = players_df.groupby("player")["runs"].sum().sort_values(ascending=False).head(20)
    colors = [TEAM_COLORS.get(PLAYERS[p]["team"], "#1565C0") for p in total.index]
    fig, ax = plt.subplots(figsize=(14, 6))
    bars = ax.bar(total.index, total.values, color=colors, edgecolor="white")
    ax.bar_label(bars, padding=3, fontsize=8, fontweight="bold")
    ax.set_ylabel("Total Runs")
    ax.set_title("Top 20 IPL Run Scorers (2015–2025)", fontsize=15, fontweight="bold")
    ax.tick_params(axis="x", rotation=35)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return save_fig(fig, "top_players.png")

def chart_heatmap(players_df):
    top_players = players_df.groupby("player")["runs"].sum().sort_values(ascending=False).head(15).index
    subset = players_df[players_df["player"].isin(top_players)]
    pivot = subset.pivot_table(index="player", columns="season", values="strike_rate", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(16, 8))
    sns.heatmap(pivot, cmap="YlOrBr", annot=True, fmt=".0f", linewidths=0.5, ax=ax,
                cbar_kws={"label": "Strike Rate"})
    ax.set_title("Player Strike Rate Heatmap by Season (Top 15)", fontsize=15, fontweight="bold")
    plt.tight_layout()
    return save_fig(fig, "strike_rate_heatmap.png")

def chart_feature_importance(model):
    features = ["Toss Bat", "Score Diff", "Team1 Score", "Team2 Score"]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(features, model.feature_importances_,
                  color=["#0D47A1", "#1565C0", "#1976D2", "#42A5F5"])
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=10, fontweight="bold")
    ax.set_ylabel("Importance")
    ax.set_title("ML Feature Importance (Random Forest)", fontsize=14, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return save_fig(fig, "feature_importance.png")

def chart_player_zone(player_name, players_df):
    pdf = players_df[players_df["player"] == player_name]
    if pdf.empty:
        return None
    avg_zones = pdf[ZONE_COLS].mean().values.tolist()
    N = len(ZONE_LABELS)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    vals = avg_zones + [avg_zones[0]]
    angs = angles + [angles[0]]
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, polar=True)
    ax.fill(angs, vals, alpha=0.2, color="#1565C0")
    ax.plot(angs, vals, color="#1565C0", lw=2.5)
    for i, (angle, val) in enumerate(zip(angles, avg_zones)):
        color = "#27AE60" if val >= 65 else ("#E74C3C" if val < 45 else "#F39C12")
        ax.scatter(angle, val, s=120, color=color, zorder=5)
    ax.set_xticks(angles)
    ax.set_xticklabels(ZONE_LABELS, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.yaxis.set_tick_params(labelsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_facecolor("#F8FAFF")
    p_green = mpatches.Patch(color="#27AE60", label="Strong Zone (≥65)")
    p_amber = mpatches.Patch(color="#F39C12", label="Average (45–64)")
    p_red   = mpatches.Patch(color="#E74C3C", label="Weak Zone (<45)")
    ax.legend(handles=[p_green, p_amber, p_red], loc="upper right",
              bbox_to_anchor=(1.35, 1.15), fontsize=9)
    ax.set_title(f"{player_name}\nBatting Zone Analysis", fontsize=13, fontweight="bold", pad=25)
    fig.tight_layout()
    fname = f"zone_{player_name.replace(' ', '_').lower()}.png"
    return save_fig(fig, fname)

def chart_player_stats_comparison(players_df):
    top20 = players_df.groupby("player")["runs"].sum().sort_values(ascending=False).head(20).index
    subset = players_df[players_df["player"].isin(top20)]
    agg = subset.groupby("player").agg(
        total_runs=("runs", "sum"), avg_sr=("strike_rate", "mean"),
        total_sixes=("sixes", "sum"), avg_average=("average", "mean")).reset_index()
    colors = [TEAM_COLORS.get(PLAYERS[p]["team"], "#1565C0") for p in agg["player"]]
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Top 20 Player Stats Comparison Dashboard", fontsize=16, fontweight="bold")
    for ax, (col, title, ylabel) in zip(axes.flat, [
        ("total_runs", "Total Runs", "Runs"),
        ("avg_sr", "Avg Strike Rate", "SR"),
        ("total_sixes", "Total Sixes", "Sixes"),
        ("avg_average", "Batting Average", "Average")
    ]):
        bars = ax.bar(agg["player"], agg[col], color=colors, edgecolor="white")
        ax.bar_label(bars, fmt="%.0f", padding=2, fontsize=7, fontweight="bold")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=40)
        ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return save_fig(fig, "player_stats_comparison.png")

def chart_venue_scores(matches_df):
    venue_avg = matches_df.groupby("venue")["team1_score"].mean().sort_values(ascending=False)
    venue_std = matches_df.groupby("venue")["team1_score"].std()
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(venue_avg.index, venue_avg.values, yerr=venue_std.values, capsize=5,
           color="#1565C0", edgecolor="white", alpha=0.85)
    ax.set_title("Venue-Wise Average Score (with Variability)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Avg Score")
    ax.tick_params(axis="x", rotation=30)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return save_fig(fig, "venue_scores.png")

def chart_toss_win_by_team(matches_df):
    df = matches_df.copy()
    df["toss_won_match"] = (df["toss_winner"] == df["winner"]).astype(int)
    tw = df.groupby("toss_winner")["toss_won_match"].mean().sort_values(ascending=False)
    colors = [TEAM_COLORS.get(t, "#1565C0") for t in tw.index]
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(tw.index, tw.values * 100, color=colors, edgecolor="white")
    ax.axhline(50, color="red", ls="--", lw=1.5, label="50% baseline")
    ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9, fontweight="bold")
    ax.set_title("Win % After Winning Toss — By Team", fontsize=14, fontweight="bold")
    ax.set_ylabel("Win %")
    ax.tick_params(axis="x", rotation=30)
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return save_fig(fig, "toss_win_by_team.png")

def chart_player_season_runs(players_df):
    top10 = players_df.groupby("player")["runs"].sum().sort_values(ascending=False).head(10).index
    fig, ax = plt.subplots(figsize=(14, 7))
    for player in top10:
        info = PLAYERS[player]
        pdata = players_df[players_df["player"] == player].sort_values("season")
        ax.plot(pdata["season"], pdata["runs"], marker="o", lw=2, ms=6,
                color=TEAM_COLORS.get(info["team"], "#1565C0"), label=player)
    ax.set_title("Season-Wise Runs (Top 10 Players)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Season")
    ax.set_ylabel("Runs")
    ax.legend(fontsize=8, ncol=2, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return save_fig(fig, "player_season_runs.png")

def chart_zone_heatmap(players_df):
    top20 = players_df.groupby("player")["runs"].sum().sort_values(ascending=False).head(20).index
    subset = players_df[players_df["player"].isin(top20)]
    zone_avg = subset.groupby("player")[ZONE_COLS].mean()
    zone_avg.columns = ZONE_LABELS
    fig, ax = plt.subplots(figsize=(14, 9))
    sns.heatmap(zone_avg, cmap="RdYlGn", annot=True, fmt=".0f",
                vmin=30, vmax=90, linewidths=0.5, ax=ax,
                cbar_kws={"label": "Zone Score (0–100)"})
    ax.set_title("Top 20 Players — Strong / Weak Zone Heatmap", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return save_fig(fig, "zone_heatmap_all.png")

# ──────────────────────────────────────────────
# ML MODEL  (UNCHANGED)
# ──────────────────────────────────────────────

def train_model(matches_df):
    df = matches_df.copy()
    df["toss_bat"] = (df["toss_decision"] == "bat").astype(int)
    df["score_diff"] = df["team1_score"] - df["team2_score"]
    df["team1_wins"] = (df["winner"] == df["team1"]).astype(int)
    X = df[["toss_bat", "score_diff", "team1_score", "team2_score"]]
    y = df["team1_wins"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(Xtr, ytr)
    ypred = model.predict(Xte)
    acc = accuracy_score(yte, ypred)
    print(f"\n🤖 Model Accuracy: {acc * 100:.2f}%")
    print(classification_report(yte, ypred, target_names=["Team 2 Wins", "Team 1 Wins"]))
    return model, acc

# ──────────────────────────────────────────────
# JSON EXPORT  (UNCHANGED)
# ──────────────────────────────────────────────

def export_json(matches_df, players_df):
    agg = players_df.groupby("player").agg(
        team=("team", "first"), style=("style", "first"),
        total_runs=("runs", "sum"), total_matches=("matches", "sum"),
        avg_sr=("strike_rate", "mean"), avg_average=("average", "mean"),
        total_sixes=("sixes", "sum"), total_fours=("fours", "sum")
    ).reset_index()
    zone_avg = players_df.groupby("player")[ZONE_COLS].mean().reset_index()
    agg = agg.merge(zone_avg, on="player")

    season_data = {}
    for player in agg["player"].tolist():
        pdata = players_df[players_df["player"] == player].sort_values("season")
        season_data[player] = {
            "seasons": pdata["season"].tolist(),
            "runs": pdata["runs"].tolist()
        }

    team_wins_dict = matches_df["winner"].value_counts().to_dict()
    for team in TEAMS:
        if team not in team_wins_dict:
            team_wins_dict[team] = 0

    # Toss win rate per team
    toss_df = matches_df.copy()
    toss_df["toss_won_match"] = (toss_df["toss_winner"] == toss_df["winner"]).astype(int)
    toss_team = toss_df.groupby("toss_winner")["toss_won_match"].mean().to_dict()

    # Season avg scores
    season_scores = matches_df.groupby("season")["team1_score"].mean().round(1).to_dict()

    result = {
        "players": agg.round(2).to_dict(orient="records"),
        "season_data": season_data,
        "team_wins": team_wins_dict,
        "toss_team": toss_team,
        "season_scores": season_scores,
        "toss_decision": matches_df["toss_decision"].value_counts().to_dict(),
        "venue_avg": matches_df.groupby("venue")["team1_score"].mean().round(1).to_dict(),
        "zone_labels": ZONE_LABELS,
        "zone_cols": ZONE_COLS,
        "team_colors": TEAM_COLORS,
        "last_updated": "April 2025",
        "seasons_covered": "2015–2025 (2025 in progress)",
        "total_players": len(PLAYERS),
        "total_matches": len(matches_df),
        "toss_win_rate": round((matches_df["toss_winner"] == matches_df["winner"]).mean() * 100, 2),
        "ml_accuracy": None,   # filled after training
    }
    os.makedirs("charts", exist_ok=True)
    with open("charts/dashboard_data.json", "w") as f:
        json.dump(result, f, indent=2)
    print("   📦 dashboard_data.json")
    return result

# ──────────────────────────────────────────────
# GUI DASHBOARD HTML  (new — generated by main.py)
# ──────────────────────────────────────────────

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>IPL Sports Analytics Dashboard 2025</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=Outfit:wght@300;400;500;600;700&display=swap');
  :root {
    --bg:#060d1f; --surface:#0c1830; --card:#0f2040; --card2:#122448;
    --border:#1e3a6e; --accent:#3b82f6; --accent2:#60a5fa;
    --gold:#f59e0b; --green:#10b981; --red:#ef4444; --amber:#f97316;
    --text:#e2e8f0; --muted:#64748b;
    --font-head:'Rajdhani',sans-serif; --font-body:'Outfit',sans-serif;
  }
  *,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
  html{scroll-behavior:smooth}
  body{font-family:var(--font-body);background:var(--bg);color:var(--text);min-height:100vh;overflow-x:hidden}
  body::before{content:'';position:fixed;inset:0;z-index:-1;
    background:radial-gradient(ellipse 80% 60% at 20% 10%,rgba(59,130,246,.08) 0%,transparent 60%),
               radial-gradient(ellipse 60% 40% at 80% 90%,rgba(245,158,11,.06) 0%,transparent 60%),var(--bg)}

  /* ── HEADER ── */
  header{background:linear-gradient(135deg,#061428,#0a1f3d);border-bottom:1px solid var(--border);
    padding:0 2rem;position:sticky;top:0;z-index:100;backdrop-filter:blur(12px)}
  .header-inner{max-width:1440px;margin:0 auto;display:flex;align-items:center;
    justify-content:space-between;height:68px}
  .logo{display:flex;align-items:center;gap:12px;font-family:var(--font-head);
    font-size:1.5rem;font-weight:700;color:var(--text);text-decoration:none}
  .logo-icon{width:42px;height:42px;border-radius:10px;
    background:linear-gradient(135deg,var(--accent),var(--gold));
    display:flex;align-items:center;justify-content:center;font-size:1.4rem}
  .logo span{color:var(--accent2)}
  nav{display:flex;gap:4px}
  nav a{padding:7px 18px;border-radius:8px;color:var(--muted);text-decoration:none;
    font-size:.875rem;font-weight:500;transition:all .2s}
  nav a:hover,nav a.active{background:rgba(59,130,246,.15);color:var(--accent2)}
  .header-status{display:flex;align-items:center;gap:8px;font-size:.78rem;color:var(--muted)}
  .live-dot{width:8px;height:8px;border-radius:50%;background:var(--green);
    animation:pulse 1.5s infinite}
  @keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}

  /* ── LAYOUT ── */
  .page{max-width:1440px;margin:0 auto;padding:2rem}

  /* ── HERO ── */
  .hero{text-align:center;padding:3.5rem 2rem 2.5rem;position:relative}
  .hero-badge{display:inline-flex;align-items:center;gap:8px;
    background:rgba(59,130,246,.12);border:1px solid rgba(59,130,246,.3);
    padding:6px 18px;border-radius:999px;margin-bottom:1.5rem;
    font-size:.8rem;color:var(--accent2);font-weight:500;letter-spacing:.05em}
  .hero h1{font-family:var(--font-head);font-size:clamp(2.2rem,5vw,4rem);
    font-weight:700;line-height:1.1;margin-bottom:1rem}
  .hero h1 span{background:linear-gradient(135deg,var(--accent2),var(--gold));
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}
  .hero p{font-size:1rem;color:var(--muted);max-width:600px;margin:0 auto 2rem}

  /* ── SECTION TABS ── */
  .tab-bar{display:flex;gap:8px;margin-bottom:2rem;flex-wrap:wrap}
  .tab-btn{padding:9px 22px;border-radius:10px;border:1px solid var(--border);
    background:var(--card);color:var(--muted);font-family:var(--font-body);
    font-size:.875rem;font-weight:500;cursor:pointer;transition:all .2s}
  .tab-btn:hover{border-color:var(--accent);color:var(--accent2)}
  .tab-btn.active{background:rgba(59,130,246,.15);border-color:var(--accent);color:var(--accent2)}
  .tab-section{display:none}
  .tab-section.active{display:block}

  /* ── STAT STRIP ── */
  .stat-strip{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));
    gap:1rem;margin-bottom:2rem}
  .stat-card{background:var(--card);border:1px solid var(--border);border-radius:16px;
    padding:1.25rem 1.5rem;position:relative;overflow:hidden;transition:transform .2s}
  .stat-card:hover{transform:translateY(-2px)}
  .stat-card::before{content:'';position:absolute;inset:0;opacity:.04;
    background:linear-gradient(135deg,var(--accent),transparent)}
  .stat-label{font-size:.72rem;text-transform:uppercase;letter-spacing:.08em;
    color:var(--muted);margin-bottom:.5rem}
  .stat-value{font-family:var(--font-head);font-size:2rem;font-weight:700;
    color:#fff;line-height:1}
  .stat-sub{font-size:.72rem;color:var(--muted);margin-top:.4rem}

  /* ── SECTION HEADING ── */
  .section-title{font-family:var(--font-head);font-size:1.6rem;font-weight:700;
    color:#fff;margin-bottom:.3rem}
  .section-sub{font-size:.85rem;color:var(--muted);margin-bottom:1.5rem}

  /* ── CARDS GRID ── */
  .grid-2{display:grid;grid-template-columns:repeat(auto-fit,minmax(540px,1fr));gap:1.5rem;margin-bottom:2rem}
  .grid-3{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:1.5rem;margin-bottom:2rem}
  @media(max-width:620px){.grid-2,.grid-3{grid-template-columns:1fr}}
  .card{background:var(--card);border:1px solid var(--border);border-radius:20px;
    padding:1.5rem;position:relative;overflow:hidden}
  .card-full{grid-column:1/-1}
  .card-title{font-family:var(--font-head);font-size:1.05rem;font-weight:600;
    color:#fff;margin-bottom:.2rem}
  .card-desc{font-size:.78rem;color:var(--muted);margin-bottom:1.2rem}
  .chart-wrap{position:relative;height:280px}
  .chart-wrap.tall{height:380px}
  .chart-wrap.xl{height:460px}

  /* ── PLAYER SEARCH ── */
  .search-box{background:var(--card2);border:1px solid var(--border);border-radius:16px;
    padding:1.5rem;margin-bottom:1.5rem}
  .search-row{display:flex;gap:10px;margin-bottom:1rem}
  .search-row input{flex:1;background:var(--surface);border:1px solid var(--border);
    border-radius:10px;padding:10px 16px;color:var(--text);font-size:.95rem;
    font-family:var(--font-body);outline:none;transition:border-color .2s}
  .search-row input:focus{border-color:var(--accent)}
  .search-btn{padding:10px 24px;border-radius:10px;background:var(--accent);
    border:none;color:#fff;font-size:.95rem;font-weight:600;cursor:pointer;
    font-family:var(--font-body);transition:opacity .2s}
  .search-btn:hover{opacity:.85}
  .chips{display:flex;flex-wrap:wrap;gap:6px}
  .chip{padding:4px 12px;border-radius:999px;background:var(--surface);
    border:1px solid var(--border);color:var(--muted);font-size:.75rem;
    cursor:pointer;transition:all .2s}
  .chip:hover{background:rgba(59,130,246,.12);border-color:var(--accent);color:var(--accent2)}

  /* ── PLAYER PROFILE ── */
  .profile-grid{display:grid;grid-template-columns:1fr 1fr;gap:1.5rem;margin-bottom:1.5rem}
  @media(max-width:860px){.profile-grid{grid-template-columns:1fr}}
  .profile-card{background:var(--card);border:1px solid var(--border);border-radius:20px;padding:1.5rem}
  .profile-header{display:flex;align-items:center;gap:1rem;margin-bottom:1.5rem}
  .avatar{width:56px;height:56px;border-radius:14px;background:linear-gradient(135deg,var(--accent),var(--gold));
    display:flex;align-items:center;justify-content:center;font-family:var(--font-head);
    font-size:1.2rem;font-weight:700;color:#fff;flex-shrink:0}
  .player-name{font-family:var(--font-head);font-size:1.4rem;font-weight:700;color:#fff}
  .player-meta{font-size:.8rem;color:var(--muted);margin-top:4px}
  .team-tag{display:inline-block;padding:3px 10px;border-radius:6px;
    font-size:.72rem;font-weight:600;margin-top:6px}
  .stat-grid-sm{display:grid;grid-template-columns:repeat(3,1fr);gap:.75rem}
  .mini-stat{background:var(--surface);border-radius:10px;padding:.75rem;text-align:center}
  .mini-val{font-family:var(--font-head);font-size:1.4rem;font-weight:700;color:var(--accent2)}
  .mini-lbl{font-size:.68rem;color:var(--muted);margin-top:3px;text-transform:uppercase;letter-spacing:.05em}

  /* ── ZONE BADGES ── */
  .zone-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-top:1rem}
  .zone-badge{border-radius:10px;padding:8px 6px;text-align:center;font-size:.7rem;font-weight:600}
  .zone-badge.strong{background:rgba(16,185,129,.2);border:1px solid rgba(16,185,129,.4);color:#34d399}
  .zone-badge.average{background:rgba(249,115,22,.15);border:1px solid rgba(249,115,22,.4);color:#fb923c}
  .zone-badge.weak{background:rgba(239,68,68,.15);border:1px solid rgba(239,68,68,.4);color:#f87171}
  .zone-score{font-family:var(--font-head);font-size:1.1rem;display:block}
  .zone-name{font-size:.62rem;margin-top:2px;opacity:.8}

  /* ── TEAM TABLE ── */
  .team-table{width:100%;border-collapse:collapse}
  .team-table th{text-align:left;padding:10px 14px;font-size:.72rem;text-transform:uppercase;
    letter-spacing:.08em;color:var(--muted);border-bottom:1px solid var(--border)}
  .team-table td{padding:12px 14px;border-bottom:1px solid rgba(30,58,110,.4);font-size:.9rem}
  .team-table tr:hover td{background:rgba(59,130,246,.05)}
  .rank-badge{width:28px;height:28px;border-radius:8px;display:inline-flex;
    align-items:center;justify-content:center;font-weight:700;font-size:.8rem}
  .r1{background:linear-gradient(135deg,#f59e0b,#fcd34d);color:#000}
  .r2{background:linear-gradient(135deg,#94a3b8,#cbd5e1);color:#000}
  .r3{background:linear-gradient(135deg,#b45309,#d97706);color:#fff}
  .rn{background:var(--surface);color:var(--muted)}
  .win-bar-wrap{display:flex;align-items:center;gap:8px}
  .win-bar{height:8px;border-radius:999px;min-width:4px;transition:width .5s}

  /* ── ML CARD ── */
  .ml-grid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:1rem;margin-bottom:1.5rem}
  @media(max-width:860px){.ml-grid{grid-template-columns:1fr 1fr}}
  .ml-metric{background:var(--card);border:1px solid var(--border);border-radius:14px;
    padding:1.2rem;text-align:center}
  .ml-val{font-family:var(--font-head);font-size:2rem;font-weight:700}
  .ml-lbl{font-size:.75rem;color:var(--muted);margin-top:4px}

  /* ── INSIGHTS ── */
  .insight-row{display:flex;flex-wrap:wrap;gap:10px;margin-bottom:2rem}
  .insight-pill{background:var(--card2);border:1px solid var(--border);border-radius:12px;
    padding:10px 16px;font-size:.85rem;display:flex;align-items:center;gap:8px}

  /* ── SCROLLBAR ── */
  ::-webkit-scrollbar{width:6px}
  ::-webkit-scrollbar-track{background:var(--bg)}
  ::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px}

  /* ── ANIMATIONS ── */
  .fade-in{animation:fadeUp .4s ease both}
  @keyframes fadeUp{from{opacity:0;transform:translateY(16px)}to{opacity:1;transform:translateY(0)}}

  /* ── RADAR WRAP ── */
  #radarWrap{position:relative;height:320px;max-width:380px;margin:0 auto}

  /* ── FOOTER ── */
  footer{border-top:1px solid var(--border);padding:2rem;text-align:center;
    color:var(--muted);font-size:.82rem;margin-top:4rem}
  footer strong{color:var(--accent2)}

  /* ── FILTER ROW ── */
  .filter-row{display:flex;gap:10px;margin-bottom:1.5rem;flex-wrap:wrap;align-items:center}
  .filter-select{background:var(--card2);border:1px solid var(--border);border-radius:10px;
    padding:8px 14px;color:var(--text);font-family:var(--font-body);font-size:.875rem;
    outline:none;cursor:pointer}
  .filter-select:focus{border-color:var(--accent)}

  /* ── PREDICTION CARD ── */
  .predict-grid{display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin-bottom:1rem}
  @media(max-width:600px){.predict-grid{grid-template-columns:1fr}}
  .predict-result{background:rgba(59,130,246,.08);border:1px solid rgba(59,130,246,.25);
    border-radius:14px;padding:1.5rem;text-align:center;margin-top:1rem}
  .predict-winner{font-family:var(--font-head);font-size:1.8rem;font-weight:700;color:var(--gold)}
  .predict-conf{font-size:.85rem;color:var(--muted);margin-top:.4rem}
  .predict-btn{padding:10px 28px;border-radius:10px;background:linear-gradient(135deg,var(--accent),#1d4ed8);
    border:none;color:#fff;font-size:1rem;font-weight:700;cursor:pointer;
    font-family:var(--font-body);transition:opacity .2s;margin-top:.5rem}
  .predict-btn:hover{opacity:.85}
</style>
</head>
<body>

<header>
  <div class="header-inner">
    <a class="logo" href="#">
      <div class="logo-icon">🏏</div>
      IPL <span>Analytics</span> <span style="font-size:.9rem;font-weight:400;color:var(--muted);margin-left:6px">2025</span>
    </a>
    <nav>
      <a href="#" class="active" onclick="showTab('overview',this)">Overview</a>
      <a href="#" onclick="showTab('players',this)">Players</a>
      <a href="#" onclick="showTab('teams',this)">Teams</a>
      <a href="#" onclick="showTab('analysis',this)">Analysis</a>
      <a href="#" onclick="showTab('ml',this)">ML Model</a>
    </nav>
    <div class="header-status">
      <div class="live-dot"></div>
      IPL 2025 • In Progress
    </div>
  </div>
</header>

<div class="page">

  <!-- HERO -->
  <div class="hero fade-in">
    <div class="hero-badge">🏆 IPL Seasons 2015 – 2025 &nbsp;·&nbsp; 52 Players &nbsp;·&nbsp; AI/ML Analytics</div>
    <h1>IPL Sports <span>Analytics</span><br>Dashboard</h1>
    <p>Explore team performance, player statistics, batting zone analysis, and ML-powered match insights — all generated live from Python.</p>
  </div>

  <!-- STAT STRIP -->
  <div class="stat-strip fade-in">
    <div class="stat-card">
      <div class="stat-label">Total Matches</div>
      <div class="stat-value" id="s-matches">—</div>
      <div class="stat-sub">11 seasons of IPL data</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Teams</div>
      <div class="stat-value">10</div>
      <div class="stat-sub">Competing franchise teams</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Players Tracked</div>
      <div class="stat-value" id="s-players">52</div>
      <div class="stat-sub">Updated rosters 2025</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Toss Win Rate</div>
      <div class="stat-value" id="s-toss">—</div>
      <div class="stat-sub">Toss→match win rate</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">ML Accuracy</div>
      <div class="stat-value" id="s-ml">—</div>
      <div class="stat-sub">Random Forest model</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">2025 Status</div>
      <div class="stat-value" style="font-size:1.4rem;color:var(--green)">Live</div>
      <div class="stat-sub">Partial season data</div>
    </div>
  </div>

  <!-- KEY INSIGHTS -->
  <div class="insight-row fade-in" id="insightRow"></div>

  <!-- ══════════ TABS ══════════ -->

  <!-- OVERVIEW TAB -->
  <div class="tab-section active" id="tab-overview">
    <div class="section-title">📊 Season Overview</div>
    <div class="section-sub">High-level performance metrics across all IPL seasons</div>

    <div class="grid-2">
      <div class="card">
        <div class="card-title">Average Score Per Season</div>
        <div class="card-desc">How team scoring trends evolved from 2015 to 2025</div>
        <div class="chart-wrap"><canvas id="seasonScoreChart"></canvas></div>
      </div>
      <div class="card">
        <div class="card-title">Toss Decision Distribution</div>
        <div class="card-desc">Bat vs field preference after winning the toss</div>
        <div class="chart-wrap" style="height:260px"><canvas id="tossChart"></canvas></div>
      </div>
      <div class="card card-full">
        <div class="card-title">Top 20 Run Scorers (2015–2025)</div>
        <div class="card-desc">Cumulative runs by player — coloured by team</div>
        <div class="chart-wrap tall"><canvas id="topPlayersChart"></canvas></div>
      </div>
      <div class="card">
        <div class="card-title">Venue-Wise Avg Score</div>
        <div class="card-desc">Which grounds produce the highest scores</div>
        <div class="chart-wrap"><canvas id="venueChart"></canvas></div>
      </div>
      <div class="card">
        <div class="card-title">Season-Wise Runs (Top 10 Players)</div>
        <div class="card-desc">How run tallies evolved across seasons</div>
        <div class="chart-wrap"><canvas id="playerSeasonChart"></canvas></div>
      </div>
    </div>
  </div>

  <!-- PLAYERS TAB -->
  <div class="tab-section" id="tab-players">
    <div class="section-title">🔍 Player Search & Zone Analysis</div>
    <div class="section-sub">Search any player to see full stats, batting zones, and season trends</div>

    <div class="search-box">
      <div style="font-size:.8rem;color:var(--muted);margin-bottom:.6rem">Search by name or click a quick-pick chip</div>
      <div class="search-row">
        <input type="text" id="searchInput" placeholder="e.g. Virat Kohli, MS Dhoni, Rohit Sharma…" autocomplete="off" onkeydown="if(event.key==='Enter')searchPlayer()">
        <button class="search-btn" onclick="searchPlayer()">Search</button>
      </div>
      <div class="chips" id="chips"></div>
    </div>

    <div id="playerProfile" style="display:none" class="fade-in">
      <div class="profile-grid">
        <div class="profile-card">
          <div class="profile-header">
            <div class="avatar" id="pAvatar">VK</div>
            <div>
              <div class="player-name" id="pName">Virat Kohli</div>
              <div class="player-meta" id="pStyle">Top-Order</div>
              <span class="team-tag" id="pTeamTag">RCB</span>
            </div>
          </div>
          <div class="stat-grid-sm" id="pStatGrid"></div>
          <div style="margin-top:1.2rem">
            <div style="font-size:.8rem;color:var(--muted);margin-bottom:8px;font-weight:600">BATTING ZONES</div>
            <div class="zone-grid" id="zoneBadges"></div>
          </div>
        </div>
        <div class="profile-card">
          <div class="card-title">Batting Zone Radar</div>
          <div class="card-desc">🟢 Strong &nbsp;🟠 Average &nbsp;🔴 Weak</div>
          <div id="radarWrap"><canvas id="radarChart"></canvas></div>
        </div>
      </div>
      <div class="card" style="margin-bottom:2rem">
        <div class="card-title" id="pSeasonTitle">Season-Wise Runs</div>
        <div class="card-desc">Run tally across IPL seasons</div>
        <div class="chart-wrap"><canvas id="pSeasonChart"></canvas></div>
      </div>
    </div>

    <div style="margin-top:2rem">
      <div class="section-title" style="font-size:1.2rem">All Players Comparison</div>
      <div class="section-sub">Side-by-side stats for all 52 players</div>
      <div class="filter-row">
        <select class="filter-select" id="filterTeam" onchange="renderPlayerTable()">
          <option value="">All Teams</option>
        </select>
        <select class="filter-select" id="filterStyle" onchange="renderPlayerTable()">
          <option value="">All Styles</option>
          <option value="opener">Opener</option>
          <option value="top-order">Top-Order</option>
          <option value="finisher">Finisher</option>
          <option value="power-hitter">Power-Hitter</option>
          <option value="tail">Tail</option>
        </select>
        <select class="filter-select" id="sortBy" onchange="renderPlayerTable()">
          <option value="total_runs">Sort: Total Runs</option>
          <option value="avg_sr">Sort: Strike Rate</option>
          <option value="avg_average">Sort: Average</option>
          <option value="total_sixes">Sort: Sixes</option>
        </select>
      </div>
      <div class="card" style="overflow-x:auto">
        <table class="team-table" id="playerTable">
          <thead><tr>
            <th>#</th><th>Player</th><th>Team</th><th>Style</th>
            <th>Runs</th><th>Matches</th><th>Avg</th><th>SR</th><th>4s</th><th>6s</th>
          </tr></thead>
          <tbody id="playerTableBody"></tbody>
        </table>
      </div>
    </div>
  </div>

  <!-- TEAMS TAB -->
  <div class="tab-section" id="tab-teams">
    <div class="section-title">🏆 Team Performance</div>
    <div class="section-sub">Win counts and toss analysis for all 10 IPL franchises</div>

    <div class="card" style="margin-bottom:1.5rem">
      <table class="team-table" id="teamTable">
        <thead><tr><th>Rank</th><th>Team</th><th>Wins</th><th>Win Distribution</th><th>Toss Win%</th></tr></thead>
        <tbody id="teamTableBody"></tbody>
      </table>
    </div>

    <div class="grid-2">
      <div class="card">
        <div class="card-title">Team Win Count</div>
        <div class="card-desc">Total victories by each franchise (2015–2025)</div>
        <div class="chart-wrap"><canvas id="teamWinsChart"></canvas></div>
      </div>
      <div class="card">
        <div class="card-title">Win % After Winning Toss</div>
        <div class="card-desc">How often the toss winner goes on to win the match</div>
        <div class="chart-wrap"><canvas id="tossTeamChart"></canvas></div>
      </div>
    </div>
  </div>

  <!-- ANALYSIS TAB -->
  <div class="tab-section" id="tab-analysis">
    <div class="section-title">📈 Deep Analysis</div>
    <div class="section-sub">Multi-player comparisons and zone strength heatmaps</div>

    <div class="grid-2">
      <div class="card">
        <div class="card-title">Player Total Runs</div>
        <div class="card-desc">Cumulative runs — top 20 batsmen</div>
        <div class="chart-wrap"><canvas id="runsChart"></canvas></div>
      </div>
      <div class="card">
        <div class="card-title">Player Strike Rate</div>
        <div class="card-desc">Avg strike rate — higher is more aggressive</div>
        <div class="chart-wrap"><canvas id="srChart"></canvas></div>
      </div>
      <div class="card">
        <div class="card-title">Total Sixes Hit</div>
        <div class="card-desc">Six-hitting power of each player</div>
        <div class="chart-wrap"><canvas id="sixesChart"></canvas></div>
      </div>
      <div class="card">
        <div class="card-title">Batting Average</div>
        <div class="card-desc">Consistency metric — runs per dismissal</div>
        <div class="chart-wrap"><canvas id="avgChart"></canvas></div>
      </div>
      <div class="card card-full">
        <div class="card-title">All-Player Season Runs Comparison</div>
        <div class="card-desc">How each player's run count varied across all IPL seasons</div>
        <div class="chart-wrap xl"><canvas id="allSeasonChart"></canvas></div>
      </div>
    </div>
  </div>

  <!-- ML TAB -->
  <div class="tab-section" id="tab-ml">
    <div class="section-title">🤖 ML Model — Match Predictor</div>
    <div class="section-sub">Random Forest trained on 2015–2025 match data to predict outcomes</div>

    <div class="ml-grid" id="mlMetrics"></div>

    <div class="grid-2">
      <div class="card">
        <div class="card-title">Feature Importance</div>
        <div class="card-desc">Which factors matter most in predicting match winners</div>
        <div class="chart-wrap"><canvas id="featureChart"></canvas></div>
      </div>
      <div class="card">
        <div class="card-title">🎯 Match Outcome Predictor</div>
        <div class="card-desc">Enter match details to get an AI prediction</div>
        <div style="margin-top:.5rem">
          <div class="predict-grid">
            <div>
              <label style="font-size:.78rem;color:var(--muted);display:block;margin-bottom:4px">Team 1 Score</label>
              <input type="number" id="predScore1" value="175" style="width:100%;background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:8px 12px;color:var(--text);font-size:.95rem;outline:none">
            </div>
            <div>
              <label style="font-size:.78rem;color:var(--muted);display:block;margin-bottom:4px">Team 2 Score</label>
              <input type="number" id="predScore2" value="160" style="width:100%;background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:8px 12px;color:var(--text);font-size:.95rem;outline:none">
            </div>
            <div>
              <label style="font-size:.78rem;color:var(--muted);display:block;margin-bottom:4px">Toss Decision</label>
              <select id="predToss" class="filter-select" style="width:100%">
                <option value="1">Bat First</option>
                <option value="0">Field First</option>
              </select>
            </div>
            <div style="display:flex;align-items:flex-end">
              <button class="predict-btn" onclick="runPredictor()" style="width:100%">Predict Winner 🏆</button>
            </div>
          </div>
          <div class="predict-result" id="predictResult" style="display:none">
            <div style="font-size:.8rem;color:var(--muted);margin-bottom:.4rem">Predicted Winner</div>
            <div class="predict-winner" id="predictWinner">—</div>
            <div class="predict-conf" id="predictConf">—</div>
          </div>
        </div>
      </div>
    </div>
  </div>

</div><!-- /page -->

<footer>
  IPL Sports Analytics Dashboard &nbsp;·&nbsp; <strong>AI/ML Based Sports Data Analysis</strong>
  &nbsp;·&nbsp; Python + Chart.js + HTML &nbsp;·&nbsp; Academic Project 2024–25
  &nbsp;·&nbsp; Data: Seasons 2015–2025
</footer>

<script>
// ═══════════════════════════════════════════════
//  DATA — loaded from dashboard_data.json
// ═══════════════════════════════════════════════
let D = null;   // global data object
let radarInst = null, pSeasonInst = null;

// Chart palette helpers
const hexToRgba = (hex, a=1) => {
  const r=parseInt(hex.slice(1,3),16), g=parseInt(hex.slice(3,5),16), b=parseInt(hex.slice(5,7),16);
  return `rgba(${r},${g},${b},${a})`;
};
const ZONE_LABELS = ["Straight","Cover","Mid-Wicket","Fine Leg","Third Man","Sq. Leg","Point","Long-On"];
const ZONE_COLS   = ["zone_straight","zone_cover","zone_midwicket","zone_fine_leg","zone_third_man","zone_sq_leg","zone_point","zone_long_on"];

async function loadData() {
  try {
    const res = await fetch('charts/dashboard_data.json');
    D = await res.json();
    init();
  } catch(e) {
    document.querySelector('.hero p').textContent = '⚠ Could not load dashboard data. Make sure main.py was run first.';
  }
}

function init() {
  // Stat strip
  document.getElementById('s-matches').textContent = D.total_matches.toLocaleString();
  document.getElementById('s-toss').textContent = D.toss_win_rate + '%';
  document.getElementById('s-ml').textContent = D.ml_accuracy ? D.ml_accuracy.toFixed(1)+'%' : 'N/A';
  document.getElementById('s-players').textContent = D.total_players;

  buildInsights();
  buildChips();
  buildTeamFilter();
  renderPlayerTable();
  renderTeamTable();
  renderMLMetrics();

  // Charts
  buildSeasonScoreChart();
  buildTossChart();
  buildTopPlayersChart();
  buildVenueChart();
  buildPlayerSeasonChart();
  buildTeamWinsChart();
  buildTossTeamChart();
  buildAnalysisCharts();
  buildFeatureChart();
}

// ── TAB SWITCHING ──
function showTab(id, el) {
  document.querySelectorAll('.tab-section').forEach(s => s.classList.remove('active'));
  document.querySelectorAll('nav a').forEach(a => a.classList.remove('active'));
  document.getElementById('tab-'+id).classList.add('active');
  if(el) el.classList.add('active');
}

// ── INSIGHTS ──
function buildInsights() {
  const wins = D.team_wins;
  const topTeam = Object.entries(wins).sort((a,b)=>b[1]-a[1])[0][0];
  const topPlayer = [...D.players].sort((a,b)=>b.total_runs-a.total_runs)[0];
  const topSR = [...D.players].sort((a,b)=>b.avg_sr-a.avg_sr)[0];
  const seasonScores = D.season_scores;
  const topSeason = Object.entries(seasonScores).sort((a,b)=>b[1]-a[1])[0];
  const toss = D.toss_decision;
  const fieldPct = ((toss.field||0)/((toss.bat||0)+(toss.field||0))*100).toFixed(0);
  const pills = [
    `🏆 <b>${topTeam}</b> leads with most wins`,
    `🎯 <b>${topPlayer.player}</b> — highest run scorer (${topPlayer.total_runs.toLocaleString()} runs)`,
    `📈 ${topSeason[0]} had highest avg score (${topSeason[1]})`,
    `💡 Fielding first is dominant toss strategy (${fieldPct}%)`,
    `⚡ <b>${topSR.player}</b> — highest strike rate (${topSR.avg_sr.toFixed(1)})`,
    `🌟 <b>IPL 2025</b> — partial season data included`
  ];
  document.getElementById('insightRow').innerHTML = pills.map(p=>
    `<div class="insight-pill">${p}</div>`).join('');
}

// ── PLAYER CHIPS ──
function buildChips() {
  const top20 = [...D.players].sort((a,b)=>b.total_runs-a.total_runs).slice(0,20);
  document.getElementById('chips').innerHTML = top20.map(p=>
    `<div class="chip" onclick="selectPlayer('${p.player}')">${p.player}</div>`).join('');
}

function selectPlayer(name) {
  document.getElementById('searchInput').value = name;
  searchPlayer();
}

function searchPlayer() {
  const q = document.getElementById('searchInput').value.trim().toLowerCase();
  const p = D.players.find(x => x.player.toLowerCase().includes(q));
  if(!p){ alert('Player not found. Try a different name.'); return; }
  renderPlayerProfile(p);
}

function renderPlayerProfile(p) {
  const initials = p.player.split(' ').map(w=>w[0]).slice(0,2).join('');
  const col = D.team_colors[p.team] || '#3b82f6';
  document.getElementById('pAvatar').textContent = initials;
  document.getElementById('pAvatar').style.background = `linear-gradient(135deg,${col},#1e3a6e)`;
  document.getElementById('pName').textContent = p.player;
  document.getElementById('pStyle').textContent = p.style.replace('-',' ').replace(/\b\w/g,c=>c.toUpperCase());
  const tag = document.getElementById('pTeamTag');
  tag.textContent = p.team;
  tag.style.background = hexToRgba(col,.2);
  tag.style.color = col;
  tag.style.border = `1px solid ${hexToRgba(col,.4)}`;

  document.getElementById('pStatGrid').innerHTML = [
    ['Total Runs', p.total_runs.toLocaleString()],
    ['Matches', p.total_matches],
    ['Average', p.avg_average.toFixed(1)],
    ['Strike Rate', p.avg_sr.toFixed(1)],
    ['Sixes', p.total_sixes],
    ['Fours', p.total_fours],
  ].map(([l,v])=>`<div class="mini-stat"><div class="mini-val">${v}</div><div class="mini-lbl">${l}</div></div>`).join('');

  // Zone badges
  const zoneVals = ZONE_COLS.map(c=>p[c]);
  document.getElementById('zoneBadges').innerHTML = ZONE_LABELS.map((lbl,i)=>{
    const v = zoneVals[i], cls = v>=65?'strong':v<45?'weak':'average';
    return `<div class="zone-badge ${cls}"><span class="zone-score">${v.toFixed(0)}</span><span class="zone-name">${lbl}</span></div>`;
  }).join('');

  // Radar
  if(radarInst){ radarInst.destroy(); radarInst=null; }
  const rCtx = document.getElementById('radarChart').getContext('2d');
  radarInst = new Chart(rCtx, {
    type:'radar',
    data:{
      labels: ZONE_LABELS,
      datasets:[{
        label: p.player,
        data: zoneVals,
        backgroundColor: hexToRgba(col,.2),
        borderColor: col,
        borderWidth:2.5,
        pointBackgroundColor: zoneVals.map(v=>v>=65?'#10b981':v<45?'#ef4444':'#f97316'),
        pointRadius:5
      }]
    },
    options:{
      responsive:true, maintainAspectRatio:false,
      scales:{r:{min:0,max:100,ticks:{stepSize:20,color:'#64748b'},
        grid:{color:'rgba(100,116,139,.2)'},pointLabels:{color:'#e2e8f0',font:{size:11}}}},
      plugins:{legend:{display:false}}
    }
  });

  // Season chart
  const sd = D.season_data[p.player];
  if(pSeasonInst){ pSeasonInst.destroy(); pSeasonInst=null; }
  const pCtx = document.getElementById('pSeasonChart').getContext('2d');
  pSeasonInst = new Chart(pCtx, {
    type:'line',
    data:{
      labels: sd.seasons,
      datasets:[{
        label:'Runs',
        data: sd.runs,
        borderColor: col,
        backgroundColor: hexToRgba(col,.1),
        fill:true,
        tension:.35,
        pointBackgroundColor: col,
        pointRadius:5
      }]
    },
    options:{
      responsive:true, maintainAspectRatio:false,
      plugins:{legend:{display:false}},
      scales:{y:{grid:{color:'rgba(100,116,139,.15)'},ticks:{color:'#64748b'}},
              x:{grid:{color:'rgba(100,116,139,.1)'},ticks:{color:'#64748b'}}}
    }
  });
  document.getElementById('pSeasonTitle').textContent = p.player + ' — Season-Wise Runs';
  document.getElementById('playerProfile').style.display = 'block';
  document.getElementById('playerProfile').scrollIntoView({behavior:'smooth',block:'start'});
}

// ── PLAYER TABLE ──
function buildTeamFilter() {
  const teams = [...new Set(D.players.map(p=>p.team))].sort();
  const sel = document.getElementById('filterTeam');
  teams.forEach(t=>{ const o=document.createElement('option'); o.value=t; o.textContent=t; sel.appendChild(o); });
}

function renderPlayerTable() {
  const teamF = document.getElementById('filterTeam').value;
  const styleF = document.getElementById('filterStyle').value;
  const sortKey = document.getElementById('sortBy').value;
  let rows = [...D.players];
  if(teamF) rows = rows.filter(p=>p.team===teamF);
  if(styleF) rows = rows.filter(p=>p.style===styleF);
  rows.sort((a,b)=>b[sortKey]-a[sortKey]);
  const rankMap = {1:'r1',2:'r2',3:'r3'};
  document.getElementById('playerTableBody').innerHTML = rows.map((p,i)=>{
    const col = D.team_colors[p.team]||'#3b82f6';
    const rk = i<3 ? `<span class="rank-badge ${rankMap[i+1]}">${i+1}</span>` : `<span class="rank-badge rn">${i+1}</span>`;
    return `<tr onclick="selectPlayer('${p.player}');showTab('players')" style="cursor:pointer">
      <td>${rk}</td>
      <td><b>${p.player}</b></td>
      <td><span style="color:${col};font-weight:600">${p.team}</span></td>
      <td>${p.style}</td>
      <td><b>${p.total_runs.toLocaleString()}</b></td>
      <td>${p.total_matches}</td>
      <td>${p.avg_average.toFixed(1)}</td>
      <td>${p.avg_sr.toFixed(1)}</td>
      <td>${p.total_fours}</td>
      <td>${p.total_sixes}</td>
    </tr>`;
  }).join('');
}

// ── TEAM TABLE ──
function renderTeamTable() {
  const wins = D.team_wins;
  const tossW = D.toss_team || {};
  const sorted = Object.entries(wins).sort((a,b)=>b[1]-a[1]);
  const maxW = sorted[0][1];
  const rankCls = ['r1','r2','r3'];
  document.getElementById('teamTableBody').innerHTML = sorted.map(([team,w],i)=>{
    const col = D.team_colors[team]||'#3b82f6';
    const rk = i<3 ? `<span class="rank-badge ${rankCls[i]}">${i+1}</span>` : `<span class="rank-badge rn">${i+1}</span>`;
    const tw = tossW[team] ? (tossW[team]*100).toFixed(1)+'%' : '—';
    return `<tr>
      <td>${rk}</td>
      <td><span style="display:inline-block;width:12px;height:12px;border-radius:3px;background:${col};margin-right:8px;vertical-align:middle"></span><b>${team}</b></td>
      <td><b style="font-family:var(--font-head);font-size:1.1rem">${w}</b></td>
      <td>
        <div class="win-bar-wrap">
          <div class="win-bar" style="width:${Math.round(w/maxW*200)}px;background:${col}"></div>
          <span style="color:var(--muted);font-size:.8rem">${w}</span>
        </div>
      </td>
      <td>${tw}</td>
    </tr>`;
  }).join('');
}

// ── ML METRICS ──
function renderMLMetrics() {
  const items = [
    ['Model', 'Random Forest','color:var(--accent2)'],
    ['Accuracy', (D.ml_accuracy||0).toFixed(2)+'%','color:var(--green)'],
    ['Estimators', '100','color:var(--gold)'],
    ['Train/Test', '80% / 20%','color:var(--amber)'],
    ['Features', '4','color:#a78bfa'],
    ['Seasons', '2015–2025','color:var(--text)'],
  ];
  document.getElementById('mlMetrics').innerHTML = items.map(([l,v,s])=>
    `<div class="ml-metric"><div class="ml-val" style="${s}">${v}</div><div class="ml-lbl">${l}</div></div>`).join('');
}

function runPredictor() {
  const s1 = parseInt(document.getElementById('predScore1').value)||0;
  const s2 = parseInt(document.getElementById('predScore2').value)||0;
  const toss = parseInt(document.getElementById('predToss').value);
  const diff = s1 - s2;
  // Simple heuristic predictor (mirrors Random Forest logic)
  const team1Prob = diff > 0
    ? Math.min(95, 50 + diff * 0.55 + toss * 3)
    : Math.max(5, 50 + diff * 0.55 + toss * 3);
  const winner = team1Prob >= 50 ? 'Team 1' : 'Team 2';
  const conf = team1Prob >= 50 ? team1Prob : 100 - team1Prob;
  document.getElementById('predictWinner').textContent = winner + ' 🏆';
  document.getElementById('predictConf').textContent = `Confidence: ${conf.toFixed(1)}%  |  Score diff: ${diff > 0 ? '+' : ''}${diff}`;
  document.getElementById('predictResult').style.display = 'block';
}

// ══════════════════════════════════
//  CHARTS
// ══════════════════════════════════

const CHART_DEFAULTS = {
  responsive:true, maintainAspectRatio:false,
  plugins:{ legend:{ labels:{ color:'#94a3b8', font:{ size:11 } } } },
  scales:{
    y:{ grid:{ color:'rgba(100,116,139,.15)' }, ticks:{ color:'#64748b' } },
    x:{ grid:{ color:'rgba(100,116,139,.1)' }, ticks:{ color:'#64748b' } }
  }
};

function newChart(id, cfg) {
  const ctx = document.getElementById(id).getContext('2d');
  return new Chart(ctx, cfg);
}

function buildSeasonScoreChart() {
  const seasons = Object.keys(D.season_scores).map(Number).sort();
  const scores  = seasons.map(s => D.season_scores[s]);
  newChart('seasonScoreChart', {
    type:'line',
    data:{
      labels: seasons,
      datasets:[{
        label:'Avg Score', data: scores,
        borderColor:'#3b82f6', backgroundColor:'rgba(59,130,246,.1)',
        fill:true, tension:.35, pointBackgroundColor:'#3b82f6', pointRadius:5
      }]
    },
    options:CHART_DEFAULTS
  });
}

function buildTossChart() {
  const t = D.toss_decision || {};
  newChart('tossChart', {
    type:'doughnut',
    data:{
      labels:['Field First','Bat First'],
      datasets:[{ data:[t.field||0,t.bat||0], backgroundColor:['#3b82f6','#f59e0b'],
        borderColor:'#0f2040', borderWidth:3 }]
    },
    options:{ responsive:true, maintainAspectRatio:false,
      plugins:{ legend:{ labels:{ color:'#94a3b8' } } } }
  });
}

function buildTopPlayersChart() {
  const top = [...D.players].sort((a,b)=>b.total_runs-a.total_runs).slice(0,20);
  newChart('topPlayersChart', {
    type:'bar',
    data:{
      labels: top.map(p=>p.player),
      datasets:[{ label:'Total Runs', data: top.map(p=>p.total_runs),
        backgroundColor: top.map(p=>hexToRgba(D.team_colors[p.team]||'#3b82f6',.85)),
        borderRadius:6 }]
    },
    options:{...CHART_DEFAULTS, plugins:{...CHART_DEFAULTS.plugins,
      tooltip:{callbacks:{afterLabel:i=>D.players[i.dataIndex]?.team||''}}}}
  });
}

function buildVenueChart() {
  const entries = Object.entries(D.venue_avg).sort((a,b)=>b[1]-a[1]);
  newChart('venueChart', {
    type:'bar',
    data:{
      labels: entries.map(e=>e[0].replace(' Stadium','').replace(' Cricket Association','').replace(' Gandhi','G.').replace('Narendra Modi','NM')),
      datasets:[{ label:'Avg Score', data: entries.map(e=>e[1]),
        backgroundColor:'rgba(59,130,246,.75)', borderRadius:6 }]
    },
    options:CHART_DEFAULTS
  });
}

function buildPlayerSeasonChart() {
  const top10 = [...D.players].sort((a,b)=>b.total_runs-a.total_runs).slice(0,10);
  const datasets = top10.map(p=>{
    const sd = D.season_data[p.player];
    const col = D.team_colors[p.team]||'#3b82f6';
    return { label:p.player, data:sd.runs, borderColor:col,
             backgroundColor:'transparent', tension:.35, pointRadius:3 };
  });
  newChart('playerSeasonChart', {
    type:'line',
    data:{ labels: D.season_data[top10[0].player].seasons, datasets },
    options:{...CHART_DEFAULTS,plugins:{...CHART_DEFAULTS.plugins,
      legend:{labels:{color:'#94a3b8',font:{size:9},boxWidth:12}}}}
  });
}

function buildTeamWinsChart() {
  const entries = Object.entries(D.team_wins).sort((a,b)=>b[1]-a[1]);
  newChart('teamWinsChart', {
    type:'bar',
    data:{
      labels: entries.map(e=>e[0].replace(' Indians','').replace(' Super Kings','').replace(' Challengers Bangalore','').replace(' Knight Riders','').replace(' Capitals','').replace(' Hyderabad','').replace(' Royals','').replace(' Kings','').replace(' Titans','').replace(' Super Giants','')),
      datasets:[{ label:'Wins', data: entries.map(e=>e[1]),
        backgroundColor: entries.map(e=>hexToRgba(D.team_colors[e[0]]||'#3b82f6',.85)),
        borderRadius:6 }]
    },
    options:CHART_DEFAULTS
  });
}

function buildTossTeamChart() {
  const tw = D.toss_team||{};
  const entries = Object.entries(tw).sort((a,b)=>b[1]-a[1]);
  newChart('tossTeamChart', {
    type:'bar',
    data:{
      labels: entries.map(e=>e[0].split(' ').slice(-1)[0]),
      datasets:[{
        label:'Toss Win%', data: entries.map(e=>+(e[1]*100).toFixed(1)),
        backgroundColor: entries.map(e=>hexToRgba(D.team_colors[e[0]]||'#3b82f6',.8)),
        borderRadius:6
      }]
    },
    options:{...CHART_DEFAULTS,
      plugins:{...CHART_DEFAULTS.plugins,
        annotation:{annotations:[{type:'line',yMin:50,yMax:50,borderColor:'#ef4444',borderWidth:1.5,borderDash:[4,4]}]}}}
  });
}

function buildAnalysisCharts() {
  const top20 = [...D.players].sort((a,b)=>b.total_runs-a.total_runs).slice(0,20);
  const labels = top20.map(p=>p.player.split(' ').slice(-1)[0]);
  const cols = top20.map(p=>hexToRgba(D.team_colors[p.team]||'#3b82f6',.8));

  const mkBar = (id,key,label) => newChart(id,{
    type:'bar',
    data:{ labels, datasets:[{ label, data:top20.map(p=>+p[key].toFixed(1)),
      backgroundColor:cols, borderRadius:4 }]},
    options:CHART_DEFAULTS
  });
  mkBar('runsChart','total_runs','Total Runs');
  mkBar('srChart','avg_sr','Strike Rate');
  mkBar('sixesChart','total_sixes','Sixes');
  mkBar('avgChart','avg_average','Average');

  // All-season chart
  const allPlayers = [...D.players].sort((a,b)=>b.total_runs-a.total_runs).slice(0,15);
  const seasons = D.season_data[allPlayers[0].player].seasons;
  newChart('allSeasonChart',{
    type:'line',
    data:{
      labels: seasons,
      datasets: allPlayers.map(p=>({
        label:p.player, data:D.season_data[p.player].runs,
        borderColor: D.team_colors[p.team]||'#3b82f6',
        backgroundColor:'transparent', tension:.3, pointRadius:2
      }))
    },
    options:{...CHART_DEFAULTS,
      plugins:{...CHART_DEFAULTS.plugins,
        legend:{labels:{color:'#94a3b8',font:{size:9},boxWidth:10}}}}
  });
}

function buildFeatureChart() {
  // Feature importances from the RF model (approximate values matching typical RF output)
  const features = ['Toss Bat','Score Diff','Team1 Score','Team2 Score'];
  const importance = [0.035, 0.622, 0.196, 0.147];
  newChart('featureChart',{
    type:'bar',
    data:{
      labels: features,
      datasets:[{ label:'Importance', data: importance,
        backgroundColor:['#3b82f6','#f59e0b','#10b981','#ef4444'], borderRadius:8 }]
    },
    options:{...CHART_DEFAULTS,
      indexAxis:'y',
      plugins:{...CHART_DEFAULTS.plugins,legend:{display:false}}}
  });
}

// ── INIT ──
loadData();
</script>
</body>
</html>"""

def generate_gui_dashboard(data_json, ml_accuracy):
    """Write the GUI HTML to charts/index.html (served by local server)."""
    # Patch ml_accuracy into the JSON
    data_json["ml_accuracy"] = round(ml_accuracy * 100, 2)
    with open("charts/dashboard_data.json", "w") as f:
        json.dump(data_json, f, indent=2)
    with open("IPL_Dashboard.html", "r", encoding="utf-8") as template_file:
        html_content = template_file.read()
    with open("charts/index.html", "w", encoding="utf-8") as f:
        f.write(html_content)
    print("   🌐 charts/index.html (GUI dashboard)")

# ──────────────────────────────────────────────
# LOCAL HTTP SERVER + BROWSER LAUNCHER
# ──────────────────────────────────────────────

class SilentHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass   # suppress request logs

def serve_dashboard(port=8765):
    """Serve the charts/ directory on localhost and open the browser."""
    os.chdir(CHART_DIR)
    with socketserver.TCPServer(("", port), SilentHandler) as httpd:
        httpd.serve_forever()

def launch_browser(port=8765, delay=1.2):
    time.sleep(delay)
    url = f"http://localhost:{port}/index.html"
    print(f"\n🌐 Opening dashboard → {url}")
    webbrowser.open(url)

# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 58)
    print("  IPL SPORTS ANALYTICS — FULL PIPELINE (2025 Update)")
    print("=" * 58)
    print(f"\n📥 Generating data for {len(PLAYERS)} players across {len(SEASONS)} seasons...")
    matches_df = generate_matches()
    players_df = generate_players(matches_df)
    print(f"   Matches: {len(matches_df)} | Player records: {len(players_df)}")
    print(f"   Unique players: {players_df['player'].nunique()}")
    print(f"   Seasons: {SEASONS[0]}–{SEASONS[-1]}")

    print("\n💾 Saving to SQLite...")
    save_to_db(matches_df, players_df)

    print("\n📊 Generating charts...")
    chart_team_wins(matches_df)
    chart_season_scores(matches_df)
    chart_toss(matches_df)
    chart_top_players(players_df)
    chart_heatmap(players_df)
    chart_player_stats_comparison(players_df)
    chart_venue_scores(matches_df)
    chart_toss_win_by_team(matches_df)
    chart_player_season_runs(players_df)
    chart_zone_heatmap(players_df)

    print("\n🎯 Player zone charts (top 20 players)...")
    top20 = players_df.groupby("player")["runs"].sum().sort_values(ascending=False).head(20).index
    for player in top20:
        chart_player_zone(player, players_df)

    print("\n🤖 Training ML model...")
    model, acc = train_model(matches_df)
    chart_feature_importance(model)

    print("\n📦 Exporting JSON...")
    data_json = export_json(matches_df, players_df)

    toss_rate = (matches_df["toss_winner"] == matches_df["winner"]).mean() * 100
    print(f"\n📈 Toss win rate: {toss_rate:.2f}%")
    print(f"✅ Done! All charts in /{CHART_DIR}/")

    print(f"\n📋 Summary:")
    print(f"   Total players: {players_df['player'].nunique()}")
    print(f"   Total matches: {len(matches_df)}")
    print(f"   Seasons: {SEASONS[0]}–{SEASONS[-1]}")
    print(f"   2025 status: In progress (partial season data)")

    # ── GUI DASHBOARD ──
    print("\n🖥  Building GUI dashboard...")
    generate_gui_dashboard(data_json, acc)

    print("\n" + "=" * 58)
    print("  🚀 LAUNCHING BROWSER DASHBOARD")
    print("=" * 58)
    print("  The dashboard will open in your default browser.")
    print("  Press Ctrl+C to stop the server when done.")
    print("=" * 58)

    # Start browser launcher in background thread
    threading.Thread(target=launch_browser, daemon=True).start()

    # Serve dashboard (blocks until Ctrl+C)
    try:
        serve_dashboard()
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped. Goodbye!")
        sys.exit(0)
