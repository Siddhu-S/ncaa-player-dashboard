"""
data_engine.py — NCAA Player Atlas
Loads and prepares player datasets for D-II (data.csv) and D-I (mbb_with_pca.csv).

Both loaders return the exact same dict shape so app.py needs zero branching.

D-II column set  → load_data()     (original schema)
D-I  column set  → load_d1_data()  (barttorvik/kenpom schema, different names & scales)
"""

from pathlib import Path
import pandas as pd
import numpy as np
import re
from scipy.spatial.distance import cdist

POS_COLOR = {
    "G":   "#2c5e7a",
    "G/F": "#6c8a3a",
    "F":   "#c47a1d",
    "F/C": "#9a3b6a",
    "C":   "#5c4a8a",
}

POS_LABEL = {
    "G":   "Guard",
    "G/F": "Guard / Forward",
    "F":   "Forward",
    "F/C": "Forward / Center",
    "C":   "Center",
}

POSITIONS = ["G", "G/F", "F", "F/C", "C"]
CLASSES   = ["R", "FR", "SO", "JR", "SR"]

SIM_KEYS = ["PC1", "PC2", "PC3", "PC4"]

TRANSFER_COLUMNS = [
    "transfer_in_portal",
    "transfer_available",
    "transfer_status",
    "transfer_from",
    "transfer_to",
    "transfer_year",
    "transfer_exp",
]

RECRUITING_COLUMNS = [
    "former_247_top_100",
    "former_247_top_150",
    "former_rivals_top_100",
    "former_rivals_ranked",
    "former_ranked_hs_prospect",
    "recruiting_summary",
]

CLASS_TO_RECRUIT_YEAR = {
    "FR": 2025,
    "SO": 2024,
    "JR": 2023,
    "SR": 2022,
    "R": 2025,
}

RECRUITING_ALIAS_GROUPS = [
    ("Rob Dillingham", "Robert Dillingham"),
    ("Solo Ball", "Solomon Ball"),
    ("Nikolas Khamenia", "Nik Khamenia"),
    ("Jacob Wilkins", "Jake Wilkins"),
    ("Cameron Williams", "Cam Williams"),
    ("Taylen Kinney", "Tay Kinney"),
]

# D-I role column → our 5-position system
D1_CONF_NAMES = {
    "A10":  "Atlantic 10",        "ACC":  "ACC",
    "AE":   "America East",       "ASun": "ASUN",
    "Amer": "American Athletic",  "B10":  "Big Ten",
    "B12":  "Big 12",             "BE":   "Big East",
    "BSky": "Big Sky",            "BSth": "Big South",
    "BW":   "Big West",           "CAA":  "Coastal Athletic",
    "CUSA": "Conference USA",     "Horz": "Horizon",
    "Ivy":  "Ivy League",         "MAAC": "MAAC",
    "MAC":  "MAC",                "MEAC": "MEAC",
    "MVC":  "Missouri Valley",    "MWC":  "Mountain West",
    "NEC":  "Northeast",          "OVC":  "Ohio Valley",
    "Pat":  "Patriot",            "SB":   "Sun Belt",
    "SC":   "Southern",           "SEC":  "SEC",
    "SWAC": "Southwestern Athletic", "Slnd": "Southland",
    "Sum":  "Summit",             "WAC":  "WAC",
    "WCC":  "West Coast",
}

D1_ROLE_MAP = {
    "Pure PG":    "G",
    "Scoring PG": "G",
    "Combo G":    "G",
    "Wing G":     "G/F",
    "Wing F":     "F",
    "Stretch 4":  "F/C",
    "PF/C":       "F/C",
    "C":          "C",
}


# ─────────────────────────────────────────────────────────────────────────
# SHARED UTILITIES
# ─────────────────────────────────────────────────────────────────────────

def flip_name(s: str) -> str:
    """'Last, First' -> 'First Last' (D-II names). D-I names are already natural order."""
    if not isinstance(s, str):
        return ""
    s = s.strip()
    idx = s.find(",")
    if idx == -1:
        return s
    last  = s[:idx].strip()
    first = s[idx+1:].strip()
    return f"{first} {last}".strip()


def normalize_class(s: str) -> str:
    if pd.isna(s):
        return "SR"
    v = str(s).lower().replace(".", "").strip()
    if v.startswith("fr"): return "FR"
    if v.startswith("so"): return "SO"
    if v.startswith("jr"): return "JR"
    if v.startswith("sr"): return "SR"
    if v.startswith("r"):  return "R"
    return "SR"


def refine_position(raw: str) -> str:
    r = ("" if pd.isna(raw) else str(raw)).strip().upper()
    if r in ("G", "G/F", "F", "F/C", "C"):
        return r
    if r.startswith("G/F") or r.startswith("GF"): return "G/F"
    if r.startswith("F/C") or r.startswith("FC"): return "F/C"
    if r.startswith("G"): return "G"
    if r.startswith("F"): return "F"
    if r.startswith("C"): return "C"
    return "G"


def height_str(inches: int) -> str:
    ft   = int(inches) // 12
    inch = int(inches) % 12
    return f"{ft}'{inch}\""


def clean_text_series(series, default: str = "") -> pd.Series:
    if series is None:
        return pd.Series(dtype="object")
    return series.fillna(default).astype(str).str.strip()


def conf_abbr(name: str) -> str:
    if pd.isna(name):
        return "—"
    name = str(name).strip()
    if not name:
        return "—"
    words = re.sub(r"[^A-Za-z ]", "", name).split()
    if len(words) == 1:
        return words[0][:5].upper()
    return "".join(w[0] for w in words)[:5].upper()


def match_key(value: str) -> str:
    value = "" if pd.isna(value) else str(value)
    value = value.lower().strip()
    value = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b\.?", "", value)
    return re.sub(r"[^a-z0-9]+", "", value)


def recruiting_match_keys(value: str) -> list[str]:
    base = match_key(value)
    keys = [base] if base else []
    for alias_group in RECRUITING_ALIAS_GROUPS:
        group_keys = [match_key(name) for name in alias_group]
        if base in group_keys:
            keys.extend(key for key in group_keys if key and key not in keys)
    return keys


def _empty_transfer_frame(index) -> pd.DataFrame:
    out = pd.DataFrame(index=index)
    out["transfer_in_portal"] = False
    out["transfer_available"] = False
    out["transfer_status"] = "Not in portal"
    out["transfer_from"] = ""
    out["transfer_to"] = ""
    out["transfer_year"] = np.nan
    out["transfer_exp"] = ""
    return out


def _load_transfer_tags(transfer_path: str | None, raw: pd.DataFrame) -> pd.DataFrame:
    tags = _empty_transfer_frame(raw.index)
    if not transfer_path:
        return tags
    path = Path(transfer_path)
    if not path.exists():
        return tags

    portal = pd.read_csv(path)
    if portal.empty or "player" not in portal.columns:
        return tags

    portal = portal.copy()
    for col in ["from", "to", "exp", "status"]:
        if col not in portal.columns:
            portal[col] = ""
    if "available" not in portal.columns:
        to_text = portal["to"].fillna("").astype(str).str.strip().str.lower()
        portal["available"] = to_text.isin(["", "na", "nan", "none", "uncommitted", "undecided", "tbd"])

    portal["_player_key"] = portal["player"].apply(match_key)
    portal["_from_key"] = portal["from"].apply(match_key)
    raw_keys = pd.DataFrame({
        "_idx": raw.index,
        "_player_key": raw["player_name"].apply(match_key),
        "_team_key": raw["team"].apply(match_key),
    })

    portal["year"] = pd.to_numeric(portal.get("year", np.nan), errors="coerce")
    portal = portal.sort_values(["available", "year"], ascending=[False, False])
    from_match = raw_keys.merge(
        portal,
        how="left",
        left_on=["_player_key", "_team_key"],
        right_on=["_player_key", "_from_key"],
        suffixes=("", "_portal"),
    ).drop_duplicates("_idx")
    portal["_to_key"] = portal["to"].apply(match_key)
    to_match = raw_keys.merge(
        portal,
        how="left",
        left_on=["_player_key", "_team_key"],
        right_on=["_player_key", "_to_key"],
        suffixes=("", "_portal"),
    ).drop_duplicates("_idx")

    from_match = from_match.set_index("_idx")
    to_match = to_match.set_index("_idx")
    for idx in raw.index:
        row = from_match.loc[idx] if idx in from_match.index else None
        if row is None or pd.isna(row.get("player")):
            row = to_match.loc[idx] if idx in to_match.index else None
        if row is None or pd.isna(row.get("player")):
            continue
        available = bool(row.get("available", False))
        to_team = "" if pd.isna(row.get("to")) else str(row.get("to")).strip()
        tags.at[idx, "transfer_in_portal"] = True
        tags.at[idx, "transfer_available"] = available
        tags.at[idx, "transfer_status"] = "Available transfer" if available else "Portal committed"
        tags.at[idx, "transfer_from"] = "" if pd.isna(row.get("from")) else str(row.get("from")).strip()
        tags.at[idx, "transfer_to"] = to_team
        tags.at[idx, "transfer_year"] = row.get("year")
        tags.at[idx, "transfer_exp"] = "" if pd.isna(row.get("exp")) else str(row.get("exp")).strip()
    return tags


def _empty_recruiting_frame(index) -> pd.DataFrame:
    out = pd.DataFrame(index=index)
    out["former_247_top_100"] = False
    out["former_247_top_150"] = False
    out["former_rivals_top_100"] = False
    out["former_rivals_ranked"] = False
    out["former_ranked_hs_prospect"] = False
    out["recruiting_summary"] = ""
    return out


def _candidate_recruit_years(cls: str) -> set[int]:
    base = CLASS_TO_RECRUIT_YEAR.get(cls)
    if base is None:
        return set()
    return {base - 1, base, base + 1}


def _load_recruiting_tags(recruiting_path: str | None, raw: pd.DataFrame) -> pd.DataFrame:
    tags = _empty_recruiting_frame(raw.index)
    if not recruiting_path:
        return tags
    path = Path(recruiting_path)
    if not path.exists():
        return tags

    rankings = pd.read_csv(path)
    if rankings.empty or "player" not in rankings.columns:
        return tags

    rankings = rankings.copy()
    rankings["source"] = rankings.get("source", "").astype(str).str.lower().str.strip()
    rankings["rank"] = pd.to_numeric(rankings.get("rank", np.nan), errors="coerce")
    rankings["class_year"] = pd.to_numeric(rankings.get("class_year", np.nan), errors="coerce").astype("Int64")
    rankings["_player_key"] = rankings["player"].apply(match_key)
    rankings = rankings[rankings["_player_key"].str.len() > 0]

    raw_keys = pd.DataFrame({
        "_idx": raw.index,
        "_player_key": raw["player_name"].apply(match_key),
        "_class": raw["yr"].apply(normalize_class),
    })
    grouped = {k: g.copy() for k, g in rankings.groupby("_player_key")}

    for _, row in raw_keys.iterrows():
        match_frames = [
            grouped[key]
            for key in recruiting_match_keys(row["_player_key"])
            if key in grouped
        ]
        if not match_frames:
            continue
        matches = pd.concat(match_frames, ignore_index=True).drop_duplicates(
            ["source", "class_year", "rank", "player"]
        )

        years = _candidate_recruit_years(row["_class"])
        if years:
            year_matches = matches[matches["class_year"].isin(years)]
            if year_matches.empty:
                continue
            matches = year_matches

        labels = []
        top_247 = matches[matches["source"].eq("247") & matches["rank"].notna()]
        rivals_sources = ["rivals_industry", "on3_rivals"]
        top_rivals = matches[matches["source"].isin(rivals_sources) & matches["rank"].notna()]

        if not top_247.empty:
            best = top_247.sort_values("rank").iloc[0]
            rank = int(best["rank"])
            class_year = "" if pd.isna(best["class_year"]) else int(best["class_year"])
            tags.at[row["_idx"], "former_247_top_100"] = rank <= 100
            tags.at[row["_idx"], "former_247_top_150"] = rank <= 150
            labels.append(f"247 Composite No. {rank}" + (f" ({class_year})" if class_year else ""))

        if not top_rivals.empty:
            best = top_rivals.sort_values("rank").iloc[0]
            rank = int(best["rank"])
            class_year = "" if pd.isna(best["class_year"]) else int(best["class_year"])
            tags.at[row["_idx"], "former_rivals_top_100"] = rank <= 100
            tags.at[row["_idx"], "former_rivals_ranked"] = True
            labels.append(f"Rivals Industry No. {rank}" + (f" ({class_year})" if class_year else ""))

        tags.at[row["_idx"], "former_ranked_hs_prospect"] = bool(labels)
        tags.at[row["_idx"], "recruiting_summary"] = "; ".join(labels)

    return tags


def _build_output(df: pd.DataFrame, id_prefix: str) -> dict:
    """
    Common finalisation: assign IDs, z-score PCs, compute league avgs,
    build conference table, attach similarity function.
    Called by both loaders after they've normalised column names.
    """
    df = df[df["name"].str.len() > 0].copy().reset_index(drop=True)
    df["id"] = [id_prefix + str(i) for i in range(len(df))]

    # ── Similarity setup ─────────────────────────────────────────


    PC_mat = df[SIM_KEYS].values.astype(float)
    cov    = np.cov(PC_mat, rowvar=False)
    # Regularise: add small diagonal to avoid singular matrix
    # (can happen with tiny datasets or near-constant PCs)
    cov   += np.eye(len(SIM_KEYS)) * 1e-6
    VI     = np.linalg.inv(cov)          # inverse covariance matrix
    
    
    # z-score PCs for similarity
    for k in SIM_KEYS:
        mu = df[k].mean()
        sd = df[k].std() or 1
        df[f"_z_{k}"] = (df[k] - mu) / sd

    Z_cols = [f"_z_{k}" for k in SIM_KEYS]
    Z      = df[Z_cols].values

    # league averages
    avg_cols = ["ppg","rpg","apg","spg","bpg","tov","fg","tp","ft","ts","usg","mpg"]
    league_avg = {c: float(df[c].mean()) for c in avg_cols}

    # conference table
    conf_df = (
        df[["conf","confName","team"]]
        .drop_duplicates()
        .groupby(["conf","confName"])["team"]
        .apply(lambda s: sorted(s.unique().tolist()))
        .reset_index()
        .rename(columns={"team": "teams"})
        .sort_values("confName")
        .reset_index(drop=True)
    )
    conferences = conf_df.to_dict("records")

    # similarity closure
    def similar_to(player_id: str, n_sim: int = 5, metric: str = "mahalanobis"):
        idx = df.index[df["id"] == player_id]
        if len(idx) == 0:
            return []
        i    = idx[0]
        vec  = PC_mat[i].reshape(1, -1)          # (1, 4)

        if metric == "euclidean":
            dists = cdist(vec, PC_mat, metric="euclidean").flatten()
        else:
            dists = cdist(vec, PC_mat, metric="mahalanobis", VI=VI).flatten()
        dists[i] = np.inf

        sorted_idx = np.argsort(dists)
        ref_idx    = min(len(dists) - 1, max(20, n_sim * 4))
        ref_dist   = dists[sorted_idx[ref_idx]] or 1.0

        results = []
        for j in sorted_idx[:n_sim]:
            row = df.iloc[j]
            results.append({
                "id":         row["id"],
                "name":       row["name"],
                "pos":        row["pos"],
                "team":       row["team"],
                "cls":        row["cls"],
                "ppg":        row["ppg"],
                "rpg":        row["rpg"],
                "apg":        row["apg"],
                "similarity": float(max(0, 1 - dists[j] / ref_dist)),
                "distance":   float(dists[j]),
            })
        return results
    
    return {
        "df":          df,
        "conferences": conferences,
        "positions":   POSITIONS,
        "classes":     CLASSES,
        "league_avg":  league_avg,
        "similar_to":  similar_to,
        "height_str":  height_str,
    }

# ─────────────────────────────────────────────────────────────────────────
# D-II LOADER  (original schema)
# ─────────────────────────────────────────────────────────────────────────

def load_data(csv_path: str, id_prefix: str = "d2p") -> dict:
    """
    Load D-II dataset (data.csv).
    Column names match the original D-II schema produced by the cleaning pipeline.
    Percentages are already 0-1 fractions.
    """
    raw = pd.read_csv(csv_path)

    def n(col):
        return pd.to_numeric(raw.get(col, 0), errors="coerce").fillna(0)

    def t(col, default=""):
        return clean_text_series(raw.get(col), default=default)

    df = pd.DataFrame()
    df["name"]         = t("Player Name").apply(flip_name)
    df["pos"]          = t("Position").apply(refine_position)
    df["cls"]          = raw.get("Year", pd.Series(index=raw.index, dtype="object")).apply(normalize_class)
    df["team"]         = t("Team")
    df["confName"]     = t("Conference")
    df["conf"]         = df["confName"].apply(conf_abbr)
    df["heightIn"]     = pd.to_numeric(raw["Height"], errors="coerce").fillna(72).round().astype(int)
    df["gp"]           = n("GP").round().astype(int)
    df["mpg"]          = n("MPG")
    df["ppg"]          = n("PPG")
    df["rpg"]          = n("RPG")
    df["apg"]          = n("APG")
    df["spg"]          = n("SPG")
    df["bpg"]          = n("BPG")
    df["tov"]          = n("TOPG")
    df["orb"]          = n("ORBPG")
    df["drb"]          = n("DRBPG")
    df["fg"]           = n("FG%")          # 0-1
    two_made           = n("FGM") - n("3PTM")
    two_att            = n("FGA") - n("3PTA")
    df["two_pct"]      = (two_made / two_att.replace(0, np.nan)).fillna(0).clip(0, 1)
    df["tp"]           = n("3PT%")         # 0-1
    df["ft"]           = n("FT%")          # 0-1
    df["ts"]           = n("TS_pct")       # 0-1
    df["usg"]          = n("usg")          # 0-1 or percentage — keep as-is
    df["efg"]          = n("eFG")
    df["three_share"]  = n("three_share")
    df["ast_tov"]      = n("AST_TOV")
    df["assist_creation"] = n("ast_per_40")
    df["assist_source"]   = "ast_per_40 proxy"
    df["dreb_arch"]       = n("DRBPG")
    df["dreb_source"]     = "DRBPG proxy"
    df["PC1"]          = n("PC1")
    df["PC2"]          = n("PC2")
    df["PC3"]          = n("PC3")
    df["PC4"]          = n("PC4")
    df["bpm"]          = np.nan
    df["porpag"]       = np.nan
    for col, values in _empty_transfer_frame(df.index).items():
        df[col] = values
    for col, values in _empty_recruiting_frame(df.index).items():
        df[col] = values

    return _build_output(df, id_prefix)


# ─────────────────────────────────────────────────────────────────────────
# D-I LOADER  (barttorvik/kenpom schema)
# ─────────────────────────────────────────────────────────────────────────
#
# Key differences from D-II schema:
#   - column names are snake_case, all lowercase
#   - player_name is already "First Last" (no flip needed)
#   - conf is already abbreviated (e.g. "B10", "SEC") — no full name available
#   - no Position column; use role -> 5-category mapping
#   - TS_pct, eFG are 0-100 scale  →  divide by 100
#   - 3P_pct, FT_pct are 0-1 scale →  keep as-is
#   - usg is 0-100 scale           →  divide by 100
#   - PCs are arch_PC1/2/3 + val_PC1 (four components, different names)
#   - No FG% column; use eFG/100 as proxy (best available overall shooting %)

def load_d1_data(
    csv_path: str,
    id_prefix: str = "d1p",
    transfer_path: str | None = None,
    recruiting_path: str | None = None,
) -> dict:
    """
    Load D-I dataset (mbb_with_pca.csv).
    Remaps column names and rescales percentages to match the shared schema
    expected by app.py helpers.
    """
    raw = pd.read_csv(csv_path)

    def n(col):
        return pd.to_numeric(raw.get(col, 0), errors="coerce").fillna(0)

    def t(col, default=""):
        return clean_text_series(raw.get(col), default=default)

    df = pd.DataFrame()

    # Identity
    df["name"]     = t("player_name")
    df["team"]     = t("team")
    df["conf"]     = t("conf")
    df["confName"] = df["conf"].map(D1_CONF_NAMES).fillna(df["conf"])

    # Position — map role to 5-category system
    df["pos"] = t("role").map(D1_ROLE_MAP).fillna("G")

    # Class
    df["cls"] = raw.get("yr", pd.Series(index=raw.index, dtype="object")).apply(normalize_class)

    # Height — already in inches
    df["heightIn"] = pd.to_numeric(raw["height_inches"], errors="coerce").fillna(78).round().astype(int)

    # Counting / rate stats (already per-game)
    df["gp"]  = n("GP").round().astype(int)
    df["mpg"] = n("mins_per_game")
    df["ppg"] = n("pts_per_game")
    df["rpg"] = n("treb_per_game")
    df["apg"] = n("ast_per_game")
    df["spg"] = n("stl_per_game")
    df["bpg"] = n("blk_per_game")
    df["orb"] = n("oreb_per_game")
    df["drb"] = n("dreb_per_game")

    # Assist-to-turnover (raw tov col is TOV_per_24; use AST_TOV ratio directly)
    df["ast_tov"] = n("AST_TOV")
    # Reconstruct tov per-game from AST_TOV ratio and apg (apg / ast_tov, guarded)
    df["tov"] = (df["apg"] / df["ast_tov"].replace(0, np.nan)).fillna(0)

    # Shooting percentages — D-I has mixed scales:
    #   eFG, TS_pct  → 0-100  →  /100
    #   3P_pct, FT_pct → 0-1  →  keep
    df["fg"]  = n("eFG") / 100.0        # best proxy for overall FG; eFG scaled 0-100
    df["two_pct"] = n("2P_pct")          # already 0-1
    df["tp"]  = n("3P_pct")             # already 0-1
    df["ft"]  = n("FT_pct")             # already 0-1
    df["ts"]  = n("TS_pct") / 100.0     # scaled 0-100 → 0-1
    df["efg"] = n("eFG") / 100.0        # 0-1 for slider

    # Usage — 0-100 in D-I → /100 for consiwaistency with D-II (both end up ~0.19 mean)
    df["usg"] = n("usg") / 100.0

    # 3P share — already 0-1
    df["three_share"] = n("three_share")
    df["assist_creation"] = n("AST_pct")
    df["assist_source"]   = "AST_pct"
    df["dreb_arch"]       = n("DRB_pct")
    df["dreb_source"]     = "DRB_pct"

    # PCs — four components with different names in this dataset
    #   arch_PC1, arch_PC2, arch_PC3  →  PC1, PC2, PC3   (archetype / style)
    #   val_PC1                        →  PC4             (value / performance)
    df["PC1"] = n("arch_PC1")
    df["PC2"] = n("arch_PC2")
    df["PC3"] = n("arch_PC3")
    df["PC4"] = n("val_PC1")

    df["bpm"] = n("bpm")
    df["porpag"] = n("PORPAG")
    for col, values in _load_transfer_tags(transfer_path, raw).items():
        df[col] = values
    for col, values in _load_recruiting_tags(recruiting_path, raw).items():
        df[col] = values

    return _build_output(df, id_prefix)
