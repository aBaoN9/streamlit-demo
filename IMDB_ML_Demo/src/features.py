import pandas as pd, numpy as np, re

def parse_year(s):
    if pd.isna(s): return np.nan
    m = re.search(r"(19|20)\d{2}", str(s))
    return int(m.group(0)) if m else np.nan

def parse_duration(s):
    if pd.isna(s): return np.nan
    t = str(s).lower()
    h = re.search(r"(\d+)\s*h", t)
    mins = re.findall(r"(\d+)\s*min", t)
    total = 0
    if h: total += int(h.group(1)) * 60
    if mins: total += int(mins[-1])
    if total == 0:
        m = re.search(r"(\d+)\s*min", t)
        if m: total = int(m.group(1))
    return total if total>0 else np.nan

def clean_votes(v):
    if pd.isna(v): return np.nan
    return pd.to_numeric(str(v).replace(",",""), errors="coerce")

def count_stars(s):
    if pd.isna(s): return 0
    return max(1, str(s).count(",") + 1)

def primary_genre(s):
    if pd.isna(s): return np.nan
    return str(s).split(",")[0].strip()

def desc_len(s):
    if pd.isna(s): return 0
    return len(str(s).split())
