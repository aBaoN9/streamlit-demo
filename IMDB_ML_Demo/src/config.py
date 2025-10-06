from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw" / "IMBD.csv"
MODELS_DIR = ROOT / "models"

# Columns
COL_RATING = "rating"
COL_GENRE = "genre"
COL_DESC = "description"
COL_YEAR = "year"
COL_CERT = "certificate"
COL_DURATION = "duration"
COL_STARS = "stars"
COL_VOTES = "votes"
