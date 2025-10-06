import pandas as pd
from .config import DATA_RAW

def load_raw():
    return pd.read_csv(DATA_RAW, low_memory=False)
