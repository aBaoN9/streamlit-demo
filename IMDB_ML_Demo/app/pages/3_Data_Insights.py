# --- add these 3 lines at top ---
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
# ---------------------------------

from src.predict_tree import predict_one  # hoặc predict_text / load_raw tuỳ file

import streamlit as st
import pandas as pd
from src.data_utils import load_raw

st.title("📊 Data Insights")
df = load_raw()
st.write("Preview dữ liệu:")
st.dataframe(df.head(30), use_container_width=True)
