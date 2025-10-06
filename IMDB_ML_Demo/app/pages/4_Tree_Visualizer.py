# --- add these 3 lines at top ---
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
# ---------------------------------

import streamlit as st
import joblib, io
import matplotlib.pyplot as plt
import numpy as np

from src.config import MODELS_DIR
MODEL_PATH = MODELS_DIR / "decision_tree_rating_regressor.pkl"

st.title("🌳 Tree Visualizer — Decision Tree (rating)")

# Load model
try:
    pipe = joblib.load(MODEL_PATH)
except Exception as e:
    st.error(f"Không load được model: {e}\nHãy chạy: `python -m src.train_tree` trước.")
    st.stop()

# Lấy preprocessor & tree từ pipeline
try:
    preproc = pipe.named_steps["prep"]
    tree = pipe.named_steps["tree"]
except Exception as e:
    st.error(f"Pipeline không đúng cấu trúc: {e}")
    st.stop()

# Lấy tên feature sau OneHotEncoder
try:
    feat_names = preproc.get_feature_names_out()
except Exception:
    # fallback cho scikit-learn cũ
    feat_names = np.array([f"f{i}" for i in range(preproc.transformers_[0][2].__len__())])

# Sidebar controls
max_depth_show = st.sidebar.slider("Độ sâu hiển thị", 1, int(tree.get_depth()) if hasattr(tree, "get_depth") else 6, 3)
dpi = st.sidebar.slider("Độ phân giải (dpi)", 80, 300, 120, step=20)

# Vẽ cây (giới hạn độ sâu để dễ nhìn)
from sklearn.tree import plot_tree
fig, ax = plt.subplots(figsize=(16, 10), dpi=dpi)
plot_tree(
    tree,
    feature_names=feat_names,
    filled=True,
    rounded=True,
    max_depth=max_depth_show,
    impurity=True,   # với regression sẽ là MSE
    fontsize=8,
    ax=ax
)
st.pyplot(fig, use_container_width=True)

# Tải ảnh
buf = io.BytesIO()
fig.savefig(buf, format="png", bbox_inches="tight")
st.download_button("⬇️ Tải ảnh cây (PNG)", data=buf.getvalue(), file_name=f"decision_tree_depth_{max_depth_show}.png", mime="image/png")

st.markdown("---")

# Hiển thị độ quan trọng đặc trưng
if hasattr(tree, "feature_importances_"):
    importances = tree.feature_importances_
    # Gộp các one-hot của cùng một trường nếu muốn (advanced) — tạm thời hiển thị trực tiếp
    top_k = st.slider("Hiện top-k đặc trưng quan trọng", 5, min(30, len(importances)), 10)
    order = np.argsort(importances)[::-1][:top_k]
    st.subheader("🔥 Top đặc trưng quan trọng")
    for idx in order:
        if importances[idx] > 0:
            st.write(f"- **{feat_names[idx]}**: {importances[idx]:.4f}")
    # (tuỳ chọn) vẽ bar chart
    try:
        import plotly.express as px
        import pandas as pd
        df_imp = pd.DataFrame({"feature": feat_names[order], "importance": importances[order]})
        st.plotly_chart(px.bar(df_imp, x="importance", y="feature", orientation="h"), use_container_width=True)
    except Exception:
        pass
else:
    st.info("Model không có thuộc tính feature_importances_.")
