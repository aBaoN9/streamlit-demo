import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

def plot_confusion_matrix(cm, labels):
    fig, ax = plt.subplots(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    st.pyplot(fig)
