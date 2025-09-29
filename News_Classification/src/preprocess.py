# src/preprocess.py
import re

_whitespace_re = re.compile(r"\s+")

def clean_text(text: str) -> str:
    if text is None:
        return ""
    # giữ nguyên chữ cái/ số, thay xuống dòng/ tab => khoảng trắng đơn
    text = text.replace("\u00A0", " ").replace("\n", " ").replace("\t", " ")
    text = _whitespace_re.sub(" ", text).strip()
    return text
