import streamlit as st
st.set_page_config(page_title="Giới thiệu", layout="wide")
st.title("ℹ️ Giới thiệu & Cách sử dụng")
st.markdown("""
Demo hướng đến việc xây dựng một ứng dụng phân loại và tìm kiếm tin tức dựa trên Mình xuất phát từ một nhu cầu rất quen thuộc: mỗi ngày có hàng trăm, hàng nghìn bài báo được xuất bản. Làm sao để tự động xác định chủ đề của một bài báo mới, hoặc tìm những bài tương tự chỉ bằng cách gõ vài từ khóa?
**Mục tiêu**: minh họa pipeline TF-IDF (VSM) + K-NN cho phân loại và tìm kiếm văn bản.  
**Cách dùng nhanh**:
- **KNN Classifier**: dán một đoạn tin → bấm *Phân loại* → xem nhãn dự đoán, phiếu bầu từ các láng giềng, từ khóa TF-IDF nổi bật, và đoạn trích của láng giềng có **từ khóa trùng**.
- **VSM Search**: nhập query → bấm *Tìm kiếm* → xem top-k bài viết gần nhất, từ khóa của query và **từ trùng** trong từng kết quả.
- **Giải thích dữ liệu**: xem phân phối nhãn, độ dài văn bản.
- **Lịch sử**: theo dõi các truy vấn/đoạn văn đã thử.
""")

