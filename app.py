import streamlit as st
import pandas as pd

# Lệnh này giúp trang web hiển thị tràn màn hình (wide) và đặt tiêu đề trên tab trình duyệt
st.set_page_config(page_title="Dashboard Phân Tích Sản Phẩm", layout="wide")

# 2. ĐỌC DỮ LIỆU
# Lệnh @st.cache_data giúp web ghi nhớ dữ liệu, không phải đọc lại từ đầu mỗi khi em bấm nút lọc
@st.cache_data
def load_data():
    return pd.read_csv('products.csv')

df = load_data()

# 3. TẠO THANH BÊN TRÁI (SIDEBAR) ĐỂ LỌC DỮ LIỆU NHƯ TRONG ẢNH
st.sidebar.header("⚙️ Bộ Lọc Dữ Liệu")

# Lấy danh sách các Category và Segment độc nhất từ file data để làm menu thả xuống
danh_sach_category = df['category'].unique()
danh_sach_segment = df['segment'].unique()

# Tạo 2 hộp chọn nhiều mục (Multiselect). Mặc định sẽ chọn tất cả.
chon_category = st.sidebar.multiselect("Chọn Category:", danh_sach_category, default=danh_sach_category)
chon_segment = st.sidebar.multiselect("Chọn Segment:", danh_sach_segment, default=danh_sach_segment)

# Máy tính sẽ lọc cái bảng df ban đầu chỉ giữ lại những hàng thỏa mãn điều kiện em vừa bấm chọn
df_da_loc = df[(df['category'].isin(chon_category)) & (df['segment'].isin(chon_segment))]

# 4. TRÌNH BÀY PHẦN CHÍNH CỦA TRANG WEB (MAIN AREA)
st.title("📊 Dashboard Phân Tích Giá Bán Sản Phẩm")
st.markdown("Chào mừng em đến với giao diện Web phân tích dữ liệu đầu tiên!")

# Phân chia màn hình làm 2 cột
cot_trai, cot_phai = st.columns(2)

with cot_trai:
    st.subheader("📈 Trung bình Giá Bán theo Danh Mục")
    # Tính toán nhanh bằng groupby như em đã học
    chart_data = df_da_loc.groupby('category')['price'].mean().reset_index()
    # Vẽ biểu đồ Cột trực tiếp bằng lệnh cực ngắn của Streamlit
    st.bar_chart(chart_data, x="category", y="price")

with cot_phai:
    st.subheader("💰 Trung bình Giá Vốn (COGS) theo Danh Mục")
    chart_data_cogs = df_da_loc.groupby('category')['cogs'].mean().reset_index()
    st.bar_chart(chart_data_cogs, x="category", y="cogs")

# Hiển thị bảng dữ liệu chi tiết ở phía dưới cùng
st.subheader("📋 Bảng Dữ Liệu Chi Tiết")
# st.dataframe giúp tạo ra một bảng có thể cuộn, phóng to, thu nhỏ rất đẹp
st.dataframe(df_da_loc, use_container_width=True)