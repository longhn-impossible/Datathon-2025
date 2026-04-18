import streamlit as st
import pandas as pd
import plotly.express as px  

# 1. THIẾT LẬP TRANG WEB
st.set_page_config(page_title="Dashboard Phân Tích", layout="wide")

# 2. ĐỌC DỮ LIỆU
@st.cache_data
def load_data():
    return pd.read_csv('E:/Datathon/products.csv')

df = load_data()

# 3. THANH BỘ LỌC (SIDEBAR)
st.sidebar.header("⚙️ Bộ Lọc Dữ Liệu")
chon_category = st.sidebar.multiselect("Chọn Category:", df['category'].unique(), default=df['category'].unique())
chon_segment = st.sidebar.multiselect("Chọn Segment:", df['segment'].unique(), default=df['segment'].unique())

# --- TÍNH NĂNG TỰ LÀM: THANH TRƯỢT GIÁ TIỀN ---
st.sidebar.markdown("---") # Kẻ một đường gạch ngang cho đẹp
st.sidebar.subheader("💰 Lọc theo Giá tiền")

# 1. Tìm giá rẻ nhất và đắt nhất trong toàn bộ cửa hàng
gia_min = float(df['price'].min())
gia_max = float(df['price'].max())

# 2. Tạo thanh trượt (slider) có 2 đầu kéo
# Nó sẽ trả về 1 tuple (cặp 2 số) lưu vào biến chon_gia
chon_gia = st.sidebar.slider(
    "Kéo để chọn khoảng giá:",
    min_value=gia_min,     # Điểm bắt đầu của thanh trượt
    max_value=gia_max,     # Điểm kết thúc của thanh trượt
    value=(gia_min, gia_max) # Mặc định ban đầu là chọn tất cả
)

# Cập nhật bộ lọc: Phải thỏa mãn Category VÀ Segment VÀ Giá tiền lớn hơn mức min VÀ Giá tiền nhỏ hơn mức max
df_da_loc = df[
    (df['category'].isin(chon_category)) & 
    (df['segment'].isin(chon_segment)) &
    (df['price'] >= chon_gia[0]) & # chon_gia[0] là đầu bên trái của thanh trượt
    (df['price'] <= chon_gia[1])   # chon_gia[1] là đầu bên phải của thanh trượt
]
# ================= PHẦN HIỂN THỊ CHÍNH =================
st.title("📊 Dashboard Phân Tích Sản Phẩm Chuyên Sâu")

# --- TÍNH NĂNG MỚI 1: TÌM SẢN PHẨM NỔI BẬT THEO GIÁ TIỀN ---
st.subheader("🏆 Top Sản Phẩm Nổi Bật (Theo Giá Bán)")
cot1, cot2 = st.columns(2)

with cot1:
    st.info("💎 Top 3 Sản Phẩm Giá Cao Nhất (Premium)")
    # Lệnh sort_values(ascending=False) xếp giá từ cao xuống thấp. head(3) lấy 3 dòng đầu.
    top_cao = df_da_loc.sort_values(by='price', ascending=False).head(3)
    st.dataframe(top_cao[['product_id', 'product_name', 'category', 'price']], use_container_width=True)

with cot2:
    st.success("🏷️ Top 3 Sản Phẩm Giá Tốt Nhất (Bình dân)")
    # Lệnh sort_values(ascending=True) xếp giá từ thấp lên cao.
    top_re = df_da_loc.sort_values(by='price', ascending=True).head(3)
    st.dataframe(top_re[['product_id', 'product_name', 'category', 'price']], use_container_width=True)

st.divider() # Vẽ một đường kẻ ngang cho đẹp

# --- TÍNH NĂNG MỚI 2: VẼ BIỂU ĐỒ TRÒN VÀ CỘT TƯƠNG TÁC ---
st.subheader("📈 Phân Tích Trực Quan")
cot_trai, cot_phai = st.columns(2)

with cot_trai:
    st.markdown("**Biểu đồ Cột: Trung bình Giá bán theo Kích cỡ (Size)**")
    # Tính trung bình giá theo size
    chart_data_bar = df_da_loc.groupby('size')['price'].mean().reset_index()
    # px.bar dùng để vẽ cột. Trục x là size, y là price, color tự đổi màu theo size
    fig_bar = px.bar(chart_data_bar, x='size', y='price', color='size', text_auto='.0f')
    st.plotly_chart(fig_bar, use_container_width=True)

with cot_phai:
    st.markdown("**Biểu đồ Tròn: Tỷ trọng Sản phẩm theo Danh mục**")
    # Đếm xem mỗi category có bao nhiêu sản phẩm
    chart_data_pie = df_da_loc['category'].value_counts().reset_index()
    chart_data_pie.columns = ['category', 'so_luong'] # Đổi tên cột cho dễ hiểu
    # px.pie vẽ hình tròn. hole=0.4 tạo ra một lỗ ở giữa (Donut chart) nhìn rất hiện đại
    fig_pie = px.pie(chart_data_pie, names='category', values='so_luong', hole=0.4)
    st.plotly_chart(fig_pie, use_container_width=True)

# --- BẢNG CHI TIẾT ---
st.subheader("📋 Bảng Dữ Liệu Chi Tiết")
st.dataframe(df_da_loc, use_container_width=True)