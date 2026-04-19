import streamlit as st
import pandas as pd
import plotly.express as px
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score

# --- THIẾT LẬP GIAO DIỆN ---
st.set_page_config(page_title="AutoML Siêu Cấp", layout="wide")
st.title("🤖 Trợ Lý AI: Tự Động Phân Tích & Dự Đoán")

# Khởi tạo "Trí nhớ" cho trang web để giữ lại AI sau khi học xong
if "ai_model" not in st.session_state:
    st.session_state.ai_model = None
    st.session_state.train_cols = None
    st.session_state.cat_cols = None

# ==========================================
# MẢNH GHÉP 1: CỔNG NẠP DỮ LIỆU TỰ ĐỘNG
# ==========================================
st.sidebar.header("📁 Bước 1: Nạp Dữ Liệu")
file_tai_len = st.sidebar.file_uploader("Tải lên file CSV:", type=['csv'])

if file_tai_len is not None:
    # Máy tự đọc file vừa thả vào
    df = pd.read_csv(file_tai_len)
    
    # [TÍN HIỆU THÔNG MINH CHO AI: BÓC TÁCH THỜI GIAN]
    # AI rất thích dữ liệu chẻ nhỏ. Nếu file sales.csv có Ngày/Tháng, hãy tách cho nó xem!
    if 'Date' in df.columns:
        df_ngay = pd.to_datetime(df['Date'], errors='coerce')
        df['Month'] = df_ngay.dt.month
        df['Year'] = df_ngay.dt.year
        df['DayOfWeek'] = df_ngay.dt.dayofweek
        
    st.success(f"Ting! Đã nạp thành công bộ dữ liệu gồm {df.shape[0]:,} dòng và {df.shape[1]} cột.")
    
    with st.expander("👀 Xem trước một phần dữ liệu (5 dòng đầu)"):
        st.dataframe(df.head())

    # ==========================================
    # MẢNH GHÉP 2: BÁO CÁO PHÂN TÍCH & DASHBOARD
    # ==========================================
    st.markdown("---")
    st.header("📊 Bước 2: Báo cáo Phân tích & Dashboard")
    
    tab1, tab2 = st.tabs(["🔎 Phân tích Nhanh (Auto-EDA)", "🛒 Dashboard Cửa hàng (App Product)"])
    
    with tab1:
        # Máy tự quét đâu là cột chữ, đâu là cột số
        cot_chu = df.select_dtypes(include=['object', 'category']).columns.tolist()
        cot_so = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        cot1, cot2 = st.columns(2)
        with cot1:
            st.info("💡 Đặc điểm dữ liệu Chữ (Phân loại)")
            # Lấy thử 1 cột chữ đầu tiên để vẽ biểu đồ tròn tự động
            if len(cot_chu) > 0:
                cot_ve_chu = st.selectbox("Chọn cột chữ để vẽ biểu đồ tròn:", cot_chu, key="pie_2")
                fig_pie = px.pie(df, names=cot_ve_chu, hole=0.4, title=f"Tỷ trọng theo {cot_ve_chu}")
                st.plotly_chart(fig_pie, use_container_width=True)
                
        with cot2:
            st.success("📈 Đặc điểm dữ liệu Số (Đo lường)")
            # Vẽ biểu đồ Histogram để xem phân bố của cột số
            if len(cot_so) > 0:
                cot_ve_so = st.selectbox("Chọn cột số để xem phân bố:", cot_so, key="hist_2")
                fig_hist = px.histogram(df, x=cot_ve_so, title=f"Biểu đồ phân bố {cot_ve_so}")
                st.plotly_chart(fig_hist, use_container_width=True)
                
    with tab2:
        cac_cot_app = ['category', 'segment', 'price', 'size', 'product_id', 'product_name']
        if all(cot in df.columns for cot in cac_cot_app):
            st.info("💡 Tính năng đã được tích hợp thành công từ App Product!")
            
            st.sidebar.markdown("---")
            st.sidebar.header("⚙️ Bộ Lọc Cửa Hàng (Bước 2)")
            chon_category = st.sidebar.multiselect("Chọn Category:", df['category'].dropna().unique(), default=df['category'].dropna().unique())
            chon_segment = st.sidebar.multiselect("Chọn Segment:", df['segment'].dropna().unique(), default=df['segment'].dropna().unique())
            
            st.sidebar.subheader("💰 Lọc theo Giá tiền")
            gia_min = float(df['price'].min())
            gia_max = float(df['price'].max())
            
            chon_gia = st.sidebar.slider(
                "Kéo để chọn khoảng giá:",
                min_value=gia_min,
                max_value=gia_max,
                value=(gia_min, gia_max)
            )
            
            # Cập nhật bộ lọc dữ liệu
            df_da_loc = df[
                (df['category'].isin(chon_category)) & 
                (df['segment'].isin(chon_segment)) &
                (df['price'] >= chon_gia[0]) & 
                (df['price'] <= chon_gia[1])
            ]
            
            st.subheader("🏆 Top Sản Phẩm Nổi Bật (Theo Giá Bán)")
            cot_35_1, cot_35_2 = st.columns(2)
            
            with cot_35_1:
                st.warning("💎 Top 3 Sản Phẩm Giá Cao Nhất (Premium)")
                top_cao = df_da_loc.sort_values(by='price', ascending=False).head(3)
                st.dataframe(top_cao[['product_id', 'product_name', 'category', 'price']], use_container_width=True)
                
            with cot_35_2:
                st.success("🏷️ Top 3 Sản Phẩm Giá Tốt Nhất (Bình dân)")
                top_re = df_da_loc.sort_values(by='price', ascending=True).head(3)
                st.dataframe(top_re[['product_id', 'product_name', 'category', 'price']], use_container_width=True)
                
            st.divider()
            
            st.subheader("📈 Phân Tích Trực Quan")
            cot_trai, cot_phai = st.columns(2)
            
            with cot_trai:
                st.markdown("**Biểu đồ Cột: Trung bình Giá bán theo Kích cỡ (Size)**")
                if not df_da_loc.empty:
                    chart_data_bar = df_da_loc.groupby('size')['price'].mean().reset_index()
                    fig_bar = px.bar(chart_data_bar, x='size', y='price', color='size', text_auto='.0f')
                    st.plotly_chart(fig_bar, use_container_width=True)
                    
            with cot_phai:
                st.markdown("**Biểu đồ Tròn: Tỷ trọng Sản phẩm theo Danh mục**")
                if not df_da_loc.empty:
                    chart_data_pie = df_da_loc['category'].value_counts().reset_index()
                    chart_data_pie.columns = ['category', 'so_luong']
                    fig_pie = px.pie(chart_data_pie, names='category', values='so_luong', hole=0.4)
                    st.plotly_chart(fig_pie, use_container_width=True)
                    
            st.subheader("📋 Bảng Dữ Liệu Cửa Hàng Chi Tiết")
            st.dataframe(df_da_loc, use_container_width=True)
        elif 'Revenue' in df.columns and 'COGS' in df.columns:
            st.info("💡 Hệ thống nhận diện đây là dữ liệu Dòng tiền (Sales). Đã tự động kích hoạt Sales Dashboard!")
            
            # Tự động tính Lợi Nhuận
            df['Profit'] = df['Revenue'] - df['COGS']
            
            st.subheader("💰 Tổng quan Dòng tiền & Lãi lỗ")
            col_rev, col_cogs, col_prof = st.columns(3)
            col_rev.metric("Tổng Doanh Thu (Revenue)", f"{df['Revenue'].sum():,.0f} đ")
            col_cogs.metric("Tổng Chi Phí (COGS)", f"{df['COGS'].sum():,.0f} đ")
            col_prof.metric("Tổng Lợi Nhuận (Profit)", f"{df['Profit'].sum():,.0f} đ")
            
            if 'Date' in df.columns:
                st.markdown("---")
                st.subheader("📈 Phân tích Xu hướng theo Thời gian")
                
                # Nhóm dữ liệu theo biểu đồ Tháng
                df_thoi_gian = df.copy()
                df_thoi_gian['Time_Index'] = pd.to_datetime(df_thoi_gian['Date'], errors='coerce')
                df_thang = df_thoi_gian.groupby(df_thoi_gian['Time_Index'].dt.to_period('M')).sum(numeric_only=True).reset_index()
                df_thang['Time_Index'] = df_thang['Time_Index'].astype(str)
                
                fig_trend = px.line(df_thang, x='Time_Index', y='Revenue', title='Biểu đồ Doanh thu Tổng chốt theo Tháng', markers=True)
                st.plotly_chart(fig_trend, use_container_width=True)
                
                st.subheader("📊 Đối chiếu Doanh Thu vs Chi Phí")
                fig_area = px.area(df_thang, x='Time_Index', y=['Revenue', 'COGS', 'Profit'], title='Cấu trúc Dòng tiền Mở rộng')
                st.plotly_chart(fig_area, use_container_width=True)
                
            st.subheader("📋 Bảng Dữ Liệu Sales Gốc")
            st.dataframe(df, use_container_width=True)
            
        else:
            st.warning("⚠️ Cần nạp bộ dữ liệu chuẩn (như Products có category/price hoặc Sales có Revenue/COGS) để xem Dashboard Tuỳ Biến này.")

    # ==========================================
    # MẢNH GHÉP 3: LÒ LUYỆN AI (LIGHTGBM)
    # ==========================================
    st.markdown("---")
    st.header("🧠 Bước 3: Huấn Luyện Trí Tuệ Nhân Tạo")
    
    # Sếp chọn xem muốn dự đoán cái gì (Thường là Giá tiền, Cân nặng...)
    muc_tieu = st.selectbox("🎯 Sếp muốn AI dự đoán giá trị của cột nào?", cot_so)
    
    if st.button("🚀 Bắt đầu Bơm dữ liệu cho AI học!"):
        with st.spinner("Đang tối ưu hóa và kích hoạt LightGBM..."):
            # 0. Chỉ xóa dòng nếu CỘT MỤC TIÊU (đáp án) bị thiếu. 
            # Giữ lại các dòng thiếu dữ liệu ở cột khác vì LightGBM tự hiểu được NaN!
            df_clean = df.dropna(subset=[muc_tieu]).copy()
            
            # --- FEATURE ENGINEERING: Tự động Bóc tách Thời Gian ---
            # (Rất quan trọng nếu file chỉ có cột Date, Revenue, COGS như sales.csv)
            if 'Date' in df_clean.columns:
                df_ngay = pd.to_datetime(df_clean['Date'], errors='coerce')
                df_clean['Year'] = df_ngay.dt.year
                df_clean['Month'] = df_ngay.dt.month
                df_clean['DayOfWeek'] = df_ngay.dt.dayofweek
                df_clean['Day'] = df_ngay.dt.day
            
            # 1. Tách đề bài (X) và đáp án (y)
            # [CHỐNG RÒ RỈ DỮ LIỆU - DATA LEAKAGE PREVENTION]
            cot_bo_di = [muc_tieu]
            # Nếu dự đoán Doanh thu, tuyệt đối không cho AI nhìn thấy Chi phí hoặc Lợi nhuận (Và ngược lại)
            if muc_tieu == 'Revenue':
                cot_bo_di.extend([c for c in ['COGS', 'Profit'] if c in df_clean.columns])
            elif muc_tieu == 'Profit':
                cot_bo_di.extend([c for c in ['COGS', 'Revenue'] if c in df_clean.columns])
            elif muc_tieu == 'COGS':
                cot_bo_di.extend([c for c in ['Revenue', 'Profit'] if c in df_clean.columns])
                 
            # XÓA CỘT 'Date' NGUYÊN BẢN ĐỂ CHUẨN BỊ Dự Đoán CHO NĂM TƯƠNG LAI 2023-2024
            if 'Date' in df_clean.columns:
                cot_bo_di.append('Date')
                
            X = df_clean.drop(columns=cot_bo_di)
            # Khong dung log1p nua theo gop y de toi uu MAE truc tiep
            y = df_clean[muc_tieu]
            
            # 2. TUYỆT KỸ TĂNG TỐC: Dùng Categorical nội tại của LightGBM thay cho get_dummies
            cac_cot_chu = X.select_dtypes(include=['object', 'category']).columns.tolist()
            for cot in cac_cot_chu:
                # Chuyển chữ thành category type để LightGBM tăng tốc tự nhiên
                X[cot] = X[cot].astype('category')
            
            # LƯU Ý QUAN TRỌNG: Ghi nhớ lại cấu trúc cột để dự đoán
            st.session_state.train_cols = X.columns 
            st.session_state.cat_cols = cac_cot_chu
            
            # ---------------------------------------------------------
            # BÍ QUYẾT XỬ LÝ HÀNG TRĂM NGÀN DÒNG DỮ LIỆU
            # ---------------------------------------------------------
            # 3.1 Ép kiểu dữ liệu (Downcasting) giúp tiết kiệm đúng 50% RAM
            cac_cot_float64 = X.select_dtypes(include=['float64']).columns
            X[cac_cot_float64] = X[cac_cot_float64].astype('float32')
            
            cac_cot_int64 = X.select_dtypes(include=['int64']).columns
            X[cac_cot_int64] = X[cac_cot_int64].astype('int32')
            
            # 3.2 Tách tập huấn luyện và đề thi thử (80% Học - 20% Thi)
            # (Đảm bảo máy có căn cứ đánh giá và không bị học vẹt)
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # ========================================================
            # CHUẨN DATATHON: CROSS-VALIDATION ĐỂ XÁC THỰC HIỆU NĂNG
            # ========================================================
            from sklearn.metrics import mean_absolute_error, mean_squared_error
            st.info("🔄 Đang chạy xác thực chéo (5-Fold Cross-Validation) đánh giá chuẩn 3 Tiêu chí Datathon (MAE, RMSE, R²)...")
            thanh_cv = st.progress(0)
            
            # Thay kf = KFold bang tscv = TimeSeriesSplit
            tscv = TimeSeriesSplit(n_splits=5)
            diem_r2 = []
            diem_mae = []
            diem_rmse = []
            
            mo_hinh_thu = lgb.LGBMRegressor(n_estimators=100, objective='regression_l1', random_state=42, n_jobs=-1)
            
            for index_vong, (train_idx, test_idx) in enumerate(tscv.split(X)):
                X_train_cv, X_test_cv = X.iloc[train_idx], X.iloc[test_idx]
                y_train_cv, y_test_cv = y.iloc[train_idx], y.iloc[test_idx]
                
                mo_hinh_thu.fit(X_train_cv, y_train_cv)
                pred_cv = mo_hinh_thu.predict(X_test_cv)
                
                diem_r2.append(r2_score(y_test_cv, pred_cv))
                diem_mae.append(mean_absolute_error(y_test_cv, pred_cv))
                diem_rmse.append(np.sqrt(mean_squared_error(y_test_cv, pred_cv)))
                
                # Cập nhật thanh loading bar (20% -> 40% -> 60%...)
                thanh_cv.progress(int(((index_vong + 1) / 5.0) * 100))
                
            cot_cv1, cot_cv2, cot_cv3 = st.columns(3)
            cot_cv1.success(f"📉 MAE Trung bình:\n### {np.mean(diem_mae):.2f}")
            cot_cv2.success(f"📉 RMSE Trung bình:\n### {np.mean(diem_rmse):.2f}")
            cot_cv3.success(f"📈 R² Trung bình:\n### {np.mean(diem_r2):.4f}")
            
            # ========================================================
            # CHUẨN DATATHON: KHAI BÁO PIPELINE & EARLY STOPPING
            # ========================================================
            # 1. Gói gọn mô hình lõi vào một ML Pipeline chuẩn xác (Toi uu L1)
            mo_hinh_loi = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.05, objective='regression_l1', random_state=42, n_jobs=-1)
            pipeline = Pipeline([
                ('ai_core', mo_hinh_loi)
            ])
            
            # 2. Khởi chạy Pipeline với quy trình Dừng Sớm
            # Truyền tham số fit (early stopping) xuống khối 'ai_core' ở trong pipeline
            kwargs_fit = {'ai_core__eval_set': [(X_val, y_val)]}
            if hasattr(lgb, 'early_stopping'):
                kwargs_fit['ai_core__callbacks'] = [lgb.early_stopping(stopping_rounds=30, verbose=False)]
            else:
                kwargs_fit['ai_core__early_stopping_rounds'] = 30
                
            pipeline.fit(X_train, y_train, **kwargs_fit)
            
            # Ghi nhớ ống dẫn (Pipeline) vào bộ não của web để dự đoán ở Bước 4
            st.session_state.ai_model = pipeline
            st.success("✅ Tuyệt vời! Pipeline học máy đã được huấn luyện xong!")
            
            # ========================================================
            # CHUẨN DATATHON: SHAP & FEATURE IMPORTANCE (INTERPRETABILITY)
            # ========================================================
            st.markdown("---")
            st.subheader("💡 Giải mã Tư duy của AI (Minh bạch hoá)")
            
            cot_fi, cot_shap = st.columns(2)
            # Rút lõi AI ra khỏi Pipeline để vẽ biểu đồ não bộ
            mo_hinh_da_hoc = pipeline.named_steps['ai_core'] 
            
            with cot_fi:
                st.markdown("**1. Nhóm Yếu tố Chi phối Nhất (Feature Importance)**")
                diem_quan_trong = mo_hinh_da_hoc.feature_importances_
                df_fi = pd.DataFrame({'Yếu Tố': X.columns, 'Độ Quan Trọng': diem_quan_trong})
                df_fi = df_fi.sort_values(by='Độ Quan Trọng', ascending=True).tail(10)
                
                fig_fi = px.bar(df_fi, x='Độ Quan Trọng', y='Yếu Tố', orientation='h', title="Top 10 Phân loại ảnh hưởng")
                st.plotly_chart(fig_fi, use_container_width=True)
                
            with cot_shap:
                st.markdown("**2. Giải thích cơ chế nội tại (SHAP Summary Plot)**")
                try:
                    import shap
                    import matplotlib.pyplot as plt
                    
                    # Lấy 500 mẫu ngẫu nhiên để thuật toán SHAP chạy nhanh mượt, không đơ web
                    X_hien_thi = X_train.sample(min(len(X_train), 500), random_state=42)
                    
                    explainer = shap.TreeExplainer(mo_hinh_da_hoc)
                    shap_values = explainer.shap_values(X_hien_thi)
                    
                    # Cấp phát lưới ảnh cho matplotlib
                    fig, ax = plt.subplots(figsize=(6, 4))
                    shap.summary_plot(shap_values, X_hien_thi, show=False)
                    st.pyplot(fig)
                except Exception as e:
                    st.warning("⚠️ Đang thiết lập Môi trường. Hãy cài terminal `pip install shap matplotlib` trên máy tính để kích hoạt biểu đồ SHAP.")



    # ==========================================
    # MẢNH GHÉP 4: GIAO DIỆN DỰ ĐOÁN TƯƠNG LAI
    # ==========================================
    # Chỉ hiện phần này nếu AI đã được huấn luyện xong
    if st.session_state.ai_model is not None:
        st.markdown("---")
        st.header(f"🔮 Bước 4: Dự Đoán {muc_tieu} Cho Mốc Thời Gian Mới")
        
        tab_don, tab_hang_loat = st.tabs(["🧍 Dự đoán Từng ngày", "📦 Dự đoán Hàng loạt Datathon (1/1/2023 - 7/1/2024)"])
        
        with tab_don:
            # Bỏ cột mục tiêu ra vì nó là thứ mình đang muốn tìm
            cac_cot_nhap = [c for c in df.columns if c != muc_tieu]
            thong_tin_moi = {}
            
            st.write("Sếp hãy nhập thông số cho món hàng mới tinh dưới đây:")
            
            # Tự động sinh ra ô nhập liệu thành 3 cột cho gọn
            cols = st.columns(3)
            for i, cot in enumerate(cac_cot_nhap):
                with cols[i % 3]:
                    if cot in cot_chu:
                        # Cột chữ -> Chọn từ danh sách
                        thong_tin_moi[cot] = st.selectbox(f"{cot}:", df[cot].dropna().unique())
                    else:
                        # Cột số -> Nhập số
                        trung_binh = float(df[cot].mean())
                        thong_tin_moi[cot] = st.number_input(f"{cot}:", value=trung_binh)

            if st.button("✨ Đưa ra Dự đoán!"):
                # 1. Biến thông tin sếp nhập thành bảng DataFrame 1 dòng
                df_moi = pd.DataFrame([thong_tin_moi])
                
                # 2. Xử lý kiểu dữ liệu (Category) khớp hoàn toàn với lúc học
                if st.session_state.cat_cols:
                    for cot in st.session_state.cat_cols:
                        if cot in df_moi.columns:
                            tap_hop = df[cot].dropna().unique()
                            df_moi[cot] = pd.Categorical(df_moi[cot], categories=tap_hop)
                            
                # Nếu có cột Date, cần phân rã ra y hệt bước 1
                if 'Date' in df_moi.columns:
                    df_ngay_moi = pd.to_datetime(df_moi['Date'], errors='coerce')
                    df_moi['Month'] = df_ngay_moi.dt.month
                    df_moi['Year'] = df_ngay_moi.dt.year
                    df_moi['DayOfWeek'] = df_ngay_moi.dt.dayofweek
                            
                # Đồng bộ hóa với cột đã học (Nếu thiếu thì gán 0)
                df_thi_don = df_moi.copy()
                for cot_can in st.session_state.train_cols:
                    if cot_can not in df_thi_don.columns:
                        df_thi_don[cot_can] = 0
                
                # Sắp xếp và chỉ lấy định dạng cột chuẩn
                df_thi_don = df_thi_don[st.session_state.train_cols]
                
                # 3. Phán quyết!
                ket_qua = st.session_state.ai_model.predict(df_thi_don)
                
                st.balloons() # Thả bóng bay ăn mừng
                st.metric(label=f"💰 Mức {muc_tieu} dự kiến sẽ là:", value=f"{ket_qua[0]:,.2f}")
                
        with tab_hang_loat:
            st.markdown("**Nộp bài thi Datathon:** Hãy thả file CSV chứa danh sách ngày tháng tương lai (ví dụ `sample_submission.csv` kéo dài từ 2023-2024) vào đây.")
            file_thi = st.file_uploader("Kéo thả file cần dự đoán:", type=['csv'], key='file_thi_datathon')
            
            if file_thi is not None:
                df_thi = pd.read_csv(file_thi)
                
                # 1. Bóc tách thời gian y hệt như AI học
                if 'Date' in df_thi.columns:
                    df_ngay_thi = pd.to_datetime(df_thi['Date'], errors='coerce')
                    df_thi['Month'] = df_ngay_thi.dt.month
                    df_thi['Year'] = df_ngay_thi.dt.year
                    df_thi['DayOfWeek'] = df_ngay_thi.dt.dayofweek
                    
                # 2. Đảm bảo Category (Nếu có)
                if st.session_state.cat_cols:
                    for cot in st.session_state.cat_cols:
                        if cot in df_thi.columns:
                           tap_hop = df[cot].dropna().unique()
                           df_thi[cot] = pd.Categorical(df_thi[cot], categories=tap_hop)
                           
                # 3. Đồng bộ cột với model
                df_predict = df_thi.copy()
                for cot_can in st.session_state.train_cols:
                    if cot_can not in df_predict.columns:
                        df_predict[cot_can] = 0
                df_predict = df_predict[st.session_state.train_cols]
                
                with st.spinner(f"🚀 AI đang nhắm mắt dự đoán hàng loạt cho {len(df_predict)} ngày..."):
                    # 4. Dự đoán toàn bộ dataset
                    ket_qua_log = st.session_state.ai_model.predict(df_predict)
                    ket_qua_that = ket_qua_log # Khong con su dung ham expm1 vi khong dung log1p nua
                    df_ket_qua = df_thi.copy()
                    
                    # Tự động thay thế giá trị {muc_tieu} trong tệp
                    df_ket_qua[muc_tieu] = ket_qua_that
                    
                    # Dọn dẹp lại rác (Những cột Month, Year tạm bợ)
                    cot_moi_tao = ['Month', 'Year', 'DayOfWeek']
                    df_ket_qua = df_ket_qua.drop(columns=[c for c in cot_moi_tao if c in df_ket_qua.columns])
                    
                    st.success(f"✅ Đã chạy xong siêu tốc toàn bộ tập tương lai!")
                    st.dataframe(df_ket_qua.head(10), use_container_width=True)
                    
                    # Sinh ra nút Download định dạng gốc
                    csv_bytes = df_ket_qua.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Tải File prediction.csv",
                        data=csv_bytes,
                        file_name=f"prediction.csv",
                        mime='text/csv'
                    )