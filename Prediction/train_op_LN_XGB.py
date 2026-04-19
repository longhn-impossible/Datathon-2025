import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings

warnings.filterwarnings('ignore')

INPUT_DATA_PATH = r'e:\Datathon\sales.csv'
TEST_DATA_PATH  = r'e:\Datathon\sample_submission.csv'
OUTPUT_PATH     = r'e:\Datathon\Prediction\prediction2.csv'

# ==========================================
# BUOC 1: DOC DU LIEU
# ==========================================
def load_data():
    df_train = pd.read_csv(INPUT_DATA_PATH)
    df_test  = pd.read_csv(TEST_DATA_PATH)

    df_train['Date_obj'] = pd.to_datetime(df_train['Date'])
    df_test['Date_obj']  = pd.to_datetime(df_test['Date'])

    return df_train, df_test

# ==========================================
# BUOC 1.5: CLEAN DATA CO BAN
# ==========================================
def clean_data(df):
    df = df.copy().sort_values('Date').reset_index(drop=True)

    if 'Revenue' in df.columns:
        # Xu ly Local Outliers truoc (Noi suy)
        r_mean = df['Revenue'].rolling(window=7, center=True, min_periods=1).mean()
        mask_outlier = df['Revenue'] < (r_mean * 0.05)
        if mask_outlier.sum() > 0:
            df.loc[mask_outlier, 'Revenue'] = r_mean[mask_outlier]
            print(f'  [Clean] Outliers < 5% rmean: sua {mask_outlier.sum()} dong thap bat thuong')

    if 'Revenue' in df.columns and 'COGS' in df.columns:
        mask = df['COGS'] > df['Revenue']
        if mask.sum() > 0:
            med = (df.loc[~mask, 'COGS'] / df.loc[~mask, 'Revenue']).median()
            df.loc[mask, 'COGS'] = df.loc[mask, 'Revenue'] * med
            print(f'  [Clean] COGS > Revenue: sua {mask.sum()} dong (margin={med:.3f})')

    return df

# ==========================================
# BUOC 2: TAO DAC TRUNG THOI GIAN (CYCLIC)
# ==========================================
def make_time_features(df):
    d = df['Date_obj']
    
    # Time Index tuyen tinh tong quat cho Linear Regression
    df['Time_Idx']    = d.apply(lambda x: x.toordinal())

    # Dac trung thoi gian cho XGBoost
    df['Year']        = d.dt.year
    df['Month']       = d.dt.month
    df['Day']         = d.dt.day
    df['DayOfWeek']   = d.dt.dayofweek
    df['Quarter']     = d.dt.quarter
    df['WeekOfYear']  = d.dt.isocalendar().week.astype(int)
    df['Is_Weekend']  = d.dt.dayofweek.isin([5, 6]).astype(int)
    df['Is_MonthEnd'] = d.dt.is_month_end.astype(int)
    df['Is_MonthStart']= d.dt.is_month_start.astype(int)
    df['Is_QuarterEnd']= d.dt.is_quarter_end.astype(int)
    df['Is_YearEnd']  = d.dt.is_year_end.astype(int)

    # Chuoi Fourier Bac 1, 2, 3 de model bat gợn sóng
    df['Month_sin']   = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_cos']   = np.cos(2 * np.pi * df['Month'] / 12)
    df['DOW_sin']     = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
    df['DOW_cos']     = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
    
    df['Month_sin_2'] = np.sin(4 * np.pi * df['Month'] / 12)
    df['Month_cos_2'] = np.cos(4 * np.pi * df['Month'] / 12)
    df['DOW_sin_2']   = np.sin(4 * np.pi * df['DayOfWeek'] / 7)
    df['DOW_cos_2']   = np.cos(4 * np.pi * df['DayOfWeek'] / 7)
    
    df['Month_sin_3'] = np.sin(6 * np.pi * df['Month'] / 12)
    df['Month_cos_3'] = np.cos(6 * np.pi * df['Month'] / 12)
    df['DOW_sin_3']   = np.sin(6 * np.pi * df['DayOfWeek'] / 7)
    df['DOW_cos_3']   = np.cos(6 * np.pi * df['DayOfWeek'] / 7)

    df['Days_to_MonthEnd']   = (d + pd.offsets.MonthEnd(0) - d).dt.days
    df['Days_to_QuarterEnd'] = (d + pd.offsets.QuarterEnd(0) - d).dt.days

    return df


# ==========================================
# BUOC 3: TRAIN LINEAR REGRESSION + XGBOOST
# ==========================================
def train_predict_revenue_ln_xgb(df_train, df_test):
    print("  [INFO] Huan luyen dac biet: Linear Regression (Trend) + XGBoost (Residual Seasonality)")

    train_clean = df_train.dropna(subset=['Revenue']).copy()
    
    # --- PHẦN 1: LINEAR REGRESSION BẮT XU HƯỚNG MA CRO ---
    # Chỉ mượn dữ liệu từ năm 2019 trở đi để lấy khuynh hướng giảm gần nhất (tránh bong bóng 2012-2018)
    # Tuy nhiên, neu dùng toordinal() từ năm 2019-2022 thì sẽ ra 1 đường chéo xuốn. 
    # De khach quan hon ta dung tu nam 2019:
    trend_df = train_clean[train_clean['Year'] >= 2019].copy()
    
    X_trend_train = trend_df[['Time_Idx']]
    y_trend_train = trend_df['Revenue']
    lr_model = LinearRegression()
    lr_model.fit(X_trend_train, y_trend_train)
    
    # Dự đoán trend cho TOÀN BỘ dữ liệu train và test
    train_clean['Trend_Rev'] = lr_model.predict(train_clean[['Time_Idx']])
    df_test['Trend_Rev']     = lr_model.predict(df_test[['Time_Idx']])
    
    # Tinh phan du (Residual) = Doanh thu thuc te - Duong Trend
    train_clean['Residual'] = train_clean['Revenue'] - train_clean['Trend_Rev']

    # --- PHẦN 2: XGBOOST BẮT MÙA VỤ VÀ SÓNG CỤC BỘ ---
    drop = ['Date', 'Date_obj', 'Revenue', 'COGS', 'Profit', 'Time_Idx', 'Trend_Rev', 'Residual']
    feat_cols = [c for c in train_clean.columns if c not in drop]
    
    X = train_clean[feat_cols]
    y = train_clean['Residual']  # Học cách đoán Phần dư!
    
    # Time Weighting: Ưu tiên dữ liệu gần
    min_year = train_clean['Year'].min()
    w = train_clean['Year'] - min_year + 1

    # Chia Val 180 ngày để Early Stopping
    val_size = 180
    X_tr, y_tr, w_tr = X.iloc[:-val_size], y.iloc[:-val_size], w.iloc[:-val_size]
    X_va, y_va, w_va = X.iloc[-val_size:], y.iloc[-val_size:], w.iloc[-val_size:]

    print("  [Ensemble] Khoi tao XGBoost hoc Residual...")
    m_xgb = xgb.XGBRegressor(
        objective='reg:absoluteerror', eval_metric='mae', n_estimators=2500,
        learning_rate=0.03, max_depth=6, random_state=42, n_jobs=-1,
        early_stopping_rounds=200
    )
    m_xgb.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=[(X_va, y_va)], verbose=False)

    print(f'  --> [Val MAE Residual]: {mean_absolute_error(y_va, m_xgb.predict(X_va)):,.0f}')

    # --- PHẦN 3: GỘP KẾT QUẢ CHO TEST SET ---
    X_test = df_test[feat_cols]
    xgb_residual_preds = m_xgb.predict(X_test)
    
    # Du doan cuoi cung = Baseline Trend + Bien dong Residual
    final_preds = df_test['Trend_Rev'] + xgb_residual_preds
    
    # Tránh trường hợp tiền bị âm do Residual dự đoán quá lố
    final_preds = np.maximum(final_preds, 0)
    
    return final_preds

# ==========================================
# MAIN FUNCTION
# ==========================================
def main():
    print("==============================================================")
    print("  DATATHON REVENUE v6 - Linear Regression Trend + XGBoost")
    print("==============================================================\n")

    print('[1] Doc du lieu...')
    df_train, df_test = load_data()
    print(f'  Train: {len(df_train):,} | Test: {len(df_test):,}')

    print('\n[1.5] Clean du lieu...')
    df_train = clean_data(df_train)

    print('\n[2] Time features...')
    df_train = make_time_features(df_train)
    df_test  = make_time_features(df_test)

    # COGS ratio
    cogs_ratio = (df_train['COGS'] / df_train['Revenue']).median()
    print(f'  [INFO] COGS/Revenue ratio de tao COGS tu Revenue: {cogs_ratio:.4f}')

    # Train
    print('\n[3] Huan luyen...')
    rev_preds = train_predict_revenue_ln_xgb(df_train, df_test)
    cogs_preds = rev_preds * cogs_ratio

    # Output
    print('\n[4] Luu file...')
    df_out = df_test[['Date']].copy()
    df_out['Revenue'] = rev_preds
    df_out['COGS'] = cogs_preds
    
    # Sắp xếp lại đúng thứ tự Date, Revenue, COGS 
    df_out = df_out[['Date', 'Revenue', 'COGS']]
    df_out.to_csv(OUTPUT_PATH, index=False)

    print(f'  Luu: {OUTPUT_PATH}')
    print(f"  Rev: min={df_out['Revenue'].min():,.0f} | max={df_out['Revenue'].max():,.0f} | mean={df_out['Revenue'].mean():,.0f}")

    # Danh gia tuong doi vs sample_submission (Dummy check)
    df_sample = pd.read_csv(TEST_DATA_PATH)
    if 'Revenue' in df_sample.columns:
        mae_sample = mean_absolute_error(df_sample['Revenue'], rev_preds)
        print(f'\n  MAE vs sample_submission: {mae_sample:,.0f}')

    print('\n[HOAN TAT]')

if __name__ == '__main__':
    main()
