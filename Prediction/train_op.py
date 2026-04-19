"""
DATATHON - REVENUE PREDICTION v3 (FIXED)
==========================================
Root cause fixes:
1. Removed IQR clipping -> model can now learn end-of-month spikes (8-9M)
2. Fixed NaN lag cascade: test lag features look up ACTUAL training dates
3. Added historical seasonality features (avg by Month, DayOfWeek, DayOfYear)
   so test rows always have valid features, not NaN
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, TimeSeriesSplit
import warnings

warnings.filterwarnings('ignore')

TRAIN_PATH  = 'e:/Datathon/sales.csv'
TEST_PATH   = 'e:/Datathon/sample_submission.csv'
OUTPUT_PATH = 'e:/Datathon/Prediction/prediction.csv'

N_FOLDS       = 5
N_ESTIMATORS  = 3000
LEARNING_RATE = 0.02

# Horizon cua test xap xi 548 ngay (1 nam rưỡi)
# Cac lag < 548 se bi NaN rat nhieu o tap test neu dung direct forecasting.
# Vi vay, ta huan luyen THEO MO HINH KHONG DUNG LAG (giong tro_li.py) nhung manh hon
# nho su dung Tinh toan Thong ke Lich su (Target Encoding).


# ==========================================
# BUOC 1: LAM SACH (CHI FIX LOI DU LIEU THAT)
# ==========================================
def clean_data(df):
    """
    Chi sua loi du lieu that su:
    - COGS > Revenue: bat thuong kinh te -> sua bang median margin
    - Local Outliers: Doanh thu = 0 hoac < 5% trung binh 7 ngay -> Smoothen (Interpolation)
    """
    df = df.copy().sort_values('Date').reset_index(drop=True)

    if 'Revenue' in df.columns:
        # Xu ly Local Outliers truoc (Noi suy - Interpolation)
        # Dung min_periods=1 va center=True de khong bi loi o 2 dau
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
# BUOC 2: TIME FEATURES
# ==========================================
def make_time_features(df):
    df = df.copy()
    d = pd.to_datetime(df['Date'], errors='coerce')
    df['Date_obj'] = d

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

    # Sin/cos bat quy luat tuan hoan (Bac 1)
    df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
    df['DOW_sin']   = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
    df['DOW_cos']   = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
    
    # 🌟 GIAI PHAP: Chuoi Fourier Bac 2 va Bac 3 🌟
    # Giup AI ôm sát được những quy luật gợn sóng đan xen phức tạp
    df['Month_sin_2'] = np.sin(4 * np.pi * df['Month'] / 12)
    df['Month_cos_2'] = np.cos(4 * np.pi * df['Month'] / 12)
    df['DOW_sin_2']   = np.sin(4 * np.pi * df['DayOfWeek'] / 7)
    df['DOW_cos_2']   = np.cos(4 * np.pi * df['DayOfWeek'] / 7)
    
    df['Month_sin_3'] = np.sin(6 * np.pi * df['Month'] / 12)
    df['Month_cos_3'] = np.cos(6 * np.pi * df['Month'] / 12)
    df['DOW_sin_3']   = np.sin(6 * np.pi * df['DayOfWeek'] / 7)
    df['DOW_cos_3']   = np.cos(6 * np.pi * df['DayOfWeek'] / 7)

    # Ngay con lai den cuoi thang/quy
    df['Days_to_MonthEnd']   = (d + pd.offsets.MonthEnd(0) - d).dt.days
    df['Days_to_QuarterEnd'] = (d + pd.offsets.QuarterEnd(0) - d).dt.days

    return df





# ==========================================
# BUOC 3: TRAIN + PREDICT MANG TINH DE QUY (RECURSIVE)
# ==========================================
def train_predict_revenue(df_train, df_test):
    print("  [INFO] Bat dau Huan luyen Ensemble + Du doan De quy (Recursive Forecasting)...")

    # 1. TẠO LAG FEATURES CHO TẬP TRAIN
    # Cần phải sort dữ liệu theo thời gian chuẩn trước khi tạo Lags
    df_train = df_train.sort_values('Date').reset_index(drop=True)
    df_train['Lag_1']     = df_train['Revenue'].shift(1)
    df_train['Lag_2']     = df_train['Revenue'].shift(2)
    df_train['Lag_7']     = df_train['Revenue'].shift(7)
    df_train['Rolling_7'] = df_train['Revenue'].shift(1).rolling(7).mean()

    # Drop các dông chưa đủ độ trễ (7 ngày đầu tiên)
    train_clean = df_train.dropna(subset=['Revenue', 'Lag_7', 'Rolling_7']).copy()
    
    # Dinh nghia cot bo di
    drop = ['Date', 'Date_obj', 'Revenue', 'COGS', 'Profit']
    feat_cols = [c for c in train_clean.columns if c not in drop]
    
    # 2. CHIA TẬP TRAIN VÀ VAL (LAST 180 DAYS) ĐỂ LÀM EARLY STOPPING
    min_year = train_clean['Year'].min()
    weights  = train_clean['Year'] - min_year + 1
    
    X = train_clean[feat_cols]
    y = train_clean['Revenue']
    w = weights
    
    val_size = 180
    X_tr, y_tr, w_tr = X.iloc[:-val_size], y.iloc[:-val_size], w.iloc[:-val_size]
    X_va, y_va, w_va = X.iloc[-val_size:], y.iloc[-val_size:], w.iloc[-val_size:]

    # 3. KHOI TAO VA HUAN LUYEN 3 MO HINH (ENSEMBLE)
    print("  [Ensemble] 1. Khoi tao LightGBM...")
    m_lgb = lgb.LGBMRegressor(
        objective='regression_l1', metric='mae', n_estimators=2000,
        learning_rate=0.03, num_leaves=63, random_state=42, n_jobs=-1, verbose=-1
    )
    m_lgb.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=[(X_va, y_va)], eval_sample_weight=[w_va],
              callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(-1)])

    print("  [Ensemble] 2. Khoi tao XGBoost...")
    m_xgb = xgb.XGBRegressor(
        objective='reg:absoluteerror', eval_metric='mae', n_estimators=2000,
        learning_rate=0.03, max_depth=6, random_state=42, n_jobs=-1,
        early_stopping_rounds=150
    )
    m_xgb.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=[(X_va, y_va)], verbose=False)

    print("  [Ensemble] 3. Khoi tao CatBoost...")
    m_cat = CatBoostRegressor(
        objective='MAE', eval_metric='MAE', iterations=2000,
        learning_rate=0.03, depth=6, random_seed=42, verbose=False
    )
    m_cat.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=150)

    # Kiem tra MAE tren tap Val
    val_preds = (
        0.4 * m_lgb.predict(X_va) + 
        0.4 * m_cat.predict(X_va) + 
        0.2 * m_xgb.predict(X_va)
    )
    val_mae = mean_absolute_error(y_va, val_preds)
    print(f"  --> [Validation MAE] (Ensemble): {val_mae:,.0f}")

    # 4. DU DOAN DE QUY TRÊN TAP TEST (RECURSIVE INFERENCE)
    print("  [Recursive] Dang chay vong lap du doan cuon chieu tung ngay cho Test Set...")
    # Ghep noi tap Train (vua du de lay lich su 7 ngay cuoi) va tap Test
    hist_df = train_clean[['Date', 'Revenue'] + feat_cols].tail(10).copy()
    test_df = df_test[['Date'] + [c for c in feat_cols if not c.startswith('Lag') and c != 'Rolling_7']].copy()
    test_df['Revenue'] = np.nan
    
    # Ghep thanh 1 bang chung
    full_df = pd.concat([hist_df, test_df], axis=0, ignore_index=True)
    start_idx = len(hist_df)
    
    # Loop de quy tung dong
    import sys
    for i in range(start_idx, len(full_df)):
        # Tinh toan Lags tu lich su ngay truoc do
        full_df.loc[i, 'Lag_1'] = full_df.loc[i-1, 'Revenue']
        full_df.loc[i, 'Lag_2'] = full_df.loc[i-2, 'Revenue']
        full_df.loc[i, 'Lag_7'] = full_df.loc[i-7, 'Revenue']
        full_df.loc[i, 'Rolling_7'] = full_df.loc[i-7:i-1, 'Revenue'].mean()
        
        # Lay Dong hien tai (chi chua features)
        row_feat = full_df.loc[[i], feat_cols]
        
        # Du doan (Predict)
        p_lgb = m_lgb.predict(row_feat)[0]
        p_xgb = m_xgb.predict(row_feat)[0]
        p_cat = m_cat.predict(row_feat)[0]
        
        # Hợp thể (Blend)
        p_final = (0.4 * p_lgb) + (0.4 * p_cat) + (0.2 * p_xgb)
        
        # Dien nguoc ket qua vao bang de phuc vu vong lap sau
        full_df.loc[i, 'Revenue'] = p_final
        
        # In tien do
        if (i - start_idx + 1) % 50 == 0:
            print(f"      .. da hoan thanh {i - start_idx + 1}/{len(test_df)} ngay.")

    # Trich xuat phan Test da duoc du doan
    test_preds = full_df.loc[start_idx:, 'Revenue'].values
    return test_preds


# ==========================================
# MAIN
# ==========================================
def main():
    print('=' * 62)
    print('  DATATHON REVENUE v3 - NaN Lag Fixed + IQR Removed')
    print('=' * 62)

    # Doc
    print('\n[1] Doc du lieu...')
    df_train = pd.read_csv(TRAIN_PATH)
    df_test  = pd.read_csv(TEST_PATH)
    print(f'  Train: {len(df_train):,} | Test: {len(df_test):,}')

    # Clean (chi fix COGS > Revenue)
    print('\n[1.5] Clean du lieu...')
    df_train = clean_data(df_train)

    # Time features
    print('\n[2] Time features...')
    df_train = make_time_features(df_train)
    df_test  = make_time_features(df_test)

    # COGS ratio
    cogs_ratio = (df_train['COGS'] / df_train['Revenue']).median()
    print(f'  [INFO] COGS/Revenue ratio de tao COGS tu Revenue: {cogs_ratio:.4f}')

    n_feat = len([c for c in df_train.columns
                  if c not in ['Date','Date_obj','Revenue','COGS','Profit']])
    print(f'  Tong features: {n_feat}')

    # Train
    print('\n[3] Huan luyen...')
    rev_preds  = train_predict_revenue(df_train, df_test)
    cogs_preds = rev_preds * cogs_ratio

    # Output
    print('\n[4] Luu file...')
    df_out = df_test[['Date']].copy()
    df_out['Revenue'] = rev_preds
    df_out['COGS'] = cogs_preds
    
    # Sắp xếp lại đúng thứ tự Date, Revenue, COGS như Cấu trúc đề thi yêu cầu
    df_out = df_out[['Date', 'Revenue', 'COGS']]
    df_out.to_csv(OUTPUT_PATH, index=False)

    print(f'  Luu: {OUTPUT_PATH}')
    print(f'  Rev: min={rev_preds.min():,.0f} | max={rev_preds.max():,.0f} | mean={rev_preds.mean():,.0f}')

    # So sanh nhanh vs sample_submission
    try:
        sample = pd.read_csv(TEST_PATH)
        mae_vs_sample = np.abs(rev_preds - sample['Revenue'].values).mean()
        print(f'\n  MAE vs sample_submission: {mae_vs_sample:,.0f}')
        print(f'  (tro_li dat ~407,803 | train_op cu dat ~1,160,325)')
    except Exception:
        pass

    print('\n[HOAN TAT]')


if __name__ == '__main__':
    main()
