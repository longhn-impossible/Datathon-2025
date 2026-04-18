import pandas as pd
from sklearn.ensemble import RandomForestRegressor 
from sklearn.model_selection import train_test_split 

df = pd.read_csv("products.csv")

print("1. Tối ưu hóa Tiền xử lý (One-Hot Encoding)...")
df_encoded = pd.get_dummies(df[['cogs', 'category']], columns=['category'], drop_first=True)

# Gắn biến X, y
X = df_encoded 
y = df['price'] 

print("2. Cắt dữ liệu...")
X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(X, y, df, test_size=0.2, random_state=42)

print("3. Khởi động AI: Rừng Quyết Định (Random Forest)...")
# n_estimators=100: Trồng 100 cây quyết định
# n_jobs=-1: Lệnh siêu tối ưu! Yêu cầu máy tính dùng TẤT CẢ các nhân CPU để chạy song song, tốc độ bàn thờ!
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

print("4. Dự đoán và Xuất kết quả...")
predictions = model.predict(X_test)

result_df = df_test[['product_id', 'product_name', 'category', 'cogs']].copy()
result_df['Gia_Thuc_Te'] = y_test.values
result_df['Gia_Du_Doan_Pro'] = predictions
result_df['Gia_Du_Doan_Pro'] = result_df['Gia_Du_Doan_Pro'].clip(lower=0)

result_df = result_df.sort_values(by='product_id')
result_df.to_csv('du_doan_gia_ban_PRO.csv', index=False)
print("Hoàn thành xuất sắc! Em hãy mở file PRO ra để xem nhé.")