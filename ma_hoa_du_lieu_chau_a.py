import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import scipy.sparse as sp

# ==========================
# 1. ĐỌC VÀ XỬ LÝ DỮ LIỆU
# ==========================

df = pd.read_csv("Du_lieu_Chau_A.csv", encoding="utf-8")
df.columns = df.columns.str.strip()  # bỏ khoảng trắng dư

# Các cột đúng theo file của bạn:
# ['Giới tính', 'Tuổi', 'Chiều cao (m)', 'Cân nặng (kg)', ..., 'Mức độ béo phì']

# Xử lý cột số
df["Chiều cao (m)"] = pd.to_numeric(df["Chiều cao (m)"], errors="coerce")
df["Cân nặng (kg)"] = pd.to_numeric(df["Cân nặng (kg)"], errors="coerce")
df["Tuổi"] = pd.to_numeric(df["Tuổi"], errors="coerce")

# Bỏ các dòng thiếu dữ liệu quan trọng
df = df.dropna(subset=["Tuổi", "Chiều cao (m)", "Cân nặng (kg)"])

# Thêm BMI (tuỳ bạn có dùng sau hay không)
df["BMI"] = df["Cân nặng (kg)"] / (df["Chiều cao (m)"] ** 2)

# ==========================
# 2. TÁCH X, y
# ==========================

target_col = "Mức độ béo phì"
X = df.drop(columns=[target_col])
y = df[target_col]

numeric_features = ["Tuổi", "Chiều cao (m)", "Cân nặng (kg)", "BMI"]
categorical_features = [c for c in X.columns if c not in numeric_features]

# ==========================
# 3. COLUMNTRANSFORMER + ONEHOTENCODER
# ==========================

# KHÔNG dùng tham số 'sparse' để tránh lỗi version
ohe = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numeric_features),
        ("cat", ohe, categorical_features),
    ]
)

# Fit và transform toàn bộ X
X_encoded_sparse = preprocessor.fit_transform(X)

# Nếu kết quả là sparse matrix thì chuyển sang numpy array
if sp.issparse(X_encoded_sparse):
    X_encoded_array = X_encoded_sparse.toarray()
else:
    X_encoded_array = X_encoded_sparse

# Lấy tên cột sau khi mã hóa
cat_encoder = preprocessor.named_transformers_["cat"]

try:
    # sklearn mới
    cat_feature_names = cat_encoder.get_feature_names_out(categorical_features)
except AttributeError:
    # sklearn cũ
    cat_feature_names = cat_encoder.get_feature_names(categorical_features)

all_feature_names = numeric_features + list(cat_feature_names)

# Đưa về DataFrame
X_encoded = pd.DataFrame(X_encoded_array, columns=all_feature_names)

# ==========================
# 4. MÃ HÓA NHÃN y
# ==========================

le = LabelEncoder()
y_encoded = le.fit_transform(y)

y_df = pd.DataFrame({
    "label_text": y,
    "label_encoded": y_encoded
})

# ==========================
# 5. LƯU RA FILE CSV
# ==========================

X_encoded.to_csv("X_ChauA_encoded.csv", index=False, encoding="utf-8-sig")
y_df.to_csv("y_ChauA_encoded.csv", index=False, encoding="utf-8-sig")

print("Đã lưu:")
print("- X_ChauA_encoded.csv (đặc trưng đã mã hóa)")
print("- y_ChauA_encoded.csv (nhãn đã mã hóa)")
